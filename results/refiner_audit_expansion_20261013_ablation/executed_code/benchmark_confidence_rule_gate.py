"""Development-only matched-window replay of rules and ASR score policies.

Frozen consumed ASR observations feed the real existing refinement chain.
There are no audio gold labels and no production gate changes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
import json
from math import isfinite
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.refinement_gate import RefinementGate, RefinementGateMode
from system.zh_itn_bridge import ZhITNNormalizer
from experiments.scripts.benchmark_fixed_gate_pipeline import ReplayASR, receive, sha, write

POLICIES = ("all_refine", "native", "confidence_supplement", "confidence_fusion")


def valid_profile(profile):
    confidence = profile["confidence"]
    return (confidence.get("coverage") == 1. and confidence.get("score") is not None
            and all(isinstance(confidence.get(key), (int, float))
                    and not isinstance(confidence[key], bool)
                    and isfinite(confidence[key]) and 0 <= confidence[key] <= 1
                    for key in ("score", "min_fragment_score")))


class PilotConfidenceRuleGate(RefinementGate):
    """Uncalibrated, explicit experiment policy; never enabled by the app."""
    def __init__(self, mode, *, profiles, policy, normalizer, low=.8,
                 high_window=.98, high_fragment=.95, **kwargs):
        super().__init__(mode, **kwargs)
        self.policy, self.profiles, self.normalizer = policy, {}, normalizer
        self.low, self.high_window, self.high_fragment = low, high_window, high_fragment
        self.records = []
        for profile in profiles:
            text = profile["text"].strip()
            if text in self.profiles:
                raise ValueError("Repeated source text cannot identify a unique score window")
            self.profiles[text] = profile

    def config_dict(self):
        return {**super().config_dict(), "pilot_policy": self.policy,
                "score_thresholds": dict(low_fragment=self.low, high_window=self.high_window,
                                         high_fragment=self.high_fragment),
                "calibrated": False, "production_eligible": False}

    def decide(self, text, **kwargs):
        native = super().decide(text, **kwargs)
        profile = self.profiles.get(text.strip())
        result, reason = native, "native_preserved"
        signals=set(native.cleanup_signals)-{"numeric_normalization"}
        weak_structure=(native.action=="refine" and "entity_hint_present" not in native.reasons
            and not kwargs.get("same_as_last_refined",False)
            and ((native.reasons==("sentence_incomplete",) and not signals)
                 or signals=={"isolated_single_character_clause"}))
        if self.policy=="weak_structure_rule_only" and profile is not None and weak_structure:
            reason="experimental_weak_structure_rule_only_keep"
            result=replace(native,should_refine=False,action="keep",reasons=(reason,))
        elif profile is None or not valid_profile(profile):
            reason = "score_unavailable_or_incomplete"
        elif self.policy in {"confidence_supplement", "confidence_fusion"}:
            confidence = profile["confidence"]
            signals = set(native.cleanup_signals) - {"numeric_normalization"}
            hints_present = "entity_hint_present" in native.reasons
            low_plain_fragments = [fragment for fragment in profile["fragments"]
                if fragment["confidence"]["coverage"] == 1.
                and fragment["confidence"]["score"] is not None
                and fragment["confidence"]["score"] < self.low
                and not self.normalizer.normalize(fragment["text"]).changes]
            # A numeric-only window may also contain a non-numeric uncertain
            # clause. Keep numeric/lexical integrity guards in the real chain.
            if (native.action == "keep" and low_plain_fragments and not hints_present
                    and native.reasons[0] in {"numeric_handled_by_rule_pack", "stable_clean_segment",
                                               "confidence_unverified_safe_keep"}):
                reason = "experimental_low_nonnumeric_fragment_review"
                result = replace(native, should_refine=True, action="refine", reasons=(reason,),
                                 asr_confidence=confidence["score"], calibrated=False, covers_segment=True)
            elif (self.policy == "confidence_fusion" and native.action == "refine"
                    and not hints_present and not kwargs.get("same_as_last_refined", False)
                    and confidence["score"] >= self.high_window
                    and confidence["min_fragment_score"] >= self.high_fragment
                    and ((native.reasons == ("sentence_incomplete",) and not signals)
                         or signals == {"isolated_single_character_clause"})):
                reason = "experimental_high_score_weak_structure_keep"
                result = replace(native, should_refine=False, action="keep", reasons=(reason,),
                                 asr_confidence=confidence["score"], calibrated=False, covers_segment=True)
        if profile is not None:
            self.records.append(dict(source=text.strip(), window_index=profile["index"],
                is_final=kwargs.get("is_final", False), native=native.public_dict(), candidate=result.public_dict(),
                score=profile["confidence"]["score"], min_fragment_score=profile["confidence"].get("min_fragment_score"),
                coverage=profile["confidence"]["coverage"], changed=native.action != result.action,
                reason=reason, calibrated=False, production_eligible=False))
        return result


def load_cases(directory):
    summary = json.loads((directory / "summary.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    cases = []
    for item in summary["cases"]:
        final_path = directory / f"case_{item['case']:02d}" / "final.json"
        if sha(final_path) != verification["output_sha256"][str(final_path.relative_to(directory))]:
            raise ValueError("Frozen confidence observations changed")
        final = json.loads(final_path.read_text())
        profiles = []
        for window in final["windows"]:
            profiles.append({**window, "fragments": [final["fragments"][index] for index in window["fragment_indices"]]})
        cases.append(dict(seed_id=f"confidence_audio_case_{item['case']:02d}", case=item["case"],
            raw_text=final["text"], profiles=profiles, audio_provenance=item,
            source_final_sha256=sha(final_path), audio_gold=False, independent_acceptance=False))
    return cases


def text_differences(before, after):
    return [dict(operation=tag, before=before[a:b], after=after[c:d],
                 before_context=before[max(0,a-16):b+16], after_context=after[max(0,c-16):d+16])
            for tag,a,b,c,d in SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if tag != "equal"]


def run(args):
    import torch
    from fastapi.testclient import TestClient
    from threadpoolctl import threadpool_limits
    from system import web_app
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility

    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); threadpool_limits(limits=2)
    cases = load_cases(args.sources)
    write(output / "sources.json", cases)
    normalizer = ZhITNNormalizer(args.zh_itn)
    policies=tuple(args.policies)
    configuration = dict(scope="frozen_audio_ASR_replay_development_diagnostic_not_audio_acceptance",
        policies=policies, repetitions=args.repetitions, device=args.device,
        thresholds=dict(low_fragment=.8, high_window=.98, high_fragment=.95),
        thresholds_frozen_before_refiner_outputs=True, policy_design_informed_by_consumed_ASR_diagnostics=True,
        thresholds_not_calibrated=True, no_threshold_search_or_training=True,
        sources=str(args.sources), confidence_summary_sha256=sha(args.sources / "summary.json"),
        refiner_weights_sha256={file.name: sha(file) for file in args.model.glob("*.safetensors")},
        numeric_backend="zh-itn", zh_itn_sha256=sha(args.zh_itn), entity_db=None,
        learned_gate="off", final_refinement_mode="off", fresh_session_cache=True,
        no_production_services_changed=True, no_new_ASR_calls=True, no_audio_gold=True,
        full_file_time_not_measured=True, original_probe_does_not_reproduce_live_VAD=True,
        input_cases=len(cases), input_windows=sum(len(case["profiles"]) for case in cases),
        references_not_passed_to_models_or_gate=True)
    write(output / "configuration.json", configuration)
    (output / "executed_code").mkdir()
    for file in [Path(__file__), ROOT / "system/web_app.py", ROOT / "system/refinement_gate.py",
                 ROOT / "system/window_refinement.py", ROOT / "system/refinement_guard.py",
                 ROOT / "system/live_qwen_refiner.py"]:
        (output / "executed_code" / file.name).write_bytes(file.read_bytes())
    print("LOADING_REFINER", flush=True)
    refiner = TransformersRefiner(args.model, args.device, 256)
    plausibility = RepetitionPlausibility(args.device)
    if not plausibility._ensure_model():
        raise RuntimeError("Existing local repetition model unavailable")
    refiner.refine_with_numeric_rules("请确认会议地点。")
    plausibility.decide("请确认确认会议地点。",1,5,"确认")
    print("REFINER_READY", flush=True)
    results = []
    with patch.object(web_app,"TransformersRefiner",return_value=refiner), \
         patch.object(web_app,"RepetitionPlausibility",return_value=plausibility), \
         patch.object(web_app,"STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS",0):
        for repetition in range(args.repetitions):
            for index, case in enumerate(cases):
                order = list(policies[index % len(policies):] + policies[:index % len(policies)])
                if repetition % 2:
                    order.reverse()
                for policy in order:
                    trace = output / f"{index:02d}.{repetition}.{policy}.jsonl"
                    captured = []
                    def gate_factory(mode, **kwargs):
                        gate = PilotConfidenceRuleGate(mode,profiles=case["profiles"],policy=policy,
                                                       normalizer=normalizer,**kwargs)
                        captured.append(gate)
                        return gate
                    with patch.object(web_app,"RefinementGate",side_effect=gate_factory), \
                         patch.object(web_app,"_stream_request",ReplayASR(case["raw_text"])):
                        app = web_app.create_app(args.model,args.device,"http://replayed-asr","Chinese",256,trace,
                            refinement_gate_mode="off" if policy == "all_refine" else "tri_state",
                            learned_window_gate_mode="off",final_refinement_mode="off",zh_itn_library=args.zh_itn)
                        with TestClient(app) as client:
                            started = time.perf_counter()
                            with client.websocket_connect("/ws/stream?mode=streaming") as ws:
                                ready = receive(ws,"ready")
                                ws.send_bytes(b"replay-only")
                                receive(ws,"update")
                                ws.send_json({"event":"finish"})
                                final = receive(ws,"final")
                            elapsed = time.perf_counter() - started
                    with trace.with_suffix(".stream_trace.jsonl").open() as stream:
                        rows = [json.loads(line) for line in stream]
                    segments = [segment for row in rows if row["event"] == "window_result"
                                for segment in row["segments"] if segment["window_kind"] != "pending_tail"]
                    windows = [(segment["source"],segment["window_kind"],segment["window_chunk_count"])
                               for segment in segments]
                    expected = [profile["text"] for profile in case["profiles"]]
                    if list(dict.fromkeys(segment["source"] for segment in segments)) != expected:
                        raise RuntimeError("Actual pipeline windows differ from frozen confidence windows")
                    stats = final["refiner_session_stats"]
                    if not stats["stats_complete"] or stats["failed_call_count"]:
                        raise RuntimeError("Incomplete or failed real model calls")
                    result = dict(index=index,case=case["case"],seed_id=case["seed_id"],repetition=repetition,
                        policy=policy,raw_text=case["raw_text"],final_text=final["clean_text"],
                        windows=windows,window_pipeline=ready["window_pipeline"],elapsed_seconds=elapsed,
                        stats=stats,final=final,gate_audit=[item for gate in captured for item in gate.records],
                        window_segments=segments)
                    results.append(result)
                    with (output / "sessions.jsonl").open("a") as stream:
                        stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                    write(output / "progress.json",dict(completed_sessions=len(results),expected_sessions=len(cases)*args.repetitions*len(policies)))
                    print("SESSION_DONE",json.dumps(dict(case=index,repetition=repetition,policy=policy,
                        calls=stats["call_count"],seconds=round(elapsed,3),
                        changed_gate_checks=sum(r["changed"] for r in result["gate_audit"])),ensure_ascii=False),flush=True)
    groups = defaultdict(dict)
    for result in results:
        groups[(result["index"],result["repetition"])][result["policy"]] = result
    if not all(len(group)==len(policies) and all(row["windows"]==group["native"]["windows"]
               and row["window_pipeline"]==group["native"]["window_pipeline"] for row in group.values())
               for group in groups.values()):
        raise RuntimeError("Window/cache pipeline mismatch across policies")
    policies = {}
    differences = []
    for policy in policies:
        selected = [row for row in results if row["policy"]==policy]
        policies[policy] = dict(sessions=len(selected), calls=sum(row["stats"]["call_count"] for row in selected),
            refiner_wall_ms=sum(row["stats"]["completed_wall_latency_ms"] for row in selected),
            chain_total_seconds=sum(row["elapsed_seconds"] for row in selected),
            chain_median_seconds=statistics.median(row["elapsed_seconds"] for row in selected),
            identical_final_pairs_vs_native=sum(group[policy]["final_text"]==group["native"]["final_text"] for group in groups.values()),
            gate_change_reasons=dict(Counter(r["reason"] for row in selected for r in row["gate_audit"] if r["changed"])))
        for (index,repetition),group in groups.items():
            if policy != "native" and group[policy]["final_text"] != group["native"]["final_text"]:
                differences.append(dict(case=index,repetition=repetition,policy=policy,
                    changes=text_differences(group["native"]["final_text"],group[policy]["final_text"]),
                    audio_provenance=cases[index]["audio_provenance"],
                    correctness="unverified",audio_checked=False,human_audio_gold=False))
    write(output / "output_differences.json",differences)
    summary = dict(scope=configuration["scope"],matched_windows=True,matched_original_confidence_partition=True,
        cases=len(cases),sessions=len(results),pairs_per_policy=len(groups),policies=policies,
        calibrated=False,production_eligible=False,quality_improvement_not_established=True,
        whole_audio_latency_not_measured=True,output_difference_records=len(differences))
    write(output / "summary.json",summary)
    print("DONE",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sources",type=Path,default=ROOT / "results/asr_window_confidence_probe_20261013_v2")
    parser.add_argument("--model",type=Path,default=Path("/home/aim0/data/models/ASR/AgenticASR-Refiner"))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--zh-itn",type=Path,default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    parser.add_argument("--repetitions",type=int,default=2)
    parser.add_argument("--policies",nargs="+",choices=(*POLICIES,"weak_structure_rule_only"),default=list(POLICIES))
    args=parser.parse_args()
    if args.repetitions<1:
        parser.error("repetitions must be positive")
    if "native" not in args.policies or len(set(args.policies))!=len(args.policies):
        parser.error("Policies must include native and contain no duplicates")
    run(args)
