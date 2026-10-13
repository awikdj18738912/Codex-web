"""Freeze ASR decisions, run paired Web refinement, and export review sheets.

All measurements are isolated offline decision experiments. Reference distance
is a review aid, never a human benefit label or an online latency measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
import types
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/gate_pilot_20261009"
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")
DEFAULT_DB = ROOT.parent / "AgenticASR_entity_e5bedb3/data/entities.db"
DEFAULT_SOURCES = [
    "results/entity_library_20261003/asr.jsonl",
    "results/entity_context_20261006/fresh_asr.jsonl",
    "results/medical_entity_20261003/asr.jsonl",
    "results/entity_reference_vocab_20261007/asr.jsonl",
    "results/entity_chinese_rules_20261008/asr.jsonl",
    "results/entity_context_20261006/mmed_dev_asr.jsonl",
]
DEFAULT_LONG_RESULTS = [
    "results/web/offline_streaming_audio4_cache_reuse_final.json",
    "results/web/offline_streaming_audio5_repetition_review_aggregate_final.json",
]


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_row(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def normalized(text):
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in folded if unicodedata.category(c)[0] not in {"P", "S", "Z"} and not c.isspace())


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def balanced_sample(pool, count, seed):
    """Sampling uses input-side strata, never candidate reference improvement."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in pool:
        groups[item["selection_stratum"]].append(item)
    queues = []
    for name in sorted(groups):
        rng.shuffle(groups[name])
        queues.append(deque(groups[name]))
    selected = []
    while len(selected) < count and any(queues):
        for q in queues:
            if q and len(selected) < count:
                selected.append(q.popleft())
    if len(selected) != count:
        raise ValueError(f"Requested {count} cases, but only {len(selected)} unique eligible cases are available")
    rng.shuffle(selected)
    return selected


def closure_value(endpoint, name):
    """Locate a named captured Web function; fail if its internal contract moves."""
    pending, seen = [endpoint], set()
    while pending:
        fn = pending.pop()
        if id(fn) in seen or not isinstance(fn, types.FunctionType):
            continue
        seen.add(id(fn))
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        if name in cells:
            return cells[name].cell_contents
        for cell in cells.values():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, types.FunctionType):
                pending.append(value)
    raise RuntimeError(f"Web closure no longer exposes {name}; update the experiment adapter")


def load_system(output):
    frozen = output / "frozen"
    sys.path.insert(0, str(frozen))
    sys.path.insert(1, str(ROOT))
    for name in tuple(sys.modules):
        if name == "system" or name.startswith("system."):
            del sys.modules[name]
    web = importlib.import_module("system.web_app")
    if not Path(web.__file__).is_relative_to(frozen):
        raise RuntimeError("Experiment did not import the frozen system")
    return web


def routing_factory(web):
    original = web.RefinementGate

    class PairedGate:
        def __init__(self, *args, **kwargs):
            self.base = original(*args, **kwargs)
            self.policy = "rules"
            self.last_native = None

        def __getattr__(self, name):
            return getattr(self.base, name)

        def decide(self, *args, **kwargs):
            decision = self.base.decide(*args, **kwargs)
            self.last_native = decision
            if self.policy == "rules":
                return replace(decision, should_refine=False, action="keep", reasons=("paired_rules_only",))
            if self.policy == "always" and not decision.should_refine:
                return replace(decision, should_refine=True, action="refine", reasons=("paired_force_call",))
            return decision

    web.RefinementGate = PairedGate


def create_runtime(web, output, model, device, refiner=None, plausibility=None, window_edit_policy="compat"):
    if refiner is not None:
        web.TransformersRefiner = lambda *_a, **_k: refiner
    if plausibility is not None:
        web.RepetitionPlausibility = lambda *_a, **_k: plausibility
    routing_factory(web)
    app = web.create_app(model, device, "http://offline-pair.invalid", "Chinese", 256,
                         None, entity_db=output / "entities_snapshot.db",
                         entity_fuzzy_mode="auto", refinement_gate_mode="tri_state",
                         final_refinement_mode="off", zh_itn_library=output / "native/libzh_itn_bridge.so",
                         **({"window_edit_policy": window_edit_policy} if window_edit_policy != "compat" else {}))
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/ws/stream")
    refine = closure_value(endpoint, "refine_update")
    review = closure_value(endpoint, "review_final_repetition_text")
    gate = closure_value(refine, "refinement_gate")
    return refine, review, gate


def tools_for_domain(web, output, domain):
    definitions = web.EntityStore(output / "entities_snapshot.db").list_entities(domain=domain)
    protector = web.EntityProtector(definitions, enable_rule_protection=True, protect_numeric_spans=False)
    matcher = web.EntityCandidateMatcher(definitions, selected_domain=domain, mode="auto") if definitions else None
    return protector, matcher


def final_source_segments(row, web):
    events, seen = [], set()
    for payload in [*row.get("chunk_payloads", []), row.get("final_payload", {})]:
        for event in payload.get("completed_segments", []):
            key, text = event.get("segment_id"), event.get("text", "")
            if isinstance(key, int) and key not in seen and isinstance(text, str) and text.strip():
                seen.add(key)
                events.append(text.strip())
    if events and web.join_refined_segments(events) == row["raw_text"].strip():
        return tuple(events)
    return None


def local_review_native_action(web, text, candidate):
    """Record source-preservation preflight before any teacher inference.

    Old frozen runtimes lack this preflight and continue to request review.
    """
    checker = getattr(web, "repetition_preservation_violations", None)
    projected = text[:candidate.start] + candidate.target + text[candidate.end:]
    return "keep" if callable(checker) and checker(text, projected) else "refine"


def recording_identity(row):
    if row.get("data_kind") == "synthetic_tts":
        domain = row.get("domain", "general")
        group = row.get("recording_group")
        if domain not in {"general", "medical"} or not isinstance(group, str) or not group.strip():
            raise ValueError("Synthetic speech requires a declared domain and shared source group")
        return group, domain, "synthetic_tts"
    key = str(row.get("id") or row.get("record_id") or row.get("audio_path"))
    if key.startswith("BAC"):
        family, domain, kind = "AISHELL", "general", "human_news"
    elif key.startswith("MMedFD") or "TTS" in str(row.get("dataset", "")):
        family, domain, kind = "MMedFD", "medical", "public_tts"
    elif key.startswith("project_audio"):
        family, domain, kind = "project_demo", "general", "demo_without_reference"
    else:
        family, domain, kind = "MultiMed", "medical", "human_medical"
    return f"{family}:{key}", domain, kind


def prepare(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preparation never overwrites an existing protocol; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    sources = [p.resolve() for p in (args.sources or [ROOT / p for p in DEFAULT_SOURCES])]
    long_sources = [] if getattr(args, "no_demo_sources", False) else [
        p.resolve() for p in (args.long_results or [ROOT / p for p in DEFAULT_LONG_RESULTS])]
    for p in [*sources, args.entity_db, args.library]:
        if not p.is_file():
            raise FileNotFoundError(p)
    hashes = {}
    for p in sorted((ROOT / "system").rglob("*")):
        if p.is_file() and p.suffix in {".py", ".json", ".html"} and "__pycache__" not in p.parts:
            dest = out / "frozen" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            hashes[str(dest.relative_to(out))] = file_hash(dest)
    native = out / "native/libzh_itn_bridge.so"
    native.parent.mkdir()
    shutil.copy2(args.library, native)
    with sqlite3.connect(args.entity_db.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(out / "entities_snapshot.db") as dst:
            src.backup(dst)
    input_manifest, rows = [], []
    (out / "inputs").mkdir()
    for i, source in enumerate(sources):
        dest = out / "inputs" / f"asr_{i:02d}.jsonl"
        shutil.copy2(source, dest)
        input_manifest.append(dict(original=str(source), frozen=str(dest.relative_to(out)), sha256=file_hash(dest)))
        for index, row in enumerate(read_rows(dest), 1):
            if isinstance(row.get("raw_text"), str) and row["raw_text"].strip():
                rows.append({**row, "origin_file": str(source), "origin_line": index})
    for source in long_sources:
        if not source.exists():
            continue
        payload = json.loads(source.read_text())
        raw = payload.get("raw_text") or payload.get("output", {}).get("raw_text")
        if not isinstance(raw, str) or not raw.strip():
            continue
        number = "4" if "audio4" in source.name else "5"
        audio = ROOT / f"测试音频{number}.mp3"
        dest = out / "inputs" / source.name
        shutil.copy2(source, dest)
        input_manifest.append(dict(original=str(source), frozen=str(dest.relative_to(out)), sha256=file_hash(dest)))
        rows.append(dict(id=f"project_audio{number}", raw_text=raw, reference="", audio_path=str(audio),
                         variant="saved_final", origin_file=str(source), origin_line=1,
                         source_note="Previously used development/demo audio; no gold reference."))
    web = load_system(out)

    class NoModel:
        def refine(self, *_a, **_k):
            raise RuntimeError("Rules-only preparation unexpectedly attempted model inference")

    window_edit_policy = getattr(args, "window_edit_policy", None) or "compat"
    refine, _review, gate = create_runtime(web, out, ROOT, "cpu", refiner=NoModel(), window_edit_policy=window_edit_policy)
    domains, seen = {}, set()
    windows, reviews = [], []
    for row in rows:
        raw = row["raw_text"].strip()
        identity, domain, kind = recording_identity(row)
        source_key = row.get("pcm_sha256") or digest([identity, row.get("variant", ""), raw])
        if source_key in seen:
            continue
        seen.add(source_key)
        if domain not in domains:
            domains[domain] = tools_for_domain(web, out, domain)
        protector, matcher = domains[domain]
        window = web.CumulativeWindowRefinement(None, window_size=3, one_punctuation_window=True,
                    fixed_groups=True, group_numeric_self_corrections=True)
        chunks, _pending = window._fixed_chunks(raw, source_chunks=final_source_segments(row, web),
                    final=True, preserve_cached_partition=False)
        groups = [tuple(chunks[i:i+3]) for i in range(0, len(chunks), 3)]
        parts, native_decisions = [], []
        for group in groups:
            text = web.join_refined_segments(group)
            result = refine(text, "Chinese", True, protector, None, matcher, single_window=True,
                            confidence_metadata={})
            parts.append(result["clean_text"])
            native_decisions.append(gate.last_native.public_dict())
        baseline_full = web.clean_transcript_deterministically(web.join_refined_segments(parts))
        common = dict(recording_group=identity, speaker=row.get("speaker", "unknown"),
                      variant=row.get("variant", "unknown"), data_kind=kind, domain=domain,
                      pcm_sha256=row.get("pcm_sha256"), audio_path=row.get("audio_path", ""),
                      original_audio_sha256=row.get("original_audio_sha256"),
                      origin_file=row["origin_file"], origin_line=row["origin_line"],
                      raw_text_full=raw, reference_full=row.get("reference", ""),
                      baseline_full=baseline_full, baseline_parts=parts,
                      decision_time="saved_authoritative_final_snapshot",
                      reference_role="unverified_synthesis_script" if kind == "synthetic_tts" else "offline_review_only",
                      evaluation_split=row.get("synthetic_split", "development_pilot"))
        if kind == "synthetic_tts":
            # Offline metadata must never be supplied to the encoder or Refiner.
            common["synthetic_metadata"] = {k: row[k] for k in
                ["seed_id", "seed_sha256", "template_group", "intended_text", "spoken_text",
                 "synthetic_split", "audio_script_verified"]}
        cursor = 0
        for i, group in enumerate(groups):
            text = web.join_refined_segments(group)
            start = raw.find(text, cursor)
            if start < 0:
                # Preserve raw ownership even when ASCII separators were normalized.
                start, end = None, None
            else:
                end, cursor = start + len(text), start + len(text)
            if len(text) > args.max_window_chars:
                continue
            decision = native_decisions[i]
            item = dict(common, task_kind="window_refine", raw_text=text, baseline_text=parts[i],
                        window_index=i, source_start=start, source_end=end,
                        window_chunk_count=len(group), original_gate=decision,
                        context_available_before=raw[:start][-96:] if start is not None else "",
                        selection_stratum=f"{kind}|{decision['action']}")
            item["case_id"] = digest(["window", source_key, i, text])[:24]
            windows.append(item)
        for candidate in web.find_repetition_review_candidates(baseline_full, max_candidates=128, numeric_rule_mode=True):
            local_start = web._repetition_candidate_context_offset(baseline_full, candidate)
            if local_start is None or len(candidate.context) > args.max_review_chars:
                continue
            item = dict(common, task_kind="local_edit_review", baseline_text=baseline_full,
                        candidate=candidate.public_dict(), candidate_local_start=local_start,
                        native_local_review_action=local_review_native_action(web, baseline_full, candidate),
                        selection_stratum=f"{kind}|{candidate.kind}")
            item["case_id"] = digest(["review", source_key, candidate.start, candidate.end, candidate.context])[:24]
            reviews.append(item)
    selected = balanced_sample(windows, args.window_count, args.seed)
    selected += balanced_sample(reviews, args.review_count, args.seed + 1)
    random.Random(args.seed + 2).shuffle(selected)
    cases_path = out / "cases.jsonl"
    for item in selected:
        append_row(cases_path, item)
    # Snapshot initialization may add a compatible schema, only in our copy.
    hashes["entities_snapshot.db"] = file_hash(out / "entities_snapshot.db")
    hashes["native/libzh_itn_bridge.so"] = file_hash(native)
    hashes["cases.jsonl"] = file_hash(cases_path)
    manifest = dict(version=1, seed=args.seed, inputs=input_manifest, frozen_hashes=hashes,
                    entity_source=str(args.entity_db.resolve()), library_source=str(args.library.resolve()),
                    settings=dict(gate="tri_state", zh_itn=True, entity_fuzzy_mode="auto",
                                  final_refinement="off", window_size=3, max_new_tokens=256,
                                  window_edit_policy=window_edit_policy),
                    source_records_after_dedup=len(seen), pool=dict(window_refine=len(windows), local_edit_review=len(reviews)),
                    selected=dict(Counter(c["task_kind"] for c in selected)),
                    unique_recordings=len({c["recording_group"] for c in selected}),
                    unique_speakers=len({c["speaker"] for c in selected if c["speaker"] != "unknown"}),
                    strata=dict(Counter(c["selection_stratum"] for c in selected)),
                    sampling="Balanced exploratory input-side strata; not population prevalence.",
                    scope="Frozen final-snapshot decision pilot, not streaming/end-to-end latency or independent heldout acceptance.",
                    annotation_status="All human labels are initially blank.")
    write_json(out / "protocol.json", manifest)
    print(json.dumps({k:manifest[k] for k in ['pool','selected','unique_recordings','strata']},ensure_ascii=False),flush=True)


def verify_frozen(output):
    protocol = json.loads((output / "protocol.json").read_text())
    for path, expected in protocol["frozen_hashes"].items():
        if file_hash(output / path) != expected:
            raise RuntimeError(f"Frozen artifact changed: {path}")
    for item in protocol.get("inputs", []):
        if file_hash(output / item["frozen"]) != item["sha256"]:
            raise RuntimeError("Frozen ASR input changed: " + item["frozen"])
    return protocol


def compact_result(result):
    keep = ["clean_text", "raw_text", "refiner_executed", "refiner_accepted", "refiner_reject_reasons",
            "refiner_masked_outputs", "refinement_gate_decisions", "refiner_retry_count", "placeholder_retry_count",
            "entity_normalizations", "numeric_normalizations", "numeric_rule_repairs", "safe_repetition_repairs",
            "local_deletion_reviews", "repetition_reviews", "protected_entities", "structured_patch_audits"]
    keep += ["window_edit_policy", "strict_local_protocol_version"]
    return {k:result[k] for k in keep if k in result}


def run(args):
    out = args.output.resolve()
    protocol = verify_frozen(out)
    window_edit_policy = protocol.get("settings", {}).get("window_edit_policy", "compat")
    if getattr(args, "window_edit_policy", None) not in {None, window_edit_policy}:
        raise RuntimeError("Window edit policy differs from the frozen preparation protocol")
    cases = read_rows(out / "cases.jsonl")
    if args.limit:
        cases = cases[:args.limit]
    if args.retry_failed and (out/'pairs.jsonl').exists():
        previous=read_rows(out/'pairs.jsonl')
        failed=[r for r in previous if any(a.get('session_stats',{}).get('failed_call_count',0)
                   or any(c.get('error_type') for c in a['logical_calls']) for a in r['arms'].values())]
        bad_ids={r['case_id'] for r in failed}
        if failed:
            archive=out/f'pairs_failed_{time.time_ns()}.jsonl'
            shutil.copy2(out/'pairs.jsonl',archive)
            temporary=out/'pairs.valid.tmp'
            temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous if r['case_id'] not in bad_ids))
            temporary.replace(out/'pairs.jsonl')
            print(f'ARCHIVE {len(failed)} resource/operation failures; retrying those cases; original evidence={archive.name}',flush=True)
    model_path = args.model.resolve()
    import torch
    torch.set_num_threads(args.cpu_threads)
    web = load_system(out)
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    gpu = torch.device(args.device)
    if gpu.type == "cuda":
        free, _total = torch.cuda.mem_get_info(gpu)
        needed = sum(p.stat().st_size for p in model_path.glob("*.safetensors"))
        if free < needed + 512 * 1024**2:
            raise RuntimeError(f"Insufficient safe free GPU memory: {free/1024**3:.2f} GiB; need weights plus 0.5 GiB")
    refiner = TransformersRefiner(model_path, args.device, 256,
                                 **({"window_edit_policy": window_edit_policy} if window_edit_policy != "compat" else {}))
    # Preserve the model's original precision and generation settings.
    parameter = next(refiner._model.parameters())
    forward_count = {"refiner": 0, "plausibility": 0}
    refiner._model.register_forward_hook(lambda *_a: forward_count.__setitem__("refiner", forward_count["refiner"]+1))
    logical_calls = []
    plausibility = RepetitionPlausibility('cpu')

    def stage_plausibility_on_cpu():
        # Shared GPU has limited memory. Preserve CUDA compute, but move the
        # masked model's weights off GPU while the large Refiner is scoring.
        if plausibility._model is not None:
            plausibility._model.to('cpu')
        plausibility.device='cpu'
        if gpu.type=='cuda':torch.cuda.empty_cache()

    class AuditedRefiner:
        def __getattr__(self, name):
            value = getattr(refiner, name)
            if not callable(value) or not (name.startswith("refine") or name.startswith("review_")):
                return value
            def invoke(*a, **k):
                stage_plausibility_on_cpu()
                before = forward_count["refiner"]
                if gpu.type == "cuda":torch.cuda.synchronize(gpu)
                started = time.perf_counter()
                entry=dict(method=name,input=list(a),kwargs=k)
                try:
                    result = value(*a, **k)
                    if gpu.type == "cuda":torch.cuda.synchronize(gpu)
                    entry['output']=list(result)
                    return result
                except BaseException as error:
                    entry.update(error_type=type(error).__name__,error=str(error))
                    print('MODEL_ERROR '+json.dumps(entry,ensure_ascii=False),flush=True)
                    raise
                finally:
                    entry.update(wall_ms=round((time.perf_counter()-started)*1000,3),
                                 forward_passes=forward_count['refiner']-before)
                    logical_calls.append(entry)
            return invoke

    ppl_calls = []

    class AuditedPlausibility:
        def __getattr__(self, name):return getattr(plausibility, name)
        def decide(self, *a, **k):return self.invoke("decide", *a, **k)
        def decide_short_context(self, *a, **k):return self.invoke("decide_short_context", *a, **k)
        def invoke(self, name, *a, **k):
            # Hook the underlying masked-model forward after lazy loading.
            if plausibility._ensure_model() and not getattr(plausibility,"_pilot_hook",False):
                plausibility._model.register_forward_hook(lambda *_a: forward_count.__setitem__("plausibility", forward_count["plausibility"]+1))
                plausibility._pilot_hook=True
            transfer_started=time.perf_counter()
            if plausibility._model is not None:
                plausibility._model.to(args.device)
                plausibility.device=args.device
            if gpu.type=='cuda':torch.cuda.synchronize(gpu)
            transfer_ms=(time.perf_counter()-transfer_started)*1000
            before=forward_count["plausibility"];started=time.perf_counter()
            entry=dict(method=name,weights_transfer_ms=round(transfer_ms,3))
            try:
                result=getattr(plausibility,name)(*a,**k)
                if gpu.type=="cuda":torch.cuda.synchronize(gpu)
                entry.update(decision=result.decision,reason=result.reason)
                return result
            except BaseException as error:
                entry.update(error_type=type(error).__name__,error=str(error))
                raise
            finally:
                entry.update(wall_ms=round((time.perf_counter()-started)*1000,3),
                             forward_passes=forward_count['plausibility']-before)
                ppl_calls.append(entry)
                stage_plausibility_on_cpu()

    refine, review, gate = create_runtime(web,out,model_path,args.device,
                                         refiner=AuditedRefiner(),plausibility=AuditedPlausibility(),
                                         window_edit_policy=window_edit_policy)
    domains = {}
    # Exclude one generation warmup from comparison costs.
    warm_started=time.perf_counter()
    refiner.refine_with_numeric_rules("今天下午开会。")
    configuration=dict(model=str(model_path),device=args.device,dtype=str(parameter.dtype),
        model_parameters=sum(p.numel() for p in refiner._model.parameters()),torch=torch.__version__,
        config_sha256=file_hash(model_path/"config.json"),weights_sha256={p.name:file_hash(p) for p in model_path.glob("*.safetensors")},
        protocol_sha256=file_hash(out/"protocol.json"),cpu_threads=args.cpu_threads,
        warmup_ms=round((time.perf_counter()-warm_started)*1000,3),
        scope="Real isolated Refiner at original precision; GPU may be shared. Decision cost is not full transcription latency.",
        plausibility_memory_policy="CPU weight staging; unchanged float32 CUDA scoring. Transfers are recorded separately and are experiment overhead.",
        adapter_sha256=file_hash(Path(__file__)),window_edit_policy=window_edit_policy,
        identical_arm_rule="Always arm shares current execution only when current already selects exactly the same operation.")
    config_path=out/"run_configuration.json"
    if config_path.exists():
        old=json.loads(config_path.read_text())
        for k in ['model','dtype','config_sha256','weights_sha256','protocol_sha256']:
            if old[k]!=configuration[k]:raise RuntimeError("Resume configuration mismatch: "+k)
    else:write_json(config_path,configuration)
    append_row(out/'execution_runs.jsonl',configuration)
    completed={r["case_id"] for r in read_rows(out/"pairs.jsonl")}
    normal_candidates = web.find_repetition_review_candidates
    for index,case in enumerate(cases,1):
        if case["case_id"] in completed:continue
        if case["domain"] not in domains:
            domains[case["domain"]]=tools_for_domain(web,out,case["domain"])
        protector,matcher=domains[case["domain"]]
        arms={}
        for policy in ['rules','current','always']:
            # No extra inference for mathematically identical policy actions.
            shared=(policy=='always' and (case['task_kind']=='local_edit_review' or arms['current']['result'].get('refiner_executed')))
            if shared:
                arms[policy]=json.loads(json.dumps(arms['current'],ensure_ascii=False))
                arms[policy]['physical_execution']='shared_identical_current_action'
                arms[policy]['physical_calls']=0
                continue
            gate.policy=policy;logical_calls.clear();ppl_calls.clear()
            stats=web._RefinerSessionStats();started=time.perf_counter()
            if case['task_kind']=='window_refine':
                result=refine(case['raw_text'],'Chinese',True,protector,None,matcher,
                              single_window=True,confidence_metadata={},call_stats=stats)
            elif policy=='rules':
                result=dict(clean_text=case['baseline_text'],refiner_executed=False,refiner_accepted=True,
                            repetition_reviews=[],refiner_reject_reasons=[])
            else:
                from system.refinement_guard import RepetitionReviewCandidate
                c=case['candidate']
                candidate=RepetitionReviewCandidate(**{k:c[k] for k in ['index','start','end','source','target','kind','context']})
                web.find_repetition_review_candidates=lambda *_a,**_k:(candidate,)
                try:
                    text,records,_latency,calls,_hits=review(case['baseline_text'],final=True,call_stats=stats)
                finally:web.find_repetition_review_candidates=normal_candidates
                result=dict(clean_text=text,refiner_executed=bool(calls),refiner_accepted=True,
                            repetition_reviews=records,refiner_reject_reasons=[])
            stats.close_new_calls()
            arm=dict(result=compact_result(result),wall_ms=round((time.perf_counter()-started)*1000,3),
                     logical_calls=list(logical_calls),plausibility_calls=list(ppl_calls),
                     physical_execution='executed',physical_calls=len(logical_calls)+len(ppl_calls),
                     session_stats=stats.snapshot())
            if case['task_kind']=='window_refine':
                parts=list(case['baseline_parts']);parts[case['window_index']]=result['clean_text']
                arm['comparison_text']=web.clean_transcript_deterministically(web.join_refined_segments(parts))
            else:arm['comparison_text']=result['clean_text']
            arms[policy]=arm
        if arms['rules']['result']['clean_text'] != case['baseline_text']:
            raise RuntimeError('Prepared baseline drift: '+case['case_id'])
        if any(a['session_stats']['failed_call_count'] or any(c.get('error_type') for c in a['logical_calls'])
               or any(c.get('error_type') for c in a['plausibility_calls']) for a in arms.values()):
            append_row(out/'operation_failures.jsonl',dict(case_id=case['case_id'],arms=arms))
            raise RuntimeError('Operation failed; not a valid neutral label: '+case['case_id'])
        ref=case['reference_full']
        proxy={}
        if ref:
            norm=web.ZhITNNormalizer(out/'native/libzh_itn_bridge.so')
            target=normalized(norm.normalize(ref).text)
            for policy,arm in arms.items():
                proxy[policy]=edit_distance(target,normalized(norm.normalize(arm['comparison_text']).text))
        row=dict(case_id=case['case_id'],task_kind=case['task_kind'],recording_group=case['recording_group'],
            data_kind=case['data_kind'],domain=case['domain'],arms=arms,
            changed_vs_rules={k:arms[k]['comparison_text']!=arms['rules']['comparison_text'] for k in ['current','always']},
            reference_distance_proxy=proxy,reference_proxy_status='review_aid_not_gold_label',
            human_label=None,physical_elapsed_note='Shared identical arms are not counted twice as actual execution.')
        append_row(out/'pairs.jsonl',row)
        print(f"PAIR {index}/{len(cases)} {case['task_kind']} calls={sum(a['physical_calls'] for a in arms.values())} changed={row['changed_vs_rules']}",flush=True)
        if gpu.type=='cuda':torch.cuda.empty_cache()
        if args.pause:time.sleep(args.pause)
    verify_frozen(out)
    report(argparse.Namespace(output=out))


def report(args):
    out=args.output.resolve();protocol=verify_frozen(out)
    cases={c['case_id']:c for c in read_rows(out/'cases.jsonl')}
    pairs=read_rows(out/'pairs.jsonl')
    if len({p['case_id'] for p in pairs})!=len(pairs):raise ValueError('Duplicate completed cases')
    if any(a.get('session_stats',{}).get('failed_call_count',0)
           or any(c.get('error_type') for c in a['logical_calls']+a['plausibility_calls'])
           for p in pairs for a in p['arms'].values()):
        raise RuntimeError('Failed operations are not valid pairs; archive and rerun with --retry-failed')
    annotation=out/'annotations.csv';existing={}
    if annotation.exists():
        with annotation.open(encoding='utf-8-sig',newline='') as f:existing={r['case_id']:r for r in csv.DictReader(f)}
    summary=dict(total=len(pairs),planned=len(cases),
                 human_reviewed=sum(bool(existing.get(p['case_id'],{}).get('human_label','').strip()) for p in pairs),
                 provable_neutral=sum(p['arms']['always']['comparison_text']==p['arms']['rules']['comparison_text'] for p in pairs),
                 scope=protocol['scope'],sampling=protocol['sampling'],by_task={})
    summary['unique_recordings']=len({p['recording_group'] for p in pairs})
    summary['data_kinds']=dict(Counter(cases[p['case_id']]['data_kind'] for p in pairs))
    summary['reference_available']=sum(bool(cases[p['case_id']]['reference_full']) for p in pairs)
    summary['audio_available']=sum(Path(cases[p['case_id']]['audio_path']).is_file() for p in pairs)
    for task in ['window_refine','local_edit_review']:
        rows=[p for p in pairs if p['task_kind']==task]
        stats=dict(cases=len(rows),unique_recordings=len({p['recording_group'] for p in rows}),arms={})
        for policy in ['rules','current','always']:
            selected=[p['arms'][policy] for p in rows]
            stats['arms'][policy]=dict(
                logical_refiner_calls=sum(len(a['logical_calls']) for a in selected),
                logical_plausibility_calls=sum(len(a['plausibility_calls']) for a in selected),
                measured_operation_cost_ms=round(sum(sum(c['wall_ms'] for c in a['logical_calls']) + sum(c['wall_ms'] for c in a['plausibility_calls']) for a in selected),3),
                refiner_forward_passes=sum(sum(c['forward_passes'] for c in a['logical_calls']) for a in selected),
                plausibility_forward_passes=sum(sum(c['forward_passes'] for c in a['plausibility_calls']) for a in selected),
                changed_vs_rules=sum(p['arms'][policy]['comparison_text']!=p['arms']['rules']['comparison_text'] for p in rows),
                rejected_windows=sum(a['result'].get('refiner_accepted') is False for a in selected),
                proxy_improved=sum(p.get('reference_distance_proxy',{}).get(policy,math.inf)<p.get('reference_distance_proxy',{}).get('rules',-math.inf) for p in rows),
                proxy_regressed=sum(p.get('reference_distance_proxy',{}).get(policy,-math.inf)>p.get('reference_distance_proxy',{}).get('rules',math.inf) for p in rows))
        summary['by_task'][task]=stats
        selected=[r for r in rows if r['arms']['current']['result'].get('refiner_executed')]
        no_effect=[r for r in selected if r['arms']['current']['comparison_text']==r['arms']['rules']['comparison_text']]
        stats['current_selected_decisions']=len(selected)
        stats['current_selected_no_effect']=len(no_effect)
        stats['current_no_effect_cost_ms']=round(sum(sum(c['wall_ms'] for c in r['arms']['current']['logical_calls']+r['arms']['current']['plausibility_calls']) for r in no_effect),3)
    summary['physical_logical_calls']=sum(a['physical_calls'] for p in pairs for a in p['arms'].values())
    summary['limits']=['No human labels have been assigned automatically.',
        'Reference edit distance is a full-utterance substitution proxy, not a verified intention or correction label.',
        'Isolated final snapshots and exploratory sampling do not certify streaming latency or generalization.',
        'Costs for identical current/always actions share the same measured execution; real shared executions are counted once.',
        'Warmup and lazy masked-model load are not representative steady-state operation costs; GPU contention can affect timing.']
    write_json(out/'summary.json',summary)
    fields=['case_id','task_kind','data_kind','recording_group','variant','domain','audio_path','origin_file',
            'raw_window','raw_full','reference_full','baseline','current_output','always_output',
            'candidate_source','candidate_target','candidate_context','rules_proxy','current_proxy','always_proxy',
            'reference_status','automatic_pair_label','automatic_label_basis','human_label','rationale','reviewer']
    with annotation.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in pairs:
            c=cases[p['case_id']];old=existing.get(p['case_id'],{})
            row={k:c.get(k,'') for k in ['case_id','task_kind','data_kind','recording_group','variant','domain','audio_path','origin_file','reference_full']}
            row.update(raw_window=c.get('raw_text',''),raw_full=c['raw_text_full'],
                baseline=p['arms']['rules']['comparison_text'],current_output=p['arms']['current']['comparison_text'],
                always_output=p['arms']['always']['comparison_text'],
                candidate_source=c.get('candidate',{}).get('source',''),candidate_target=c.get('candidate',{}).get('target',''),
                candidate_context=c.get('candidate',{}).get('context',''),
                rules_proxy=p['reference_distance_proxy'].get('rules',''),
                current_proxy=p['reference_distance_proxy'].get('current',''),always_proxy=p['reference_distance_proxy'].get('always',''),
                reference_status=old.get('reference_status') or ('unverified_reference' if c['reference_full'] else 'missing_reference'),
                automatic_pair_label='neutral' if p['arms']['always']['comparison_text']==p['arms']['rules']['comparison_text'] else '',
                automatic_label_basis='identical_final_output' if p['arms']['always']['comparison_text']==p['arms']['rules']['comparison_text'] else '',
                human_label=old.get('human_label',''),rationale=old.get('rationale',''),reviewer=old.get('reviewer',''))
            writer.writerow(row)
    with annotation.open(encoding='utf-8-sig',newline='') as f:
        annotation_rows=list(csv.DictReader(f))
    with (out/'changed_for_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(r for r in annotation_rows if not r['automatic_pair_label'])
    neutral_path=out/'neutral_pairs.jsonl'
    neutral_path.write_text('')
    for p in pairs:
        if p['arms']['always']['comparison_text']!=p['arms']['rules']['comparison_text']:continue
        c=cases[p['case_id']]
        inputs={k:c[k] for k in ['task_kind','domain','data_kind','baseline_text']}
        if c['task_kind']=='window_refine':
            inputs.update(raw_text=c['raw_text'],available_context=c['context_available_before'],
                          window_chunk_count=c['window_chunk_count'])
        else:
            inputs.update(available_context=c['candidate']['context'],
                          candidate={k:c['candidate'][k] for k in ['source','target','kind']},
                          candidate_local_start=c['candidate_local_start'])
            # The review baseline itself is full text for audit. The encoder
            # needs only the already-available bounded candidate context.
            inputs['baseline_text']=c['candidate']['context']
        append_row(neutral_path,dict(case_id=c['case_id'],recording_group=c['recording_group'],
                   input=inputs,label='neutral',label_basis='identical_final_output',
                   supervision_scope='incremental_operation_effect_only',split='development_pilot'))
    guide='''# 人工收益标注说明

此表的 human_label 初始为空，不存在自动生成的真实收益标签。
automatic_pair_label 只对两条分支最终输出逐字一致的样本填 neutral；这是相对操作无效果的证据，不代表原 ASR 正确。
标签评价 always_output 相对 baseline 的变化：benefit、neutral、harm、uncertain。
先核对原音及参考，再判断最终意图；不得只按 CER 下降、发生修改或验证器接受打标签。
错误实体、数字、否定变化、关键内容删除或新增事实优先记 harm；参考／原音存疑记 uncertain。
reference_status 可填写 verified_verbatim、verified_intended、uncertain 或 missing_reference。
candidate_context 是局部复核时已有上下文；reference_full 仅用于离线审核，不进入模型输入。
窗口输出的 comparison_text 是把该局部结果替入共同全文基线后的距离代理，不能当作完整有状态会话结果。
已有资料已用于开发，公开 TTS 与真人数据须分别报告；噪声版本不是独立录音。
report 子命令会保留 human_label、rationale、reviewer 和 reference_status 的已有填写。
'''
    (out/'ANNOTATION_GUIDE.md').write_text(guide)
    lines=['# 门控成对试验初步报告','',f"完成 {len(pairs)}/{len(cases)} 个决策样本；人工已标注 {summary['human_reviewed']} 条。",'',
           f"{summary['provable_neutral']} 条 always 与 rules 最终输出逐字相同，可确定为相对操作无收益；它们不证明原始转录正确。",'',
           '这是冻结 ASR 最终快照的真实 Refiner 对照；不是在线速度测试或独立泛化验收。','',
           '| 任务 | 策略 | 逻辑 Refiner 调用 | 模型成本累计（秒） | 相对规则输出变化 | 参考距离改善／退化（待核对） |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for task,st in summary['by_task'].items():
        for policy,a in st['arms'].items():
            lines.append(f"| {task} | {policy} | {a['logical_refiner_calls']} | {a['measured_operation_cost_ms']/1000:.3f} | {a['changed_vs_rules']} | {a['proxy_improved']} / {a['proxy_regressed']} |")
    lines+=['','相同 current/always 操作共享一次真实执行，表内各策略成本是映射到该策略的实测操作成本，不能将各行相加当成实验墙钟。',
            '语义、实体与数字误改需要人工复核；未标注前不能称为已验证改善率或训练收益数据。','',
            f"样本来自 {summary['unique_recordings']} 个来源；{summary['reference_available']} 条有配套参考，音频路径可用 {summary['audio_available']}/{len(pairs)}。",'',
            '## 当前策略仍可筛选的无效操作','',
            '| 任务 | 当前选中的决策 | 最终无变化 | 对应实测模型成本累计（秒） |',
            '| --- | ---: | ---: | ---: |']
    for task,st in summary['by_task'].items():
        lines.append(f"| {task} | {st['current_selected_decisions']} | {st['current_selected_no_effect']} | {st['current_no_effect_cost_ms']/1000:.3f} |")
    cost=sum(st['current_no_effect_cost_ms'] for st in summary['by_task'].values())
    lines += ['',f"上述最终无变化操作对应 {cost/1000:.3f} 秒累计模型时间。这是按结果回看的局部 oracle 空间，不代表学习型门控已取得同等收益。",'',
              '剔除／重跑异常记录保留在 pairs_failed_*.jsonl；辅助模型在 CPU 暂存权重，评分仍使用原 CUDA 精度，转移开销单独记录，不作为正式服务稳态成本。',
              '本轮值得继续审核变化样本，建立真实正收益与退化标签；不能仅凭无效调用比例宣布新门控能提升准确率或速度。','',
              '产物：protocol.json、cases.jsonl、pairs.jsonl、summary.json、annotations.csv、changed_for_review.csv、neutral_pairs.jsonl 和 ANNOTATION_GUIDE.md。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--sources',type=Path,nargs='+')
    parser.add_argument('--long-results',type=Path,nargs='+')
    parser.add_argument('--no-demo-sources',action='store_true',help='Exclude default project demos when using a dedicated source dataset')
    parser.add_argument('--window-edit-policy',choices=['compat','strict_local'],default=None,
                        help='Freeze a window output policy at prepare; run inherits the frozen setting')
    parser.add_argument('--entity-db',type=Path,default=DEFAULT_DB)
    parser.add_argument('--library',type=Path,default=ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    parser.add_argument('--window-count',type=int,default=400)
    parser.add_argument('--review-count',type=int,default=100)
    parser.add_argument('--max-window-chars',type=int,default=200)
    parser.add_argument('--max-review-chars',type=int,default=360)
    parser.add_argument('--seed',type=int,default=20261009)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--pause',type=float,default=0.05)
    parser.add_argument('--retry-failed',action='store_true',help='Archive failed pairs and retry them; retain original failure evidence')
    args=parser.parse_args()
    if args.window_count<0 or args.review_count<0 or args.cpu_threads<1 or args.pause<0:
        parser.error('Counts, pause and thread limits must be valid')
    {'prepare':prepare,'run':run,'report':report}[args.stage](args)


if __name__=='__main__':main()
