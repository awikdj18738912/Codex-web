"""Build a bounded, audio-linked diagnostic queue without creating gold labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import digest, review_basis
from system.edit_location_detector import EditLocationDetector
from system.refinement_gate import RefinementGate
from system.zh_itn_bridge import ZhITNNormalizer
from experiments.scripts.prepare_edit_detection_data import Preparer, make_records, sha, write_jsonl


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def packet_basis(row):
    return digest(dict(input_basis=review_basis(row), audio_provenance=row["audio_provenance"],
                       historical_output=row.get("historical_output"), model_scores=row.get("model_scores")))


def verify_clip(path, expected_pcm=None):
    import soundfile as sf

    info = sf.info(path)
    pcm_hash = None
    if expected_pcm is not None:
        pcm, rate = sf.read(path, dtype="float32")
        if rate != 16000 or pcm.ndim != 1:
            raise ValueError("Unexpected historical PCM format")
        pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
        if pcm_hash != expected_pcm:
            raise ValueError("Audio/ASR PCM provenance mismatch: " + str(path))
    return dict(audio_sha256=sha(path), decoded_float32_pcm_sha256=pcm_hash,
                duration_seconds=info.duration, sample_rate=info.samplerate, channels=info.channels)


def load_podcast_sources(directory):
    manifest_path = directory / "manifest.json"
    result_path = directory / "control_replay.jsonl"
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["dataset"])
    if sha(source) != manifest["dataset_sha256"]:
        raise ValueError("Podcast original audio changed")
    lookup = {row["id"]: row for row in manifest["samples"]}
    rows = read_rows(result_path)
    if len(rows) != len(lookup) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Podcast result identities differ from manifest")
    sources = []
    for index, row in enumerate(rows):
        sample = lookup[row["id"]]
        if row["pcm_sha256"] != sample["pcm_sha256"]:
            raise ValueError("Podcast ASR/manifest PCM hashes differ")
        path = Path(sample["audio_path"])
        audio = verify_clip(path, sample["pcm_sha256"])
        if abs(audio["duration_seconds"] - sample["duration_seconds"]) > .01:
            raise ValueError("Podcast clip duration changed")
        sources.append(dict(seed_id=row["id"], raw_text=row["result"]["raw_text"],
            historical_final_text=row["result"]["clean_text"], audio_path=str(path), audio_sha256=audio["audio_sha256"],
            recording_group="audio:" + manifest["dataset_sha256"], template_group="podcast_existing_development",
            split="regression", category="existing_podcast", context="",
            audio_provenance=dict(**audio, original_path=str(source), original_sha256=manifest["dataset_sha256"],
                clip_start_seconds=sample["start_seconds"], clip_end_seconds=sample["end_seconds"],
                timing_granularity="whole_audio_clip_not_word_or_text_window_alignment",
                source_kind="published_medical_discussion_podcast", source_origin_verified=True,
                transcript_file=str(result_path), transcript_file_sha256=sha(result_path), transcript_row=index,
                trace_id=row["ready"]["trace_id"], already_consumed_development=True),
            historical_stats=dict(refiner_calls=row["result"].get("session_refiner_call_count"),
                refiner_reject_reasons=row["result"].get("refiner_reject_reasons", []))))
    return sources, [manifest_path, result_path]


def load_test_sources(directory):
    result_path = directory / "vllm_results.jsonl"
    manifest_path = directory / "chunks_manifest.json"
    hash_path = directory / "input_sha256.txt"
    expected = {}
    for line in hash_path.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        expected[Path(name).name] = checksum
    manifest = json.loads(manifest_path.read_text())
    rows = read_rows(result_path)
    if [(r["file"], r["source"], r["chunk_index"]) for r in rows] != [
            (r["file"], r["source"], r["chunk_index"]) for r in manifest]:
        raise ValueError("ASR chunk identities differ from manifest")
    sources, verified = [], {}
    offsets = Counter()
    for index, (row, entry) in enumerate(zip(rows, manifest)):
        number = int(row["source"].split("_")[1])
        original = ROOT / f"测试音频{number}.mp3"
        full = directory / "audio" / row["source"]
        if number not in verified:
            if sha(original) != expected[original.name] or sha(full) != expected[full.name]:
                raise ValueError("Test recording original/converted audio changed")
            verified[number] = expected[original.name]
        path = directory / "chunks" / row["file"]
        audio = verify_clip(path)
        if abs(audio["duration_seconds"] - entry["duration_s"]) > .001 or row["duration_s"] != entry["duration_s"]:
            raise ValueError("Test clip duration mismatch")
        # Verify actual samples against the corresponding converted WAV.
        import soundfile as sf
        clip_pcm, rate = sf.read(path, dtype="int16")
        if rate != 16000 or clip_pcm.ndim != 1:
            raise ValueError("Unexpected benchmark chunk format")
        start = offsets[row["source"]]
        full_pcm, _ = sf.read(full, dtype="int16", start=start, frames=len(clip_pcm))
        if not (clip_pcm == full_pcm).all():
            raise ValueError("Benchmark chunk is not the declared original slice")
        offsets[row["source"]] += len(clip_pcm)
        sources.append(dict(seed_id="benchmark:" + row["file"], raw_text=row["text"], historical_final_text=None,
            audio_path=str(path), audio_sha256=audio["audio_sha256"], recording_group="audio:" + verified[number],
            template_group="existing_user_recording", split="regression", category="existing_test_recording", context="",
            audio_provenance=dict(**audio, original_path=str(original), original_sha256=verified[number],
                clip_start_seconds=start / rate, clip_end_seconds=(start + len(clip_pcm)) / rate,
                timing_granularity="whole_audio_clip_not_word_or_text_window_alignment",
                source_kind="existing_user_test_recording_speaker_and_generation_origin_unverified", source_origin_verified=False,
                transcript_file=str(result_path), transcript_file_sha256=sha(result_path), transcript_row=index,
                already_consumed_development=True)))
    return sources, [manifest_path, result_path, hash_path]


def source_windows(preparer, source):
    historical = source["historical_final_text"]
    rows = make_records(preparer, source, origin="audio_diagnostic_proposal", target=historical,
                        status="historical_pipeline_output_unverified" if historical is not None else "unverified_audio_ASR")
    for row in rows:
        row["historical_output"] = dict(
            whole_clip_final_text=historical,
            projected_window_text=row["desired_text"], projected_window_edits=row["edits"],
            projection_status=row["label_status"],
            attribution="historical_whole_pipeline_output_not_isolated_current_refiner_effect",
            stats=source.get("historical_stats"), audio_checked=False)
        # Model proposals, including unchanged outputs, never become labels.
        row.update(desired_text=None, edits=None, character_tags=None, insertion_gaps=None,
                   offset_alternatives=[], ambiguous_offsets=None, label_status="awaiting_audio_review",
                   audio_provenance=source["audio_provenance"], training_eligible=False,
                   human_audio_gold=False, independent_acceptance_eligible=False)
        row["review_basis_sha256"] = review_basis(row)
    return rows


def classify(row):
    scores = row["model_scores"]
    large = scores["courtesy_teacher"]
    tags = []
    if large["status"] == "unsupported":
        tags.append("unsupported_input")
    elif bool(large["candidates"]) != (row["native_gate"]["action"] == "refine"):
        tags.append("teacher_native_disagreement")
    statuses = {score["status"] for score in scores.values() if score["status"] != "unsupported"}
    if len(statuses) > 1:
        tags.append("detector_disagreement")
    projection = row["historical_output"]["projected_window_text"]
    if projection is not None and projection != row["baseline_text"]:
        tags.append("historical_pipeline_changed_text")
        if not large["candidates"] and large["status"] != "unsupported":
            tags.append("teacher_no_candidate_historical_change")
    if any("repetition" in signal or "disfluency" in signal for signal in row["native_gate"]["cleanup_signals"]):
        tags.append("oral_cleanup_candidate")
    if row["upstream_numeric_changes"]:
        tags.append("numeric_rule_control")
    if (row["native_gate"]["action"] == "keep" and large["status"] == "no_candidate"
            and scores["context_teacher"]["status"] == "no_candidate"):
        tags.append("no_detected_issue_control_unverified")
    if not tags:
        tags.append("no_detected_issue_control_unverified")
    return tags


def select_queue(rows, count=40):
    """Bound clip redundancy and retain disagreement plus unverified controls."""
    selected, seen, clip_counts, origin_counts = [], set(), Counter(), Counter()

    def add(row, maximum_per_clip=1):
        key = digest([row["baseline_text"], row["left_context"], row["right_context"]])
        origin = row["category"]
        if key in seen or clip_counts[row["audio_path"]] >= maximum_per_clip:
            return False
        if origin_counts[origin] >= (32 if origin == "existing_podcast" else count - 32):
            return False
        selected.append(row)
        seen.add(key)
        clip_counts[row["audio_path"]] += 1
        origin_counts[origin] += 1
        return True

    # Four deliberate controls; they are not pre-labelled KEEP.
    for tag in ("no_detected_issue_control_unverified", "numeric_rule_control"):
        for row in rows:
            if len(selected) >= (2 if tag.startswith("no_detected") else 4):
                break
            if tag in row["selection_tags"]:
                add(row)
    priority = {"teacher_no_candidate_historical_change": 0, "teacher_native_disagreement": 1,
                "historical_pipeline_changed_text": 2, "oral_cleanup_candidate": 3, "detector_disagreement": 4}
    ordered = sorted(rows, key=lambda row: (min((priority.get(tag, 5) for tag in row["selection_tags"]), default=5), row["id"]))
    for cap in (1, 2):
        for row in ordered:
            if len(selected) == count:
                return selected
            add(row, cap)
    if len(selected) != count:
        raise ValueError(f"Insufficient diverse audio windows: {len(selected)}/{count}")
    return selected


def build(args):
    import torch
    from threadpoolctl import threadpool_limits

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    podcasts, podcast_files = load_podcast_sources(args.podcast)
    tests, test_files = load_test_sources(args.benchmark)
    preparer = Preparer(ZhITNNormalizer(args.zh_itn), [])
    rows = [row for source in podcasts + tests for row in source_windows(preparer, source)]
    native = RefinementGate("tri_state")
    for row in rows:
        row["native_gate"] = native.decide(row["raw_asr"], is_final=True, stable=True,
            entity_hints=row["entity_hints"], numeric_rule_mode=True, numeric_rule_text=row["baseline_text"]).public_dict()
        row["model_scores"] = {}
    models = {"context_teacher": args.context_teacher, "courtesy_teacher": args.courtesy_teacher, "rbt3": args.rbt3}
    model_manifest = {}
    for name, path in models.items():
        detector = EditLocationDetector(path, args.device)
        model_manifest[name] = dict(path=str(path.resolve()), threshold=detector.metadata["threshold"],
                                   weights_sha256=detector.metadata["weights_sha256"])
        for row in rows:
            score = detector.detect(row)
            row["model_scores"][name] = {key: score[key] for key in ("status", "candidates", "requires_fallback", "production_skip_allowed")}
            if score["status"] == "unsupported":
                row["model_scores"][name]["reason"] = score["reason"]
        del detector
        torch.cuda.empty_cache()
        print("SCORED", name, len(rows), flush=True)
    for row in rows:
        row["selection_tags"] = classify(row)
        row["packet_basis_sha256"] = packet_basis(row)
    selected = select_queue(rows, args.count)
    (output / "audio").mkdir()
    copied = {}
    for row in selected:
        original = Path(row["audio_path"])
        name = row["audio_sha256"][:24] + original.suffix
        if name not in copied:
            shutil.copyfile(original, output / "audio" / name)
            if sha(output / "audio" / name) != row["audio_sha256"]:
                raise ValueError("Packaged audio differs from source")
            copied[name] = row["audio_sha256"]
        row["packaged_audio"] = "audio/" + name
    write_jsonl(output / "all_candidates.jsonl", rows)
    write_jsonl(output / "review_queue.jsonl", selected)
    # Deliberately blank; never precheck an audio review or copy a model target.
    write_jsonl(output / "review_drafts.jsonl", [dict(id=row["id"], review_basis_sha256=row["review_basis_sha256"],
        packet_basis_sha256=row["packet_basis_sha256"], reviewer="", audio_checked=False, review_kind=None,
        desired_text=None, edit_tasks=None, historical_pipeline_effect="unassessable") for row in selected])
    queue_hash = sha(output / "review_queue.jsonl")
    template = (ROOT / "experiments/fixtures/audio_edit_review.html").read_text()
    payload = json.dumps(dict(queue=selected, queue_sha256=queue_hash), ensure_ascii=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__REVIEW_DATA__", payload))
    manifest = dict(scope="consumed_audio_diagnostic_queue_not_training_or_independent_acceptance",
        candidate_windows=len(rows), selected_windows=len(selected), selected_audio_clips=len(copied),
        sources=Counter(row["category"] for row in selected), recording_groups=len({row["recording_group"] for row in selected}),
        selection_tags=Counter(tag for row in selected for tag in row["selection_tags"]),
        human_audio_gold=0, training_eligible=0, model_thresholds_frozen=True, models=model_manifest,
        no_new_asr_or_refiner_calls=True, native_gate_scope="offline_recomputed_stable_final_tri_state_not_historical_decision",
        entity_definitions=[], entity_scope="no_library_matching_original_podcast_control; not_current_live_entity_configuration",
        numeric_backend="zh-itn", numeric_library_sha256=sha(args.zh_itn),
        source_sha256={str(path.resolve()): sha(path) for path in podcast_files + test_files},
        output_sha256={name: sha(output / name) for name in ("all_candidates.jsonl", "review_queue.jsonl", "review_drafts.jsonl", "review.html")},
        audio_sha256=copied,
        limitations=["Podcast windows share one original recording; no random window train/test split",
                     "Test-recording generation origin/speakers unverified; do not claim all sources are natural speech",
                     "Audio bounds cover whole clips; no word/window timestamps",
                     "Historical changes include rules and local reviews, not isolated current Refiner gains",
                     "No error labels, accuracy, false-positive rate, or useful-skip benefit established"])
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "executed_code").mkdir()
    for path in (Path(__file__), ROOT / "system/edit_detection_data.py", ROOT / "experiments/fixtures/audio_edit_review.html"):
        (output / "executed_code" / path.name).write_bytes(path.read_bytes())
    print(json.dumps({k: manifest[k] for k in ("candidate_windows", "selected_windows", "selected_audio_clips", "sources", "selection_tags")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=[40], default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--podcast", type=Path, default=ROOT / "results/oncology_podcast_20261008")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "results/qwen3_asr_speed_fresh_20260929")
    parser.add_argument("--context-teacher", type=Path, default=ROOT / "results/contextual_edit_teacher_20261013")
    parser.add_argument("--courtesy-teacher", type=Path, default=ROOT / "results/contextual_courtesy_teacher_20261013")
    parser.add_argument("--rbt3", type=Path, default=ROOT / "results/contextual_rbt3_detector_20261013")
    parser.add_argument("--zh-itn", type=Path, default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    build(parser.parse_args())
