"""Build fixed-window text supervision and a separate ASR review queue.

Only existing training partitions are imported. Scripts and model outputs
never become human audio gold; no training or production routing is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import TASKS, align_edits, apply_edits, attach_review, digest, review_basis
from system.refinement_guard import join_refined_segments, restore_rule_numeric_surfaces
from system.window_refinement import CumulativeWindowRefinement

TASK_BY_CATEGORY = {"filler": "disfluency", "stutter": "repetition"}
PIPELINE = dict(version="fixed_k3_v1", window_size=3, one_punctuation_window=True,
                fixed_groups=True, stream_reprocess_rejected=False,
                final_reprocess_rejected=False, group_numeric_self_corrections=True)


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def lexical_positions(text):
    # Preserve code points and offsets. Do not casefold/NFKC: these can change
    # character length and would invalidate the source-coordinate mapping.
    positions = [index for index, char in enumerate(text)
                 if not char.isspace() and unicodedata.category(char)[0] not in {"P", "Z", "S"}]
    return "".join(text[index] for index in positions), positions


def project_script_deletions(spoken, intended, baseline):
    """Weak deletion-only proposals; punctuation and residual ASR errors stay.

    The returned target is explicitly not a complete cleaned transcript.
    Matching deletion strings at multiple ASR locations remains unresolved.
    """
    before, _ = lexical_positions(spoken)
    after, _ = lexical_positions(intended)
    current, positions = lexical_positions(baseline)
    changes = [op for op in SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not changes:
        return (baseline, "script_keep_candidate") if current == after else (None, "reference_mismatch")
    if any(op[0] != "delete" for op in changes):
        return None, "script_non_deletion_requires_review"
    blocks = SequenceMatcher(None, before, current, autojunk=False).get_matching_blocks()
    edits = []
    for _, start, end, _, _ in changes:
        matches = [block.b + start - block.a for block in blocks
                   if block.a <= start and end <= block.a + block.size]
        if len(matches) != 1:
            return None, "script_edit_not_aligned"
        offset = matches[0]
        # Repeated complete phrases can make the location ambiguous even
        # when SequenceMatcher chooses a deterministic matching block.
        literal = before[start:end]
        other = [index for index in range(len(current)) if current.startswith(literal, index)]
        if len(other) > 1 and current != before:
            return None, "ambiguous_script_projection"
        left, right = positions[offset], positions[offset + end - start - 1] + 1
        edits.append(dict(start=left, end=right, source=baseline[left:right], target=""))
    edits.sort(key=lambda edit: edit["start"])
    return apply_edits(baseline, edits), "projected_script_deletion_unverified_audio"


class Preparer:
    def __init__(self, normalizer, definitions):
        self.normalizer, self.definitions = normalizer, definitions

    def windows(self, raw, source_segments=None, extra_candidates=()):
        from system.entity_matcher import EntityCandidateMatcher
        from system.entity_pipeline import prepare_entity_segment
        from system.protection import EntityProtector
        from system.session_memory import SessionEntityMemory

        splitter = CumulativeWindowRefinement(None, one_punctuation_window=True,
                    fixed_groups=True, group_numeric_self_corrections=True)
        chunks, _ = splitter._fixed_chunks(raw, source_chunks=source_segments or None,
                                          final=True, preserve_cached_partition=False)
        protector = EntityProtector(self.definitions, session_memory=SessionEntityMemory(),
                                    enable_rule_protection=True, protect_numeric_spans=False)
        matcher = EntityCandidateMatcher(self.definitions, selected_domain="general", mode="auto") if self.definitions else None
        result = []
        for offset in range(0, len(chunks), 3):
            source = join_refined_segments(chunks[offset:offset + 3])
            numeric = self.normalizer.normalize(source)
            prepared = prepare_entity_segment(numeric.text, protector, matcher, allow_auto=True)
            result.append(dict(raw_asr=source, baseline_text=prepared.baseline_text,
                               chunk_count=min(3, len(chunks) - offset),
                               entity_candidates=[match.public_dict() for match in prepared.report.matches] + list(extra_candidates),
                               entity_hints=list(prepared.hints),
                               protection={"coordinate_space": "itn_text", "input": numeric.text,
                                           "spans": [span.public_dict() for span in prepared.protection.spans]},
                               upstream_entity_changes=list(prepared.normalizations),
                               upstream_numeric_changes=[change.public_dict() for change in numeric.changes]))
        return result


def make_records(preparer, source, *, origin, target, status, task=None, source_segments=None):
    windows = preparer.windows(source["raw_text"], source_segments, source.get("entity_candidates", []))
    baseline = join_refined_segments(window["baseline_text"] for window in windows)
    if origin == "asr_script_proposal":
        spoken = preparer.normalizer.normalize(source["spoken_text"]).text
        intended = preparer.normalizer.normalize(source["intended_text"]).text
        target, status = project_script_deletions(spoken, intended, baseline)
    elif target is not None:
        # Apply the same deterministic preparation to the declared target;
        # otherwise an already-normalized alias could be mislabeled as a
        # neural edit that restores its old spelling.
        target = join_refined_segments(window["baseline_text"] for window in
                    preparer.windows(target, extra_candidates=source.get("entity_candidates", [])))
    if target is not None and restore_rule_numeric_surfaces(baseline, target) != target:
        target, status = None, "numeric_integrity_requires_review"
    full_edits = align_edits(baseline, target, task)["edits"] if target is not None else []
    records, cursor = [], 0
    for index, window in enumerate(windows):
        text = window["baseline_text"]
        start = baseline.find(text, cursor)
        if start < 0:
            raise ValueError("Window cannot be located in prepared recording")
        end = start + len(text)
        cursor = end
        crossing = any(edit["start"] < end and edit["end"] > start
                       and not start <= edit["start"] <= edit["end"] <= end for edit in full_edits)
        local_edits = [{**edit, "start": edit["start"] - start, "end": edit["end"] - start}
                       for edit in full_edits if start <= edit["start"] < end and edit["end"] <= end]
        # EOF insertions belong to the final window.
        local_edits += [{**edit, "start": len(text), "end": len(text)} for edit in full_edits
                        if index == len(windows) - 1 and edit["start"] == edit["end"] == end]
        desired = apply_edits(text, local_edits) if target is not None and not crossing else None
        label_status = "cross_window_edit_requires_review" if crossing else status
        record = dict(schema_version="edit_locations_v1",
                      id=digest([origin, source["seed_id"], source.get("variant"), index])[:24],
                      seed_id=source["seed_id"], recording_group=source["recording_group"],
                      split=source["split"], split_group=source["template_group"], window_index=index,
                      baseline_start_in_recording=start, baseline_end_in_recording=end,
                      raw_recording_text=source["raw_text"], prepared_recording_text=baseline,
                      desired_text=desired, label_status=label_status, label_origin=origin,
                      suggested_tasks=[task] if task else [], category=source.get("category"),
                      audio_path=source.get("audio_path"), audio_sha256=source.get("audio_sha256"),
                      human_audio_gold=False, audio_script_verified=False,
                      training_eligible=origin == "controlled_text" and desired is not None,
                      allowed_usage="text_pretraining_only" if origin == "controlled_text" else "review_only",
                      independent_acceptance_eligible=False,
                      available_context=source.get("context", ""),
                      left_context=baseline[max(0, start - 80):start], right_context=baseline[end:end + 80],
                      window_pipeline=PIPELINE, **window)
        if desired is not None:
            record.update(align_edits(text, desired, task))
        else:
            record.update(edits=None, character_tags=None, insertion_gaps=None)
        record["review_basis_sha256"] = review_basis(record)
        records.append(record)
    return records


def completed_segments(source):
    events, seen = [], set()
    for payload in (*source.get("chunk_payloads", []), source.get("final_payload", {})):
        for event in payload.get("completed_segments", []):
            key = event.get("segment_id")
            if isinstance(key, int) and key not in seen and isinstance(event.get("text"), str):
                events.append(event["text"])
                seen.add(key)
    return events


def build(args):
    from system.entity_store import EntityStore
    from system.zh_itn_bridge import ZhITNNormalizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = read(args.sources)
    if any(row.get("synthetic_split") != "train" for row in sources):
        raise ValueError("Only original training-source ASR views may be imported")
    with sqlite3.connect(f"file:{args.entities.resolve()}?mode=ro", uri=True) as original, \
         sqlite3.connect(output / "entities_snapshot.db") as snapshot:
        original.backup(snapshot)
    definitions = EntityStore(output / "entities_snapshot.db").list_entities(domain="general")
    preparer = Preparer(ZhITNNormalizer(args.zh_itn), definitions)
    records = []
    for row in read(args.scripts):
        if row["synthetic_split"] != "train":
            continue
        source = {**row, "raw_text": row["spoken_text"], "split": "train",
                  "recording_group": "synthetic_text:" + row["seed_id"]}
        records += make_records(preparer, source, origin="controlled_text", target=row["intended_text"],
                                status="controlled_script_text", task=TASK_BY_CATEGORY.get(row["category"]))
    for row in read(args.probes):
        source = dict(seed_id=row["id"], raw_text=row["input"], recording_group="controlled_probe:" + row["id"],
                      split="train", template_group="controlled_probe:" + row["id"], category=row["task"] or "keep",
                      context=row["context"], entity_candidates=row.get("entity_candidates", []))
        records += make_records(preparer, source, origin="controlled_text", target=row["expected"],
                                status="controlled_task_definition", task=row["task"])
    for row in sources:
        source = {**row, "split": "train"}
        records += make_records(preparer, source, origin="asr_script_proposal", target=None,
                                status="unverified_asr", task=TASK_BY_CATEGORY.get(row["category"]),
                                source_segments=completed_segments(row))
    for index, row in enumerate(read(args.model_replay)):
        source = dict(seed_id=f"model_replay:{index}", raw_text=row["raw_text"],
                      recording_group="consumed_replay:" + row["seed_id"],
                      split="regression", template_group="consumed_replay:" + row["seed_id"],
                      category="model_output", context="")
        proposals = make_records(preparer, source, origin="refiner_output_proposal", target=row["final_text"],
                                 status="refiner_final_output_unverified")
        for proposal in proposals:
            proposal["teacher_provenance"] = dict(policy=row["policy"], repetition=row["repetition"],
                source_file=str(args.model_replay), source_row=index, final_output=row["final_text"],
                rejected_reasons=row["final"]["refiner_reject_reasons"])
        records += proposals
    # Numeric fixtures are consumed regression expectations, not detector train.
    numeric_excluded = 0
    for row in read(args.numeric_fixtures):
        if row["behavior"] != "cleanup":
            numeric_excluded += 1
            continue
        source = dict(seed_id=row["id"], raw_text=row["input"], recording_group="numeric_regression:" + row["id"],
                      split="regression", template_group="numeric_cleanup", category=row["category"], context="")
        records += make_records(preparer, source, origin="text_regression", target=row["expected"],
                                status="text_fixture_not_audio_gold", task="self_correction")
    index = {record["id"]: record for record in records}
    if len(index) != len(records):
        raise ValueError("Duplicate record identities")
    if args.reviews:
        reviewed = set()
        for review in read(args.reviews):
            if review["id"] in reviewed or review["id"] not in index:
                raise ValueError("Duplicate or unknown review")
            reviewed.add(review["id"])
            index[review["id"]] = attach_review(index[review["id"]], review)
        records = list(index.values())
    collections = {
        "records": records,
        "controlled_text_train": [r for r in records if r["label_origin"] == "controlled_text" and r["training_eligible"]],
        "asr_review_queue": [r for r in records if r["label_origin"] == "asr_script_proposal" and not r["human_audio_gold"]],
        "verified_audio_train": [r for r in records if r["human_audio_gold"] and r["training_eligible"]],
        "refiner_output_review": [r for r in records if r["label_origin"] == "refiner_output_proposal"],
        "text_regression": [r for r in records if r["label_origin"] == "text_regression"],
    }
    for name, rows in collections.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    # Small, diverse queue first; repeated model policies/views should not
    # make a reviewer inspect the same text edit dozens of times.
    priority = []
    seen_proposals = set()
    candidates = sorted(collections["refiner_output_review"] + collections["asr_review_queue"],
                        key=lambda row: (not bool(row["edits"]), row["label_origin"] != "refiner_output_proposal"))
    for row in candidates:
        signature = digest([row["baseline_text"], row["desired_text"], row["available_context"]])
        if signature in seen_proposals:
            continue
        seen_proposals.add(signature)
        priority.append(row)
        if len(priority) == 40:
            break
    write_jsonl(output / "priority_review.jsonl", priority)
    code_files = [Path(__file__), ROOT / "system/edit_detection_data.py", ROOT / "system/window_refinement.py",
                  ROOT / "system/entity_pipeline.py", ROOT / "system/refinement_guard.py"]
    (output / "executed_code").mkdir()
    for file in code_files:
        (output / "executed_code" / file.name).write_bytes(file.read_bytes())
    manifest = dict(scope="training_text_supervision_and_unverified_ASR_review_proposals", no_training_started=True,
                    records=len(records), files={name: len(rows) for name, rows in collections.items()},
                    label_status=dict(Counter(r["label_status"] for r in records)),
                    origin=dict(Counter(r["label_origin"] for r in records)),
                    controlled_edit_tasks=dict(Counter(edit["task"] for r in collections["controlled_text_train"] for edit in r["edits"])),
                    asr_proposed_edit_tasks=dict(Counter(edit["task"] for r in collections["asr_review_queue"] for edit in (r["edits"] or []))),
                    controlled_keep_windows=sum(not r["edits"] for r in collections["controlled_text_train"]),
                    controlled_unique_baseline_target_pairs=len({(r["baseline_text"], r["desired_text"]) for r in collections["controlled_text_train"]}),
                    multi_edit_ambiguity_unchecked=sum(r.get("alignment_ambiguity_checked") is False for r in records),
                    priority_review_records=len(priority),
                    numeric_rule_only_fixtures_excluded=numeric_excluded,
                    asr_source_views=len(sources), human_audio_gold=sum(r["human_audio_gold"] for r in records),
                    original_dev_test_not_used_for_training=True, consumed_replay_is_review_only=True,
                    no_independent_acceptance_data=True,
                    window_pipeline=PIPELINE,
                    source_sha256={str(path): sha(path) for path in (args.sources, args.scripts, args.probes, args.numeric_fixtures, args.model_replay, args.entities, args.zh_itn)},
                    code_sha256={str(path.relative_to(ROOT)): sha(path) for path in code_files},
                    output_sha256={file.name: sha(file) for file in output.glob("*.jsonl")})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=ROOT / "results/gate_terminal_gain_round4_20261012/train_collection/sources.jsonl")
    parser.add_argument("--scripts", type=Path, default=ROOT / "results/gate_synthetic_20261009/texts.jsonl")
    parser.add_argument("--probes", type=Path, default=ROOT / "experiments/fixtures/edit_detection_controlled.jsonl")
    parser.add_argument("--numeric-fixtures", type=Path, default=ROOT / "experiments/fixtures/numeric_refinement_gold.jsonl")
    parser.add_argument("--entities", type=Path, default=ROOT / "results/gate_terminal_gain_round4_20261012/train_collection/runtime/entities_snapshot.db")
    parser.add_argument("--zh-itn", type=Path, default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--model-replay", type=Path, default=ROOT / "results/gate_fixed_pipeline_replay_20261013_v2/sessions.jsonl")
    build(parser.parse_args())
