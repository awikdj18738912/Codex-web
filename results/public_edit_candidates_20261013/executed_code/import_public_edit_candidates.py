"""Import author data without turning standardization into minimal ASR gold.

Spoken pairs are positive-only review proposals. Unedited characters have no
KEEP labels. SIGHAN2015 generated benchmark variants are never fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import align_edits, digest
from system.refinement_guard import reject_reasons, restore_rule_numeric_surfaces


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate_pair(source, target, normalizer):
    record = dict(raw_source=source, author_target=target, baseline_text=None, prepared_target=None,
                  edits=None, positive_only_character_labels=None, untouched_characters_labelled_keep=False,
                  human_audio_gold=False, training_eligible=False, production_eligible=False,
                  allowed_usage="review_only", project_intent_verified=False,
                  entity_preparation_applied=False, baseline_segment_scope="whole_author_line")
    if not source.strip() or not target.strip():
        return {**record, "status": "empty_pair"}
    if "<" in source or ">" in source or "<" in target or ">" in target:
        return {**record, "status": "unsupported_or_malformed_markup"}
    baseline = normalizer.normalize(source).text
    prepared_target = normalizer.normalize(target).text
    record.update(baseline_text=baseline, prepared_target=prepared_target)
    aligned = align_edits(baseline, prepared_target)
    record.update(edits=aligned["edits"], ambiguous_offsets=aligned["ambiguous_offsets"],
                  alignment_ambiguity_checked=aligned["alignment_ambiguity_checked"],
                  offset_alternatives=aligned["offset_alternatives"],
                  offset_unit="unicode_code_point", interval="[start,end)")
    if not aligned["edits"]:
        return {**record, "status": "author_unchanged_not_global_keep_gold"}
    if restore_rule_numeric_surfaces(baseline, prepared_target) != prepared_target:
        return {**record, "status": "numeric_integrity_conflict"}
    if any(edit["operation"] != "delete" for edit in aligned["edits"]):
        return {**record, "status": "non_deletion_standardization_requires_review"}
    if any(edit["end"] - edit["start"] > 16 for edit in aligned["edits"]):
        return {**record, "status": "large_deletion_requires_review"}
    if len(prepared_target) / len(baseline) < .75:
        return {**record, "status": "high_deletion_ratio_requires_review"}
    reasons = reject_reasons(baseline, prepared_target, numeric_rule_mode=True)
    if reasons:
        return {**record, "status": "project_integrity_guard_conflict", "guard_reasons": list(reasons)}
    if len(baseline) > 240:
        return {**record, "status": "long_author_line_requires_partition_review"}
    labels = [-100] * len(baseline)
    for edit in aligned["edits"]:
        labels[edit["start"]:edit["end"]] = [1] * (edit["end"] - edit["start"])
    return {**record, "status": "bounded_deletion_candidate_not_gold", "positive_only_character_labels": labels}


def build(args):
    from system.zh_itn_bridge import ZhITNNormalizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    normalizer = ZhITNNormalizer(args.zh_itn)
    spoken = args.raw / "chinese_disfluency/data/chinese_disfluency"
    source_file, target_file = spoken / "dis.raw.3k", spoken / "flu.raw.3k"
    sources, targets = source_file.read_text().splitlines(), target_file.read_text().splitlines()
    if len(sources) != len(targets):
        raise ValueError("Parallel file row counts differ; no zip truncation")
    rows = []
    for index, (source, target) in enumerate(zip(sources, targets)):
        row = candidate_pair(source, target, normalizer)
        row.update(id=digest(["chinese_disfluency", index, source, target])[:24], source_line=index + 1,
                   source_dataset="Translation_Disfluency_Detection",
                   upstream_annotation_role="speech_translation_standardization_not_minimal_cleanup",
                   source_commit="0456baded348fa9f8617c6e0a14757fe7b7c31d0",
                   source_url="https://github.com/dqqcasia/Translation_Disfluency_Detection",
                   license_status="not_declared_in_downloaded_repository", split="unassigned_review",
                   speaker_or_document_group=None, document_boundary_verified=False)
        rows.append(row)
    benchmark_file = args.raw / "decopy/data/output/sighan2015/char_ty_ys_xs.json"
    benchmark = []
    for index, row in enumerate(json.loads(benchmark_file.read_text())):
        error, expected = row["replaced_text"], row["original_text"]
        aligned = align_edits(error, expected, "local_typo")
        benchmark.append(dict(id=digest(["decopy_test_variant", index, error, expected])[:24],
            source_row=index, raw_source=error, author_target=expected, edits=aligned["edits"],
            source_dataset="Decopy_SIGHAN2015_generated_test_variant", source_file=str(benchmark_file),
            upstream_wrong_ids=row.get("wrong_ids"), split="benchmark_quarantine",
            source_role="generated_variant_of_1100_SIGHAN2015_benchmark_sentences",
            human_audio_gold=False, human_annotation_of_injected_error_verified=False,
            training_eligible=False, production_eligible=False, allowed_usage="benchmark_review_only",
            license_status="repository_data_license_not_declared"))
    candidates = [row for row in rows if row["status"] == "bounded_deletion_candidate_not_gold"]
    for name, values in (("spoken_pairs", rows), ("bounded_deletion_review", candidates), ("spelling_benchmark_quarantine", benchmark)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    files = [source_file, target_file, spoken / "chinese_annotation_rules_2018.pdf", benchmark_file,
             args.raw / "source_registry.json", args.raw / "encoder_registry.json"]
    manifest = dict(spoken_pairs=len(rows), statuses=dict(Counter(row["status"] for row in rows)),
                    bounded_deletion_candidates=len(candidates),
                    untyped_candidate_operations=sum(len(row["edits"]) for row in candidates),
                    distinct_deleted_strings=len({edit["source"] for row in candidates for edit in row["edits"]}),
                    spelling_benchmark_quarantine=len(benchmark), benchmark_used_for_training=False,
                    positive_only_supervision_proposals=True, global_KEEP_gold_created=False,
                    human_audio_gold=0, training_eligible_records=0, document_groups_unknown=True,
                    source_sha256={str(file): sha(file) for file in files},
                    code_sha256=sha(Path(__file__)),
                    output_sha256={file.name: sha(file) for file in output.glob("*.jsonl")},
                    limitations=["Public spoken targets include simplification and grammatical rewriting",
                                 "Source license not declared; candidates remain review-only",
                                 "No audio or verified speaker/document boundaries",
                                 "Algorithmic integrity pass is not proof of intended deletion",
                                 "Full mixed Decopy train file not downloaded or hash-verified"])
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "executed_code").mkdir()
    (output / "executed_code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/public_edit_sources_20261013")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zh-itn", type=Path, default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    build(parser.parse_args())
