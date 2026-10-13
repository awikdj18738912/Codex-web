import csv
import json

import pytest

from experiments.scripts import export_gate_reviews as exporter


def annotation(case_id="a", baseline="现在来来做推演。", output="现在来做推演。"):
    row = dict.fromkeys(exporter.BASIS_FIELDS, "")
    row.update(case_id=case_id, task_kind="window_refine", data_kind="human_news",
               recording_group="recording:1", baseline=baseline, always_output=output,
               current_output=output, audio_path="recording.wav",
               automatic_pair_label="neutral" if baseline == output else "",
               automatic_label_basis="identical_final_output" if baseline == output else "",
               human_label="", reviewer="", rationale="", reference_status="unverified_reference")
    return row


def review(row, label="benefit"):
    return dict(case_id=row["case_id"], assistant_label=label,
                assistant_confidence="medium", assistant_category="repetition_cleanup",
                assistant_reason="删除明显起头重复，仍需人工确认。",
                assistant_harm_severity="none", assistant_needs_audio=label == "uncertain",
                assistant_audio_checked=False, assistant_status="auxiliary_unconfirmed",
                assistant_train_eligible=False, review_basis_sha256=exporter.basis_hash(row),
                baseline_sha256=exporter.text_hash(row["baseline"]),
                always_output_sha256=exporter.text_hash(row["always_output"]))


@pytest.mark.parametrize("field", ["baseline", "always_output", "reference_full", "raw_window", "candidate_context"])
def test_changed_evidence_invalidates_auxiliary_review(field):
    row = annotation()
    decision = review(row)
    row[field] += "新的内容"
    with pytest.raises(ValueError, match="Stale review"):
        exporter.attach_reviews([row], [decision])


def test_human_decision_is_preserved_even_when_disagreeing_with_assistant():
    row = annotation()
    decision = review(row)
    row.update(human_label="harm", reviewer="human", rationale="回听后的结论",
               reference_status="verified_intended")
    merged = exporter.attach_reviews([row], [decision])[0]
    assert merged["human_label"] == "harm"
    assert merged["assistant_label"] == "benefit"
    assert merged["rationale"] == "回听后的结论"
    assert merged["reviewer"] == "human"
    assert merged["reference_status"] == "verified_intended"
    assert merged["assistant_train_eligible"] is False
    assert "assistant_label" not in row


@pytest.mark.parametrize("mutation", [
    {"assistant_train_eligible": True}, {"assistant_status": "gold"},
    {"assistant_audio_checked": True}, {"assistant_confidence": 0.99},
    {"assistant_label": "uncertain", "assistant_needs_audio": False},
    {"assistant_label": "harm", "assistant_harm_severity": "none"},
])
def test_text_review_cannot_claim_audio_verification_or_training_gold(mutation):
    row = annotation()
    decision = {**review(row), **mutation}
    with pytest.raises(ValueError):
        exporter.attach_reviews([row], [decision])


def test_complete_coverage_required_and_identical_outputs_stay_separate():
    changed = annotation()
    neutral = annotation("b", baseline="明天开会。", output="明天开会。")
    with pytest.raises(ValueError, match="cover exactly"):
        exporter.attach_reviews([changed, neutral], [])
    with pytest.raises(ValueError, match="cover exactly"):
        exporter.attach_reviews([changed, neutral], [review(changed), review(neutral)])
    merged = exporter.attach_reviews([changed, neutral], [review(changed)])
    assert merged[1]["automatic_pair_label"] == "neutral"
    assert merged[1]["assistant_label"] == ""
    assert all(row["human_label"] == "" for row in merged)
    with pytest.raises(ValueError, match="Duplicate"):
        exporter.attach_reviews([changed], [review(changed), review(changed)])


def put_fixture(tmp_path):
    row = annotation()
    exporter.write_csv(tmp_path / "annotations.csv", [row], list(row))
    arms = {key: dict(comparison_text=row[field]) for key, field in
            (("rules", "baseline"), ("current", "current_output"), ("always", "always_output"))}
    pair = dict(case_id=row["case_id"], arms=arms)
    case = {key: row[key] for key in ("case_id", "task_kind", "data_kind", "recording_group", "audio_path", "reference_full")}
    case.update(raw_text=row["raw_window"], raw_text_full=row["raw_full"])
    (tmp_path / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n")
    manifest = dict(review_method="individual_text_comparison_no_audio", reviews=[review(row, "uncertain")])
    (tmp_path / "assistant_review_decisions.json").write_text(json.dumps(manifest))
    return pair


def test_export_writes_review_queue_and_does_not_touch_original_annotations(tmp_path):
    put_fixture(tmp_path)
    before = (tmp_path / "annotations.csv").read_bytes()
    summary = exporter.export(tmp_path)
    assert summary["audio_checked"] == summary["auxiliary_train_eligible"] == summary["human_labels_present"] == 0
    assert summary["auxiliary_label_counts"] == {"uncertain": 1}
    assert (tmp_path / "annotations.csv").read_bytes() == before
    with (tmp_path / "audio_review_queue.csv").open(encoding="utf-8-sig", newline="") as stream:
        queue = list(csv.DictReader(stream))
    assert queue[0]["review_priority"] == "1"
    assert queue[0]["audio_scope"] == "full_recording"
    assert queue[0]["human_label"] == ""
    assert queue[0]["assistant_label"] == "uncertain"
    decision = json.loads((tmp_path / "assistant_reviews.jsonl").read_text())
    assert decision["assistant_audio_checked"] is False
    assert decision["assistant_train_eligible"] is False
    assert "human_label" not in decision


@pytest.mark.parametrize("failure", ["stale_pair", "failed_operation"])
def test_invalid_pair_stops_export_before_writing(tmp_path, failure):
    pair = put_fixture(tmp_path)
    if failure == "stale_pair":
        pair["arms"]["always"]["comparison_text"] = "不同输出。"
    else:
        pair["arms"]["always"]["session_stats"] = dict(failed_call_count=1)
    (tmp_path / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError):
        exporter.export(tmp_path)
    assert not (tmp_path / "assistant_reviews.jsonl").exists()
