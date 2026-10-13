from __future__ import annotations

import unittest

from system.edit_detection_data import align_edits, apply_edits, attach_review, review_basis
from experiments.scripts.prepare_edit_detection_data import project_script_deletions, make_records, Preparer


class IdentityNormalizer:
    def normalize(self, text):
        from types import SimpleNamespace
        return SimpleNamespace(text=text, changes=[])


class EditDetectionDataTest(unittest.TestCase):
    def test_offsets_roundtrip_and_repeated_copy_ambiguity(self):
        source, target = "请确确认编号。", "请确认编号。"
        result = align_edits(source, target, "repetition")
        self.assertEqual(apply_edits(source, result["edits"]), target)
        self.assertTrue(result["ambiguous_offsets"])
        self.assertEqual(result["offset_alternatives"], [{"start": 1, "end": 2}, {"start": 2, "end": 3}])
        self.assertEqual(len(result["character_tags"]), len(source))

    def test_insertions_have_gap_labels_including_end_of_text(self):
        result = align_edits("材料齐备", "材料齐备。", "boundary_punctuation")
        self.assertEqual(result["insertion_gaps"], [{"offset": 4, "kind": "CORRECTION_CANDIDATE"}])
        self.assertEqual(result["character_tags"], ["KEEP"] * 4)

    def test_disjoint_edits_keep_source_coordinates(self):
        result = align_edits("嗯，请确确认编号。。", "请确认编号。", "repetition")
        self.assertEqual(apply_edits("嗯，请确确认编号。。", result["edits"]), "请确认编号。")
        self.assertGreater(len(result["edits"]), 1)
        self.assertFalse(result["alignment_ambiguity_checked"])
        self.assertIsNone(result["ambiguous_offsets"])

    def test_keep_and_invalid_offsets(self):
        self.assertEqual(align_edits("一张一张清点。", "一张一张清点。") ["edits"], [])
        with self.assertRaises(ValueError):
            apply_edits("甲乙", [{"start": 0, "end": 1, "source": "乙", "target": ""}])

    def test_script_projection_preserves_unrelated_asr_errors(self):
        target, status = project_script_deletions("需要需要周清确认", "需要周清确认", "需要需要周青确认")
        self.assertIsNone(target)  # ambiguous duplicate position + unrelated substitution
        self.assertEqual(status, "ambiguous_script_projection")
        target, status = project_script_deletions("嗯，需要周清确认", "需要周清确认", "嗯，需要周青确认")
        self.assertEqual(target, "，需要周青确认")
        self.assertEqual(status, "projected_script_deletion_unverified_audio")

    def test_spelling_or_numeric_changes_are_not_script_deletion_labels(self):
        self.assertIsNone(project_script_deletions("五元", "六元", "五元")[0])
        self.assertIsNone(project_script_deletions("材料备齐", "材料备齐", "资料备齐")[0])

    def record(self, split="train"):
        record = dict(id="sample", raw_asr="请确确认编号。", baseline_text="请确确认编号。",
                      entity_candidates=[], available_context="", recording_group="recording",
                      audio_path="recording.wav", audio_sha256="example_digest", split=split, window_pipeline={})
        record["review_basis_sha256"] = review_basis(record)
        return record

    def review(self, record, kind="assistant_text_only"):
        return dict(id=record["id"], review_basis_sha256=review_basis(record), reviewer="test_reviewer",
                    review_kind=kind, audio_checked=kind == "human_audio_cleaned",
                    desired_text="请确认编号。", edit_tasks=["repetition"])

    def test_text_review_never_creates_audio_gold(self):
        record = self.record()
        result = attach_review(record, self.review(record))
        self.assertFalse(result["human_audio_gold"])
        self.assertFalse(result["training_eligible"])

    def test_stale_review_cannot_change_new_source(self):
        record = self.record()
        review = self.review(record)
        record["baseline_text"] = "请查阅附件。"
        with self.assertRaisesRegex(ValueError, "Stale"):
            attach_review(record, review)

    def test_context_changes_also_invalidate_review(self):
        record = self.record()
        review = self.review(record)
        record["right_context"] = "这是引用原话，要求保留重复。"
        with self.assertRaisesRegex(ValueError, "Stale"):
            attach_review(record, review)

    def test_audio_review_requires_provenance_and_never_moves_regression_into_train(self):
        record = self.record("regression")
        reviewed = attach_review(record, self.review(record, "human_audio_cleaned"))
        self.assertTrue(reviewed["human_audio_gold"])
        self.assertFalse(reviewed["training_eligible"])
        self.assertFalse(reviewed["independent_acceptance_eligible"])
        record["audio_path"] = None
        with self.assertRaises(ValueError):
            attach_review(record, self.review(record, "human_audio_cleaned"))

    def test_reviewed_numeric_value_changes_are_rejected(self):
        record = self.record()
        record["baseline_text"] = "预算12万元。"
        review = self.review(record)
        review.update(desired_text="预算13万元。", edit_tasks=["local_typo"])
        with self.assertRaisesRegex(ValueError, "numeric"):
            attach_review(record, review)

    def test_fixed_window_partition_and_source_owned_targets(self):
        source = dict(seed_id="seed", raw_text="材料备齐，会议开始，嗯，可以发言了。",
                      recording_group="recording", split="train", template_group="family", category="filler")
        rows = make_records(Preparer(IdentityNormalizer(), []), source, origin="controlled_text",
                            target="材料备齐，会议开始，可以发言了。", status="controlled_script_text", task="disfluency")
        self.assertEqual([row["chunk_count"] for row in rows], [3, 1])
        self.assertTrue(all(row["training_eligible"] for row in rows))
        self.assertTrue(all(not row["human_audio_gold"] for row in rows))
        self.assertEqual("".join(row["desired_text"] for row in rows), "材料备齐，会议开始，可以发言了。")
        for row in rows:
            self.assertEqual(apply_edits(row["baseline_text"], row["edits"]), row["desired_text"])

    def test_upstream_numeric_formatting_does_not_create_detection_edits(self):
        class NumericNormalizer(IdentityNormalizer):
            def normalize(self, text):
                return super().normalize(text.replace("百分之五", "5%"))
        source = dict(seed_id="numeric", raw_text="增长了百分之五。", recording_group="numeric",
                      split="train", template_group="numeric", category="clean")
        row = make_records(Preparer(NumericNormalizer(), []), source, origin="controlled_text",
                           target="增长了5%。", status="controlled_script_text")[0]
        self.assertEqual(row["edits"], [])
        self.assertEqual(row["baseline_text"], "增长了5%。")


if __name__ == "__main__":
    unittest.main()
