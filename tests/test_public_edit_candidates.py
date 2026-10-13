from __future__ import annotations

from types import SimpleNamespace
import unittest

from experiments.scripts.import_public_edit_candidates import candidate_pair
from system.edit_detection_data import apply_edits


class IdentityNormalizer:
    def normalize(self, text):
        return SimpleNamespace(text=text)


class PublicEditCandidatesTest(unittest.TestCase):
    def pair(self, source, target):
        return candidate_pair(source, target, IdentityNormalizer())

    def test_malformed_or_unknown_tags_are_not_interpreted(self):
        for source, target in (("请确认。", "<restart>请确认。"), ("请<确认。", "请确认。")):
            self.assertEqual(self.pair(source, target)["status"], "unsupported_or_malformed_markup")

    def test_unchanged_author_target_is_not_complete_KEEP_supervision(self):
        row = self.pair("名单，名单交给调度员。", "名单，名单交给调度员。")
        self.assertEqual(row["status"], "author_unchanged_not_global_keep_gold")
        self.assertIsNone(row["positive_only_character_labels"])
        self.assertFalse(row["training_eligible"])

    def test_numeric_value_changes_are_not_language_cleanup_labels(self):
        self.assertEqual(self.pair("不能超过55度。", "不能超过5度。")["status"], "numeric_integrity_conflict")

    def test_short_deletion_only_has_positive_locations_and_unknown_elsewhere(self):
        row = self.pair("材料都已备齐，嗯，可以开始。", "材料都已备齐，可以开始。")
        self.assertEqual(row["status"], "bounded_deletion_candidate_not_gold")
        self.assertEqual(set(row["positive_only_character_labels"]), {-100, 1})
        self.assertEqual(apply_edits(row["baseline_text"], row["edits"]), row["prepared_target"])
        self.assertFalse(row["human_audio_gold"])
        self.assertFalse(row["untouched_characters_labelled_keep"])

    def test_subject_loss_or_extensive_rewrite_is_not_safe_candidate(self):
        row = self.pair("耳环比较好是加六的。", "比较好是加六的。")
        self.assertNotEqual(row["status"], "bounded_deletion_candidate_not_gold")
        row = self.pair("会议安排我们之后再通知。", "我们之后通知会议安排。")
        self.assertEqual(row["status"], "non_deletion_standardization_requires_review")

    def test_emphatic_repetition_loss_is_rejected(self):
        row = self.pair("这个支点很重要，很重要，不能随意移动。", "这个支点很重要，不能随意移动。")
        self.assertEqual(row["status"], "project_integrity_guard_conflict")


if __name__ == "__main__":
    unittest.main()
