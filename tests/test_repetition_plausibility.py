"""Threshold and failure-mode checks for the generic repetition scorer."""

import unittest
from unittest.mock import patch

from system.repetition_plausibility import RepetitionPlausibility


class RepetitionPlausibilityTest(unittest.TestCase):
    def test_only_clear_model_margin_authorizes_deletion(self) -> None:
        scorer = RepetitionPlausibility()
        with patch.object(scorer, "_ensure_model", return_value=True):
            for edited_score, expected in (
                (-1.0, "remove"),
                (-1.9, "unresolved"),
                (-2.8, "keep"),
            ):
                with self.subTest(edited_score=edited_score):
                    with patch.object(scorer, "_mean_log_probability",
                                      side_effect=[edited_score, -2.0]):
                        result = scorer.decide("首首先。", 0, 2, "首")
                    self.assertEqual(result.decision, expected)

    def test_missing_model_never_deletes(self) -> None:
        scorer = RepetitionPlausibility()
        with patch.object(scorer, "_ensure_model", return_value=False):
            result = scorer.decide("首首先。", 0, 2, "首")
        self.assertEqual(result.decision, "unresolved")

    def test_long_sentence_is_not_truncated_into_false_evidence(self) -> None:
        scorer = RepetitionPlausibility()
        result = scorer.decide("首首" + "先" * 100, 0, 2, "首")
        self.assertEqual(result.decision, "unresolved")
