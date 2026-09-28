"""Threshold and failure-mode checks for the generic repetition scorer."""

import unittest
from unittest.mock import patch

from system.repetition_plausibility import RepetitionPlausibility


class RepetitionPlausibilityTest(unittest.TestCase):
    def test_short_context_recovers_local_repeat_without_changing_lexical_pair(self) -> None:
        scorer = RepetitionPlausibility()
        source = "天下熙熙皆为利来，天下攘攘皆为利利往，所以没有正义。"
        cases = (
            ("利利", "利", "攘攘皆为利利往，所以", "攘攘皆为利往，所以", -1.6, "remove"),
            ("攘攘", "攘", "来，天下攘攘皆为利利", "来，天下攘皆为利利", -2.9, "keep"),
        )
        for repeated, target, original_window, edited_window, edited_score, expected in cases:
            with self.subTest(repeated=repeated):
                start = source.index(repeated)
                score_by_text = {original_window: -2.0, edited_window: edited_score}
                with (
                    patch.object(scorer, "_ensure_model", return_value=True),
                    patch.object(scorer, "_mean_log_probability",
                                 side_effect=score_by_text.__getitem__) as score,
                ):
                    result = scorer.decide_short_context(
                        source, start, start + len(repeated), target
                    )
                self.assertEqual(result.decision, expected)
                self.assertEqual(score.call_count, 2)

    def test_short_context_never_collapses_longer_run_to_one_character(self) -> None:
        scorer = RepetitionPlausibility()
        with patch.object(scorer, "_ensure_model") as load_model:
            result = scorer.decide_short_context("谢谢谢谢老师。", 0, 4, "谢")
        self.assertEqual(result.decision, "unresolved")
        load_model.assert_not_called()

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
