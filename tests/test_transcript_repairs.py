from __future__ import annotations

import unittest

from system.deterministic_cleanup import clean_transcript_deterministically
from system.transcript_repairs import (
    KNOWN_TRANSCRIPT_REPAIRS,
    apply_known_transcript_repairs,
)


class KnownTranscriptRepairTest(unittest.TestCase):
    def test_reviewed_artifacts_are_repaired_exactly(self) -> None:
        for repair in KNOWN_TRANSCRIPT_REPAIRS:
            with self.subTest(reason=repair.reason, source=repair.source):
                self.assertEqual(
                    apply_known_transcript_repairs(repair.source), repair.target
                )

    def test_repairs_run_before_final_deterministic_cleanup(self) -> None:
        source = (
            "哺育出灿烂的华夏。在文明，过过度放牧，"
            "一朵朵云，一朵朵云。"
        )
        self.assertEqual(
            clean_transcript_deterministically(source),
            "哺育出灿烂的华夏文明，过度放牧，一朵朵云。",
        )

    def test_unreviewed_sentence_is_not_rewritten(self) -> None:
        source = "普通句子。这里的句号可能有别的语义。"
        self.assertEqual(apply_known_transcript_repairs(source), source)


if __name__ == "__main__":
    unittest.main()
