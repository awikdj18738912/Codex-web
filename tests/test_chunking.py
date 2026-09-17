import unittest

from system.chunking import (
    Chunk,
    ChunkManager,
    find_self_correction_antecedent,
    is_filler_only,
    merge_boundary_anomaly_chunks,
    merge_self_correction_chunks,
)


class ChunkManagerTest(unittest.TestCase):
    def test_one_punctuation_mode_emits_comma_delimited_chunks(self):
        source = "第一段，第二段。第三段！"
        chunks = ChunkManager(80, one_punctuation_window=True).update(
            source, vad_boundary=True
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["第一段，", "第二段。", "第三段！"],
        )

    def test_short_self_correction_stays_together(self):
        source = "我有一个苹果，不对，我有一个梨。"
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual([chunk.text for chunk in chunks], [source])

    def test_long_self_corrections_use_soft_clause_boundaries(self):
        source = (
            "你好，我有一个苹果，不对，我有一个梨，我有一箱苹果，不对，我有一箱梨，"
            "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。"
        )
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "你好，我有一个苹果，不对，我有一个梨，",
                "我有一箱苹果，不对，我有一箱梨，",
                "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。",
            ],
        )
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))

    def test_nearest_clause_boundary_keeps_numeric_correction_together(self):
        # A sentence terminator must close the corrected clause.  Searching
        # only soft punctuation jumps to a far-away comma, producing an
        # oversized chunk that pushes the corrected value out of the
        # K-window refinement range.
        source = (
            "你好，我有一个苹果。不对，我有一个梨。我有百分之三十七的概率在"
            "二零一五年十二月五日这天获得一千二百三十四万五千四百三十一元。"
            "不对，应该是一百五十块。不对，应该是一百块。"
        )
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "你好，我有一个苹果。不对，我有一个梨。",
                "我有百分之三十七的概率在二零一五年十二月五日这天获得"
                "一千二百三十四万五千四百三十一元。不对，应该是一百五十块。"
                "不对，应该是一百块。",
            ],
        )
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))

    def test_sentence_before_correction_is_merged(self):
        source = "你好，我有一个苹果。不对，我有一个梨。我有一千元。"
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["你好，我有一个苹果。不对，我有一个梨。", "我有一千元。"],
        )

    def test_self_correction_backtracks_over_filler_chunks(self):
        chunks = [
            Chunk(0, "然后我有一箱苹果，"),
            Chunk(1, "嗯，"),
            Chunk(2, "不对我有一箱梨。"),
        ]
        self.assertTrue(is_filler_only("嗯，"))
        self.assertEqual(find_self_correction_antecedent(chunks, 2), 0)
        self.assertEqual(
            [chunk.text for chunk in merge_self_correction_chunks(chunks)],
            ["然后我有一箱苹果，嗯，不对我有一箱梨。"],
        )

    def test_self_correction_does_not_cross_hard_boundary_after_filler(self):
        chunks = [
            Chunk(0, "我昨天买了苹果。"),
            Chunk(1, "嗯，"),
            Chunk(2, "不对，我今天买的是梨。"),
        ]
        self.assertIsNone(find_self_correction_antecedent(chunks, 2))
        self.assertEqual(
            [chunk.text for chunk in merge_self_correction_chunks(chunks)],
            [chunk.text for chunk in chunks],
        )

    def test_boundary_anomaly_is_merged_across_punctuation_chunks(self):
        chunks = [
            Chunk(0, "但是三番五次的。"),
            Chunk(1, "的提醒就不太好了，"),
        ]
        self.assertEqual(
            [chunk.text for chunk in merge_boundary_anomaly_chunks(chunks)],
            ["但是三番五次的。的提醒就不太好了，"],
        )

    def test_boundary_anomaly_lexical_continuation_is_not_merged(self):
        chunks = [Chunk(0, "是的。"), Chunk(1, "的确如此。")]
        self.assertEqual(
            [chunk.text for chunk in merge_boundary_anomaly_chunks(chunks)],
            [chunk.text for chunk in chunks],
        )


if __name__ == "__main__":
    unittest.main()
