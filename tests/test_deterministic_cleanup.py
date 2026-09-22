from __future__ import annotations

import unittest

from system.deterministic_cleanup import (
    collapse_boundary_echo_fields,
    collapse_repeated_character_stutters,
    clean_transcript_deterministically,
    collapse_repeated_comma_items,
    collapse_repeated_short_utterances,
    collapse_sentence_boundary_fillers,
    collapse_standalone_fillers,
    detect_boundary_echo_repairs,
)


class DeterministicCleanupTest(unittest.TestCase):
    def test_three_identical_short_exclamations_collapse_to_one(self) -> None:
        self.assertEqual(
            collapse_repeated_short_utterances("前文。可恶！可恶！可恶！后文。"),
            "前文。可恶！后文。",
        )

    def test_trailing_unpunctuated_repetition_is_included(self) -> None:
        self.assertEqual(
            collapse_repeated_short_utterances("可恶！可恶！可恶"),
            "可恶！",
        )

    def test_two_repetitions_collapse_to_one(self) -> None:
        self.assertEqual(
            collapse_repeated_short_utterances("快跑！快跑！"),
            "快跑！",
        )

    def test_comma_separated_short_repetitions_collapse_to_one(self) -> None:
        self.assertEqual(
            collapse_repeated_comma_items("不，不，不，不，我赔。"),
            "不，我赔。",
        )
        self.assertEqual(
            collapse_repeated_comma_items("少主，少主，少主！"),
            "少主！",
        )

    def test_boundary_echo_after_clause_suffix_is_removed(self) -> None:
        self.assertEqual(
            collapse_boundary_echo_fields(
                "如果说你要询问我的话，话，你可以来我们这套"
            ),
            "如果说你要询问我的话，你可以来我们这套",
        )
        self.assertEqual(
            collapse_boundary_echo_fields("如果说你要询问我的话，话，"),
            "如果说你要询问我的话，",
        )
        self.assertEqual(
            clean_transcript_deterministically(
                "如果说你要询问我的话，话，你可以来我们这套"
            ),
            "如果说你要询问我的话，你可以来我们这套",
        )

    def test_boundary_echo_repairs_are_auditable(self) -> None:
        self.assertEqual(
            detect_boundary_echo_repairs("如果说你要询问我的话，话，你可以来"),
            (
                {
                    "source": "话，话，",
                    "target": "话，",
                    "echo": "话",
                    "start": 9,
                    "end": 13,
                },
            ),
        )

    def test_boundary_echo_run_is_collapsed_and_is_idempotent(self) -> None:
        source = "如果说你要询问我的话，话，话，你可以来"
        expected = "如果说你要询问我的话，你可以来"
        self.assertEqual(collapse_boundary_echo_fields(source), expected)
        self.assertEqual(collapse_boundary_echo_fields(expected), expected)

    def test_boundary_echo_does_not_remove_standalone_repetition(self) -> None:
        self.assertEqual(
            collapse_boundary_echo_fields("慢，慢，你再说"),
            "慢，慢，你再说",
        )
        self.assertEqual(
            collapse_boundary_echo_fields("这句话，话题很重要"),
            "这句话，话题很重要",
        )

    def test_sentence_boundary_echo_is_merged(self) -> None:
        self.assertEqual(
            clean_transcript_deterministically(
                "鬼灵门输。输给落云宗了。你。你竟结成了元婴。"
            ),
            "鬼灵门输给落云宗了。你竟结成了元婴。",
        )

    def test_latest_transcript_comma_repetitions_are_cleaned(self) -> None:
        source = (
            "嗯，这嘿，我，呀，不，嘿，不，不，不，不，我赔。"
            "少主，少主！你你来！少主，少主，少主！"
        )
        self.assertEqual(
            clean_transcript_deterministically(source),
            "嗯，这嘿，我，呀，不，嘿，不，我赔。少主！你来！少主！",
        )

    def test_nonidentical_comma_fields_are_preserved(self) -> None:
        source = "他说少主，少主快走，人人平等，天天向上。"
        self.assertEqual(collapse_repeated_comma_items(source), source)

    def test_different_or_long_utterances_are_preserved(self) -> None:
        source = "可恶！快跑！可恶！这是一句超过八个字的重复！" * 3
        self.assertEqual(collapse_repeated_short_utterances(source), source)

    def test_repeated_pronoun_stutter_is_collapsed(self) -> None:
        self.assertEqual(
            clean_transcript_deterministically("你你竟结成了元婴，我我不知道。"),
            "你竟结成了元婴，我不知道。",
        )

    def test_ambiguous_repeated_characters_wait_for_refiner(self) -> None:
        self.assertEqual(
            clean_transcript_deterministically(
                "儒儒家的，跟孟孟子争争论的人呢。"
            ),
            "儒儒家的，跟孟孟子争争论的人呢。",
        )

    def test_lexical_reduplication_is_preserved(self) -> None:
        source = "人人平等，天天向上，看看这里，爸爸妈妈，哈哈笑。"
        self.assertEqual(collapse_repeated_character_stutters(source), source)

    def test_distributive_and_continuative_reduplication_is_preserved(self) -> None:
        source = (
            "一根根线条，一条条天河，一座座城市，一道道防线，"
            "一代代人，源源不断，生生不息。"
        )
        self.assertEqual(collapse_repeated_character_stutters(source), source)

    def test_broad_classifier_reduplication_inventory_is_preserved(self) -> None:
        source = (
            "一朵朵云，一层层山，一艘艘船，一本本书，一串串灯，"
            "一群群人，一阵阵风，一缕缕水汽，一排排树，一束束光，"
            "一簇簇花，一波波人潮，一轮轮明月，一蓬蓬雾。"
        )
        self.assertEqual(collapse_repeated_character_stutters(source), source)

    def test_standalone_fillers_are_removed_without_touching_words(self) -> None:
        self.assertEqual(
            collapse_standalone_fillers("饭，呃，要上床，呃，这是本性的"),
            "饭，要上床，这是本性的",
        )
        self.assertEqual(collapse_standalone_fillers("嗯哼，呃逆。"), "嗯哼，呃逆。")

    def test_sentence_boundary_嗯_filler_is_removed(self) -> None:
        source = "调查一下。嗯，那我这样一听，你可能是要询问我了。"
        expected = "调查一下。那我这样一听，你可能是要询问我了。"
        self.assertEqual(collapse_sentence_boundary_fillers(source), expected)
        self.assertEqual(clean_transcript_deterministically(source), expected)

    def test_sentence_boundary_嗯_filler_does_not_match_lexical_or_ambiguous_forms(
        self,
    ) -> None:
        source = "调查一下。嗯哼，那我这样一听。调查一下。嗯嗯，那我这样一听。嗯，那我这样一听。"
        expected = "调查一下。嗯哼，那我这样一听。调查一下。嗯嗯，那我这样一听。那我这样一听。"
        self.assertEqual(collapse_sentence_boundary_fillers(source), expected)
        self.assertEqual(
            collapse_sentence_boundary_fillers("嗯，那我这样一听，调查一下，嗯，那我这样一听"),
            "嗯，那我这样一听，调查一下，嗯，那我这样一听",
        )

    def test_normal_chinese_reduplication_is_preserved(self) -> None:
        source = "人人平等，天天向上，好好学习，让我看看。"
        self.assertEqual(clean_transcript_deterministically(source), source)

    def test_classifier_reduplication_requires_one_prefix(self) -> None:
        self.assertEqual(
            clean_transcript_deterministically("一首首歌，一座座城市。"),
            "一首首歌，一座座城市。",
        )
        self.assertEqual(
            clean_transcript_deterministically("首首先说明，条条件不同。"),
            "首首先说明，条条件不同。",
        )

    def test_latest_transcript_lexical_repetitions_survive_cleanup(self) -> None:
        source = (
            "天下熙熙皆为利来，天下攘攘皆为利往。"
            "赤裸裸的利益，通通的人都没有利益。"
            "谢谢罗老师，韩跑跑打打杀杀。"
        )
        self.assertEqual(clean_transcript_deterministically(source), source)


if __name__ == "__main__":
    unittest.main()
