import unittest
from system.window_refinement import (
    CumulativeWindowRefinement,
    StreamingRefinementDisplay,
)


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def refine(text, *args, **kwargs):
            self.calls.append(text)
            return dict(raw_text=text, clean_text=text.replace('苹果', '梨'),
                        refiner_latency_ms=1, refiner_accepted=True,
                        refiner_reject_reasons=[], entity_audit_issues=[],
                        entity_refinement_hints=[], entity_normalizations=[],
                        protected_entities=[], entity_candidates=[],
                        entity_matcher_latency_ms=0)
        self.session = CumulativeWindowRefinement(refine)

    def update(self, text, final=False):
        return self.session.update(text, 'Chinese', final, None, None)

    def test_one_punctuation_window_keeps_each_clause_independent(self):
        session = CumulativeWindowRefinement(
            self.session.refine,
            window_size=1,
            one_punctuation_window=True,
        )
        text = "第一段，第二段。第三段！"

        result = session.update(text, "Chinese", True, None, None)

        self.assertEqual(result["clean_text"], text)
        self.assertEqual(self.calls, ["第一段，", "第二段。", "第三段！"])
        self.assertEqual(result["window_size"], 1)

    def test_one_punctuation_chunks_use_three_block_active_window(self):
        session = CumulativeWindowRefinement(
            self.session.refine,
            window_size=3,
            one_punctuation_window=True,
        )
        text = "第一段，第二段。第三段！第四段？"

        result = session.update(text, "Chinese", True, None, None)

        self.assertEqual(result["clean_text"], text)
        self.assertEqual(
            self.calls,
            ["第一段，", "第二段。第三段！第四段？"],
        )
        self.assertEqual(result["window_size"], 3)

    def test_fixed_groups_wait_for_three_chunks_and_keep_cross_boundary_edit(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            return dict(
                raw_text=text,
                clean_text=text.replace("喂，喂，", "喂，"),
                refiner_latency_ms=1,
                refiner_accepted=True,
                refiner_reject_reasons=[],
                entity_audit_issues=[],
                entity_refinement_hints=[],
                entity_normalizations=[],
                protected_entities=[],
                entity_candidates=[],
                entity_matcher_latency_ms=0,
            )

        session = CumulativeWindowRefinement(
            refine, window_size=3, one_punctuation_window=True,
            fixed_groups=True,
        )
        prefix = "模拟普通人接到警察电话：“喂，喂，"
        self.assertEqual(
            session.update("模拟普通人接到警察电话：“喂，", "Chinese", False, None, None)["clean_text"],
            "模拟普通人接到警察电话：“喂，",
        )
        self.assertEqual(calls, [])

        first = session.update(prefix, "Chinese", False, None, None)
        self.assertEqual(calls, [prefix])
        self.assertEqual(first["clean_text"], "模拟普通人接到警察电话：“喂，")
        self.assertEqual(first["committed_chunks"], 3)
        display = StreamingRefinementDisplay()
        display.accept(first, 1)
        revised_tail = display.compose(prefix + "是郭庆子是吧？")
        self.assertEqual(
            revised_tail["display_refined_text"],
            "模拟普通人接到警察电话：“喂，",
        )
        self.assertEqual(revised_tail["pending_raw_text"], "是郭庆子是吧？")

        extended = session.update(
            prefix + "是郭庆子是吧？", "Chinese", False, None, None,
        )
        self.assertEqual(calls, [prefix])
        self.assertEqual(extended["clean_text"], "模拟普通人接到警察电话：“喂，是郭庆子是吧？")
        final = session.update(
            prefix + "是郭庆子是吧？", "Chinese", True, None, None,
        )
        self.assertEqual(calls, [prefix, "是郭庆子是吧？"])
        self.assertEqual(final["clean_text"], extended["clean_text"])

    def test_fixed_groups_reuse_exact_source_after_group_index_shift(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            return {
                "raw_text": text,
                "clean_text": text,
                "refiner_latency_ms": 1,
                "refiner_accepted": True,
                "refiner_reject_reasons": [],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(
            refine, window_size=3, one_punctuation_window=True,
            fixed_groups=True,
        )
        session.update(
            "甲。乙。丙。丁。戊。己。", "Chinese", True, None, None,
        )
        calls.clear()

        result = session.update(
            "新甲。新乙。新丙。新丁。新戊。新己。丁。戊。己。",
            "Chinese", True, None, None,
        )

        self.assertEqual(calls, [
            "新甲。新乙。新丙。",
            "新丁。新戊。新己。",
        ])
        self.assertIn("丁。戊。己。", result["clean_text"])

    def test_vad_segments_are_refined_as_one_cross_boundary_window(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            clean = text.replace("甲。乙", "甲乙").replace("你。你", "你")
            return {
                "raw_text": text,
                "clean_text": clean,
                "refiner_latency_ms": 1,
                "refiner_accepted": True,
                "refiner_reject_reasons": [],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(refine, window_size=3)
        result = session.update_segments(
            ("前文甲。", "乙补充。", "你。你来了。"),
            "Chinese",
            True,
            None,
            None,
        )

        self.assertEqual(calls, ["前文甲。乙补充。你。你来了。"])
        self.assertEqual(result["clean_text"], "前文甲乙补充。你来了。")
        self.assertEqual(
            result["boundary_reviews"][0]["source_chunk_indices"],
            [1, 2, 3],
        )
        self.assertTrue(result["boundary_reviews"][0]["accepted"])

    def test_post_merge_boundary_repair_handles_keep_chunks(self):
        """Reviewed repairs still run when a bad mark split two KEEP chunks."""

        session = CumulativeWindowRefinement(
            self.session.refine,
            window_size=3,
            one_punctuation_window=True,
        )
        source = "前文。补充。正义就是那种。最好的东西。结尾。"

        result = session.update(source, "Chinese", True, None, None)

        self.assertEqual(result["raw_text"], source)
        self.assertEqual(
            result["clean_text"],
            "前文。补充。正义就是那种最好的东西。结尾。",
        )

    def test_shift_preserves_prefix_and_suffix_without_duplicate(self):
        for text in ['苹果。', '苹果。天气好。', '苹果。天气好。出门。',
                     '苹果。天气好。出门。结束。']:
            result = self.update(text)
            self.assertEqual(result['clean_text'], text.replace('苹果', '梨'))
        count = len(self.calls)
        result = self.update(text, final=True)
        self.assertEqual(result['event'], 'final')
        self.assertEqual(len(self.calls), count)

    def test_three_identical_short_sentence_chunks_are_deduplicated(self):
        text = "前文。可恶！可恶！可恶！后文。"

        result = self.update(text, final=True)

        self.assertEqual(result["raw_text"], text)
        self.assertEqual(result["clean_text"], "前文。可恶！后文。")

    def test_asr_revision_invalidates_cached_source(self):
        self.update('苹果。天气好。出门。结束。')
        text = '香蕉。天气好。出门。新末尾。'
        self.assertEqual(self.update(text, True)['clean_text'], text)

    def test_prefix_revision_reuses_unchanged_committed_chunks(self):
        self.session.window_size = 2
        self.update('甲。乙。丙。丁。')
        self.calls.clear()

        result = self.update('新甲。乙。丙。丁。', True)

        self.assertEqual(result['clean_text'], '新甲。乙。丙。丁。')
        self.assertEqual(self.calls, ['新甲。'])

    def test_rejected_active_window_recovers_per_chunk(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            rejected = text == "丙。丁。"
            return {
                "raw_text": text,
                "clean_text": text,
                "refiner_latency_ms": 1,
                "refiner_accepted": not rejected,
                "refiner_reject_reasons": ["segment_1:severe_content_loss"]
                if rejected
                else [],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "refiner_masked_outputs": [],
                "refiner_retry_reasons": [],
                "refinement_gate_decisions": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(refine, window_size=2)
        result = session.update("甲。乙。丙。丁。", "Chinese", True, None, None)

        self.assertEqual(result["clean_text"], "甲。乙。丙。丁。")
        self.assertTrue(result["refiner_accepted"])
        self.assertEqual(calls, ["甲。", "乙。", "丙。丁。", "丙。", "丁。"])

    def test_long_text_keeps_all_content_and_bounds_model_input(self):
        text = '甲乙丙丁' * 300
        self.assertEqual(self.update(text, True)['clean_text'], text)
        self.assertTrue(all(len(call) <= 80 for call in self.calls))

    def test_active_window_uses_sentence_chunks_with_eighty_char_limit(self):
        text = "第一句。第二句内容较长但仍然完整。第三句。第四句。"
        result = self.update(text, True)

        self.assertEqual(result["clean_text"], text)
        self.assertEqual(result["window_max_chars"], 80)
        self.assertTrue(all(len(call) <= 80 for call in self.calls))
        self.assertTrue(any(call.endswith("第三句。第四句。") for call in self.calls))

    def test_self_correction_chunks_are_refined_independently(self):
        text = (
            "你好，我有一个苹果，不对，我有一个梨，我有一箱苹果，不对，我有一箱梨，"
            "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。"
        )
        result = self.update(text, True)

        self.assertEqual(result["clean_text"], text.replace("苹果", "梨"))
        self.assertEqual(
            self.calls,
            [
                "你好，我有一个苹果，不对，我有一个梨，",
                "我有一箱苹果，不对，我有一箱梨，",
                "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。",
            ],
        )

    def test_one_punctuation_window_keeps_self_correction_context(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            clean = text
            if "我有一个苹果，不对，我有一个梨。" in text:
                clean = text.replace("我有一个苹果，不对，我有一个梨。", "我有一个梨。")
            return {
                "raw_text": text,
                "clean_text": clean,
                "refiner_latency_ms": 1,
                "refiner_accepted": True,
                "refiner_reject_reasons": [],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(
            refine,
            window_size=3,
            one_punctuation_window=True,
        )
        source = "我有一个苹果，不对，我有一个梨。为什么不转换？"

        result = session.update(source, "Chinese", True, None, None)

        self.assertEqual(result["clean_text"], "我有一个梨。为什么不转换？")
        self.assertTrue(
            any("我有一个苹果，不对，我有一个梨。" in call for call in calls)
        )

    def test_self_correction_reopens_committed_antecedent_after_filler(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            clean = text.replace(
                "然后我有一箱苹果，嗯，不对我有一箱梨。",
                "然后我有一箱梨。",
            )
            return {
                "raw_text": text,
                "clean_text": clean,
                "refiner_latency_ms": 1,
                "refiner_accepted": True,
                "refiner_reject_reasons": [],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(
            refine,
            window_size=1,
            one_punctuation_window=True,
        )
        session.update("甲。然后我有一箱苹果，嗯，", "Chinese", True, None, None)
        result = session.update(
            "甲。然后我有一箱苹果，嗯，不对我有一箱梨。",
            "Chinese",
            True,
            None,
            None,
        )

        self.assertEqual(result["clean_text"], "甲。然后我有一箱梨。")
        self.assertIn("然后我有一箱苹果，嗯，不对我有一箱梨。", calls)

    def test_correction_after_sentence_boundary_is_not_committed_separately(self):
        text = "你好，我有一个苹果。不对，我有一个梨。我有一千元。"
        self.update(text, True)

        self.assertIn("你好，我有一个苹果。不对，我有一个梨。", self.calls)
        self.assertNotIn("你好，我有一个苹果。", self.calls)

    def test_final_refreshes_rejected_intermediate_window(self):
        calls = []

        def refine(text, *args, **kwargs):
            final = bool(args[1])
            calls.append((text, final))
            return {
                "raw_text": text,
                "clean_text": text if final else "错误的短输出。",
                "refiner_latency_ms": 1,
                "refiner_accepted": final,
                "refiner_reject_reasons": [] if final else [
                    "segment_1:severe_content_loss"
                ],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(refine)
        first = session.update("第一句。", "Chinese", False, None, None)
        self.assertFalse(first["refiner_accepted"])

        final = session.update("第一句。", "Chinese", True, None, None)
        self.assertTrue(final["refiner_accepted"])
        self.assertEqual(final["clean_text"], "第一句。")
        self.assertEqual(calls, [("第一句。", False), ("第一句。", True)])

    def test_display_retains_refinement_and_appends_pending_raw_tail(self):
        self.session.window_size = 1
        result = self.update("第一句苹果。第二句苹果。")
        display = StreamingRefinementDisplay()
        self.assertTrue(display.accept(result, 1))

        payload = display.compose("第一句苹果。第二句苹果。第三句原始文本。")

        self.assertEqual(payload["committed_clean_text"], "第一句梨。")
        self.assertEqual(payload["active_clean_text"], "第二句梨。")
        self.assertEqual(payload["display_refined_text"], "第一句梨。第二句梨。")
        self.assertEqual(payload["pending_raw_text"], "第三句原始文本。")
        self.assertEqual(
            payload["display_text"],
            "第一句梨。第二句梨。第三句原始文本。",
        )
        self.assertTrue(payload["has_pending_refinement"])

    def test_streaming_display_removes_boundary_echo_from_pending_tail(self):
        committed = "如果说你要询问我的话，"
        source = committed + "话，"
        display = StreamingRefinementDisplay()
        self.assertTrue(
            display.accept(
                {
                    "raw_text": committed,
                    "clean_text": committed,
                    "refinement_source_spans": [
                        {
                            "source_text": committed,
                            "clean_text": committed,
                            "state": "active",
                        }
                    ],
                },
                1,
            )
        )

        payload = display.compose(source)

        self.assertEqual(payload["display_text"], "如果说你要询问我的话，")
        self.assertEqual(payload["pending_raw_text"], "话，")

    def test_display_drops_revised_active_span_but_keeps_committed_prefix(self):
        self.session.window_size = 1
        result = self.update("第一句苹果。第二句苹果。")
        display = StreamingRefinementDisplay()
        display.accept(result, 3)

        payload = display.compose("第一句苹果。第二句香蕉。")

        self.assertEqual(payload["committed_clean_text"], "第一句梨。")
        self.assertEqual(payload["active_clean_text"], "")
        self.assertEqual(payload["display_refined_text"], "第一句梨。")
        self.assertEqual(payload["pending_raw_text"], "第二句香蕉。")
        self.assertEqual(payload["display_revision"], 3)

    def test_display_rejects_stale_refinement_revision(self):
        first = self.update("第一句苹果。")
        newer = self.update("第一句苹果。第二句苹果。")
        display = StreamingRefinementDisplay()
        self.assertTrue(display.accept(newer, 2))
        self.assertFalse(display.accept(first, 1))

        payload = display.compose("第一句苹果。第二句苹果。")

        self.assertEqual(payload["display_revision"], 2)
        self.assertEqual(payload["display_text"], "第一句梨。第二句梨。")
        self.assertFalse(payload["has_pending_refinement"])
