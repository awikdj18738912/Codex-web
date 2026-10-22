import json
import unittest

from system.window_refinement import CumulativeWindowRefinement, StreamingRefinementDisplay
from system.refinement_guard import window_context_reject_reasons
from system.refinement_protocol import apply_window_context_patch


FIRST = "换言之，就这个世界上有没有正义的问题，正义。"
SECOND = "是人类的发现，还是人类的发明，还是一套纯粹的说辞？"


class WindowAdjacentContextTest(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def refine(text, *args, **kwargs):
            self.calls.append((text, kwargs["window_context"]))
            return dict(raw_text=text, clean_text=text, refiner_latency_ms=1,
                        refiner_accepted=True, refiner_reject_reasons=[])

        self.session = CumulativeWindowRefinement(
            refine, window_size=3, one_punctuation_window=True,
            fixed_groups=True, adjacent_context=True,
        )

    def update(self, text, final=False, **kwargs):
        return self.session.update(text, "Chinese", final, None, None, **kwargs)

    def test_waits_for_complete_predicate_block(self):
        pending = self.update(FIRST)
        self.assertEqual(self.calls, [])
        self.assertEqual(pending["clean_text"], FIRST)
        self.assertEqual(pending["refinement_state"], "waiting_for_context")
        self.update(FIRST + "是人类")
        self.assertEqual(self.calls, [])
        result = self.update(FIRST + "是人类的发现，")
        self.assertEqual(self.calls, [(FIRST, {"left_context": "", "right_context": "是人类的发现，"})])
        self.assertEqual(result["clean_text"], FIRST + "是人类的发现，")
        self.assertEqual(result["committed_chunks"], 3)
        self.assertEqual(result["refinement_source_spans"][0]["source_text"], FIRST)

    def test_final_tail_owns_only_its_source_and_reuses_first_window(self):
        self.update(FIRST + SECOND)
        result = self.update(FIRST + SECOND, final=True)
        self.assertEqual(self.calls, [
            (FIRST, {"left_context": "", "right_context": "是人类的发现，"}),
            (SECOND, {"left_context": "正义。", "right_context": ""}),
        ])
        self.assertEqual(result["clean_text"], FIRST + SECOND)
        self.assertEqual(result["window_cache_reused_group_count"], 1)
        self.assertEqual([span["source_text"] for span in result["refinement_source_spans"]], [FIRST, SECOND])

    def test_changed_read_only_context_invalidates_cache_and_signature(self):
        initial = FIRST + "是人类的发现，"
        self.update(initial)
        signature = self.session.streaming_input_signature(initial)
        revised = FIRST + "是人类的发明，"
        self.assertNotEqual(self.session.streaming_input_signature(revised)[0], signature[0])
        result = self.update(revised)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result["window_cache_reprocess_reasons"], {"window_context_changed": 1})
        self.update(revised + "未结束的尾巴")
        self.assertEqual(len(self.calls), 2)

    def test_left_context_revision_invalidates_unchanged_second_group(self):
        text = "甲，乙，丙。丁，戊，己。庚，"
        self.update(text)
        result = self.update(text.replace("丙", "新丙"))
        self.assertEqual(result["window_cache_reprocess_reasons"], {
            "no_exact_source_group_match": 1, "window_context_changed": 1,
        })

    def test_display_drops_refinement_when_its_read_only_context_is_revised(self):
        initial = FIRST + "是人类的发现，"
        result = self.update(initial)
        result["clean_text"] = initial.replace("就这个", "这个")
        result["refinement_source_spans"][0]["clean_text"] = FIRST.replace("就这个", "这个")
        display = StreamingRefinementDisplay()
        display.accept(result, 1)
        self.assertIn("这个世界", display.compose(initial)["display_refined_text"])
        revised = FIRST + "是人类的发明，"
        composed = display.compose(revised)
        self.assertEqual(composed["display_refined_text"], "")
        self.assertEqual(composed["pending_raw_text"], revised)

    def test_vad_boundaries_do_not_remove_adjacent_context(self):
        result = self.update(FIRST + SECOND, final=True, source_chunks=[FIRST, SECOND])
        self.assertEqual(self.calls[0][1]["right_context"], "是人类的发现，")
        self.assertEqual(self.calls[1][1]["left_context"], "正义。")
        self.assertEqual(result["clean_text"], FIRST + SECOND)

    def test_final_without_lookahead_flushes_and_caches_short_tail(self):
        for source in (FIRST, "就这个问题，正义"):
            with self.subTest(source=source):
                self.session.group_cache = []
                self.calls.clear()
                result = self.update(source, final=True)
                self.assertEqual(result["clean_text"], source)
                self.assertEqual(self.calls[0][1]["right_context"], "")
                self.update(source, final=True)
                self.assertEqual(len(self.calls), 1)

    def test_dropped_subject_is_rejected_but_punctuation_repair_is_allowed(self):
        proposed = "换言之，这个世界上有没有正义的问题？"
        self.assertIn("window_boundary_subject_loss", window_context_reject_reasons(
            FIRST, proposed, right_context="是人类的发现，"))
        self.assertEqual(window_context_reject_reasons(
            FIRST, FIRST[:-1], right_context="是人类的发现，"), ())
        self.assertEqual(window_context_reject_reasons(
            "换言之，嗯嗯。", "换言之，", right_context="是人类的发现，"), ())
        self.assertEqual(window_context_reject_reasons(
            "换言之，正意。", "换言之，正义。", right_context="是人类的发现，"), ())

    def test_scoped_patch_cannot_edit_context_or_return_whole_transcript(self):
        for response in ('{"action":"replace","source":"是人类的发现",'
                         '"target":"是人类的发明","reason":"local_typo"}',):
            candidate, _, issue = apply_window_context_patch(FIRST, response)
            self.assertIsNotNone(issue)
            self.assertEqual(candidate, FIRST)
        patch = json.dumps(dict(action="replace", source="正义。", target="正义，", reason="boundary_punctuation"), ensure_ascii=False)
        candidate, _, issue = apply_window_context_patch(FIRST, patch)
        self.assertIsNone(issue)
        self.assertEqual(candidate, FIRST[:-1] + "，")
        self.assertIn("window_context_copied", window_context_reject_reasons(
            FIRST, FIRST + "是人类的发现，", right_context="是人类的发现，"))


if __name__ == "__main__":
    unittest.main()
