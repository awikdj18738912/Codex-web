from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from system.entity_pipeline import finalize_entity_segment, prepare_entity_segment
from system.protection import EntityProtector, strip_refiner_key_suffix


W003 = "而在中国的春秋时期，也有过这样的讨论，关于义和利的讨论。"
W003_AFTER = "在中国的春秋时期，也有过这样的讨论，关于义和利的讨论。"


def finalize(source: str, model_output: str):
    protector = EntityProtector((), enable_rule_protection=False)
    prepared = prepare_entity_segment(source, protector, None, allow_auto=False)
    # The real refiner may append an entity key annotation; remove it before
    # validating the proposed transcript, matching the production boundary.
    candidate = strip_refiner_key_suffix(model_output)
    return finalize_entity_segment(candidate, prepared, protector)


class LeadingConnectiveGuardTest(unittest.TestCase):
    def test_confirmed_w003_leading_er_deletion_is_accepted(self):
        result = finalize(W003, W003_AFTER + "<KEY>[义和利]")
        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.text, W003_AFTER)
        self.assertEqual(result.reject_reasons, ())

    def test_punctuation_and_whitespace_only_differences_are_ignored(self):
        source = "而在北京，会议继续。"
        proposed = "在北京 会议继续。<KEY>[会议]"
        result = finalize(source, proposed)
        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.text, "在北京 会议继续。")

    def test_independent_location_sentence_keeps_er_zai_when_model_preserves_it(self):
        source = "而在北京，会议继续。"
        result = finalize(source, source)
        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.text, source)

    def test_old_connective_deletion_cases_are_not_blocked_by_removed_rule(self):
        examples = (
            ("而是应该继续讨论。", "是应该继续讨论。"),
            ("而且应该继续讨论。", "且应该继续讨论。"),
            ("而非应该继续讨论。", "非应该继续讨论。"),
            ("这不是甲，而在乙地发生。", "这不是甲，在乙地发生。"),
        )
        for source, proposed in examples:
            with self.subTest(source=source):
                result = finalize(source, proposed)
                self.assertTrue(result.accepted, result.reject_reasons)
                self.assertNotIn("semantic_content_loss", result.reject_reasons)

    def test_other_content_deletions_are_not_blocked_by_removed_rule(self):
        examples = (
            # Remove the place, the sentence subject, or the negation too.
            (W003, "在春秋时期，也有过这样的讨论，关于义和利的讨论。"),
            ("而在春秋时期，孔子讨论义利。", "在春秋时期，讨论义利。"),
            ("而在春秋时期，孔子并未同意。", "在春秋时期，孔子同意。"),
            # Deleting both leading characters is outside the confirmed case.
            (W003, "中国的春秋时期，也有过这样的讨论，关于义和利的讨论。"),
        )
        for source, proposed in examples:
            with self.subTest(proposed=proposed):
                result = finalize(source, proposed)
                self.assertTrue(result.accepted, result.reject_reasons)
                self.assertNotIn("semantic_content_loss", result.reject_reasons)


try:
    from tests.test_web_app_finish import web_app
except ModuleNotFoundError:
    web_app = None


@unittest.skipIf(web_app is None, "FastAPI is not installed")
class LeadingConnectiveWebTest(unittest.TestCase):
    def test_web_accepts_plain_deletions_once_without_retry_or_fallback(self):
        from experiments.scripts.collect_gate_pairs import closure_value

        class FrozenW003Refiner:
            calls = []
            outputs = {
                W003: W003_AFTER + "<KEY>[义和利]",
                "就是关于义和利的讨论。": "关于义和利的讨论。",
            }

            def __init__(self, *args, **kwargs):
                type(self).calls = []

            def refine(self, text, **kwargs):
                type(self).calls.append((text, kwargs.get("strict_placeholders", False)))
                return self.outputs[text], 1.0

        with patch.object(web_app, "TransformersRefiner", FrozenW003Refiner):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"), "cpu", "http://fake-asr", "Chinese", 32,
                None, refinement_gate_mode="off", window_edit_policy="compat")
        endpoint = next(route.endpoint for route in app.routes
                        if getattr(route, "path", None) == "/ws/stream")
        refine_update = closure_value(endpoint, "refine_update")
        sources = (W003, "就是关于义和利的讨论。")
        expected = (W003_AFTER, "关于义和利的讨论。")
        results = [
            refine_update(
                source, "Chinese", True,
                web_app.EntityProtector([], enable_rule_protection=False),
                None, single_window=True)
            for source in sources
        ]

        self.assertEqual(FrozenW003Refiner.calls,
                         [(source, False) for source in sources])
        for result, text in zip(results, expected):
            with self.subTest(text=text):
                self.assertEqual(result["clean_text"], text)
                self.assertTrue(result["refiner_accepted"])
                self.assertEqual(result["refiner_retry_count"], 0)
                self.assertEqual(result["refiner_reject_reasons"], [])
                self.assertEqual(result["safe_repetition_repairs"], [])


if __name__ == "__main__":
    unittest.main()
