from __future__ import annotations

import unittest

from system.refinement_protocol import (
    apply_structured_patch,
    permits_boundary_punctuation_repair,
)


class StructuredPatchTest(unittest.TestCase):
    def test_keep_patch_returns_source(self) -> None:
        source = "今天天气很好。"
        refined, payload, issue = apply_structured_patch(
            source,
            '{"action":"keep","source":"","target":"","reason":"no_change"}',
        )
        self.assertEqual(refined, source)
        self.assertEqual(payload["action"], "keep")
        self.assertIsNone(issue)

    def test_replace_patch_only_replaces_unique_source_span(self) -> None:
        refined, _, issue = apply_structured_patch(
            "今天在黄河园区。",
            '{"action":"replace","source":"黄河园区","target":"黄河源区","reason":"term"}',
        )
        self.assertEqual(refined, "今天在黄河源区。")
        self.assertIsNone(issue)

    def test_invalid_patch_falls_back_to_source(self) -> None:
        source = "这里不只有公路。"
        refined, _, issue = apply_structured_patch(
            source,
            '{"action":"replace","source":"不存在","target":"东西","reason":"guess"}',
        )
        self.assertEqual(refined, source)
        self.assertEqual(issue, "structured_patch_source_not_unique")

    def test_legacy_plain_text_remains_supported(self) -> None:
        source = "今天很好。"
        refined, payload, issue = apply_structured_patch(source, "今天真好。")
        self.assertEqual(refined, "今天真好。")
        self.assertIsNone(payload)
        self.assertIsNone(issue)

    def test_boundary_repair_requires_explicit_reason_and_small_punctuation_edit(self) -> None:
        self.assertTrue(
            permits_boundary_punctuation_repair(
                {
                    "action": "replace",
                    "source": "华夏。在文明",
                    "target": "华夏文明",
                    "reason": "boundary_repair",
                }
            )
        )
        self.assertFalse(
            permits_boundary_punctuation_repair(
                {
                    "action": "replace",
                    "source": "第一。第二。第三。",
                    "target": "第一第二第三",
                    "reason": "typo",
                }
            )
        )
        self.assertTrue(
            permits_boundary_punctuation_repair(
                {
                    "action": "replace",
                    "source": "三番五次的。的提醒",
                    "target": "三番五次的提醒",
                    "reason": "boundary_punctuation",
                },
                source_text="三番五次的。的提醒",
            )
        )
        self.assertFalse(
            permits_boundary_punctuation_repair(
                {
                    "action": "replace",
                    "source": "第一段。第二段",
                    "target": "第一段第二段",
                    "reason": "boundary_punctuation",
                },
                source_text="第一段。第二段",
            )
        )


if __name__ == "__main__":
    unittest.main()
