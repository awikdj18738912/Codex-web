import json
import unittest
from pathlib import Path

from system.zh_itn_bridge import DEFAULT_LIBRARY, ZhITNNormalizer


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(DEFAULT_LIBRARY.is_file(), "Build the native zh-itn bridge first")
class NativeNumericGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalizer = ZhITNNormalizer()
        cls.cases = [json.loads(line) for line in (
            ROOT / "experiments/fixtures/zh_itn_numeric_grammar_v90.jsonl"
        ).read_text().splitlines()]
        cls.cases.extend(dict(category="bare_degree_v91", source=source, expected=expected)
            for source, expected in (
                ("三十七度。", "37度。"),
                ("大概有三十七度。", "大概有37度。"),
                ("今天天气大概有三十七度。", "今天天气大概有37度。"),
                ("旋转三十七度。", "旋转37度。"),
                ("用了三十七度电。", "用了37度电。"),
                ("大概三十七点五度。", "大概37.5度。"),
                ("三十七摄氏度。", "37摄氏度。"),
                ("一度认为这个方案可行。", "一度认为这个方案可行。"),
                ("几度春秋。", "几度春秋。"),
                ("梅开二度。", "梅开二度。"),
                ("三度夺冠。", "三度夺冠。"),
                ("第十度假村。", "第十度假村。"),
                ("品牌叫三十七度。", "品牌叫三十七度。"),
                ("《三十七度》", "《三十七度》"),
                ("原文写作“三十七度”。", "原文写作“三十七度”。"),
            ))

    def test_formats_and_lexical_protections(self):
        for case in self.cases:
            with self.subTest(category=case["category"], source=case["source"]):
                result = self.normalizer.normalize(case["source"])
                self.assertEqual(result.text, case["expected"])

    def test_idempotence(self):
        for case in self.cases:
            with self.subTest(source=case["source"]):
                once = self.normalizer.normalize(case["source"]).text
                self.assertEqual(self.normalizer.normalize(once).text, once)

    def test_mapping_spans_reconstruct_output(self):
        for case in self.cases:
            with self.subTest(source=case["source"]):
                source = case["source"]
                result = self.normalizer.normalize(source)
                last = 0
                for change in result.changes:
                    self.assertGreaterEqual(change.start, last)
                    self.assertEqual(source[change.start:change.end], change.original)
                    self.assertTrue(change.kind.startswith("zh."))
                    last = change.end
                rebuilt = source
                for change in reversed(result.changes):
                    rebuilt = rebuilt[:change.start] + change.replacement + rebuilt[change.end:]
                self.assertEqual(rebuilt, result.text)

    def test_native_decision_is_observable(self):
        applied = self.normalizer.normalize("价格是三元。")
        self.assertEqual(applied.text, "价格是3元。")
        self.assertGreater(applied.candidate_count, 0)
        self.assertTrue(applied.decision_reason.startswith("applied"))
        untouched = self.normalizer.normalize("普通中文文本。")
        self.assertEqual(untouched.candidate_count, 0)
        self.assertEqual(untouched.decision_reason, "no_candidates_detected")


if __name__ == "__main__":
    unittest.main()
