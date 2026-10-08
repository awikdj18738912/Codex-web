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
