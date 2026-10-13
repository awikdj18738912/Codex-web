from __future__ import annotations

import json
from pathlib import Path
import unittest

from system.edit_detection_data import align_edits
from system.edit_location_detector import make_tokenizer, encode_input, collate, build_network, target_options, candidate_spans
from experiments.scripts.train_edit_location_detector import split_rows, metrics, choose_policy

ROOT = Path(__file__).resolve().parents[1]


class EditLocationDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_tokenizer(ROOT / "models/entity_ranker/encoder")

    def test_source_tokens_preserve_every_character_and_context_is_separate(self):
        row = dict(baseline_text="请确确认编号5%。", left_context="他正在核对附件。",
                   entity_candidates=[dict(observed="周清", canonical="周青")])
        encoded = encode_input(self.tokenizer, row)
        self.assertEqual(len(encoded["source_positions"]), len(row["baseline_text"]))
        self.assertEqual(len(set(encoded["source_positions"])), len(row["baseline_text"]))
        self.assertEqual(encoded["gap_left"][1:], encoded["source_positions"])
        self.assertEqual(encoded["gap_right"][:-1], encoded["source_positions"])

    def test_unrepresentable_spaces_and_long_inputs_fail_explicitly(self):
        for text in ("请 确认", "请" * 600):
            with self.subTest(text=text[:8]), self.assertRaises(ValueError):
                encode_input(self.tokenizer, dict(baseline_text=text))

    def test_equal_copy_offsets_have_alternative_training_targets(self):
        row = dict(baseline_text="请确确认编号。", **align_edits("请确确认编号。", "请确认编号。", "repetition"))
        options = target_options(row)
        self.assertEqual(len(options), 2)
        self.assertEqual([{i for i, label in enumerate(chars) if label} for chars, _ in options], [{1}, {2}])

    def test_candidates_close_at_eof_and_insertions_have_gap_positions(self):
        spans = candidate_spans([[.9, .1, 0], [.1, .8, .1]], [[1, 0], [1, 0], [.2, .8]], .5)
        self.assertEqual([(s["start"], s["end"]) for s in spans], [(1, 2), (2, 2)])
        self.assertEqual(candidate_spans([[1, 0, 0]], [], 0)[0]["end"], 1)

    def test_padded_batch_heads_match_character_and_gap_shapes(self):
        import torch
        torch.set_num_threads(2)
        model = build_network(ROOT / "models/entity_ranker/encoder", pretrained=False).eval()
        rows = [dict(baseline_text="请确认。"), dict(baseline_text="需要需要确认编号。")]
        encoded = [encode_input(self.tokenizer, row) for row in rows]
        with torch.inference_mode():
            chars, gaps = model(**collate(self.tokenizer, encoded))
        self.assertEqual(tuple(chars.shape), (2, len(rows[1]["baseline_text"]), 3))
        self.assertEqual(tuple(gaps.shape), (2, len(rows[1]["baseline_text"]) + 1, 2))

    def test_metrics_accept_equivalent_delete_location(self):
        row = dict(id="example", baseline_text="请确确认编号。", **align_edits("请确确认编号。", "请确认编号。", "repetition"))
        chars = [[1., 0, 0] for _ in row["baseline_text"]]
        chars[2] = [0, 1., 0]
        result = metrics([row], [dict(chars=chars, gaps=[])], .5, False)
        self.assertEqual(result["location_recall"], 1.)
        self.assertEqual(result["location_precision"], 1.)
        self.assertEqual(result["by_task"]["repetition"]["recall"], 1.)

    def test_only_touching_punctuation_does_not_locate_a_filler_edit(self):
        row = dict(id="filler", baseline_text="嗯，请确认。", **align_edits("嗯，请确认。", "请确认。", "disfluency"))
        chars = [[1., 0, 0] for _ in row["baseline_text"]]
        chars[1] = [0, 1., 0]
        result = metrics([row], [dict(chars=chars, gaps=[])], .5, False)
        self.assertEqual(result["by_task"]["disfluency"]["recall"], 0.)

    def test_review_location_policy_need_not_recover_every_deleted_separator(self):
        positive = dict(id="filler", baseline_text="嗯，请确认。", **align_edits("嗯，请确认。", "请确认。", "disfluency"))
        negative = dict(id="keep", baseline_text="请确认。", **align_edits("请确认。", "请确认。"))
        chars = [[.9, .1, 0] for _ in positive["baseline_text"]]
        chars[0] = [.1, .9, 0]
        predictions = [dict(chars=chars, gaps=[]), dict(chars=[[.9, .1, 0]] * len(negative["baseline_text"]), gaps=[])]
        (_, threshold, result), _ = choose_policy([positive, negative], predictions, False, False)
        self.assertGreater(threshold, .1)
        self.assertEqual(result["false_positive_keep_windows"], 0)
        self.assertEqual(result["by_task"]["disfluency"]["recall"], 1.)
        self.assertEqual(result["location_recall"], .5)

    def test_group_splits_keep_task_probes_in_train_and_inputs_disjoint(self):
        rows = [json.loads(line) for line in (ROOT / "results/edit_detection_data_20261013_v3/controlled_text_train.jsonl").open()]
        partitions, info = split_rows(rows, 20261013)
        self.assertEqual(info["independent_components"], 38)
        for subset in (partitions["validation"], partitions["test"]):
            self.assertTrue(all(row["label_status"] != "controlled_task_definition" for row in subset))
            self.assertEqual({edit["task"] for row in subset for edit in row["edits"]}, {"disfluency", "repetition"})
        self.assertEqual(sum(row["label_status"] == "controlled_task_definition" for row in partitions["train"]), 18)
        sets = [{row["baseline_text"] for row in partitions[name]} for name in ("train", "validation", "test")]
        self.assertTrue(all(not sets[i] & sets[j] for i in range(3) for j in range(i)))


if __name__ == "__main__":
    unittest.main()
