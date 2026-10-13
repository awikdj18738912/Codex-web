"""Regression checks for weak-label leakage, source ownership and split integrity."""
import copy
from pathlib import Path
import unittest

from experiments.scripts.train_weak_refinement_gate import (assign_partitions,
    choose_thresholds, input_signature, policy_metrics)
from system.weak_refinement_gate import encode_input, input_text, make_tokenizer, select_action


def window(text="明天开会。"):
    return dict(task_kind="window_refine", domain="general", raw_text=text,
                baseline_text=text, available_context="")


class CountingTokenizer:
    def __init__(self):
        self.targets = []

    def __call__(self, first, second, truncation):
        assert truncation is False
        self.targets.append(second)
        size = len(first) + len(second) + 3
        return dict(input_ids=list(range(10, size + 10)), attention_mask=[1] * size)


class WeakGateTests(unittest.TestCase):
    def test_gold_and_future_fields_never_change_input(self):
        value = window()
        before = input_text(value)
        value.update(intended_text="泄漏答案", label="harm", always_output="未来结果",
                     template_group="未来组", reference="参考答案", confidence=1)
        self.assertEqual(before, input_text(value))

    def test_local_candidate_uses_authoritative_offset(self):
        value = dict(task_kind="local_edit_review", baseline_text="慢慢走，慢慢说。", candidate_local_start=4,
                     candidate=dict(source="慢慢", target="慢", kind="character_run"))
        _first, marked, _context = input_text(value)
        self.assertEqual(marked, "慢慢走，[unused1]慢慢[unused2]说。")
        value["candidate_local_start"] = 3
        with self.assertRaisesRegex(ValueError, "invalid_candidate_offset"):
            input_text(value)

    def test_token_budget_crops_only_context(self):
        value = window()
        value["available_context"] = "旧上下文" * 1000
        tokenizer = CountingTokenizer()
        encoded = encode_input(tokenizer, value, max_tokens=100)
        self.assertLessEqual(len(encoded["input_ids"]), 100)
        self.assertEqual(len(set(tokenizer.targets)), 1)
        self.assertIn(value["baseline_text"], tokenizer.targets[-1])

    def test_oversize_target_fails_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "target_exceeds_token_budget"):
            encode_input(CountingTokenizer(), window("这是完整目标" * 100), max_tokens=100)

    def test_group_split_joins_template_recording_and_identical_input(self):
        rows = []
        for group in range(12):
            for index, label in enumerate(("benefit", "neutral", "harm")):
                rows.append(dict(case_id=f"{group}-{index}", split_group=f"basic:{group}",
                                 recording_group=f"recording-{group}", label=label, eligible_for_weak_pretraining=True,
                                 input=window(f"第{group}组第{index}个输入。")))
            rows.append(dict(case_id=f"{group}-local", split_group=f"basic:{group}",
                             recording_group=f"recording-{group}", label="neutral", eligible_for_weak_pretraining=True,
                             input=dict(task_kind="local_edit_review", baseline_text=f"第{group}组慢慢走。",
                                        candidate_local_start=len(f"第{group}组"),
                                        candidate=dict(source="慢慢", target="慢", kind="character_run"))))
        rows[4]["input"] = copy.deepcopy(rows[0]["input"])
        assigned, _ = assign_partitions(rows, seed=33, attempts=100)
        again, _ = assign_partitions(rows, seed=33, attempts=100)
        self.assertEqual(assigned, again)
        self.assertEqual(assigned[rows[0]["case_id"]], assigned[rows[4]["case_id"]])
        for key in ("split_group", "recording_group"):
            for value in {r[key] for r in rows}:
                self.assertEqual(len({assigned[r["case_id"]] for r in rows if r[key] == value}), 1)
        self.assertEqual(set(assigned.values()), {"train", "validation", "test"})

    def test_threshold_selection_excludes_observed_validation_harm(self):
        rows, probs = [], []
        for task in ("window_refine", "local_edit_review"):
            for label, scores in (("benefit", [0.8, 0.15, 0.05]), ("harm", [0.6, 0.1, 0.3]),
                                  ("neutral", [0.1, 0.8, 0.1])):
                rows.append(dict(input=dict(task_kind=task), label=label))
                probs.append(scores)
        thresholds, evidence = choose_thresholds(rows, probs)
        self.assertEqual(evidence["utility"], 1.9)
        for row, scores in zip(rows, probs):
            self.assertEqual(select_action(scores, thresholds[row["input"]["task_kind"]]),
                             "refine" if row["label"] == "benefit" else "keep")
        self.assertFalse(evidence["calibrated"])

    def test_intended_and_spoken_metrics_have_distinct_targets(self):
        row = dict(case_id="a", label="benefit", eligible_for_weak_pretraining=True)
        common = dict(logical_refiner_calls=1, logical_plausibility_calls=0, operation_cost_ms=1,
                      executed=True, changed=True)
        rules = dict(**common, intended_distance=2, spoken_distance=0, exact_intended=False, effect="neutral")
        always = dict(**common, intended_distance=0, spoken_distance=2, exact_intended=True, effect="benefit")
        outcomes = dict(a=dict(intended_chars=8, spoken_chars=10, arms=dict(rules=rules, always=always)))
        metrics = policy_metrics([row], outcomes, lambda _: "always")
        self.assertEqual(metrics["captured_benefit"], 1)
        self.assertEqual(metrics["intended_micro_cer"], 0)
        self.assertEqual(metrics["spoken_micro_cer"], 0.2)

    def test_invalid_probabilities_keep(self):
        for scores in ([float("nan"), 0, 0], [0.9, 0.9, 0.1], [0.8, -0.1, 0.3]):
            self.assertEqual(select_action(scores, dict(benefit=0.5, harm=0.2)), "keep")

    def test_real_tokenizer_preserves_candidate_span(self):
        encoder = Path(__file__).resolve().parents[1] / "models/entity_ranker/encoder"
        tokenizer = make_tokenizer(encoder)
        value = dict(task_kind="local_edit_review", baseline_text="陈晓慢慢走。", candidate_local_start=2,
                     candidate=dict(source="慢慢", target="慢", kind="character_run"))
        encoded = encode_input(tokenizer, value)
        masked_ids = [i for i, mask in zip(encoded["input_ids"], encoded["source_mask"]) if mask]
        self.assertEqual(tokenizer.decode(masked_ids).replace(" ", ""), "慢慢")


if __name__ == "__main__":
    unittest.main()
