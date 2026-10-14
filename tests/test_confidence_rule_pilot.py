import unittest

from experiments.scripts.benchmark_confidence_rule_gate import PilotConfidenceRuleGate, load_cases, ROOT
from tests.test_edit_detection_data import IdentityNormalizer


def profile(text,score=.99,fragment_score=.99,coverage=1.):
    confidence=dict(score=score,min_fragment_score=fragment_score,coverage=coverage)
    return dict(index=0,text=text,confidence=confidence,
                fragments=[dict(text=text,confidence=dict(score=fragment_score,coverage=coverage))])


class ConfidenceRulePilotTest(unittest.TestCase):
    def gate(self,text,policy="confidence_fusion",**values):
        return PilotConfidenceRuleGate("tri_state",profiles=[profile(text,**values)],policy=policy,
                                       normalizer=IdentityNormalizer(),refine_on_unverified_confidence=False)

    def test_high_score_does_not_skip_repetition_or_correction(self):
        for text in ["需要需要确认材料。","我说错了，应该是这里。"]:
            with self.subTest(text=text):
                result=self.gate(text).decide(text,is_final=True)
                self.assertEqual(result.action,"refine")
                self.assertEqual(result.reasons,("cleanup_signal_present",))

    def test_high_score_does_not_skip_entity_hints(self):
        text="请交给周青。"
        result=self.gate(text).decide(text,entity_hints=["周青候选"],is_final=True)
        self.assertEqual(result.action,"refine")
        self.assertEqual(result.reasons,("entity_hint_present",))

    def test_low_plain_fragment_can_add_an_experimental_review(self):
        text="请确认材料。"
        result=self.gate(text,score=.9,fragment_score=.4).decide(text,is_final=True)
        self.assertEqual(result.action,"refine")
        self.assertEqual(result.reasons,("experimental_low_nonnumeric_fragment_review",))
        self.assertFalse(result.calibrated)

    def test_high_incomplete_fragment_keep_is_only_in_fusion(self):
        text="会议"
        self.assertEqual(self.gate(text).decide(text,is_final=True).action,"keep")
        self.assertEqual(self.gate(text,policy="confidence_supplement").decide(text,is_final=True).action,"refine")

    def test_incomplete_score_coverage_keeps_original_routing(self):
        text="会议"
        self.assertEqual(self.gate(text,coverage=.5).decide(text,is_final=True).action,"refine")

    def test_rule_only_ablation_does_not_use_score_thresholds(self):
        text="会议"
        result=self.gate(text,policy="weak_structure_rule_only",score=.1,fragment_score=.1,coverage=.1).decide(text,is_final=True)
        self.assertEqual(result.action,"keep")
        self.assertEqual(result.reasons,("experimental_weak_structure_rule_only_keep",))
        self.assertFalse(result.calibrated)
        self.assertEqual(self.gate("需要需要确认。",policy="weak_structure_rule_only").decide("需要需要确认。",is_final=True).action,"refine")

    def test_unstable_and_already_refined_hypotheses_are_not_promoted(self):
        text="请确认材料。"
        gate=self.gate(text,score=.9,fragment_score=.4)
        self.assertEqual(gate.decide(text,is_final=False,stable=False).action,"defer")
        self.assertEqual(gate.decide(text,is_final=True,same_as_last_refined=True).action,"keep")

    def test_unknown_source_falls_back_to_native(self):
        gate=self.gate("请确认材料。",score=.9,fragment_score=.4)
        self.assertEqual(gate.decide("请查阅文件。",is_final=True).action,"keep")

    def test_frozen_probe_cases_have_unique_window_identity(self):
        cases=load_cases(ROOT / "results/asr_window_confidence_probe_20261013_v2")
        self.assertEqual(sum(len(case["profiles"]) for case in cases),16)
        for case in cases:
            self.assertEqual(len({p["text"] for p in case["profiles"]}),len(case["profiles"]))


if __name__=="__main__":
    unittest.main()
