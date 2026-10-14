import unittest

from experiments.scripts.prepare_confidence_audit_expansion import spaced_indices
from experiments.scripts.audit_expanded_refiner_results import edit_distance,lexical_content
from tests.test_edit_detection_data import IdentityNormalizer
from experiments.scripts.summarize_saved_confidence_replay import summarize_records


class ExpandedRefinerAuditTest(unittest.TestCase):
    def test_temporal_selection_unique_and_spread(self):
        self.assertEqual(spaced_indices(11,6),[0,2,4,6,8,10])
        self.assertEqual(len(set(spaced_indices(38,8))),8)
        self.assertEqual(spaced_indices(2,2),[0,1])
        with self.assertRaises(ValueError):spaced_indices(3,4)

    def test_content_comparison_ignores_punctuation_not_words(self):
        normalizer=IdentityNormalizer()
        before=lexical_content('我有三元，嗯。',normalizer)
        after=lexical_content('我有三元。',normalizer)
        self.assertEqual(edit_distance(before,after),1)
        self.assertEqual(edit_distance(lexical_content('会议，开始。',normalizer),lexical_content('会议开始',normalizer)),0)

    def test_cer_distance_preserves_wrong_digit_values(self):
        self.assertEqual(edit_distance('预算12元','预算13元'),1)
        self.assertEqual(edit_distance('', '甲乙'),2)
        self.assertEqual(edit_distance('甲乙',''),2)

    def test_saved_policy_summary_cannot_silently_be_empty(self):
        base=dict(case=0,repetition=0,windows=[['原文','K3',3]],window_pipeline={'K':3},
                  stats={'call_count':2,'completed_wall_latency_ms':10},elapsed_seconds=.1,gate_audit=[])
        rows=[{**base,'policy':'native','final_text':'原文'},
              {**base,'policy':'candidate','final_text':'修改'}]
        summary,differences=summarize_records(rows,{'policies':['native','candidate'],'scope':'test_fixture'})
        self.assertEqual(set(summary['policies']),{'native','candidate'})
        self.assertEqual(summary['policies']['candidate']['identical_final_pairs_vs_native'],0)
        self.assertEqual(len(differences),1)
        with self.assertRaises(ValueError):
            summarize_records(rows[:1],{'policies':['native','candidate'],'scope':'test_fixture'})


if __name__=='__main__':unittest.main()
