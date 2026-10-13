import unittest

from system.edit_detection_data import align_edits
from experiments.scripts.audit_edit_detector_thresholds import audit_operating_points, recall_constraints
from experiments.scripts.train_edit_location_detector import metrics


class ThresholdAuditTest(unittest.TestCase):
    def test_between_grid_points_avoids_all_keep_false_positives(self):
        positive = dict(id="positive", baseline_text="请请确认。", **align_edits("请请确认。", "请确认。", "repetition"))
        negative = dict(id="negative", baseline_text="请确认。", **align_edits("请确认。", "请确认。"))
        scores = [dict(chars=[[.996, .004, 0]] + [[1., 0, 0]] * 4, gaps=[]),
                  dict(chars=[[.999, .001, 0]] * 4, gaps=[])]
        rows = [positive, negative]
        result = audit_operating_points(rows, scores)
        self.assertGreater(result["boundary_threshold"], .001)
        self.assertLess(result["boundary_threshold"], .01)
        self.assertEqual(result["boundary_metrics"]["false_positive_keep_windows"], 0)
        self.assertTrue(recall_constraints(result["boundary_metrics"]))
        self.assertFalse(recall_constraints(result["next_event_metrics"]))
        # Verify against all breakpoints, independently of binary search.
        feasible = [t for t in (0., 1., 1. - .996, 1. - .999)
                    if recall_constraints(metrics(rows, scores, t, False))]
        self.assertEqual(result["boundary_threshold"], max(feasible))

    def test_punctuation_only_does_not_satisfy_meaningful_edit_recall(self):
        row = dict(id="filler", baseline_text="嗯，请确认。", **align_edits("嗯，请确认。", "请确认。", "disfluency"))
        scores = [dict(chars=[[.996, .004, 0], [.1, .9, 0]] + [[1., 0, 0]] * 4, gaps=[])]
        result = audit_operating_points([row], scores)
        self.assertLess(result["boundary_threshold"], .01)

    def test_no_positive_examples_cannot_define_recall(self):
        row = dict(id="keep", baseline_text="请确认。", **align_edits("请确认。", "请确认。"))
        with self.assertRaises(ValueError):
            audit_operating_points([row], [dict(chars=[[1., 0, 0]] * 4, gaps=[])])


if __name__ == "__main__":
    unittest.main()
