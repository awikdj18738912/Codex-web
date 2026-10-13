import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

from experiments.scripts.train_gate_round2 import check_partition_links,choose_veto
from experiments.scripts.stage_gate_partition import stage
from system.weak_refinement_gate import WeakRefinementGate,select_veto_action


class RoundTwoTests(unittest.TestCase):
    def test_uncertain_and_invalid_scores_preserve_native_action(self):
        limits=dict(neutral=0.9,harm=0.9)
        self.assertEqual(select_veto_action([0.3,0.5,0.2],limits,'refine'),'refine')
        self.assertEqual(select_veto_action([float('nan'),0,0],limits,'refine'),'refine')
        self.assertEqual(select_veto_action([0,0,1],limits,'keep'),'keep')
        self.assertEqual(select_veto_action([0,0,1],limits,'defer'),'defer')

    def test_confident_neutral_or_harm_can_veto(self):
        limits=dict(neutral=0.9,harm=0.8)
        self.assertEqual(select_veto_action([0.02,0.96,0.02],limits,'refine'),'keep')
        self.assertEqual(select_veto_action([0.05,0.05,0.9],limits,'refine'),'keep')

    def test_native_keep_and_defer_do_not_run_encoder(self):
        gate=WeakRefinementGate.__new__(WeakRefinementGate)
        gate.score=Mock(side_effect=AssertionError('should not score'))
        for action in ['keep','defer']:
            self.assertEqual(gate.decide_with_fallback({},action)['action'],action)
        gate.score.assert_not_called()

    def test_unsupported_target_preserves_existing_refinement(self):
        gate=WeakRefinementGate.__new__(WeakRefinementGate)
        gate.metadata=dict(veto_thresholds=dict(window_refine=dict(neutral=0.9,harm=0.9)))
        gate.score=Mock(side_effect=ValueError('target_exceeds_token_budget'))
        self.assertEqual(gate.decide_with_fallback(dict(task_kind='window_refine'),'refine')['action'],'refine')

    def test_veto_search_preserves_at_least_95percent_native_benefit(self):
        rows,probs,outcomes=[],[],{}
        for index in range(31):
            label='benefit' if index<20 else 'neutral' if index<30 else 'harm'
            key=str(index)
            rows.append(dict(case_id=key,label=label,native_action='refine',input=dict(task_kind='window_refine')))
            probs.append([0.01,0.98,0.01] if index==0 or label=='neutral' else [0.02,0.03,0.95] if label=='harm' else [0.8,0.15,0.05])
            outcomes[key]=dict(arms=dict(current=dict(effect=label,operation_cost_ms=10),rules=dict(effect='neutral',operation_cost_ms=0)))
        thresholds,evidence=choose_veto(rows,probs,outcomes)
        e=evidence['by_task']['window_refine']
        self.assertEqual(e['required_benefit'],19)
        self.assertEqual(e['retained_benefit'],19)
        self.assertEqual(e['selected_harm'],0)
        self.assertLess(e['recorded_operation_cost_ms'],310)

    def test_duplicate_model_input_across_split_is_rejected(self):
        value=dict(task_kind='window_refine',raw_text='我们 开会。',baseline_text='我们开会。',available_context='')
        a=dict(case_id='a',split_group='one',recording_group='one',input=value)
        b=dict(case_id='b',split_group='two',recording_group='two',input=value.copy())
        b['input']['raw_text']='我们开会。'
        with self.assertRaisesRegex(ValueError,'encoder-input leakage'):
            check_partition_links(dict(train=[a],validation=[b]))

    def test_source_guard_does_not_hide_new_source_misses_in_old_majority(self):
        rows,probs,outcomes=[],[],{}
        for i in range(42):
            key=str(i);label='benefit' if i<22 else 'neutral'
            cohort='new_dev' if i in [20,21] else 'old_validation'
            rows.append(dict(case_id=key,label=label,native_action='refine',validation_cohort=cohort,input=dict(task_kind='window_refine')))
            probs.append([0.01,0.98,0.01] if i==20 else [0.05,0.9,0.05] if i>=22 else [0.8,0.15,0.05])
            outcomes[key]=dict(arms=dict(current=dict(effect=label,operation_cost_ms=10),rules=dict(effect='neutral',operation_cost_ms=0)))
        _,unprotected=choose_veto(rows,probs,outcomes,source_retention=False)
        _,protected=choose_veto(rows,probs,outcomes,source_retention=True)
        self.assertEqual(unprotected['by_task']['window_refine']['retained_by_source']['new_dev'],1)
        self.assertEqual(protected['by_task']['window_refine']['retained_by_source']['new_dev'],2)

    def test_test_pack_requires_round_two_frozen_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'requires a round-two frozen selection'):
                stage(Path(directory),Path(directory)/'test','test')
            self.assertFalse((Path(directory)/'test').exists())

    def test_consumed_holdout_cannot_be_restaged_as_independent_test(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);(source/'test_consumed.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already been evaluated'):
                stage(source,source/'fresh','test')


if __name__=='__main__':unittest.main()
