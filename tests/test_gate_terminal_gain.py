from experiments.scripts.gate_terminal_gain_pilot import paired_effect,route_window,window_key,final_snapshot


def arm(text,distance,calls=1,ppl=0,tokens=100):
    return dict(final_text=text,intended_distance=distance,
                cost=dict(refiner_calls=calls,plausibility_calls=ppl,refiner_token_positions=tokens))


def test_window_operation_improvement_is_neutral_if_local_review_repairs_skip():
    effect=paired_effect(arm('请确认编号。',0),arm('请确认编号。',0,tokens=80))
    assert effect['label']=='neutral'
    assert effect['skip_saves_refiner_calls']==0
    assert effect['skip_has_compute_advantage_proxy']


def test_lower_refiner_count_cannot_hide_added_mlm_cost():
    effect=paired_effect(arm('请确认编号。',0,calls=2),arm('请确认编号。',0,calls=1,ppl=1,tokens=80))
    assert effect['label']=='neutral'
    assert effect['skip_saves_refiner_calls']==1
    assert not effect['skip_has_compute_advantage_proxy']


def test_terminal_quality_and_native_instability_have_priority_over_cost():
    assert paired_effect(arm('确认',0),arm('确确认',1,calls=0,tokens=0))['label']=='benefit'
    assert paired_effect(arm('确认',1),arm('确确认',0))['label']=='harm'
    effect=paired_effect(arm('确认',0),arm('确认',0),stable=False)
    assert effect['label']=='uncertain' and not effect['skip_has_compute_advantage_proxy']


def test_posthoc_policy_checks_joint_terminal_effect_instead_of_adding_veto_gains():
    from experiments.scripts.summarize_terminal_gain_pilot import choose_posthoc
    native=arm('确认编号。',0,calls=2,tokens=200)
    # Two individual vetoes could each be neutral, but the fully executed
    # joint skip can lose their shared repair. It must not be selected.
    joint=arm('确确认编号。',1,calls=0,tokens=0)
    row=dict(native=native,skip_all=joint,native_control_stable=True)
    assert choose_posthoc(row) is native
    assert choose_posthoc(row,False) is native


def test_intervention_only_vetoes_exact_input_and_rejects_local_routing():
    import pytest
    value=dict(task_kind='window_refine',domain='general',raw_text='请确认确认编号。',
               baseline_text='请确认确认编号。',available_context='')
    key=window_key(value)
    assert route_window(value,'refine',{key})[0]['action']=='keep'
    assert route_window(dict(value,raw_text='请查阅编号。'),'refine',{key})[0]['action']=='refine'
    with pytest.raises(AssertionError):route_window(dict(value,task_kind='local_edit_review'),'refine','all')


def test_consumed_marker_on_mixed_bank_excludes_only_heldout_rows(tmp_path,monkeypatch):
    import json
    from experiments.scripts import gate_terminal_gain_pilot as pilot
    folder=tmp_path/'results/mixed_bank';folder.mkdir(parents=True)
    (folder/'test_consumed.json').write_text('{}')
    rows=[dict(seed_id='training',template_group='train_group',synthetic_split='train'),
          dict(seed_id='heldout',template_group='test_group',synthetic_split='test')]
    (folder/'texts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    forbidden,_=pilot.heldout_boundary()
    assert forbidden['seed_id']=={'heldout'}
    assert forbidden['template_group']=={'test_group'}


def test_expansion_keeps_noise_view_and_dev_partition_for_native_discovery():
    import pytest
    from experiments.scripts.expand_terminal_gain import eligible_views
    rows=[dict(recording_group='shared',variant=v,pcm_sha256=v,synthetic_split='dev') for v in ('clean','noise')]
    assert eligible_views(rows,[],'dev')==rows
    with pytest.raises(ValueError):eligible_views(rows,[],'train')
    from experiments.scripts.gate_terminal_gain_pilot import view_id
    assert view_id(rows[0])!=view_id(rows[1])


def test_conservative_action_separates_terminal_quality_and_cost():
    from experiments.scripts.assemble_terminal_gain_round4 import conservative_target
    effect=dict(native_control_stable=True,label='neutral',skip_has_compute_advantage_proxy=False)
    assert conservative_target(effect)=='refine_required'
    assert conservative_target(dict(effect,skip_has_compute_advantage_proxy=True))=='skip_allowed'
    assert conservative_target(dict(effect,label='benefit',skip_has_compute_advantage_proxy=True))=='refine_required'
    assert conservative_target(dict(effect,label='harm'))=='skip_allowed'
    assert conservative_target(dict(effect,native_control_stable=False)) is None


def test_linear_features_do_not_encode_labels_or_references():
    from experiments.scripts.train_terminal_gain_linear import feature_text
    value=dict(task_kind='window_refine',domain='general',raw_text='请确认确认编号。',
               baseline_text='请确认确认编号。',available_context='')
    assert feature_text(value)==feature_text(dict(value,label='benefit',intended_text='未来参考答案'))


def test_linear_evaluation_uses_joint_executed_arm_and_detects_quality_loss():
    from experiments.scripts.train_terminal_gain_linear import evaluate
    from experiments.scripts.gate_terminal_gain_pilot import view_id
    def complete_arm(distance):
        return dict(intended_distance=distance,cost=dict(refiner_calls=1,plausibility_calls=0,
            refiner_token_positions=10,forward_passes=1,refiner_wall_ms=1,plausibility_wall_ms=0))
    row=dict(recording_group='fixture',variant='clean',pcm_sha256='fixture',intended_chars=10,
        native=complete_arm(0),skip_all=complete_arm(1),counterfactuals={'a':complete_arm(0),'b':complete_arm(0)},
        effects=[dict(window_key=k,label='neutral') for k in ('a','b')])
    scores={view_id(row):{'a':.9,'b':.9}}
    result=evaluate([row],scores,.5,.1)
    assert result['degraded_views']==1 and result['intended_distance']==1
    assert evaluate([row],scores,1.01,.1)['degraded_views']==0


def test_final_snapshot_runs_actual_finalizer_and_local_review_after_window_veto():
    from pathlib import Path
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import system.web_app as web
    requests=[]
    class Refiner:
        def __init__(self,*a,**k):pass
        def refine(self,text,**k):requests.append('window');return text.replace('确确认','确认'),1.
        def review_repetition_candidate_with_status(self,*a,**k):requests.append('local');return 'REMOVE',1.,True,.5
    def request(asr_url,endpoint,session_id=None,data=b'',params=None):
        if endpoint=='/stream/start':return dict(session_id='fixture')
        if endpoint=='/stream/finish':return dict(text='请确确认编号。',language='Chinese')
        if endpoint=='/stream/cancel':return dict(cancelled=True)
        raise AssertionError(endpoint)
    with patch.object(web,'TransformersRefiner',Refiner),patch.object(web,'_stream_request',request):
        app=web.create_app(Path('/tmp/fake-refiner'),'cpu','http://frozen-asr.invalid','Chinese',32,None,
                           refinement_gate_mode='tri_state',experimental_gate_router=lambda *_:dict(action='keep'))
        with TestClient(app) as client:final,_,_=final_snapshot(client)
    assert final['clean_text']=='请确认编号。'
    assert requests==['local']
    assert final['refiner_session_stats']['inflight_call_count']==0
