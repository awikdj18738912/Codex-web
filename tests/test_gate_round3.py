from system.weak_refinement_gate import GAIN_LABELS,WeakRefinementGate,select_gain_action


def test_binary_invalid_or_missing_scores_preserve_native_action():
    for scores in [None,[float('nan'),0],[0.2,0.2],[0.3,0.4,0.3]]:
        assert select_gain_action(scores,0.3,'refine')=='refine'
    assert select_gain_action([0.01,0.99],0.3,'keep')=='keep'
    assert select_gain_action([0.01,0.99],0.3,'defer')=='defer'


def test_binary_recall_filter_and_runtime_named_probabilities():
    gate=WeakRefinementGate.__new__(WeakRefinementGate)
    gate.labels=GAIN_LABELS;gate.metadata=dict(gain_thresholds=dict(window_refine=0.1))
    gate.score=lambda _: [0.08,0.92]
    result=gate.decide_with_fallback(dict(task_kind='window_refine'),'refine')
    assert result['action']=='keep' and set(result['probabilities'])==set(GAIN_LABELS)
    gate.score=lambda _: [0.11,0.89]
    assert gate.decide_with_fallback(dict(task_kind='window_refine'),'refine')['action']=='refine'


def test_single_edit_effect_can_improve_with_unrelated_residual_error():
    from experiments.scripts.assemble_gate_round3 import operation_label
    label,basis=operation_label('uncertain','需要需要林青签字','需要林青签字','需要林清签字','需要林清签字')
    assert label=='benefit' and basis=='single_lexical_edit_reduces_script_distance'


def test_equal_distance_and_mixed_changes_stay_uncertain():
    from experiments.scripts.assemble_gate_round3 import operation_label
    assert operation_label('uncertain','林青签字','林清签字','林晴签字','林晴签字')[0]=='uncertain'
    assert operation_label('uncertain','需要需要林青签字','需要王青盖章','需要林清签字','需要林清签字')[0]=='uncertain'
    assert operation_label('uncertain','需要签字','需要签字','需要签字','需要签字')[0]=='uncertain'


def test_gain_selection_retains_each_small_development_cohort():
    from experiments.scripts.train_gate_round3 import gain_thresholds
    rows=[];probs=[];outcomes={}
    for i in range(22):
        effect='benefit' if i<20 else 'neutral';key=str(i)
        rows.append(dict(case_id=key,native_action='refine',validation_cohort='new_dev' if i==19 else 'old_dev',input=dict(task_kind='window_refine')))
        probs.append([0.01,0.99] if i==19 or i>=20 else [0.9,0.1])
        outcomes[key]=dict(arms=dict(current=dict(effect=effect,operation_cost_ms=10)))
    thresholds,evidence=gain_thresholds(rows,probs,outcomes)
    assert thresholds['window_refine']==0
    assert thresholds['local_edit_review']==0
    assert evidence['window_refine']['by_cohort']['new_dev']['retained']==1


def test_streaming_confidence_filter_preserves_text_and_segments():
    from experiments.scripts.evaluate_gate_streaming import without_confidence
    value=dict(text='原始文本',confidence=.99,confidence_metadata={'confidence':.98},completed_segments=[dict(text='分段',confidence=.9)])
    result=without_confidence(value)
    assert result['confidence'] is None and result['confidence_metadata']=={}
    assert result['completed_segments'][0]['text']=='分段' and result['completed_segments'][0]['confidence'] is None
    assert value['confidence']==.99


def test_tts_default_reserve_stays_conservative_and_override_is_explicit():
    from types import SimpleNamespace
    from pipeline.src.speech.qwen_tts_service import QwenTTSService
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,mem_get_info=lambda _:(int(5.6*1024**3),24*1024**3)),bfloat16='bf16')
    import pytest
    with pytest.raises(RuntimeError):QwenTTSService(device='cuda:0')._choose_device(fake)
    assert QwenTTSService(device='cuda:0',minimum_free_gib=5.5)._choose_device(fake)==('cuda:0','bf16')
