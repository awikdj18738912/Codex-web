from types import SimpleNamespace
from experiments.scripts.audit_gate_failures import projected_cleanup, duplicate_repairs
from experiments.scripts.replay_gate_pipeline import review_after_windows


def test_cleanup_preserves_unrelated_name_error():
    assert projected_cleanup('需要需要郭新签字','需要郭新签字','需要需要郭鑫签字')=='需要郭鑫签字'


def test_non_deletion_or_missing_alignment_is_unresolved():
    assert projected_cleanup('五元','六元','五元') is None
    assert projected_cleanup('需要需要签字','需要签字','签字') is None


def test_cross_task_duplicate_requires_same_recording_and_effect():
    common=dict(baseline_normalized='工作工作群',always_normalized='工作群',weak_label='benefit')
    rows=[dict(common,recording_group='a',case_id='1',task_kind='window_refine'),
          dict(common,recording_group='a',case_id='2',task_kind='local_edit_review'),
          dict(common,recording_group='b',case_id='3',task_kind='window_refine')]
    assert duplicate_repairs(rows)[0]['case_ids']==['1','2']
    assert len(duplicate_repairs(rows))==1


def test_local_candidates_regenerated_on_actual_window_output():
    candidate=SimpleNamespace(context='工作工作群',source='工作工作',target='工作',kind='repeat')
    candidate.public_dict=lambda:dict(source=candidate.source,target=candidate.target)
    def candidates(text,**kwargs):return (candidate,) if '工作工作' in text else ()
    web=SimpleNamespace(find_repetition_review_candidates=candidates,
        _repetition_candidate_context_offset=lambda *_:0,clean_transcript_deterministically=lambda s:s)
    student=SimpleNamespace(decide_with_fallback=lambda *_:dict(action='refine'))
    def review(text,**kwargs):
        found=web.find_repetition_review_candidates(text,max_candidates=128,numeric_rule_mode=True)
        assert kwargs['review_cache']==kwargs['response_cache']=={}
        return (text.replace('工作工作','工作') if found else text,[],0,len(found),0)
    current=review_after_windows(web,review,'工作群','current',student,'general',None)
    learned=review_after_windows(web,review,'工作工作群','student',student,'general',None)
    assert current[0]==learned[0]=='工作群'
    assert current[2]==[] and len(learned[2])==1
    assert web.find_repetition_review_candidates is candidates


def test_candidate_generator_restored_on_error():
    def candidates(*_,**__):return ()
    web=SimpleNamespace(find_repetition_review_candidates=candidates)
    def review(*_,**__):raise RuntimeError('failure')
    import pytest
    with pytest.raises(RuntimeError):review_after_windows(web,review,'文本','student',None,'general',None)
    assert web.find_repetition_review_candidates is candidates
