"""Stateful incremental publication, immutable audio, and failure recovery."""
import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest
from system.entity_store import EntityDefinition

spec=importlib.util.spec_from_file_location('system._budget_test',Path(__file__).with_name('candidate.py'))
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
Repair=module.IncrementalEntityRepair
ONE='今天我们一起讨论厉飞羽的相关问题。'
TWO='接下来我们一起讨论韩莉的相关问题。'
DEFS=(EntityDefinition(1,'厉飞宇','TERM','general','normalize',0,()),
      EntityDefinition(2,'韩立','PERSON','general','normalize',0,()))


class Backend:
    def __init__(self, *, slow=0, fail=False):
        self.slow=slow;self.fail=fail;self.sessions={};self.counter=0;self.foreground=0
        self.audio={};self.calls=[];self.active=0;self.max_active=0
    async def __call__(self,endpoint,sid=None,data=b'',params=None):
        self.calls.append((endpoint,sid,bytes(data),params));self.active+=1
        self.max_active=max(self.max_active,self.active)
        try:
            await asyncio.sleep(0)
            if endpoint=='/stream/start':
                self.counter+=1;sid=str(self.counter);prompt=bool((params or {}).get('context'))
                self.sessions[sid]={'prompt':prompt,'pcm':b''};return dict(session_id=sid,context_applied=prompt)
            if endpoint=='/stream/cancel':self.sessions.pop(sid,None);return dict(cancelled=True)
            session=self.sessions[sid]
            if session['prompt']:
                if self.slow:await asyncio.sleep(self.slow)
                if self.fail:raise RuntimeError('deliberate replay failure')
            if endpoint=='/stream/chunk':
                session['pcm']+=data
                if not session['prompt']:self.foreground+=1
            if session['prompt']:
                source=ONE if session['pcm'].startswith(b'A') else TWO
                if b'B' in session['pcm'] and session['pcm'].startswith(b'A'):source=ONE+TWO
                text=source.replace('羽','宇').replace('莉','立');events=[]
                self.audio[sid]=session['pcm']
            else:
                text=ONE if self.foreground<=1 else ONE+TWO
                index=1 if self.foreground<=1 else 2
                events=[dict(segment_id=index,text=ONE if index==1 else TWO,confidence=.99)]
                if endpoint=='/stream/finish':events=[dict(segment_id=1,text=ONE),dict(segment_id=2,text=TWO)] if self.foreground>1 else [dict(segment_id=1,text=ONE)]
            if endpoint=='/stream/finish':self.sessions.pop(sid,None)
            return dict(text=text,completed_segments=events,confidence=.99,confidence_covers_full_text=True)
        finally:self.active-=1


class FinalOnlyBackend:
    """No entity candidate is visible until the authoritative finish result."""
    def __init__(self):self.sessions={};self.counter=0;self.calls=[]
    async def __call__(self,endpoint,sid=None,data=b'',params=None):
        self.calls.append((endpoint,sid,bytes(data),params))
        if endpoint=='/stream/start':
            self.counter+=1;sid=str(self.counter);prompt=bool((params or {}).get('context'))
            self.sessions[sid]={'prompt':prompt,'pcm':b''}
            return dict(session_id=sid,context_applied=prompt)
        session=self.sessions[sid]
        if endpoint=='/stream/chunk':session['pcm']+=data
        if session['prompt']:
            text=ONE.replace('羽','宇');segments=[]
        elif endpoint=='/stream/finish':
            text=ONE;segments=[dict(segment_id=1,text=ONE,confidence=.99)]
        else:
            text='目前没有名称。';segments=[]
        if endpoint in {'/stream/finish','/stream/cancel'}:self.sessions.pop(sid,None)
        return dict(text=text,completed_segments=segments,confidence=.99,confidence_covers_full_text=True)


async def settle(repair):
    for _ in range(200):
        if repair._review and repair._review.done():return
        await asyncio.sleep(.001)
    raise AssertionError('review never settled')


def test_stream_revision_is_published_before_finish_and_survives_later_raw():
    async def scenario():
        backend=Backend();r=Repair(backend,DEFS,candidate_margin=0,minimum_source_seconds=0,minimum_segments=1,minimum_saving=0)
        sid=(await r.request('/stream/start'))['session_id']
        first=await r.request('/stream/chunk',sid,b'A'*64000);assert first['text']==ONE
        await settle(r)
        second=await r.request('/stream/chunk',sid,b'B'*64000)
        assert '厉飞宇' in second['text'] and '韩莉' in second['text']
        assert second['entity_audio_repair_revision']==1
        assert len(second['completed_segments'])==2
        assert next(iter(backend.audio.values()))==b'A'*64000
        await settle(r)
        third=await r.request('/stream/chunk',sid,b'B'*64000)
        assert '厉飞宇' in third['text'] and '韩立' in third['text']
        assert third['entity_audio_repair_revision']==2
        final=await r.request('/stream/finish',sid)
        assert final['text']==third['text'] and final['entity_audio_repair_revision']==2
        assert final['confidence'] is None and not final['confidence_covers_full_text']
        assert backend.max_active==1 and not backend.sessions
    asyncio.run(scenario())


def test_no_repeated_review_when_source_stays_unchanged():
    async def scenario():
        backend=Backend();r=Repair(backend,DEFS[:1],candidate_margin=0,minimum_source_seconds=0,minimum_segments=1,minimum_saving=0)
        sid=(await r.request('/stream/start'))['session_id'];await r.request('/stream/chunk',sid,b'A'*64000)
        await settle(r);final=await r.request('/stream/finish',sid)
        assert final['entity_audio_repair']['checks']==1
        assert final['entity_audio_repair']['incremental']['completed_windows']==1
        assert final['text']==ONE.replace('羽','宇')
    asyncio.run(scenario())


@pytest.mark.parametrize('condition',['failure','timeout','oversized','budget'])
def test_failure_keeps_original_and_incomplete_evidence_uses_legacy_confirmation(condition):
    async def scenario():
        backend=Backend(fail=condition=='failure',slow=.1 if condition=='timeout' else 0)
        r=Repair(backend,DEFS[:1],candidate_margin=0,minimum_source_seconds=0,minimum_segments=1,minimum_saving=0,finish_wait=.002 if condition=='timeout' else .8,
                 window_seconds=.01 if condition=='oversized' else 20,
                 replay_budget_seconds=.01 if condition=='budget' else 60)
        sid=(await r.request('/stream/start'))['session_id'];await r.request('/stream/chunk',sid,b'A'*64000)
        final=await r.request('/stream/finish',sid)
        expected=ONE if condition=='failure' else ONE.replace('羽','宇')
        assert final['text']==expected
        assert backend.counter<=3
        await asyncio.sleep(.25)
        assert not backend.sessions
    asyncio.run(scenario())


def test_short_clip_uses_original_verification_path_without_incremental_work():
    async def scenario():
        backend=Backend();r=Repair(backend,DEFS[:1])
        sid=(await r.request('/stream/start'))['session_id']
        first=await r.request('/stream/chunk',sid,b'A'*64000)
        assert first['text']==ONE and r._review is None
        final=await r.request('/stream/finish',sid)
        assert final['text']==ONE.replace('羽','宇')
        assert final['entity_audio_repair']['incremental']['detections']==0
        assert backend.counter==2
    asyncio.run(scenario())


def test_exact_span_competition_does_not_confuse_nested_entity_with_full_name():
    def c(a,b,name,score):return dict(start=a,end=b,canonical=name,evidence_surface=name,observed='杨祝军' if b-a==3 else '祝军',retrieval_score=score)
    full=c(0,3,'杨柱军',.96);nested=c(1,3,'朱军',.95);competitor=c(0,3,'杨竹军',.95)
    assert full in [{k:v for k,v in r.items() if k!='candidate_margin'} for r in module.eligible_candidates([full,nested])]
    assert not module.eligible_candidates([full,competitor])


def test_invalid_source_withdraws_previous_revision_instead_of_moving_edits():
    backend=Backend();r=Repair(backend,DEFS[:1]);r._segments={1:dict(segment_id=1,text=ONE)}
    start=ONE.index('厉飞羽');c=dict(start=start,end=start+3,observed='厉飞羽',canonical='厉飞宇',entity_id=1)
    r._proposals=[c]
    first=r._overlay(dict(text=ONE));assert first['entity_audio_repair_revision']==1
    second=r._overlay(dict(text='不同的内容。'))
    assert second['text']=='不同的内容。' and second['entity_audio_repair_revision']==2
    assert not r.changes
    r.close()


def test_finish_does_not_schedule_new_local_review_and_final_only_candidate_uses_legacy():
    async def scenario():
        backend=FinalOnlyBackend()
        r=Repair(backend,DEFS[:1],candidate_margin=0,minimum_source_seconds=0,
                 minimum_segments=1,minimum_saving=0)
        schedule_calls=0
        original_schedule=r._schedule
        async def counted_schedule():
            nonlocal schedule_calls
            schedule_calls+=1
            await original_schedule()
        r._schedule=counted_schedule
        sid=(await r.request('/stream/start'))['session_id']
        partial=await r.request('/stream/chunk',sid,b'A'*64000)
        assert '厉飞宇' not in partial['text'] and r._review is None
        before_finish=schedule_calls
        assert before_finish==1
        final=await r.request('/stream/finish',sid)
        assert schedule_calls==before_finish
        assert '厉飞宇' in final['text']
        assert final['entity_audio_repair']['incremental']['windows']==0
        assert final['entity_audio_repair']['incremental']['legacy_final_fallbacks']==1
        assert final['entity_audio_repair']['checks']==1
        assert not backend.sessions
    asyncio.run(scenario())


def test_window_over_sixty_percent_of_received_audio_defers_to_legacy_finish():
    async def scenario():
        backend=Backend()
        r=Repair(backend,DEFS[:1],candidate_margin=0,minimum_source_seconds=0,
                 minimum_segments=1)
        sid=(await r.request('/stream/start'))['session_id']
        partial=await r.request('/stream/chunk',sid,b'A'*(10*64000))
        assert '厉飞羽' in partial['text']
        assert r._review is None and not r._incremental_used
        assert r._inc['windows']==0
        assert r._audio_size==10*64000
        assert sum(call[0]=='/stream/start' for call in backend.calls)==1
        final=await r.request('/stream/finish',sid)
        assert '厉飞宇' in final['text']
        assert final['entity_audio_repair']['incremental']['legacy_final_fallbacks']==1
        assert final['entity_audio_repair']['checks']==1
        assert not backend.sessions
    asyncio.run(scenario())


def test_legacy_final_with_no_new_change_withdraws_published_revision():
    async def scenario():
        backend=Backend();r=Repair(backend,DEFS[:1])
        source_segment=dict(segment_id=1,text=ONE,confidence=.99)
        r._segments={1:source_segment}
        start=ONE.index('厉飞羽')
        r._proposals=[dict(start=start,end=start+3,observed='厉飞羽',canonical='厉飞宇',entity_id=1)]
        first=r._overlay(dict(text=ONE,completed_segments=[dict(source_segment)]))
        assert first['text']==ONE.replace('羽','宇')
        assert first['entity_audio_repair_revision']==1
        assert r.changes

        # Final legacy text already has the canonical spelling, so it produces
        # no replacement and must publish a higher-revision withdrawal.
        final=await r._legacy_finish(dict(text=ONE.replace('羽','宇'),
                                          completed_segments=first['completed_segments']))
        assert final['text']==ONE.replace('羽','宇')
        assert final['entity_audio_repair_revision']==2
        assert final['completed_segments']==[source_segment]
        assert not r.changes and not r._proposals
        r.close()
    asyncio.run(scenario())

@pytest.mark.parametrize('unresolved_score,should_replay',[(.585,False),(.70,True)])
def test_unrelated_unresolved_span_requires_its_own_replay_score(monkeypatch,unresolved_score,should_replay):
    async def scenario():
        text=ONE+'再不行，我让他五个，怎么样啊？'
        r=Repair(Backend(),DEFS[:1]);r._segments={1:dict(segment_id=1,text=text)}
        start=text.index('厉飞羽');resolved=dict(start=start,end=start+3,observed='厉飞羽',canonical='厉飞宇',entity_id=1,retrieval_score=.966667)
        other=text.index('五个');unresolved=dict(start=other,end=other+2,observed='五个',canonical='误工',entity_id=3,retrieval_score=unresolved_score)
        r._proposals=[resolved];r._incremental_used=True
        class FrozenSelector:
            def select(self,value):
                assert value==text
                return [resolved,unresolved],['厉飞宇','误工']
        monkeypatch.setattr(module,'selector',lambda definitions:FrozenSelector())
        fallback_calls=[]
        async def fallback(original):
            fallback_calls.append(original)
            return original
        r._legacy_finish=fallback
        result=await r._repair(dict(text=text))
        assert bool(fallback_calls)==should_replay
        assert r._inc['legacy_final_fallbacks']==int(should_replay)
        if not should_replay:
            assert '厉飞宇' in result['text'] and '五个' in result['text'] and '误工' not in result['text']
            assert r._inc['weak_unresolved_kept']==1
        r.close()
    asyncio.run(scenario())


def _candidate(text, observed, canonical, entity_id, score=.92):
    start=text.index(observed)
    return dict(start=start,end=start+len(observed),observed=observed,
                canonical=canonical,entity_id=entity_id,evidence_surface=observed,
                retrieval_score=score)


def _freeze_selector(monkeypatch, candidates, calls=None):
    class FrozenSelector:
        def select(self, text):
            if calls is not None:
                calls.append(text)
            terms=list(dict.fromkeys(c['canonical'] for c in candidates))
            return [dict(c) for c in candidates],terms
    monkeypatch.setattr(module,'selector',lambda definitions:FrozenSelector())


def test_intersecting_windows_are_merged_before_total_audio_budget(monkeypatch):
    async def scenario():
        backend=Backend()
        repair=Repair(backend,DEFS,candidate_margin=0,minimum_source_seconds=0,
                      minimum_segments=2,minimum_saving=.15,replay_budget_seconds=20,
                      replay_chunk_bytes=64000)
        text=ONE+TWO
        candidates=[_candidate(text,'厉飞羽','厉飞宇',1),
                    _candidate(text,'韩莉','韩立',2)]
        _freeze_selector(monkeypatch,candidates)
        # Raw windows cost 5+4 seconds (9s), which exceeds the 8.5s budget
        # under a 15% minimum saving. Their 1s overlap makes the merged plan
        # exactly 8s, so it should be admitted once.
        repair._windows=lambda source,local:[
            [0,5*64000,[local[0]]],
            [4*64000,8*64000,[local[1]]],
        ]
        sid=(await repair.request('/stream/start'))['session_id']
        await repair.request('/stream/chunk',sid,b'A'*(5*64000))
        partial=await repair.request('/stream/chunk',sid,b'B'*(5*64000))
        assert partial['text']==text
        assert repair._inc['windows']==1
        assert repair._inc['max_planned_audio_seconds']==8.0
        await settle(repair)
        assert repair.replay_audio_bytes==8*64000
        # One primary and one merged replay session; no duplicate window cost.
        assert backend.counter==2
        assert len(backend.audio['2'])==8*64000
        replay_start=next(c for c in backend.calls if c[0]=='/stream/start' and c[3] and c[3].get('context'))
        assert '厉飞宇' in replay_start[3]['context'] and '韩立' in replay_start[3]['context']
        assert repair._inc['completed_windows']==1
        await repair.request('/stream/cancel',sid)
        assert not backend.sessions
    asyncio.run(scenario())


@pytest.mark.parametrize('uncovered',[False,True])
def test_three_disjoint_or_uncovered_high_score_windows_skip_background_asr(monkeypatch,uncovered):
    async def scenario():
        backend=Backend()
        repair=Repair(backend,DEFS,candidate_margin=0,minimum_source_seconds=0,
                      minimum_segments=2,minimum_saving=0,replay_budget_seconds=30)
        text=ONE+TWO
        candidates=[_candidate(text,'厉飞羽','厉飞宇',1),
                    _candidate(text,'韩莉','韩立',2),
                    _candidate(text,'今天','今田',3)]
        _freeze_selector(monkeypatch,candidates)
        windows=[[0,1*64000,[candidates[0]]],
                 [3*64000,4*64000,[candidates[1]]]]
        if not uncovered:
            windows.append([6*64000,7*64000,[candidates[2]]])
        repair._windows=lambda source,local:windows
        sid=(await repair.request('/stream/start'))['session_id']
        await repair.request('/stream/chunk',sid,b'A'*(5*64000))
        await repair.request('/stream/chunk',sid,b'B'*(5*64000))
        reason='unmapped_or_oversized' if uncovered else 'too_many_windows'
        assert repair._inc['preflight_reason']==reason
        assert repair._inc['windows']==0 and repair.replay_audio_bytes==0
        # Before finish there has been no backend call beyond the primary start.
        assert backend.counter==1
        assert sum(c[0]=='/stream/start' for c in backend.calls)==1
        legacy=[]
        async def legacy_finish(original):
            legacy.append(original)
            return original
        repair._legacy_finish=legacy_finish
        final=await repair.request('/stream/finish',sid)
        assert legacy and final['entity_audio_repair']['incremental']['legacy_final_fallbacks']==1
        assert backend.counter==1
        assert not backend.sessions
    asyncio.run(scenario())


def test_insufficient_saving_window_is_reconsidered_when_audio_grows_without_retrieval(monkeypatch):
    class StableBackend:
        def __init__(self):self.sessions={};self.counter=0;self.calls=[]
        async def __call__(self,endpoint,sid=None,data=b'',params=None):
            self.calls.append((endpoint,sid,bytes(data),params))
            if endpoint=='/stream/start':
                self.counter+=1;sid=str(self.counter)
                self.sessions[sid]={'prompt':bool((params or {}).get('context')),'pcm':b''}
                return dict(session_id=sid,context_applied=self.sessions[sid]['prompt'])
            if endpoint=='/stream/cancel':self.sessions.pop(sid,None);return dict(cancelled=True)
            session=self.sessions[sid]
            if endpoint=='/stream/chunk':session['pcm']+=data
            if session['prompt']:
                text=ONE.replace('羽','宇')
                events=[]
            else:
                # A stable completed segment is repeated while only its audio
                # range grows; no transcript re-retrieval should be needed.
                text=ONE
                events=[dict(segment_id=1,text=ONE,confidence=.99)]
            if endpoint=='/stream/finish':self.sessions.pop(sid,None)
            return dict(text=text,completed_segments=events,confidence=.99,
                        confidence_covers_full_text=True)

    async def scenario():
        backend=StableBackend();selector_calls=[]
        repair=Repair(backend,DEFS[:1],candidate_margin=0,minimum_source_seconds=0,
                      minimum_segments=1,minimum_saving=.4,replay_budget_seconds=30,
                      replay_chunk_bytes=64000)
        candidate=_candidate(ONE,'厉飞羽','厉飞宇',1)
        _freeze_selector(monkeypatch,[candidate],selector_calls)
        repair._windows=lambda source,local:[[0,7*64000,[local[0]]]]
        sid=(await repair.request('/stream/start'))['session_id']
        await repair.request('/stream/chunk',sid,b'A'*(10*64000))
        assert repair._inc['preflight_reason']=='insufficient_total_saving'
        assert repair._inc['windows']==0 and repair._review is None
        assert repair._inc['detections']==1 and len(selector_calls)==1
        assert backend.counter==1

        # Same completed segment text, but 5s more PCM makes the identical 7s
        # local window pass the minimum-saving preflight (7/15 <= 60%).
        await repair.request('/stream/chunk',sid,b'B'*(5*64000))
        await settle(repair)
        assert repair._inc['detections']==1 and len(selector_calls)==1
        assert repair._inc['windows']==1 and repair._inc['completed_windows']==1
        assert repair._inc['max_planned_audio_seconds']==7.0
        assert repair.replay_audio_bytes==7*64000
        assert backend.counter==2
        await repair.request('/stream/cancel',sid)
        assert not backend.sessions
    asyncio.run(scenario())


def test_cached_selector_matches_generic_and_returns_mutation_isolated_results():
    text=ONE+TWO
    expected=module.GenericEntitySelector(DEFS).select(text)
    cached=module.CachedSelector(DEFS)
    first=cached.select(text)
    assert first==expected
    assert first[0] is not expected[0]
    assert first[1] is not expected[1]

    # Mutate both the returned candidate payload and terms list; neither should
    # poison the immutable cached result returned to a later caller.
    first[0][0]['canonical']='污染缓存'
    first[1].append('额外术语')
    again=cached.select(text)
    assert again==expected
    assert again[0][0]['canonical']!='污染缓存'
    assert '额外术语' not in again[1]
