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
