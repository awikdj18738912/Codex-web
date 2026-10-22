"""Historical legacy local-replay regression tests.

These protect the archived offline fallback behavior, not the production
default streaming entity policy. No model/ASR is invoked.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
from pathlib import Path

import pytest

import system.entity_audio_repair as V2
RATE_BYTES = 64000


async def _unused_request(*args, **kwargs):
    raise AssertionError("unexpected ASR request")


def candidate(text, observed, canonical, *, score, start=None, evidence_surface=None):
    start = text.index(observed) if start is None else start
    return {
        "start": start, "end": start + len(observed), "observed": observed,
        "canonical": canonical, "entity_id": 1, "entity_type": "TERM",
        "retrieval_score": score,
        "evidence_surface": evidence_surface if evidence_surface is not None else canonical,
    }


def repair_for(text, audio_seconds, events, bounds):
    definition = V2.EntityDefinition(1, "标准实体", "TERM", "general", "normalize", 0, ())
    repair = V2._LegacyEntityAudioRepair(_unused_request, [definition])
    repair._audio_size = int(audio_seconds * RATE_BYTES)
    repair._segments = {e["segment_id"]: dict(e) for e in events}
    repair._segment_audio = dict(bounds)
    return repair


def test_v2_preserves_frozen_45_second_boundary():
    text = "abcdefgh强词甲ijklmnop"
    c = candidate(text, "强词甲", "标准词甲", score=0.9)
    event = {"segment_id": 1, "text": text}
    below = repair_for(text, 44.999, [event], {1:(10*RATE_BYTES,12*RATE_BYTES)})
    assert below._localized_windows(text, [c]) is None
    exact = repair_for(text, 45.0, [event], {1:(10*RATE_BYTES,12*RATE_BYTES)})
    assert exact._localized_windows(text, [c]) is not None


def test_v2_eligible_weak_candidate_outside_strong_window_gets_its_own_window():
    p1 = "abcdefgh强词甲ijklmnopqrst甲"
    p2 = "乙abcdefgh弱词乙ijklmnop丙"
    text = p1 + p2
    events = [{"segment_id":1,"text":p1},{"segment_id":2,"text":p2}]
    strong = candidate(text, "强词甲", "标准词甲", score=0.9)
    weak = candidate(text, "弱词乙", "标准词乙", score=0.55)
    assert weak["retrieval_score"] < V2.MIN_REPLAY_SCORE
    # Both three-character observed spans pass the existing confirmation eligibility rule.
    assert V2._short_expansion_supported(strong)
    assert V2._short_expansion_supported(weak)
    repair = repair_for(text, 100.0, events, {
        1:(10*RATE_BYTES,12*RATE_BYTES),
        2:(30*RATE_BYTES,32*RATE_BYTES),
    })
    windows = repair._localized_windows(text, [strong, weak])
    assert windows is not None and len(windows) == 2
    planned = [c for _,_,local in windows for c in local]
    assert strong in planned and weak in planned


def test_v2_ineligible_short_expansion_is_not_given_its_own_window():
    p1 = "abcdefgh强词甲ijklmnopqrst甲"
    p2 = "乙uvwx乐曲博物馆abcdefgh丙"
    text = p1 + p2
    events = [{"segment_id":1,"text":p1},{"segment_id":2,"text":p2}]
    strong = candidate(text,"强词甲","标准词甲",score=0.9)
    weak_ineligible = candidate(text,"乐曲","越秀区",score=0.55)
    assert V2._short_expansion_supported(strong)
    assert not V2._short_expansion_supported(weak_ineligible)
    repair = repair_for(text,100.0,events,{1:(10*RATE_BYTES,12*RATE_BYTES),2:(30*RATE_BYTES,32*RATE_BYTES)})
    windows = repair._localized_windows(text,[strong,weak_ineligible])
    assert windows is not None and len(windows)==1
    planned=[c for _,_,local in windows for c in local]
    assert strong in planned and weak_ineligible not in planned


def test_v2_ineligible_short_candidate_can_remain_an_anchor_inside_strong_window():
    text="abcdefgh强词甲ijklmnopQRST乐曲博物馆后续文字abcdefgh"
    strong=candidate(text,"强词甲","标准词甲",score=0.9)
    short=candidate(text,"乐曲","越秀区",score=0.55)
    assert V2._short_expansion_supported(strong)
    assert not V2._short_expansion_supported(short)
    repair=repair_for(text,100.0,[{"segment_id":1,"text":text}],{1:(10*RATE_BYTES,13*RATE_BYTES)})
    windows=repair._localized_windows(text,[strong,short])
    assert windows is not None and len(windows)==1
    assert strong in windows[0][2]
    # V2 does not plan a standalone window for this ineligible expansion, but
    # retains it in a selected window's candidate context.
    assert short in windows[0][2]


def test_v2_preserves_original_global_weak_score_gate(monkeypatch):
    text="abcdefgh弱词乙ijklmnop"
    weak=candidate(text,"弱词乙","标准词乙",score=0.55)
    assert V2._short_expansion_supported(weak)
    class FakeSelector:
        def __init__(self, definitions): pass
        def select(self, source): return [weak],[weak["canonical"]]
    monkeypatch.setattr(V2,"GenericEntitySelector",FakeSelector)
    async def no_network_request(*args,**kwargs):
        raise AssertionError("global weak gate should return before ASR replay")
    repair=V2._LegacyEntityAudioRepair(no_network_request,[V2.EntityDefinition(1,"标准实体","TERM","general","normalize",0,())])
    result=asyncio.run(repair._repair({"text":text}))
    assert result["text"]==text
    assert repair.reason=="weak_candidates"
    assert repair.replay_mode=="pending"
    # This global gate is inherited from the original implementation, not a
    # new V2 window-planning filter.


def test_v2_over_window_count_or_coverage_falls_back_for_full_replay():
    pieces=[
        "abcdefgh甲乙丙ijklmnopQRST甲",
        "乙qrstuvwx丁戊己abcdefghLMNO丙",
        "丁ijklmnop庚辛壬qrstuvwxYZAB戊",
    ]
    text="".join(pieces)
    offsets=[0,len(pieces[0]),len(pieces[0])+len(pieces[1])]
    events=[{"segment_id":i+1,"text":part} for i,part in enumerate(pieces)]
    spans={1:(0,2*RATE_BYTES),2:(10*RATE_BYTES,12*RATE_BYTES),3:(20*RATE_BYTES,22*RATE_BYTES)}
    cands=[candidate(text,s,f"标准实体{i}",score=(0.91 if i==0 else 0.55),start=offsets[i]+8)
           for i,s in enumerate(("甲乙丙","丁戊己","庚辛壬"))]
    repair=repair_for(text,100.0,events,spans)
    assert repair._localized_windows(text,cands) is None
    assert repair.local_replay_reason=="insufficient_audio_saving"

    long_text="abcdefgh强词甲ijklmnop"
    long_c=candidate(long_text,"强词甲","标准词甲",score=0.9)
    coverage=repair_for(long_text,100.0,[{"segment_id":1,"text":long_text}],{1:(0,70*RATE_BYTES)})
    assert coverage._localized_windows(long_text,[long_c]) is None
    assert coverage.local_replay_reason=="insufficient_audio_saving"


def test_v2_eligible_weak_candidates_over_window_limit_trigger_full_audio_fallback(monkeypatch):
    pieces=[
        "abcdefgh甲乙丙ijklmnopQRST甲",
        "乙qrstuvwx丁戊己abcdefghLMNO丙",
        "丁ijklmnop庚辛壬qrstuvwxYZAB戊",
    ]
    text="".join(pieces)
    offsets=[0,len(pieces[0]),len(pieces[0])+len(pieces[1])]
    events=[{"segment_id":i+1,"text":part} for i,part in enumerate(pieces)]
    cands=[candidate(text,s,f"标准实体{i}",score=(0.91 if i==0 else 0.55),start=offsets[i]+8)
           for i,s in enumerate(("甲乙丙","丁戊己","庚辛壬"))]
    assert all(V2._short_expansion_supported(c) for c in cands)

    class FakeSelector:
        def __init__(self, definitions): pass
        def select(self, source): return cands,[c["canonical"] for c in cands]
    monkeypatch.setattr(V2,"GenericEntitySelector",FakeSelector)

    async def no_network_request(*args,**kwargs):
        raise AssertionError("_repair should use the overridden replay helper")
    repair=V2._LegacyEntityAudioRepair(no_network_request,[V2.EntityDefinition(1,"标准实体","TERM","general","normalize",0,())])
    repair._audio_size=100*RATE_BYTES
    repair._segments={e["segment_id"]:dict(e) for e in events}
    repair._segment_audio={1:(0,2*RATE_BYTES),2:(10*RATE_BYTES,12*RATE_BYTES),3:(20*RATE_BYTES,22*RATE_BYTES)}
    replay_ranges=[]
    async def fake_replay(start,end,context):
        replay_ranges.append((start,end))
        return text,False
    repair._replay_range=fake_replay
    result=asyncio.run(repair._repair({"text":text}))
    assert repair.replay_mode=="full"
    assert repair.local_replay_reason=="insufficient_audio_saving"
    assert replay_ranges==[(0,100*RATE_BYTES)]
    assert result["text"]==text
    assert repair.changes==[] and repair.revision==0


def test_v2_segment_revision_is_atomic_after_a_later_local_replay_error(monkeypatch):
    part1="abcdefgh强词甲ijklmnop"
    part2="qrstuvwx弱词乙abcdefgh"
    text=part1+part2
    c1=candidate(text,"强词甲","标准词甲",score=0.9)
    c2=candidate(text,"弱词乙","标准词乙",score=0.55)
    events=[{"segment_id":1,"text":text}]
    publish_calls=[]

    class FakeSelector:
        def __init__(self, definitions): pass
        def select(self, source): return [c1,c2],[c1["canonical"],c2["canonical"]]
    monkeypatch.setattr(V2,"GenericEntitySelector",FakeSelector)

    async def fake_request(endpoint,session_id=None,data=b"",params=None):
        if endpoint=="/stream/start": return {"session_id":"session-v2","context_applied":False}
        if endpoint=="/stream/chunk": return {"completed_segments":events}
        if endpoint=="/stream/finish": return {"text":text,"completed_segments":events}
        if endpoint=="/stream/cancel": return {"cancelled":True}
        raise AssertionError(endpoint)

    repair=V2._LegacyEntityAudioRepair(fake_request,[V2.EntityDefinition(1,"标准实体","TERM","general","normalize",0,())])
    repair._localized_windows=lambda source,candidates:[[0,1,[c1]],[1,2,[c2]]]
    calls=[]
    async def fail_second_window(start,end,context):
        calls.append((start,end))
        if len(calls)==1:
            return part1.replace("强词甲","标准词甲",1),False
        raise RuntimeError("synthetic second-window failure")
    repair._replay_range=fail_second_window
    repair._publish_changes=lambda original,changes: publish_calls.append((original,changes))

    async def exercise():
        await repair.request("/stream/start")
        await repair.request("/stream/chunk","session-v2",b"pcm")
        return await repair.request("/stream/finish","session-v2")
    final=asyncio.run(exercise())
    assert len(calls)==2
    assert publish_calls==[]
    assert final["text"]==text
    assert final["completed_segments"]==events
    assert final["entity_audio_repair"]["phase"]=="fallback"
    assert final["entity_audio_repair"]["changes"]==[]
    assert final["entity_audio_repair_revision"]==0


def test_v2_preserves_short_alias_edge_and_segment_payload_fallback():
    source="今天中传开会。"
    alias=candidate(source,"中传","中国传媒大学",score=0.7,evidence_surface="中传")
    replay="今天中国传媒大学开会。"
    assert V2.verify_entity_audio_changes(source,replay,[alias])[1]==[alias]

    sound_source="风声中加喀随后又说完毕。"
    edge=candidate(sound_source,"加喀","贾卡尔",score=0.7)
    sound_replay=sound_source.replace("加喀","贾卡尔",1)
    assert V2.verify_entity_audio_changes(sound_source,sound_replay,[edge])[1]==[edge]

    # A confirmed correction crossing a segment boundary keeps the authoritative
    # top-level text but clears segment events rather than returning stale text.
    cross="abcdefgh加喀ijklmnop"
    split=cross.index("加喀")+1
    cross_c=candidate(cross,"加喀","贾卡尔",score=0.7)
    corrected=cross.replace("加喀","贾卡尔",1)
    payload={"text":cross,"completed_segments":[{"segment_id":1,"text":cross[:split]},
                                                   {"segment_id":2,"text":cross[split:]}]}
    revised=V2.revise_entity_audio_payload(payload,payload["completed_segments"],corrected,[cross_c])
    assert revised["text"]==corrected
    assert revised["completed_segments"]==[]
    assert payload["text"]==cross
