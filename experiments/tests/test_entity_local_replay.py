"""Isolated tests for the frozen local-audio replay candidate.

No ASR/GPU/model is loaded. Relative imports are resolved under system._* names
so the candidate is isolated from the production module namespace.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_as_system_submodule(name: str, path: Path):
    qualified = f"system.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


CANDIDATE = _load_as_system_submodule(
    "_entity_local_replay_candidate_test",
    ROOT / "results/entity_local_replay_20261012/candidate_entity_audio_repair.py",
)
ORIGINAL = _load_as_system_submodule(
    "_entity_local_replay_original_test",
    ROOT / "results/entity_local_replay_20261012/original_entity_audio_repair.py",
)
RATE_BYTES = 64000


async def _unused_request(*args, **kwargs):
    raise AssertionError("unexpected ASR request")


def _definition():
    return CANDIDATE.EntityDefinition(
        1, "标准实体", "TERM", "general", "normalize", 0, ()
    )


def _candidate(text: str, observed: str, canonical: str, *, score=0.9, start=None,
               evidence_surface=None):
    start = text.index(observed) if start is None else start
    return {
        "start": start,
        "end": start + len(observed),
        "observed": observed,
        "canonical": canonical,
        "entity_id": 1,
        "entity_type": "TERM",
        "retrieval_score": score,
        "evidence_surface": evidence_surface if evidence_surface is not None else canonical,
    }


def _repair_with_map(text, audio_size, segments, audio_spans):
    repair = CANDIDATE.EntityAudioRepair(_unused_request, [_definition()])
    repair._audio_size = audio_size
    repair._segments = {
        event["segment_id"]: dict(event) for event in segments
    }
    repair._segment_audio = dict(audio_spans)
    return repair


def test_local_replay_is_disabled_below_45_seconds_and_enabled_at_exactly_45():
    text = "abcdefgh错词甲ijklmnop"
    cand = _candidate(text, "错词甲", "标准词甲")
    event = {"segment_id": 1, "text": text}
    small = _repair_with_map(
        text, int(44.999 * RATE_BYTES), [event], {1: (10*RATE_BYTES, 12*RATE_BYTES)}
    )
    assert small._localized_windows(text, [cand]) is None
    assert small.local_replay_reason == "short_audio_or_disabled"

    exact = _repair_with_map(
        text, 45*RATE_BYTES, [event], {1: (10*RATE_BYTES, 12*RATE_BYTES)}
    )
    windows = exact._localized_windows(text, [cand])
    assert windows is not None
    assert len(windows) == 1
    assert windows[0][2] == [cand]


def test_localization_requires_consistent_full_text_and_valid_timeline():
    parts = ["abcdefgh错词", "甲ijklmnop"]
    text = "".join(parts)
    events = [{"segment_id": i+1, "text": part} for i, part in enumerate(parts)]
    repair = _repair_with_map(
        text, 80*RATE_BYTES, events,
        {1: (10*RATE_BYTES, 12*RATE_BYTES), 2: (12*RATE_BYTES, 14*RATE_BYTES)},
    )
    cand = _candidate(text, "错词甲", "标准词甲")
    assert repair._localized_windows(text + "多字", [cand]) is None
    assert repair.local_replay_reason == "unreliable_segment_text"

    repair._timeline_valid = False
    assert repair._localized_windows(text, [cand]) is None
    assert repair.local_replay_reason == "unreliable_segment_text"


def test_repeated_segment_rewrite_and_missing_segment_id_invalidate_timeline():
    rewritten = _repair_with_map("甲", 0, [], {})
    rewritten._audio_size = 64000
    rewritten._record_segments({"completed_segments": [{"segment_id": 1, "text": "甲"}]}, audio_before=0)
    rewritten._audio_size = 128000
    rewritten._record_segments({"completed_segments": [{"segment_id": 1, "text": "乙"}]}, audio_before=64000)
    assert rewritten._timeline_valid is False

    missing = _repair_with_map("乙", 0, [], {})
    missing._audio_size = 64000
    missing._record_segments({"completed_segments": [{"segment_id": 2, "text": "乙"}]}, audio_before=0)
    assert missing._timeline_valid is False


def test_cross_segment_candidate_keeps_eight_character_anchors_in_replay_window():
    source = "abcdefgh加喀ijklmnop"
    split = source.index("加喀") + 1  # split the candidate span across two ASR segments
    events = [
        {"segment_id": 1, "text": source[:split]},
        {"segment_id": 2, "text": source[split:]},
    ]
    cand = _candidate(source, "加喀", "贾卡尔", evidence_surface="贾卡尔")
    repair = _repair_with_map(
        source, 60*RATE_BYTES, events,
        {1: (10*RATE_BYTES, 12*RATE_BYTES), 2: (12*RATE_BYTES, 14*RATE_BYTES)},
    )
    windows = repair._localized_windows(source, [cand])
    assert windows is not None and len(windows) == 1
    start, end, local = windows[0]
    assert start <= 8*RATE_BYTES and end >= 16*RATE_BYTES
    assert local == [cand]

    replay = source.replace("加喀", "贾卡尔", 1)
    revised, confirmed = CANDIDATE.verify_entity_audio_changes(source, replay, [cand])
    assert revised == replay
    assert confirmed == [cand]


def test_weak_candidate_inside_strong_window_is_retained_but_weak_outside_is_not_planned():
    text = "abcdefgh错词甲ijklmnopQRSTUVWXYZ乙丙丁abcdef"
    strong = _candidate(text, "错词甲", "标准词甲", score=0.9)
    weak = _candidate(text, "乙丙丁", "标准词乙", score=0.55)
    whole = {"segment_id": 1, "text": text}
    repair = _repair_with_map(text, 100*RATE_BYTES, [whole], {1: (10*RATE_BYTES, 13*RATE_BYTES)})
    windows = repair._localized_windows(text, [strong, weak])
    assert windows is not None and len(windows) == 1
    assert strong in windows[0][2]
    assert weak in windows[0][2]

    pieces = [
        "abcdefgh错词甲ijklmnopQRST甲",
        "乙qrstuvwx弱词乙abcdefghLMNO丙",
    ]
    separate_text = "".join(pieces)
    separate = [
        {"segment_id": 1, "text": pieces[0]},
        {"segment_id": 2, "text": pieces[1]},
    ]
    strong2 = _candidate(separate_text, "错词甲", "标准词甲", score=0.9)
    weak2 = _candidate(separate_text, "弱词乙", "标准词乙", score=0.55)
    separated = _repair_with_map(
        separate_text, 100*RATE_BYTES, separate,
        {1: (10*RATE_BYTES, 12*RATE_BYTES), 2: (40*RATE_BYTES, 42*RATE_BYTES)},
    )
    planned = separated._localized_windows(separate_text, [strong2, weak2])
    assert planned is not None and len(planned) == 1
    assert strong2 in planned[0][2]
    assert weak2 not in planned[0][2]
    assert separated.local_replay_reason is None


def test_more_than_two_windows_or_more_than_sixty_percent_audio_falls_back():
    pieces = [
        "abcdefgh甲乙丙ijklmnopQRST甲",
        "乙qrstuvwx丁戊己abcdefghLMNO丙",
        "丁ijklmnop庚辛壬qrstuvwxYZAB戊",
    ]
    text = "".join(pieces)
    starts = [0, len(pieces[0]), len(pieces[0])+len(pieces[1])]
    events = [{"segment_id": i+1, "text": part} for i,part in enumerate(pieces)]
    spans = {1:(0,2*RATE_BYTES),2:(10*RATE_BYTES,12*RATE_BYTES),3:(20*RATE_BYTES,22*RATE_BYTES)}
    cands = [_candidate(text, value, f"标准词{i}", start=starts[i]+8, score=0.9)
             for i,value in enumerate(("甲乙丙","丁戊己","庚辛壬"))]
    repair = _repair_with_map(text,100*RATE_BYTES,events,spans)
    assert repair._localized_windows(text,cands) is None
    assert repair.local_replay_reason == "insufficient_audio_saving"

    long_text = "abcdefgh错词甲ijklmnop"
    long_event = {"segment_id":1,"text":long_text}
    long_cand = _candidate(long_text,"错词甲","标准词甲")
    coverage = _repair_with_map(long_text,100*RATE_BYTES,[long_event],{1:(0,70*RATE_BYTES)})
    assert coverage._localized_windows(long_text,[long_cand]) is None
    assert coverage.local_replay_reason == "insufficient_audio_saving"


def test_confirmed_short_alias_and_edge_sound_still_pass_but_unsupported_expansion_is_rejected():
    alias_source = "今天中传开会。"
    alias = _candidate(alias_source,"中传","中国传媒大学",evidence_surface="中传")
    alias_replay = "今天中国传媒大学开会。"
    assert ORIGINAL.verify_entity_audio_changes(alias_source,alias_replay,[alias])[1] == [alias]
    assert CANDIDATE.verify_entity_audio_changes(alias_source,alias_replay,[alias])[1] == [alias]

    edge_source = "风声中加喀随后又说完毕。"
    edge = _candidate(edge_source,"加喀","贾卡尔")
    edge_replay = edge_source.replace("加喀","贾卡尔",1)
    assert ORIGINAL.verify_entity_audio_changes(edge_source,edge_replay,[edge])[1] == [edge]
    assert CANDIDATE.verify_entity_audio_changes(edge_source,edge_replay,[edge])[1] == [edge]

    bad_source = "柏林乐曲博物馆。"
    bad = _candidate(bad_source,"乐曲","越秀区")
    bad_replay = "柏林越秀区博物馆。"
    assert ORIGINAL.verify_entity_audio_changes(bad_source,bad_replay,[bad])[1] == []
    assert CANDIDATE.verify_entity_audio_changes(bad_source,bad_replay,[bad])[1] == []


def test_payload_revision_updates_only_covered_segment_and_clears_unmappable_segments():
    source = "abcdefgh错词甲ijklmnop后续文本"
    parts = [source[:11], source[11:]]
    candidate = _candidate(source,"错词甲","标准词甲")
    corrected = source.replace("错词甲","标准词甲",1)
    payload = {"text":source,"confidence":0.8,"completed_segments":[
        {"segment_id":1,"text":parts[0],"confidence":0.8},
        {"segment_id":2,"text":parts[1],"confidence":0.7},
    ]}
    revised = CANDIDATE.revise_entity_audio_payload(payload,payload['completed_segments'],corrected,[candidate])
    assert revised['text'] == corrected
    assert CANDIDATE.join_refined_segments(e['text'] for e in revised['completed_segments']) == corrected
    assert revised['completed_segments'][0]['text'] == parts[0].replace("错词甲","标准词甲",1)
    assert revised['completed_segments'][1]['text'] == parts[1]
    assert revised['completed_segments'][0]['confidence'] is None
    assert payload['text'] == source  # source payload remains untouched

    split = source.index("错词甲") + 1
    split_payload = {"text":source,"completed_segments":[
        {"segment_id":1,"text":source[:split]},
        {"segment_id":2,"text":source[split:]},
    ]}
    split_revision = CANDIDATE.revise_entity_audio_payload(split_payload,split_payload['completed_segments'],corrected,[candidate])
    assert split_revision['text'] == corrected
    assert split_revision['completed_segments'] == []  # whole-event fallback prevents stale segment text
    assert split_payload['text'] == source


def test_local_error_after_an_earlier_proposal_falls_back_atomically():
    source1 = "abcdefgh错词甲ijklmnop"
    source2 = "qrstuvwx错词乙abcdefgh"
    source = source1 + source2
    c1 = _candidate(source,"错词甲","标准词甲",start=8)
    c2 = _candidate(source,"错词乙","标准词乙",start=len(source1)+8)
    events = [{"segment_id":1,"text":source1},{"segment_id":2,"text":source2}]
    bounds = {1:(0,2*RATE_BYTES),2:(10*RATE_BYTES,12*RATE_BYTES)}
    publish_calls=[]

    class FakeSelector:
        def __init__(self, definitions): pass
        def select(self, text): return [c1,c2],[c1['canonical'],c2['canonical']]

    old_selector = CANDIDATE.GenericEntitySelector
    CANDIDATE.GenericEntitySelector = FakeSelector
    try:
        async def request(endpoint, session_id=None, data=b"", params=None):
            if endpoint == "/stream/start": return {"session_id":"fake-session","context_applied":False}
            if endpoint == "/stream/chunk": return {"completed_segments":events}
            if endpoint == "/stream/finish": return {"text":source,"completed_segments":events}
            if endpoint == "/stream/cancel": return {"cancelled":True}
            raise AssertionError(endpoint)
        repair = CANDIDATE.EntityAudioRepair(request,[_definition()])
        repair._localized_windows = lambda text,candidates:[[0,1,[c1]],[1,2,[c2]]]
        calls=[]
        async def fake_replay(start,end,context):
            calls.append((start,end))
            if len(calls)==1:
                return source1.replace("错词甲","标准词甲",1),False
            raise RuntimeError("second local replay failed")
        repair._replay_range = fake_replay
        def publish(original,changes):
            publish_calls.append((original,changes))
            return original
        repair._publish_changes = publish

        async def exercise():
            await repair.request("/stream/start")
            await repair.request("/stream/chunk","fake-session",b"pcm")
            return await repair.request("/stream/finish","fake-session")
        final = asyncio.run(exercise())
    finally:
        CANDIDATE.GenericEntitySelector = old_selector

    assert len(calls)==2
    assert publish_calls==[]
    assert final['text']==source
    assert final['completed_segments']==events
    assert final['entity_audio_repair']['phase']=="fallback"
    assert final['entity_audio_repair']['changes']==[]
    assert final['entity_audio_repair_revision']==0
