"""Validate a type-independent replay gate without changing ASR/refiner code.

The .63 gate was fixed on the earlier 344 development examples. Fresh holdout
audio is decoded with the ungated wrapper so missed recoveries remain visible.
Both gates then consume the same recorded original PCM and real ASR responses.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import evaluate_fresh_entity_blind as base
from system.entity_audio_repair import EntityAudioRepair, MIN_REPLAY_SCORE, REPLAY_PREFIX
from system.entity_store import EntityStore

ROOT = base.ROOT
OUT = ROOT / 'results/generic_entity_optimization_20261010'
HOLDOUT = OUT / 'holdout'
ARMS = ('ungated', 'gated')


def prepare():
    old = json.loads((ROOT/'results/generic_entity_blind_20261010/metadata.json').read_text())
    new = json.loads((HOLDOUT/'metadata.json').read_text())
    assert len(new['samples']) == 96
    assert not {s['sample_id'] for s in old['samples']} & {s['sample_id'] for s in new['samples']}
    assert not {s['reference'] for s in old['samples']} & {s['reference'] for s in new['samples']}
    assert (HOLDOUT/'glossaries.json').read_bytes() == (ROOT/'results/generic_entity_blind_20261010/glossaries.json').read_bytes()
    base.OUT = HOLDOUT
    base.prepare()
    manifest = json.loads((HOLDOUT/'manifest.json').read_text())
    manifest['gate_threshold'] = MIN_REPLAY_SCORE
    manifest['threshold_chosen_from'] = 'Earlier 344 development examples; neither previous 128 nor this 96-item holdout'
    manifest['collection_gate'] = 0.0
    manifest['frozen_hashes'][str(Path(__file__).resolve().relative_to(ROOT))] = base.digest(__file__)
    base.write(HOLDOUT/'manifest.json', manifest)


async def collect():
    import system.entity_audio_repair as repair
    original = repair.EntityAudioRepair
    class Ungated(original):
        def __init__(self, *args, **kwargs):
            kwargs['minimum_replay_score'] = 0.0
            super().__init__(*args, **kwargs)
    repair.EntityAudioRepair = Ungated
    base.OUT = HOLDOUT
    try:
        await base.collect()
    finally:
        repair.EntityAudioRepair = original


async def exercise(sample, initial, replay, definitions, pcm, threshold, trace=None):
    cursor, sessions, calls = 0, {}, Counter()
    async def request(endpoint, session_id=None, data=b'', params=None):
        nonlocal cursor
        calls[endpoint] += 1
        if trace is not None:
            if endpoint == '/stream/cancel' and (cursor == len(trace) or trace[cursor]['endpoint'] != endpoint):
                return dict(cancelled=True)
            expected = trace[cursor]
            if (endpoint != expected['endpoint'] or session_id != expected['session_id']
                    or (params or {}).get('context', '') != expected['context']
                    or hashlib.sha256(data).hexdigest() != expected['data_sha256']):
                raise ValueError('Production requests differ from recorded real audio: ' + sample['id'])
            cursor += 1
            return deepcopy(expected['result'])
        if endpoint == '/stream/start':
            prompted = bool((params or {}).get('context'))
            record = replay if prompted else initial
            if record is None:
                raise ValueError('Unexpected prompt replay')
            ready = deepcopy(record['ready'])
            if prompted:
                from system.entity_asr_context import context_digest
                if ready.get('context_digest') != context_digest(params['context']):
                    raise ValueError('The unchanged term list differs from the recorded real prompt')
            sessions[ready['session_id']] = dict(record=record, index=0, pcm=bytearray())
            return ready
        if endpoint == '/stream/cancel':
            sessions.pop(session_id, None)
            return dict(cancelled=True)
        state = sessions[session_id]
        if endpoint == '/stream/chunk':
            state['pcm'].extend(data)
            response = deepcopy(state['record']['chunk_payloads'][state['index']]); state['index'] += 1
            return response
        if endpoint == '/stream/finish':
            if hashlib.sha256(state['pcm']).hexdigest() != sample['pcm_sha256']:
                raise ValueError('Replayed PCM changed')
            if state['index'] != len(state['record']['chunk_payloads']):
                raise ValueError('Incomplete stream replay')
            result = deepcopy(state['record']['final_payload']); sessions.pop(session_id)
            return result
        raise ValueError(endpoint)
    guard = EntityAudioRepair(request, definitions, minimum_replay_score=threshold)
    try:
        ready = await guard.request('/stream/start', params=dict(language='Chinese', chunk_seconds=2.0))
        for offset in range(0, len(pcm), 64000):
            await guard.request('/stream/chunk', ready['session_id'], pcm[offset:offset+64000])
        final = await guard.request('/stream/finish', ready['session_id'])
    finally:
        guard.close()
    if sessions:
        raise ValueError('Sessions were not released')
    # All authoritative downstream ASR fields must match, not just its text.
    authoritative = {k:v for k,v in final.items() if k not in ('entity_audio_repair', 'entity_audio_repair_revision')}
    return dict(final_payload=final, authoritative_sha256=hashlib.sha256(json.dumps(authoritative,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                start_count=calls['/stream/start'], trace_consumed=cursor)


async def regression():
    import soundfile as sf
    source = ROOT/'results/generic_entity_validation_20261010'
    manifest = json.loads((source/'manifest.json').read_text())
    raw = {r['id']:r for r in base.rows(source/'asr_baseline.jsonl')}
    replay = {r['id']:r for r in base.rows(ROOT/'results/generic_entity_audio_20261010/outputs_neighbor.jsonl') if r['replayed']}
    replay.update({r['id']:r['replay'] for r in base.rows(source/'new_audio_requests.jsonl') if r['replay']})
    previous = ROOT/'results/generic_entity_blind_20261010'
    fresh = {r['id']:r for r in base.rows(previous/'audio_results.jsonl')}
    fixtures = [(dict(s, cohort='old_'+s['cohort']), source/'libraries'/s['library_key']/'entities.db') for s in manifest['samples']]
    fixtures += [(dict(s, cohort='previous_128'), previous/'libraries'/s['source']/'entities.db')
                 for s in json.loads((previous/'manifest.json').read_text())['samples']]
    completed = {r['id'] for r in base.rows(OUT/'regression.jsonl')}
    definitions = {}
    for index,(sample,db) in enumerate(fixtures):
        if sample['id'] in completed:continue
        if str(db) not in definitions: definitions[str(db)] = EntityStore(db).list_entities()
        if sample['id'] in fresh:
            record = fresh[sample['id']]; initial = record['initial']; prompted = None; trace = record['request_trace']
            pcm = Path(sample['audio_path']).read_bytes()
        else:
            record = None; initial = raw[sample['id']]; prompted = replay.get(sample['id']); trace = None
            wave,rate = sf.read(sample['audio_path'],dtype='float32'); assert rate == 16000
            pcm = wave.astype('<f4').tobytes()
        if hashlib.sha256(pcm).hexdigest() != sample['pcm_sha256']:raise ValueError('Fixture audio changed')
        results = {}
        for arm,threshold in [('ungated',0.0),('gated',MIN_REPLAY_SCORE)]:
            results[arm] = await exercise(sample,initial,prompted,definitions[str(db)],pcm,threshold,trace)
        if results['ungated']['authoritative_sha256'] != results['gated']['authoritative_sha256']:
            raise ValueError('Gate changed existing authoritative ASR output: '+sample['id'])
        base.append(OUT/'regression.jsonl',dict(id=sample['id'],cohort=sample['cohort'],
            identical_authoritative_payload=True,results=results))
        print('REGRESSION',index+1,'/',len(fixtures),sample['id'],flush=True)
    base.write(OUT/'regression_summary.json',dict(samples=len(fixtures),identical_authoritative_payloads=len(fixtures)))


async def holdout():
    base.OUT = HOLDOUT
    manifest = base.verify()
    actual = {r['id']:r for r in base.rows(HOLDOUT/'audio_results.jsonl')}
    assert len(actual) == len(manifest['samples'])
    completed = {r['id'] for r in base.rows(HOLDOUT/'gate_results.jsonl')}
    definitions = {}
    for sample in manifest['samples']:
        if sample['id'] in completed:continue
        source = sample['source']
        if source not in definitions: definitions[source] = EntityStore(HOLDOUT/'libraries'/source/'entities.db').list_entities()
        record = actual[sample['id']]; pcm = Path(sample['audio_path']).read_bytes()
        assert hashlib.sha256(pcm).hexdigest() == sample['pcm_sha256']
        results = {}
        for arm,threshold in [('ungated',0.0),('gated',MIN_REPLAY_SCORE)]:
            results[arm] = await exercise(sample,record['initial'],None,definitions[source],pcm,threshold,record['request_trace'])
        assert results['ungated']['final_payload']['text'] == record['production_final']['text']
        base.append(HOLDOUT/'gate_results.jsonl',dict(id=sample['id'],source=source,cohort=sample['cohort'],results=results,
            identical_authoritative_payload=results['ungated']['authoritative_sha256']==results['gated']['authoritative_sha256'],
            avoided_prompted_seconds=record['timings'].get('prompted',0) if results['ungated']['start_count']>results['gated']['start_count'] else 0))
    base.verify()


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from evaluate_entity_library import normalized
    from evaluate_medical_entities import lexicon_occurrences
    references = json.loads((HOLDOUT/'references.json').read_text())
    glossary = json.loads((HOLDOUT/'glossaries.json').read_text())
    data = base.rows(HOLDOUT/'gate_results.jsonl'); assert len(data)==96
    converter,itn = opencc.OpenCC('t2s'),ZhITNNormalizer()
    def norm(text):return normalized(converter.convert(itn.normalize(text).text))
    summary,losses = {},[]
    for row in data:
        ref = references[row['id']]['reference']; names = [e['term'] for e in glossary[row['source']]]
        expected = Counter(e['text'] for e in lexicon_occurrences(ref,names)); scores = {}
        for arm in ARMS:
            payload = row['results'][arm]['final_payload']; status = payload['entity_audio_repair']; text = payload['text']
            have = Counter(e['text'] for e in lexicon_occurrences(text,names))
            hits=sum(min(n,have[w]) for w,n in expected.items());errors=Levenshtein.distance(norm(ref),norm(text))
            scores[arm]=(hits,errors)
            for group in ['all',row['source'],row['cohort']]:
                counter=summary.setdefault(group,{}).setdefault(arm,Counter())
                counter.update(samples=1,hits=hits,expected_mentions=sum(expected.values()),errors=errors,reference_chars=len(norm(ref)),
                    replays=status['checks'],changed_samples=int(bool(status['changes'])),changes=len(status['changes']),
                    extra_mentions=sum(max(0,n-expected[w]) for w,n in have.items()))
        if scores['gated'][0]<scores['ungated'][0] or scores['gated'][1]>scores['ungated'][1]:losses.append(row['id'])
    for group in summary.values():
        for counter in group.values():counter['CER']=counter['errors']/counter['reference_chars']
    total_avoided = sum(r['avoided_prompted_seconds'] for r in data)
    result=dict(threshold=MIN_REPLAY_SCORE,summary=summary,quality_losses=losses,
        authoritative_payloads_identical=sum(r['identical_authoritative_payload'] for r in data),
        avoided_real_request_seconds_from_recordings=total_avoided,
        note='Avoided time is the measured original prompted-ASR time of skipped trials, not a newly measured latency speedup')
    base.write(OUT/'holdout_summary.json',result)
    if losses:raise ValueError('Holdout quality loss; do not deploy the gate: '+str(losses))
    if summary['all']['gated']['replays']>=summary['all']['ungated']['replays']:raise ValueError('No useful replay saving')
    if result['authoritative_payloads_identical']!=96:raise ValueError('Unexpected downstream ASR fields changed')
    (OUT/'DEPLOY_ALLOWED').write_text('Historical regression and new holdout preserve all authoritative ASR fields; replay count reduced\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','regression','holdout','report'])
    stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='report':report()
    else:asyncio.run(dict(collect=collect,regression=regression,holdout=holdout)[stage]())
