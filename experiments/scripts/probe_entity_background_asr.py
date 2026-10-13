"""Reference-free, development-only dictionary background replay experiment.

Same frozen glossary and actual first pass; only the same-audio prompt gains
independently sourced definitions. Nothing is installed in production.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_context_confirmation_20261012'
BASE=ROOT/'results/large_entity_benchmark_v1_20261010'
PREVIOUS=ROOT/'results/entity_confirmation_20261011'
sys.path.insert(0,str(ROOT))


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    path=OUT/'background_asr_plan.json'
    if path.exists():raise FileExistsError('Background replay plan already frozen')
    sample_rows={r['id']:r for r in read(BASE/'dataset/samples.jsonl')}
    actual={r['id']:r for r in read(BASE/'full_evaluation/audio.jsonl')}
    for folder in ('holdout','holdout_news','holdout_v2'):
        sample_rows.update({r['id']:r for r in read(PREVIOUS/folder/'samples.jsonl')})
    for file in (PREVIOUS/'holdout_audio.jsonl',PREVIOUS/'short_expansion/holdout_audio.jsonl'):
        for row in read(file):
            actual[row['id']]=dict(initial=row['initial'],trace=row['baseline_trace'])
    ids=list(dict.fromkeys(r['sample_id'] for r in read(OUT/'pilot_inputs.jsonl')))
    inputs=[]
    for sid in ids:
        s=sample_rows[sid];record=actual[sid]
        inputs.append(dict(id=sid,pcm_path=s['pcm_path'],pcm_sha256=s['pcm_sha256'],
            initial=record['initial'],trace=record['trace']))
    data=OUT/'background_asr_inputs.jsonl'
    data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
    write(path,dict(development_only=True,recordings=len(inputs),input_sha256=digest(data),
        dictionary_sha256=digest(BASE/'dataset/library/entities.db'),
        background_sha256=digest(OUT/'verified_backgrounds.json'),
        entity_module_sha256=digest(ROOT/'system/entity_audio_repair.py'),
        target_reference_at_inference=False,new_names_added=False,
        selection='Unique recordings in the already-known64 review pilot; not independent accuracy evidence',
        conditions=['current_rule_same_trace','same_audio_with_verified_background']))
    print('PREPARED',len(inputs),'recordings',flush=True)


async def collect():
    import httpx
    from system.entity_audio_repair import EntityAudioRepair,GenericEntitySelector,REPLAY_PREFIX,verify_entity_audio_changes,revise_entity_audio_payload
    from system.entity_asr_context import EntityASRContext
    from system.entity_asr_echo_guard import EntityASREchoGuard
    from system.entity_store import EntityStore
    from validate_entity_confirmation import replay_request,exercise
    plan=json.loads((OUT/'background_asr_plan.json').read_text())
    for file,key in [(OUT/'background_asr_inputs.jsonl','input_sha256'),(OUT/'verified_backgrounds.json','background_sha256'),
                     (BASE/'dataset/library/entities.db','dictionary_sha256'),(ROOT/'system/entity_audio_repair.py','entity_module_sha256')]:
        assert digest(file)==plan[key]
    profiles=json.loads((OUT/'verified_backgrounds.json').read_text())['profiles']
    definitions=EntityStore(BASE/'dataset/library/entities.db').list_entities(domain='general')
    destination=OUT/'background_asr_outputs.jsonl';done={r['id'] for r in read(destination)} if destination.exists() else set()
    semaphore=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async def process(row):
            if row['id'] in done:return
            async with semaphore:
                pcm=Path(row['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
                request,remaining,_=replay_request(row['trace'])
                baseline=await exercise(EntityAudioRepair,request,definitions,pcm);assert not remaining
                text=row['initial'].get('text','')
                candidates,terms=GenericEntitySelector(definitions).select(text)
                descriptions=[dict(name=w,background=profiles[w]['description']) for w in terms if w in profiles]
                proposed=baseline;trace=[];fallback=False;attempts=0
                if descriptions and baseline['entity_audio_repair']['checks']:
                    context_text=REPLAY_PREFIX+json.dumps(terms,ensure_ascii=False)+'\n词条背景只帮助辨别名称，不代表录音内容；不要补充未听到的词：'+json.dumps(descriptions,ensure_ascii=False)
                    assert len(context_text)<=2048
                    context=EntityASRContext(context_text,tuple(terms))
                    async def actual(endpoint,session_id=None,data=b'',params=None):
                        query=dict(params or {})
                        if session_id:query['session_id']=session_id
                        result=(await client.post('http://127.0.0.1:8766'+endpoint,params=query,content=data)).raise_for_status().json()
                        trace.append(dict(endpoint=endpoint,session_id=session_id,context=query.get('context',''),
                                          data_sha256=hashlib.sha256(data).hexdigest(),result=result))
                        return result
                    guard=EntityASREchoGuard(actual,context,replay_chunk_bytes=64000);sid=None
                    try:
                        ready=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0,context=context_text))
                        assert ready['context_applied'];sid=ready['session_id'];attempts=1
                        for offset in range(0,len(pcm),64000):await guard.request('/stream/chunk',sid,pcm[offset:offset+64000])
                        replay=await guard.request('/stream/finish',sid);fallback=guard.fallback_active
                        corrected,changes=verify_entity_audio_changes(text,replay.get('text',''),candidates) if not fallback else (text,[])
                        proposed=revise_entity_audio_payload(row['initial'],row['initial'].get('completed_segments',[]),corrected,changes) if changes else row['initial']
                        proposed=dict(proposed,experimental_changes=changes)
                    finally:
                        if sid is not None:
                            try:await guard.request('/stream/cancel',sid)
                            except Exception:pass
                        guard.close()
                with destination.open('a') as f:f.write(json.dumps(dict(id=row['id'],initial=row['initial'],baseline=baseline,proposed=proposed,
                    verified_backgrounds=descriptions,prompt_terms=terms,new_replay_attempts=attempts,fallback_active=fallback,trace=trace),ensure_ascii=False)+'\n')
                print('BACKGROUND_ASR',row['id'],'checked',attempts,'changed',baseline['text']!=proposed['text'],flush=True)
        await asyncio.gather(*(process(r) for r in read(OUT/'background_asr_inputs.jsonl')))


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    cc=opencc.OpenCC('t2s');itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(s):return normalized(cc.convert(itn.normalize(s).text))
    samples={r['id']:r for r in read(BASE/'dataset/samples.jsonl')}
    for folder in ('holdout','holdout_news','holdout_v2'):samples.update({r['id']:r for r in read(PREVIOUS/folder/'samples.jsonl')})
    results=read(OUT/'background_asr_outputs.jsonl');plan=json.loads((OUT/'background_asr_plan.json').read_text())
    assert len(results)==len({r['id'] for r in results})==plan['recordings']
    summary={arm:Counter() for arm in ('baseline','proposed')};changes=[]
    for row in results:
        s=samples[row['id']];ref=norm(s['reference']);expected=Counter(norm(e['canonical']) for e in s['gold_entities'])
        for arm in summary:
            r=row[arm];value=norm(r['text']);cs=r.get('experimental_changes',r.get('entity_audio_repair',{}).get('changes',[]))
            summary[arm].update(samples=1,hits=sum(min(n,value.count(w)) for w,n in expected.items()),
                expected_mentions=sum(expected.values()),errors=Levenshtein.distance(ref,value),reference_chars=len(ref),
                absent_changes=sum(norm(c['canonical']) not in ref for c in cs))
        if row['baseline']['text']!=row['proposed']['text']:
            changes.append(dict(id=row['id'],reference=s['reference'],before=row['baseline']['text'],after=row['proposed']['text']))
    for value in summary.values():value['CER']=value['errors']/value['reference_chars']
    write(OUT/'background_asr_summary.json',summary);write(OUT/'background_asr_changes.json',changes)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','report']);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='collect':asyncio.run(collect())
    else:report()
