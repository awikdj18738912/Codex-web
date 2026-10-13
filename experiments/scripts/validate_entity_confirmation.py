"""Paired raw-ASR evaluation with only the entity confirmation module changed."""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_confirmation_20261011'
RUN=OUT
HOLDOUT_FOLDERS=('holdout','holdout_news')
BASE=ROOT/'results/large_entity_benchmark_v1_20261010'
sys.path.insert(0,str(ROOT))


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def classes():
    from system.entity_audio_repair import EntityAudioRepair
    name='system._confirmation_baseline'
    spec=importlib.util.spec_from_file_location(name,OUT/'original_entity_audio_repair.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.EntityAudioRepair,EntityAudioRepair


def check():
    plan=json.loads((RUN/'validation_plan.json').read_text())
    assert digest(ROOT/'system/entity_audio_repair.py')==plan['entity_module_sha256']
    assert digest(BASE/'dataset/library/entities.db')==plan['library_sha256']
    for folder,h in plan['holdout_manifest_sha256'].items():assert digest(OUT/folder/'manifest.json')==h
    return plan


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    path=RUN/'validation_plan.json'
    if path.exists():raise FileExistsError('Validation plan already frozen')
    old=[json.loads(s) for s in (BASE/'dataset/samples.jsonl').read_text().splitlines()]
    if RUN != OUT:
        old += [dict(r,source_audio_sha256=r['audio_sha256']) for folder in ('holdout','holdout_news') for r in read(OUT/folder/'samples.jsonl')]
    prior_audio={r['source_audio_sha256'] for r in old};prior_pcm={r['pcm_sha256'] for r in old}
    samples=[];hashes={}
    for folder in HOLDOUT_FOLDERS:
        assert (OUT/folder/'READY').exists()
        m=json.loads((OUT/folder/'manifest.json').read_text())
        assert digest(OUT/folder/'samples.jsonl')==m['samples_sha256']
        hashes[folder]=digest(OUT/folder/'manifest.json')
        for row in read(OUT/folder/'samples.jsonl'):
            assert row['audio_sha256'] not in prior_audio and row['pcm_sha256'] not in prior_pcm
            assert digest(Path(row['audio_path']))==row['audio_sha256']
            assert digest(Path(row['pcm_path']))==row['pcm_sha256']
            samples.append({k:row[k] for k in ('id','audio_path','pcm_path','pcm_sha256','audio_sha256','topics','cohort','duration_seconds')})
    assert len(samples)==len({r['audio_sha256'] for r in samples})==len({r['pcm_sha256'] for r in samples})==200
    write(path,dict(reference_at_inference=False,samples=samples,
        module_change=('Short source fragment cannot expand to a longer registered surface without a shared character'
                       if RUN != OUT else 'Ambiguous two-character spelling without shared character: narrower glossary must confirm same anchored span'),
        rule_chosen_from=('Original 2410 and first 200 diagnostic recordings; second 200 frozen before ASR'
                          if RUN != OUT else 'Previous 2410 development recordings, 18 failed lexical guards and 94 real control ASR requests'),
        entity_module_sha256=digest(ROOT/'system/entity_audio_repair.py'),
        baseline_module_sha256=digest(OUT/'original_entity_audio_repair.py'),
        library_sha256=digest(BASE/'dataset/library/entities.db'),holdout_manifest_sha256=hashes))
    print('PREPARED 200 unseen recordings, hashes verified',flush=True)


async def exercise(guard_class,request,definitions,pcm):
    guard=guard_class(request,definitions)
    try:
        ready=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0))
        for offset in range(0,len(pcm),64000):await guard.request('/stream/chunk',ready['session_id'],pcm[offset:offset+64000])
        return await guard.request('/stream/finish',ready['session_id'])
    finally:guard.close()


def replay_request(trace,extra_trace=None,real_request=None):
    remaining=list(deepcopy(trace));extra=list(deepcopy(extra_trace or []))
    async def request(endpoint,session_id=None,data=b'',params=None):
        signature=(endpoint,session_id,hashlib.sha256(data).hexdigest(),(params or {}).get('context',''))
        for pool in (remaining,extra):
            for i,row in enumerate(pool):
                if signature==(row['endpoint'],row['session_id'],row['data_sha256'],row['context']):
                    return deepcopy(pool.pop(i)['result'])
        if real_request is not None:return await real_request(endpoint,session_id,data,params)
        raise ValueError(f'Unexpected request {endpoint} {session_id}')
    return request,remaining,extra


async def development():
    from system.entity_store import EntityStore
    original,new=classes();definitions=EntityStore(BASE/'dataset/library/entities.db').list_entities(domain='general')
    controls={(r['id'],r['scenario']):r for r in read(OUT/'stability_audio.jsonl')}
    pcm={r['id']:r['pcm_path'] for r in read(BASE/'dataset/samples.jsonl')}
    destination=RUN/'development_production_guard.jsonl'
    done={r['id'] for r in read(destination)} if destination.exists() else set()
    for row in read(BASE/'full_evaluation/audio.jsonl'):
        if not row['production_final']['entity_audio_repair']['changes'] or row['id'] in done:continue
        wave=Path(pcm[row['id']]).read_bytes()
        old_request,remaining,_=replay_request(row['trace'])
        baseline=await exercise(original,old_request,definitions,wave);assert not remaining
        assert baseline['text']==row['production_final']['text']
        extra=controls.get((row['id'],'local_competitors'),{}).get('trace',[])
        new_request,remaining,_=replay_request(row['trace'],extra)
        proposed=await exercise(new,new_request,definitions,wave);assert not remaining
        with destination.open('a') as f:f.write(json.dumps(dict(id=row['id'],baseline=baseline,proposed=proposed),ensure_ascii=False)+'\n')
        print('DEV',row['id'],baseline['text']==proposed['text'],flush=True)


async def collect():
    import httpx
    from system.entity_store import EntityStore
    plan=check();original,new=classes();definitions=EntityStore(BASE/'dataset/library/entities.db').list_entities(domain='general')
    destination=RUN/'holdout_audio.jsonl';done={r['id'] for r in read(destination)} if destination.exists() else set()
    semaphore=asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async def process(row):
            if row['id'] in done:return
            async with semaphore:
                wave=Path(row['pcm_path']).read_bytes();assert hashlib.sha256(wave).hexdigest()==row['pcm_sha256']
                trace=[];additional=[];initial=None
                async def actual(endpoint,session_id=None,data=b'',params=None,*,log):
                    nonlocal initial
                    query=dict(params or {})
                    if session_id:query['session_id']=session_id
                    result=(await client.post('http://127.0.0.1:8766'+endpoint,params=query,content=data)).raise_for_status().json()
                    log.append(dict(endpoint=endpoint,session_id=session_id,context=query.get('context',''),
                                    data_sha256=hashlib.sha256(data).hexdigest(),result=deepcopy(result)))
                    if endpoint=='/stream/finish' and log is trace and initial is None:initial=deepcopy(result)
                    return result
                async def initial_request(endpoint,session_id=None,data=b'',params=None):
                    return await actual(endpoint,session_id,data,params,log=trace)
                async def additional_request(endpoint,session_id=None,data=b'',params=None):
                    return await actual(endpoint,session_id,data,params,log=additional)
                baseline=await exercise(original,initial_request,definitions,wave)
                request,remaining,_=replay_request(trace,real_request=additional_request)
                proposed=await exercise(new,request,definitions,wave);assert not remaining
                with destination.open('a') as f:f.write(json.dumps(dict(id=row['id'],initial=initial,baseline=baseline,proposed=proposed,
                    baseline_trace=trace,additional_trace=additional,pcm_sha256=row['pcm_sha256']),ensure_ascii=False)+'\n')
                print('HOLDOUT',row['id'],baseline['text']==proposed['text'],'extra',len(additional),flush=True)
        await asyncio.gather(*(process(row) for row in plan['samples']))
    check()


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so');cc=opencc.OpenCC('t2s')
    def norm(s):return normalized(cc.convert(itn.normalize(s).text))
    cohort={r['id']:r for folder in HOLDOUT_FOLDERS for r in read(OUT/folder/'samples.jsonl')}
    records=read(RUN/'holdout_audio.jsonl');assert len(records)==len({r['id'] for r in records})==len(cohort)==200
    counts={arm:Counter() for arm in ('initial','baseline','proposed')};changes=[]
    for row in records:
        gold=cohort[row['id']];ref=norm(gold['reference']);expected=Counter(norm(e['canonical']) for e in gold['gold_entities'])
        for arm in counts:
            value=row[arm];text=norm(value.get('text',''));status=value.get('entity_audio_repair',{})
            counts[arm].update(samples=1,hits=sum(min(n,text.count(w)) for w,n in expected.items()),expected_mentions=sum(expected.values()),
                errors=Levenshtein.distance(ref,text),reference_chars=len(ref),replays=status.get('checks',0),
                short_confirmation_checks=status.get('short_confirmation_checks',0),
                absent_changes=sum(norm(c['canonical']) not in ref for c in status.get('changes',[])))
        if row['baseline']['text']!=row['proposed']['text']:
            changes.append(dict(id=row['id'],reference=gold['reference'],before=row['baseline']['text'],after=row['proposed']['text']))
    for value in counts.values():
        value['CER']=value['errors']/value['reference_chars'];value['entity_recall']=value['hits']/value['expected_mentions']
    write(RUN/'holdout_summary.json',counts);write(RUN/'holdout_changes.json',changes)
    print(json.dumps(counts,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','development','collect','report'])
    parser.add_argument('--short-expansion',action='store_true');a=parser.parse_args()
    if a.short_expansion:RUN=OUT/'short_expansion';HOLDOUT_FOLDERS=('holdout_v2',)
    if a.stage=='prepare':prepare()
    elif a.stage=='report':report()
    elif a.stage=='development':asyncio.run(development())
    else:asyncio.run(collect())
