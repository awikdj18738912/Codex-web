"""Paired actual long-audio replay performance; first pass is shared.

Only original entity guard versus isolated candidate module. No production
changes and no use of the181 held-out references. Prototype windows are
chosen from prior development evidence, never from target gold at inference.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_local_replay_20261012'
OLD=ROOT/'results/generic_entity_validation_20261010'
sys.path.insert(0,str(ROOT))


def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def load(name,file):
    spec=importlib.util.spec_from_file_location('system.'+name,file)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def prepare():
    if (OUT/'plan.json').exists():raise FileExistsError('Local replay plan already frozen')
    samples=json.loads((OLD/'manifest.json').read_text())['samples']
    initial={r['id']:r for r in read(OLD/'asr_baseline.jsonl')}
    cases=[]
    for s in samples:
        if s.get('duration_seconds',0)<=45:continue
        cases.append(dict(id=s['id'],audio_path=s['audio_path'],pcm_sha256=s['pcm_sha256'],
                          library=str(OLD/'libraries'/s['library_key']/'entities.db'),
                          reference=s['reference'],initial=initial[s['id']],kind='known_real_clip_concatenation'))
    continuous=OLD/'continuous_podcast'
    protocol=json.loads((continuous/'protocol.json').read_text())
    prior=json.loads((continuous/'result.json').read_text())
    cases.append(dict(id='continuous_300',audio_path=str(continuous/'continuous_19m_24m.wav'),
                      pcm_sha256=protocol['pcm_sha256'],library=str(ROOT/'data/entity_benchmarks/oncology_repaired_20261009/entities.db'),
                      reference=None,initial=prior['baseline'],kind='continuous_real_no_gold'))
    inputs=[]
    refs={}
    for row in cases:
        pcm=subprocess.run(['ffmpeg','-v','error','-i',row['audio_path'],'-ar','16000','-ac','1','-f','f32le','pipe:1'],capture_output=True,check=True).stdout
        assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
        path=OUT/(row['id']+'.f32le');path.write_bytes(pcm)
        record=row['initial'];assert len(record['chunk_payloads'])==(len(pcm)+63999)//64000
        inputs.append(dict(id=row['id'],pcm_path=str(path),pcm_sha256=row['pcm_sha256'],
                           library=row['library'],library_sha256=sha(Path(row['library'])),
                           initial={k:record[k] for k in ('ready','chunk_payloads','final_payload')},kind=row['kind']))
        refs[row['id']]=row['reference']
    (OUT/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
    write(OUT/'references.json',refs)
    write(OUT/'plan.json',dict(development_only=True,cases=len(inputs),input_sha256=sha(OUT/'inputs.jsonl'),
          original_sha256=sha(OUT/'original_entity_audio_repair.py'),candidate_sha256=sha(OUT/'candidate_entity_audio_repair.py'),
          settings=dict(min_seconds=45,max_audio_fraction=.6,max_windows=2,audio_margin_seconds=2,strong_candidate_score=.63),
          target_reference_at_inference=False,acceptance='All existing corrections retained, no higher CER on gold cases, '
          'no new absent-canonical replacements, actual repair time improves. Need independent validation before deployment.',
          asr_url='http://127.0.0.1:8766'))
    print('PREPARED',len(inputs),flush=True)


async def collect():
    import httpx
    from system.entity_store import EntityStore
    from validate_entity_confirmation import exercise
    plan=json.loads((OUT/'plan.json').read_text());assert sha(OUT/'inputs.jsonl')==plan['input_sha256']
    assert sha(OUT/'candidate_entity_audio_repair.py')==plan['candidate_sha256']
    classes={'baseline':load('_local_baseline',OUT/'original_entity_audio_repair.py').EntityAudioRepair,
             'candidate':load('_local_candidate',OUT/'candidate_entity_audio_repair.py').EntityAudioRepair}
    destination=OUT/'results.jsonl';done={(r['id'],r['arm']) for r in read(destination)} if destination.exists() else set()
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        for n,row in enumerate(read(OUT/'inputs.jsonl')):
            pcm=Path(row['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
            assert sha(Path(row['library']))==row['library_sha256']
            definitions=EntityStore(row['library']).list_entities()
            assert definitions and any(d.enabled and d.normalization_policy=='normalize' for d in definitions)
            for arm in (('baseline','candidate') if n%2==0 else ('candidate','baseline')):
                if (row['id'],arm) in done:continue
                initial=row['initial'];initial_sid=initial['ready']['session_id'];index=0;trace=[]
                async def request(endpoint,session_id=None,data=b'',params=None):
                    nonlocal index
                    if endpoint=='/stream/start' and not (params or {}).get('context') and index==0:
                        return deepcopy(initial['ready'])
                    if session_id==initial_sid:
                        if endpoint=='/stream/chunk':
                            assert data==pcm[index*64000:(index+1)*64000]
                            result=deepcopy(initial['chunk_payloads'][index]);index+=1;return result
                        if endpoint=='/stream/finish':
                            assert index==len(initial['chunk_payloads']);return deepcopy(initial['final_payload'])
                    query=dict(params or {})
                    if session_id:query['session_id']=session_id
                    started=time.perf_counter()
                    result=(await client.post(plan['asr_url']+endpoint,params=query,content=data)).raise_for_status().json()
                    trace.append(dict(endpoint=endpoint,session_id=session_id,data_sha256=hashlib.sha256(data).hexdigest(),
                                      context=query.get('context',''),result=deepcopy(result),seconds=time.perf_counter()-started))
                    return result
                started=time.perf_counter()
                final=await exercise(classes[arm],request,definitions,pcm)
                record=dict(id=row['id'],arm=arm,final=final,trace=trace,wall_seconds=time.perf_counter()-started)
                with destination.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                print('FINISHED',row['id'],arm,round(record['wall_seconds'],3),final['entity_audio_repair'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect']);parser.add_argument('--sparse',action='store_true');parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    if args.run_dir:OUT=args.run_dir.resolve()
    if args.sparse:OUT=OUT/'sparse_development'
    if args.stage=='prepare':prepare()
    else:asyncio.run(collect())
