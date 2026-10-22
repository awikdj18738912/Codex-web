"""Execute untouched upstream generation classes and filters on 500 Oral records.

Only the harness, endpoint/output paths, seed-pool rounds and scheduling are
local. Prompts, sample constructors, validators, temperatures, token limits,
scene weights and dedup logic are imported exclusively from the frozen clone.
"""
from __future__ import annotations
import asyncio
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
REPO = OUT/'upstream'
PIPE = REPO/'pipeline'
os.environ.update(VLLM_BASE_URL='http://127.0.0.1:8010/v1', VLLM_MODEL_NAME='refiner-teacher',
                  OPENROUTER_API_KEY='', OPENROUTER_MODEL='')
sys.path.insert(0, str(PIPE))
from configs import settings

for key, name in [('SEEDS_DIR','seeds'),('RAW_DIR','raw'),('PROCESSED_DIR','processed'),('FINAL_DIR','final')]:
    path=OUT/name;path.mkdir(exist_ok=True);setattr(settings,key,str(path))

from src.generators.seed_generator import SeedGenerator
from src.generators.oral_text_generator import OralTextGenerator
from src.generators.clean_text_generator import CleanTextGenerator
from src.generators.asr_simulator import ASRSimulator
from src.pipeline.dedup import deduplicate
from src.services.llm_client import AsyncLLMClient
from src.utils.io_utils import read_jsonl, write_jsonl


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,PIPE/'scripts'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


assemble=module('upstream_assemble','04_assemble.py')
finalize=module('upstream_finalize','05_finalize.py')
CONTEXT=ContextVar('request_context',default={})
CALLS=Counter()
CURRENT='initializing'
START=0


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')


class Client(AsyncLLMClient):
    async def generate(self,**kwargs):
        # Two upstream English seed prompts loop indefinitely under the local
        # server's json_object constrained decoder. Preserve prompts/sampling,
        # but use unconstrained JSON text only for these missing seed pools.
        if (CONTEXT.get().get('stage')=='seed'
                and CONTEXT.get().get('scene') in {'english_customer_service','english_voice_search'}):
            kwargs={**kwargs,'response_format':None}
        row=dict(CONTEXT.get(),request=kwargs,endpoint=str(self.client.base_url),utc=datetime.now(timezone.utc).isoformat());t=time.monotonic()
        try:
            answer=await super().generate(**kwargs);row['response']=answer;return answer
        except Exception as exc:
            row['error']=type(exc).__name__+':'+str(exc);raise
        finally:
            row['seconds']=round(time.monotonic()-t,3);CALLS[row.get('stage','unknown')]+=1
            append(OUT/'api_calls.jsonl',row)


async def tagged(stage,scene,coroutine,**metadata):
    token=CONTEXT.set(dict(stage=stage,scene=scene,**metadata))
    try:return await coroutine
    finally:CONTEXT.reset(token)


def file_counts(name):
    return {scene:len(read_jsonl(OUT/'raw'/scene/name)) for scene in settings.SCENE_DISTRIBUTION
            if (OUT/'raw'/scene/name).exists()}


def progress(**extra):
    dump(OUT/'progress.json',dict(stage=CURRENT,elapsed_seconds=round(time.monotonic()-START,1),
         oral_by_scene=file_counts('oral.jsonl'),clean_by_scene=file_counts('clean.jsonl'),
         asr_by_scene=file_counts('asr_sim.jsonl'),api_calls=dict(CALLS),**extra))


async def heartbeat():
    while True:
        progress();await asyncio.sleep(20)


async def main():
    global CURRENT,START
    START=time.monotonic();random.seed(2026101741)
    if (OUT/'api_calls.jsonl').exists():
        for line in (OUT/'api_calls.jsonl').read_text().splitlines():
            CALLS[json.loads(line).get('stage','unknown')]+=1
    files=sorted(PIPE.rglob('*.py'))+sorted((PIPE/'configs').glob('*.json'))
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    quotas={s:int(500*w) for s,w in settings.SCENE_DISTRIBUTION.items()}
    assert sum(quotas.values())==500
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    original_manifest=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else None
    manifest=dict(upstream='https://github.com/AnXMuy/AgenticASR',commit=commit,
         source_sha256=hashes,original_prompt_files_unchanged=True,target_valid_oral=500,
         quotas=quotas,endpoint='http://127.0.0.1:8010/v1',model='refiner-teacher',
         actual_model='/home/aim0/data/models/ASR/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit',
         teacher_differs_from_paper='Paper uses Gemma-4-31B-IT; this run uses local Qwen AWQ4bit.',
         execution='Original SeedGenerator.generate_scene_seeds (one unmodified round per scene), OralTextGenerator.generate_scene until 500 valid Oral records, CleanTextGenerator.generate_scene including original keyword QC, ASRSimulator.simulate_scene, original assemble, original qc_one, original dedup.',
         harness_changes=['Output directories only','Two identical teachers, HTTP concurrency4 each; same upstream algorithms','HTTP timeout600 seconds instead of120','RNG initialized; original distributions unchanged','Fresh one-round seed pools rather than growing every scene to3000 entries','Original final logic called per item without final-quota capping (all scene counts <=quotas).'],
         no_local_training_filters=True,started_utc=datetime.now(timezone.utc).isoformat())
    if original_manifest:
        manifest['started_utc']=original_manifest['started_utc']
        manifest['restarts']=original_manifest.get('restarts',[])+[dict(utc=datetime.now(timezone.utc).isoformat(),reason='Incomplete seed JSON; resume completed seed pools, retry missing scenes with unmodified prompts.')]
        manifest['filesystem_wiring']=original_manifest.get('filesystem_wiring')
    manifest['seed_api_compatibility']='After original json_object mode exhausted three retries for two English seed scenes, only those seed calls use response_format=None. Same original prompts, temperature, max tokens and parser; all Oral/Clean/ASR/QC settings remain original.'
    manifest['parallel_teachers']=['http://127.0.0.1:8010/v1','http://127.0.0.1:8014/v1']
    dump(OUT/'manifest.json',manifest)
    client=Client(max_concurrent=4,timeout=600)
    second=Client(base_url='http://127.0.0.1:8014/v1',max_concurrent=4,timeout=600)
    scene_clients={scene:[client,second][i%2] for i,scene in enumerate(quotas)}
    # Official keyword mode is rule-based, so this additional client performs
    # no inference; point it at the same service for traceability regardless.
    cleaner=CleanTextGenerator(client);cleaner.qc.llm=client
    monitor=asyncio.create_task(heartbeat())
    try:
        CURRENT='seeds'
        seedgen=SeedGenerator(client)
        async def seeds(scene):
            for attempt in range(1,4):
                try:
                    return await tagged('seed',scene,seedgen.generate_scene_seeds(scene))
                except Exception as exc:
                    append(OUT/'seed_errors.jsonl',dict(scene=scene,attempt=attempt,error=type(exc).__name__+':'+str(exc)))
                    if attempt==3:raise
        await asyncio.gather(*(seeds(scene) for scene in quotas))
        pool_counts={}
        for scene in quotas:
            pool=seedgen.load_seeds(scene)
            pool_counts[scene]=({cat:len(values) for cat,values in pool.items() if isinstance(values,list)}
                                if isinstance(pool,dict) else {'flat_array':len(pool)})
        dump(OUT/'seed_counts.json',pool_counts)
        CURRENT='oral';progress()
        generators={scene:OralTextGenerator(scene_clients[scene],rng=random.Random(2026101741+i)) for i,scene in enumerate(quotas)}
        await asyncio.gather(*(tagged('oral',scene,generators[scene].generate_scene(scene,total,resume=True)) for scene,total in quotas.items()))
        assert sum(file_counts('oral.jsonl').values())==500
        CURRENT='clean';progress()
        cleaners={scene:CleanTextGenerator(scene_clients[scene]) for scene in quotas}
        for scene,gen in cleaners.items():gen.qc.llm=scene_clients[scene]
        await asyncio.gather(*(tagged('clean',scene,cleaners[scene].generate_scene(scene)) for scene in quotas))
        CURRENT='asr';progress()
        simulators={scene:ASRSimulator(scene_clients[scene],mode='llm_sim',rng=random.Random(2026101753+i)) for i,scene in enumerate(quotas)}
        await asyncio.gather(*(tagged('asr',scene,simulators[scene].simulate_scene(scene)) for scene in quotas))
        CURRENT='assembly';progress()
        assembled=[];assembly_stats={}
        for scene in quotas:
            rows=read_jsonl(OUT/'raw'/scene/'asr_sim.jsonl')
            records,stats=assemble._assemble_from_asr(scene,rows);assembled.extend(records);assembly_stats[scene]=stats
            write_jsonl(records,OUT/'processed'/f'{scene}.jsonl')
        dump(OUT/'assembly_stats.json',assembly_stats)
        CURRENT='quality_control';progress(assembled=len(assembled))
        sem=asyncio.Semaphore(8);checked=0;accepted=[];reasons=Counter()

        async def qc(item):
            nonlocal checked
            scene=item['meta']['scene'];cid=f"{scene}_{item.get('id')}"
            record,ok,reason=await tagged('qc',scene,finalize.qc_one(item,scene_clients[scene],sem),case_id=cid)
            entry=dict(case_id=cid,scene=scene,passed=ok,reason=reason,record=record)
            append(OUT/'qc_results.jsonl',entry);checked+=1
            if ok:accepted.append(record)
            else:reasons[str(reason)]+=1
            if checked%25==0:progress(qc_checked=checked,qc_accepted=len(accepted),assembled=len(assembled))

        await asyncio.gather(*(qc(item) for item in assembled))
        order={s:i for i,s in enumerate(quotas)}
        accepted.sort(key=lambda r:(order[r['meta']['scene']],r.get('id',0)))
        CURRENT='dedup';progress(qc_checked=checked,qc_accepted=len(accepted))
        final=deduplicate(accepted,threshold=settings.DEDUP_THRESHOLD)
        write_jsonl(final,OUT/'final'/'train_with_provenance.jsonl')
        write_jsonl([{'input':r['input'],'output':r['output'],'meta':r.get('meta',{})} for r in final],OUT/'final'/'train.jsonl')
        oral_counts=file_counts('oral.jsonl');clean_counts=file_counts('clean.jsonl');asr_counts=file_counts('asr_sim.jsonl')
        attempts=sum(json.loads(p.read_text()).get('attempted_count',0) for p in (OUT/'raw').glob('*/oral.ckpt.json'))
        per_scene={scene:dict(oral=oral_counts.get(scene,0),clean=clean_counts.get(scene,0),asr=asr_counts.get(scene,0),
            assembled=sum(r['meta']['scene']==scene for r in assembled),qc_pass=sum(r['meta']['scene']==scene for r in accepted),
            final=sum(r['meta']['scene']==scene for r in final)) for scene in quotas}
        changed=[name for name,digest in hashes.items() if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest]
        summary=dict(upstream_commit=commit,oral_generation_attempts=attempts,valid_oral=sum(oral_counts.values()),
             after_clean_and_keyword_qc=sum(clean_counts.values()),after_asr=sum(asr_counts.values()),
             after_assembly=len(assembled),qc_checked=checked,qc_pass=len(accepted),qc_rejected=checked-len(accepted),
             near_duplicates_removed=len(accepted)-len(final),final_retained=len(final),
             retained_per_valid_oral=len(final)/500,retained_per_generation_attempt=len(final)/attempts,
             per_scene=per_scene,api_calls=dict(CALLS),source_files_changed=changed,
             upstream_git_status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
             elapsed_seconds=round(time.monotonic()-START,2),rejection_reasons=dict(reasons),
             wall_seconds_including_restarts=round((datetime.now(timezone.utc)-datetime.fromisoformat(manifest['started_utc'])).total_seconds(),2),
             caveats=['Qwen4bit teacher instead of paper Gemma','Original bilingual settings, unlike previous Chinese-focused local runs','Automatic retention measures acceptance, not independently measured label accuracy','Seed pools are one unmodified round per scene, not full3000pool'])
        assert not changed
        dump(OUT/'summary.json',summary);CURRENT='complete';progress(qc_checked=checked,qc_accepted=len(accepted),final_retained=len(final))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:
        monitor.cancel()
        await asyncio.gather(monitor,return_exceptions=True)
        await client.client.close()
        await second.client.close()


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(main())
