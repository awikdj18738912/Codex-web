#!/usr/bin/env python3
"""Exact same-candidate pipeline replay using observed clean/ASR/judge responses."""
from __future__ import annotations
import asyncio, contextvars, hashlib, importlib.util, json, logging, os, random, shutil, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'results/refiner_auto_acceptance_300_20261017'
os.environ.update(VLLM_BASE_URL='http://127.0.0.1:8010/v1',VLLM_MODEL_NAME='refiner-teacher',OPENROUTER_API_KEY='',OPENROUTER_MODEL='',CLEAN_TARGET_MODE='construction')
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
from configs import settings
from src.generators.asr_simulator import ASRSimulator
from src.pipeline.dedup import deduplicate
from src.services.llm_client import AsyncLLMClient
from src.utils import numeric_policy as np
from src.pipeline import training_quality as tq


def module(name,filename):
 spec=importlib.util.spec_from_file_location(name,ROOT/'pipeline/scripts'/filename); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
assemble=module('guard_ablation_assemble','04_assemble.py')
finalize=module('guard_ablation_finalize','05_finalize.py')
exporter=module('guard_ablation_exporter','export_sft.py')

def readjsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def dump(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

# Original job order recovers the exact deterministic ASR partial target seed.
weights=settings.SCENE_DISTRIBUTION
quotas={s:int(1000*w) for s,w in weights.items()}
order=sorted(weights,key=lambda s:1000*weights[s]-quotas[s],reverse=True)
for s in order[:1000-sum(quotas.values())]: quotas[s]+=1
jobs=[(s,i) for s,n in quotas.items() for i in range(1,n+1)]
random.Random(2026101709).shuffle(jobs); jobs=jobs[:300]
serial_by_id={f'scale_{scene}_{n:04d}':serial for serial,(scene,n) in enumerate(jobs)}

old_api=readjsonl(BASE/'api_calls.jsonl')
api_lookup={}
for row in old_api: api_lookup.setdefault((row.get('stage'),row.get('id')),[]).append(row)
CONTEXT=contextvars.ContextVar('guard_replay_context',default={})
PROMPT_VARIANT=contextvars.ContextVar('guard_replay_prompt_variant',default='normal')
LIVE_CACHE={}; LIVE_LOCK=asyncio.Lock(); CALLS=[]
def preload_cached_live_responses():
 sentence='数字格式由外部规则负责。英文名称和型号沿用原文。'
 expected_removed=tq.AUDIT_SYSTEM.replace(sentence,'')
 for path in (OUT/'judge_api_calls.jsonl',OUT/'live_api_calls.jsonl'):
  if not path.exists(): continue
  for row in readjsonl(path):
   req=row.get('request'); answer=row.get('response')
   if not isinstance(req,dict) or not isinstance(answer,str): continue
   system=req.get('system_prompt','')
   if system not in {tq.AUDIT_SYSTEM,expected_removed}: continue
   key=hashlib.sha256(json.dumps(req,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   LIVE_CACHE.setdefault(key,answer)
preload_cached_live_responses()

def canonical(o): return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'))
class ReplayClient(AsyncLLMClient):
 async def generate(self,**kwargs):
  ctx=CONTEXT.get(); stage,case_id=ctx.get('stage'),ctx.get('id'); variant=PROMPT_VARIANT.get()
  key=hashlib.sha256(canonical(kwargs).encode()).hexdigest()
  # Exact request match to the saved response is required; prompt-removed judges
  # intentionally miss the cache and run live once per unique request.
  if variant=='normal':
   for saved in api_lookup.get((stage,case_id),[]):
    if canonical(saved.get('request'))==canonical(kwargs):
     return saved['response']
  async with LIVE_LOCK:
   if key in LIVE_CACHE: return LIVE_CACHE[key]
  started=time.monotonic()
  answer=await super().generate(**kwargs)
  row={'stage':stage,'id':case_id,'prompt_variant':variant,'request':kwargs,'request_hash':key,'response':answer,'seconds':round(time.monotonic()-started,3)}
  CALLS.append(row)
  async with LIVE_LOCK: LIVE_CACHE[key]=answer
  with (OUT/'live_api_calls.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
  return answer

async def call(stage,case_id,coro):
 t=CONTEXT.set({'stage':stage,'id':case_id})
 try:return await coro
 finally:CONTEXT.reset(t)

def all_jobs():
 for s,i in jobs: yield s,i

async def replay_group(label,numeric_on,english_on,*,remove_prompt=False):
 old_num,old_eng=np.lock_numeric_target,np.english_entities_preserved
 if not numeric_on: np.lock_numeric_target=lambda s,t:t
 if not english_on: np.english_entities_preserved=lambda s,t:True
 old_system=tq.AUDIT_SYSTEM
 if remove_prompt:
  sentence='数字格式由外部规则负责。英文名称和型号沿用原文。'
  assert old_system.count(sentence)==1
  tq.AUDIT_SYSTEM=old_system.replace(sentence,'')
  PROMPT_VARIANT.set('prompt_removed')
 else: PROMPT_VARIANT.set('normal')
 attempts=[]; accepted=[]; status=Counter(); t0=time.monotonic()
 baseline={r['id']:r for r in readjsonl(BASE/'attempts.jsonl')}
 client=ReplayClient(max_concurrent=4,timeout=600,max_retries=1)
 sem=asyncio.Semaphore(4)
 queue=asyncio.Queue()
 for scene,n in all_jobs():
  cid=f'scale_{scene}_{n:04d}'; queue.put_nowait((scene,cid,baseline[cid]))
 async def worker():
  while not queue.empty():
   scene,cid,old=await queue.get(); row={'id':cid,'scene':scene}
   try:
    oral=old.get('oral')
    if not oral: row['status']='oral_rejected'; continue
    row['oral']=oral; clean=old.get('clean')
    if not clean: row['status']='clean_rejected'; continue
    row['clean']=clean; selected=bool(old.get('asr_selected')); row['asr_selected']=selected
    if selected:
     asr=old.get('asr')
     if not asr:
      rng_seed=2026101723+serial_by_id[cid]*401
      simulator=ASRSimulator(client,mode='llm_sim',rng=random.Random(rng_seed))
      asr=await call('asr',cid,simulator.simulate_one(scene,clean))
     row['asr']=asr
     if not asr: row['status']='asr_rejected'; continue
     records,stats=assemble._assemble_from_asr(scene,[asr])
    else:
     records,stats=assemble._assemble_from_clean(scene,[clean])
    row['assembly_stats']=stats
    if not records: row['status']='assembly_rejected'; continue
    checked,passed,reason=await call('audit',cid,finalize.qc_one(records[0],client,sem))
    row['audit']=checked.get('meta',{}).get('quality_audit'); row['audit_reason']=reason
    if not passed: row['status']='audit_rejected'; continue
    if exporter.convert_one(checked) is None: row['status']='export_rejected'; continue
    row['status']='accepted'; accepted.append(checked)
   except Exception as exc:
    row.update(status='exception',error=f'{type(exc).__name__}:{exc}')
   finally:
    status[row['status']]+=1; attempts.append(row); queue.task_done()
 await asyncio.gather(*(worker() for _ in range(4)))
 accepted.sort(key=lambda x:x['id'])
 unique=deduplicate(accepted,threshold=settings.DEDUP_THRESHOLD)
 dump(f'{label}_attempts.json',attempts)
 dump(f'{label}_retained_ids.json',[r['id'] for r in unique])
 summary={'group':label,'numeric_guard':numeric_on,'english_guard':english_on,'prompt_sentence_removed':remove_prompt,
  'candidate_denominator':300,'status_counts':dict(status),'accepted_before_dedup':len(accepted),'unique_retained':len(unique),
  'retained_rate':len(unique)/300,'new_live_api_calls':len(CALLS),'unique_live_requests':len(LIVE_CACHE),'elapsed_seconds':round(time.monotonic()-t0,2),
  'live_call_ids':[r['id'] for r in CALLS if r['prompt_variant']==('prompt_removed' if remove_prompt else 'normal')]}
 dump(f'{label}_summary.json',summary)
 await client.client.close()
 tq.AUDIT_SYSTEM=old_system; PROMPT_VARIANT.set('normal'); np.lock_numeric_target=old_num; np.english_entities_preserved=old_eng
 return summary

async def main():
 groups=[]
 for args in [('both_on',1,1,False),('numeric_off',0,1,False),('english_off',1,0,False),('both_off',0,0,False)]:
  groups.append(await replay_group(args[0],args[1],args[2],remove_prompt=args[3]))
 base=readjsonl(BASE/'accepted_candidates.jsonl'); base.sort(key=lambda x:x['id'])
 original_unique=deduplicate(base,threshold=settings.DEDUP_THRESHOLD)
 assert (len(base),len(original_unique))==(257,255),f'baseline mismatch {len(base)}/{len(original_unique)}'
 prompt=await replay_group('prompt_removed_both_off',0,0,remove_prompt=True)
 dump('pipeline_replay_summary.json',{'groups':groups,'prompt_removed_group':prompt,'baseline_observed':{'accepted_before_dedup':257,'unique_retained':255,'retained_rate_in_300':.85},
  'fixed_denominators':{'all_attempts':300,'valid_oral':280},'teacher':'Qwen refiner-teacher 8010','runner':'Luna delegated analysis',
  'method':'Same saved 300 Oral/clean inputs; saved ASR candidates and exact matched audit response reused. Only six previously rejected ASR responses are re-evaluated through actual simulator under monkeypatch; live judge calls cached across groups.',
  'invariants':'normalize_source, _construction_plan_matches exact expected_clean, fixed partial target, valid_partial_pair, assembly, surface_issues, unsupported_edits policy, audit schema, export, and 0.75 dedup remain production code paths. Numeric off patches only lock_numeric_target; English off patches only english_entities_preserved. No restore_numeric_surfaces patch.'})
 print(json.dumps({'groups':groups,'prompt_removed_group':prompt},ensure_ascii=False,indent=2))
if __name__=='__main__':
 logging.basicConfig(level=logging.WARNING); asyncio.run(main())
