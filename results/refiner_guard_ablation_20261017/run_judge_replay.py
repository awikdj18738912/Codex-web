#!/usr/bin/env python3
"""Cached-response 300-case guard replay plus prompt sentence removal audit."""
from __future__ import annotations
import concurrent.futures, hashlib, json, os, re, threading, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
import sys
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
os.environ.update(VLLM_BASE_URL='http://127.0.0.1:8010/v1',VLLM_MODEL_NAME='refiner-teacher')
from src.utils import numeric_policy as np
from src.pipeline.training_quality import AUDIT_SYSTEM, surface_issues, unsupported_edits, validate_audit
from configs.scenes import get_enabled_scenes
from configs.settings import DEDUP_THRESHOLD

def readjsonl(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def compact(t): return re.sub(r"[\s，,。.!！?？:：;；、‘’“”\"'…—]+", "", t)
def natural_prefixes(t,minimum=4):
    out=[]
    for m in re.finditer(r"[,，;；。！？!?]",t):
        i,end=m.start(),m.end()
        if end<minimum or end>=len(t): continue
        if i and t[i-1].isdigit() and t[end].isdigit(): continue
        out.append(t[:end])
    return out
def valid_partial_pair(full_input,full_target,part_input,part_target):
    source,target,inp,out=map(compact,(full_input,full_target,part_input,part_target))
    return bool(len(inp)>=4 and 4<=len(out)<len(target) and len(inp)<len(source) and target.startswith(out)
                and out in {compact(x) for x in natural_prefixes(full_target,4)})

def api_call(kwargs):
    response_format=kwargs.get('response_format',{'type':'json_object'})
    if response_format=='json_object': response_format={'type':'json_object'}
    data=json.dumps({'model':'refiner-teacher','messages':[{'role':'system','content':kwargs['system_prompt']},{'role':'user','content':kwargs['user_prompt']}],
        'temperature':kwargs.get('temperature',0.0),'max_tokens':kwargs.get('max_tokens',1536),
        'response_format':response_format},ensure_ascii=False).encode()
    req=urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response: body=json.loads(response.read())
    return body['choices'][0]['message']['content']

def prompt_key(kwargs): return hashlib.sha256(json.dumps(kwargs,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

# Preserve every cached call verbatim as the regular-prompt evidence.
base_attempts=readjsonl(ROOT/'results/refiner_auto_acceptance_300_20261017/attempts.jsonl')
base_by={r['id']:r for r in base_attempts}
apirows=readjsonl(ROOT/'results/refiner_auto_acceptance_300_20261017/api_calls.jsonl')
auditrows=[r for r in apirows if r.get('stage')=='audit']
assert len(auditrows)==268
assert all(r['request']['system_prompt']==AUDIT_SYSTEM for r in auditrows), 'saved judge prompt differs from current AUDIT_SYSTEM'

# Old statuses/audits remain observed. Construction proof bypasses lexical guards;
# only six saved ASR partial pairs can change admission.
asr_calls={r['id']:r for r in apirows if r.get('stage')=='asr'}
new=[]
for r in base_attempts:
    if r.get('status')!='asr_rejected': continue
    call=asr_calls[r['id']]
    pair=json.loads(call['response'])
    source=r['oral']['oral_text']; target=r['clean']['clean_text']; expected=r['clean']['meta']['generation_plan']['expected_clean']
    assert pair.get('target') and valid_partial_pair(source,expected,pair.get('input',''),pair['target'])
    new.append({'id':r['id'],'scene':r['scene'],'source':source,'expected':expected,'input':pair['input'],'target':pair['target'],
                'plan':r['clean']['meta']['generation_plan'],'numeric_fail':np.lock_numeric_target(pair['input'],pair['target']) is None,
                'english_fail':not np.english_entities_preserved(pair['input'],pair['target']),
                'surface_issues':surface_issues(pair['input'],pair['target']),
                'unsupported_edits':unsupported_edits(pair['input'],pair['target'])})
assert len(new)==6

# Exact four-way runtime monkeypatch test and set membership for the six candidates.
orig_num=np.lock_numeric_target; orig_eng=np.english_entities_preserved
admitted={}
for label,numeric_on,english_on in [('both_on',1,1),('numeric_off',0,1),('english_off',1,0),('both_off',0,0)]:
    np.lock_numeric_target=orig_num if numeric_on else (lambda s,t:t)
    np.english_entities_preserved=orig_eng if english_on else (lambda s,t:True)
    admitted[label]=[]
    for item in new:
        # Existing partial-pair reference/alignment, length, and normalized prepare gate.
        if not valid_partial_pair(item['source'],item['expected'],item['input'],item['target']): continue
        if len(item['input'].strip())<2 or len(item['target'].strip())<2: continue
        if np.prepare_training_pair(item['input'],item['target'],None) is None: continue
        admitted[label].append(item['id'])
np.lock_numeric_target=orig_num; np.english_entities_preserved=orig_eng

# Exactly one normal-prompt audit per unique new candidate; response shared by all groups.
normal_calls=[]
normal_system=AUDIT_SYSTEM
for item in new:
    payload={'input':item['input'],'target':item['target']}
    kwargs={'system_prompt':normal_system,'user_prompt':json.dumps(payload,ensure_ascii=False),'temperature':0.0,'max_tokens':1536,'response_format':'json_object'}
    normal_calls.append((prompt_key(kwargs),item['id'],kwargs))

# The prompt ablation re-audits all previously eligible actual source/target calls,
# changing only the one numeric/English-preservation sentence, plus the six newly eligible ASR fragments.
sentence='数字格式由外部规则负责。英文名称和型号沿用原文。'
assert AUDIT_SYSTEM.count(sentence)==1
prompt_jobs=[]
for r in auditrows:
    kwargs=dict(r['request'])
    kwargs['system_prompt']=kwargs['system_prompt'].replace(sentence,'')
    assert kwargs['system_prompt']==AUDIT_SYSTEM.replace(sentence,'')
    prompt_jobs.append((prompt_key(kwargs),r['id'],kwargs))
for _,case_id,kwargs in normal_calls:
    k=dict(kwargs); k['system_prompt']=k['system_prompt'].replace(sentence,'')
    prompt_jobs.append((prompt_key(k),case_id,k))
assert len(prompt_jobs)==274

# Run all requests with four workers; append every raw request, response/error and request hash.
log=OUT/'judge_api_calls.jsonl'
cache={}
if log.exists():
    for r in readjsonl(log):
        if r.get('response') is not None: cache[r['request_hash']]=r['response']
lock=threading.Lock(); completed=0

def execute(job):
    global completed
    key,case_id,kwargs=job
    if key in cache: return key,case_id,kwargs,cache[key],True
    started=time.time(); row={'stage':'prompt_removed_audit','id':case_id,'request_hash':key,'request':kwargs}
    try:
        answer=api_call(kwargs); row.update(response=answer,seconds=round(time.time()-started,3))
    except Exception as exc:
        row.update(error=f'{type(exc).__name__}:{exc}',seconds=round(time.time()-started,3)); answer=None
    with lock:
        with log.open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        if answer is not None: cache[key]=answer
        completed+=1
        if completed%20==0:
            dump('judge_progress.json',{'completed_this_run':completed,'total_jobs':len(prompt_jobs),'successful_cached_or_live':len(cache),'elapsed_seconds':round(time.time()-t0,1)})
    return key,case_id,kwargs,answer,False

t0=time.time()
all_jobs=list({j[0]:j for j in prompt_jobs+normal_calls}.values())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(execute,all_jobs))
response_by_hash={k:v for k,_,_,v,_ in results if v is not None}
assert len(response_by_hash)==len(all_jobs),f'missing responses {len(all_jobs)-len(response_by_hash)}'

# Parse typed verdicts identically to the current pipeline.
def verdict(text,source,target):
    try: value=json.loads(text)
    except Exception: return False,'audit_not_json',{}
    return validate_audit(value,source,target)[0],validate_audit(value,source,target)[1],value
normal_verdict={}
for key,case_id,kwargs in normal_calls:
    item=next(i for i in new if i['id']==case_id)
    normal_verdict[case_id]=verdict(response_by_hash[key],item['input'],item['target'])

# Preserve the actual already-retained accepted candidates for matching retained-rate baseline.
accepted=readjsonl(ROOT/'results/refiner_auto_acceptance_300_20261017/accepted_candidates.jsonl')
assert len(accepted)==257

def is_exportable_fragment(item):
    meta=item['meta']; ref=meta.get('fragment_reference_clean') or (meta.get('generation_plan') or {}).get('expected_clean')
    if not ref or not valid_partial_pair(item['oral_text'],ref,item['input'],item['output']['refined_text']): return False
    if (meta.get('scene') or item.get('scene')) not in get_enabled_scenes(): return False
    if meta.get('language')=='en': return False
    return np.prepare_training_pair(item['input'],item['output']['refined_text'],None) is not None

def make_new(item):
    return {'id':item['id'],'input':item['input'],'oral_text':item['source'],'output':{'refined_text':item['target'],'keyword_list':[]},
            'meta':{'scene':item['scene'],'language':'zh','is_fragment':True,'fragment_reference_clean':item['expected'],
                    'generation_plan':item['plan'],'quality_audit':normal_verdict[item['id']][2]}}

# Compute final retained count per four groups with normal prompt judge responses.
results_groups={}
for label,ids in admitted.items():
    extras=[]
    for item in new:
        if item['id'] not in ids: continue
        ok,reason,_=normal_verdict[item['id']]
        record=make_new(item)
        if ok and is_exportable_fragment(record): extras.append(record)
    # Production duplicate threshold and fragment-aware matching.
    sys.path.insert(0,str(ROOT/'pipeline'))
    from src.pipeline.dedup import deduplicate
    unique=deduplicate(accepted+extras,threshold=DEDUP_THRESHOLD)
    results_groups[label]={'known_original_accepted_before_dedup':257,'known_original_unique':255,
       'candidate_ids_after_guards':ids,'new_semantic_pass_ids':[i['id'] for i in extras],
       'new_unique_retained':max(0,len(unique)-255),'accepted_before_dedup':257+len(extras),
       'unique_retained':len(unique),'denominator_candidates':300,'retained_rate':len(unique)/300,
       'judge_rejections':{i['id']:normal_verdict[i['id']][1] for i in new if i['id'] in ids and not normal_verdict[i['id']][0]}}

# Prompt-removal group re-audits the 268 existing audit-stage pairs and all six new partial candidates.
prompt_verdicts=[]
for key,case_id,kwargs in prompt_jobs:
    response=response_by_hash[key]
    try: payload=json.loads(kwargs['user_prompt']); source,target=payload['input'],payload['target']
    except Exception: source=target=''
    ok,reason,value=verdict(response,source,target)
    prompt_verdicts.append({'id':case_id,'request_hash':key,'passed':ok,'reason':reason,'source':source,'target':target,'audit':value})
# Record overall existing prompts population. Existing items have saved response target/input in exact audit request.
prompt_existing=[]; prompt_new=[]
for v in prompt_verdicts:
    (prompt_new if v['id'] in {i['id'] for i in new} else prompt_existing).append(v)
# Save every prompt response verdict; final unique count for known 268 is exact from newly judged responses,
# but cannot reconstruct original output record for all stages from request alone here.
dump('regular_groups.json',{'groups':results_groups,'new_asr_candidate_diagnostics':new,'normal_prompt_judges':{k:{'passed':v[0],'reason':v[1],'audit':v[2]} for k,v in normal_verdict.items()},
     'api_requests_made':sum(not reused for _,_,_,_,reused in results),'api_response_cache_size':len(response_by_hash),
     'judge_call_policy':'One normal-system audit per unique ASR candidate; shared across four ablations.'})
dump('prompt_removed_verdicts.json',{'sentence_removed':sentence,'old_audit_pairs_rejudged':len(prompt_existing),'new_asr_candidates_rejudged':len(prompt_new),
     'total_prompt_removed_audits':len(prompt_verdicts),'passed':sum(v['passed'] for v in prompt_verdicts),'rejected':sum(not v['passed'] for v in prompt_verdicts),
     'results':prompt_verdicts,'limitation':'Prompt-removal acceptance requires exact target/output reconstruction and dedup against all formerly accepted candidates; see id+input+target fields. No Clean regeneration performed.'})
summary={'teacher':'Qwen refiner-teacher @ 127.0.0.1:8010/v1','experimenter':'Luna','fixed_candidate_denominator':300,'full_clean_construction_pairs':280,
 'ordinary_prompt_guard_groups':results_groups,'saved_ASR_rejected_pairs':6,'prompt_removed_audit_count':len(prompt_verdicts),
 'prompt_removed_audit_pass':sum(v['passed'] for v in prompt_verdicts),'prompt_removed_audit_reject':sum(not v['passed'] for v in prompt_verdicts),
 'new_live_api_calls':sum(not reused for _,_,_,_,reused in results),'elapsed_seconds':round(time.time()-t0,2),
 'strict_limits':'No oral/clean regeneration. Numeric normalization retained. Construction expected_clean proof, valid_partial_pair, surface, unsupported-edit policy, semantic booleans, export, dedup threshold preserved.'}
dump('judge_replay_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
