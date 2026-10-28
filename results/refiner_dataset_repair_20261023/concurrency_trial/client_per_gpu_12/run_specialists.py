import argparse,asyncio,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
P=Path(__file__).resolve().parent
ENDPOINTS=['http://127.0.0.1:8012','http://127.0.0.1:8013']
class InvalidAudit(ValueError):
 def __init__(self,message,payload):super().__init__(message);self.payload=payload

def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sync_call(row,stage,prompt,index,notes=None):
 payload={'input':row['input']}
 if stage!='source':payload['target']=row['target']
 if notes is not None:payload['audit_hints_to_verify']=notes
 body={'model':'qwen38-teacher','messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],'temperature':0,'seed':20261023,'max_tokens':900 if stage=='repair' else 384,'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}}
 host=ENDPOINTS[index]
 tokenbody={k:body[k] for k in ['model','messages','chat_template_kwargs']};tokenbody['add_generation_prompt']=True
 def post(path,data,timeout):
  with urlopen(Request(host+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=timeout) as f:return json.load(f)
 budget=post('/tokenize',tokenbody,30);body['max_tokens']=min(body['max_tokens'],budget['max_model_len']-budget['count']-16)
 if body['max_tokens']<128:raise ValueError('insufficient_context_budget')
 raw=post('/v1/chat/completions',body,240);c=raw['choices'][0]
 if c['finish_reason']!='stop':raise ValueError('incomplete:'+str(c['finish_reason']))
 d=json.loads(c['message']['content'])
 if stage=='repair':
  if type(d.get('repairable'))is not bool or not isinstance(d.get('repaired_target'),str):raise InvalidAudit('invalid repair schema',d)
  if d['repairable'] and not d['repaired_target'].strip():raise InvalidAudit('empty repaired target',d)
 else:
  if d.get('status')not in ['pass','fail','uncertain'] or not isinstance(d.get('reason'),str):raise InvalidAudit('invalid audit schema',d)
  for key,text in [('input_quote',row['input']),('target_quote',row.get('target','') if stage!='source' else '')]:
   q=d.get(key)
   if not isinstance(q,str) or (q and q not in text):raise InvalidAudit('unverifiable '+key,d)
  if d['status']!='pass':
   if stage=='source' and not d['input_quote']:raise InvalidAudit('missing source evidence',d)
   if stage=='cleanup' and not d['target_quote']:raise InvalidAudit('missing cleanup evidence',d)
   if stage=='fidelity' and not(d['input_quote']or d['target_quote']):raise InvalidAudit('missing fidelity evidence',d)
 return {'result':d,'usage':raw.get('usage'),'endpoint':host}

async def main():
 a=argparse.ArgumentParser();a.add_argument('--sample',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--repair',action='store_true');args=a.parse_args()
 prompts=json.loads((P/'prompts.json').read_text());rows=read(args.sample);previous=read(args.output) if args.output.exists()else [];done={r['audit_id']for r in previous}
 provenance={'model':'qwen38-teacher','prompt_sha256':hashlib.sha256((P/'prompts.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sample_sha256':hashlib.sha256(args.sample.read_bytes()).hexdigest(),'endpoints':ENDPOINTS,'per_gpu_concurrency':12,'temperature':0,'seed':20261023,'repair_enabled':args.repair,'scope':'diagnostic specialist pilot; separate calls use the same teacher, not independent models; no formal training export modified'}
 f=args.output.with_suffix('.provenance.json')
 if f.exists()and json.loads(f.read_text())!=provenance:raise ValueError('resume provenance mismatch')
 f.write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
 sems=[asyncio.Semaphore(12),asyncio.Semaphore(12)];lock=asyncio.Lock();started=time.monotonic();results=list(previous)
 async def call(row,stage,index,notes=None):
  async with sems[index%2]:
   try:return await asyncio.to_thread(sync_call,row,stage,prompts[stage],index%2,notes)
   except Exception as exc:
    out={'error':str(exc)[:400]}
    if isinstance(exc,InvalidAudit):out['invalid_payload_for_review']=exc.payload
    return out
 async def one(row,index):
  stages=['source','fidelity','cleanup'];values=await asyncio.gather(*(call(row,s,index*3+j)for j,s in enumerate(stages)));checks=dict(zip(stages,values))
  status=lambda s:checks[s].get('result',{}).get('status')
  if status('source')in ['fail','uncertain']:decision='quarantine'
  elif any(x.get('error')for x in checks.values())or any(status(s)=='uncertain'for s in stages):decision='needs_review'
  elif all(status(s)=='pass'for s in stages):decision='accept_candidate'
  else:decision='repair_candidate'
  if row.get('force_quarantine'):decision='quarantine'
  elif row.get('known_issue') and status('source')=='pass' and not any(v.get('error')for v in checks.values()):decision='repair_candidate'
  out={**row,'checks':checks,'decision':decision,'formal_training_eligible':False}
  if args.repair and decision=='repair_candidate':
   fix=await call(row,'repair',index,{'checks':checks,'prior_confirmed_issue':row.get('known_issue')});out['repair']=fix
   if fix.get('result',{}).get('repairable')is True:
    newrow={**row,'target':fix['result']['repaired_target']}
    values=await asyncio.gather(*(call(newrow,s,index*2+j+1)for j,s in enumerate(['fidelity','cleanup'])));out['repair_rechecks']=dict(zip(['fidelity','cleanup'],values))
    out['decision']='repaired_candidate'if all(v.get('result',{}).get('status')=='pass'for v in values)else'repair_needs_review'
   else:out['decision']='repair_needs_review'
  async with lock:
   with args.output.open('a')as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
   done.add(row['audit_id']);results.append(out)
   from collections import Counter
   progress={'completed':len(done),'target':len(rows),'decisions':dict(Counter(r['decision']for r in results)),'api_or_evidence_error_rows':sum(any(v.get('error')for v in r['checks'].values())for r in results),'wall_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
   args.output.with_suffix('.progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2));print(json.dumps(progress,ensure_ascii=False),flush=True)
 await asyncio.gather(*(one(r,i)for i,r in enumerate(rows)if r['audit_id']not in done))
 args.output.with_suffix('.complete.json').write_text(json.dumps({'completed':len(done),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}))
if __name__=='__main__':asyncio.run(main())
