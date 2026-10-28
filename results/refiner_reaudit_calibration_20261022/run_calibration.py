import argparse,asyncio,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parent
THINKING=False
MAX_OUTPUT=1100
ENDPOINTS=['http://127.0.0.1:8012/v1/chat/completions','http://127.0.0.1:8013/v1/chat/completions']
class AuditValidationError(ValueError):
 def __init__(self,message,audit):
  super().__init__(message);self.audit=audit
def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def request(row,system,index):
 body={'model':'qwen38-teacher','messages':[{'role':'system','content':system},{'role':'user','content':json.dumps({'input':row['input'],'target':row['target'],**({'source_evidence':row['source_evidence']} if row.get('source_evidence') else {})},ensure_ascii=False)}],'temperature':0,'max_tokens':MAX_OUTPUT,'seed':20261022,'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':THINKING,'preserve_thinking':False}}
 tokenbody={'model':'qwen38-teacher','messages':body['messages'],'add_generation_prompt':True,'chat_template_kwargs':body['chat_template_kwargs']}
 with urlopen(Request(ENDPOINTS[index].replace('/v1/chat/completions','/tokenize'),data=json.dumps(tokenbody).encode(),headers={'Content-Type':'application/json'}),timeout=30) as resp: budget=json.load(resp)
 body['max_tokens']=min(MAX_OUTPUT,budget['max_model_len']-budget['count']-16)
 if body['max_tokens']<128:raise ValueError('insufficient_context_budget')
 with urlopen(Request(ENDPOINTS[index],data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as resp:r=json.load(resp)
 c=r['choices'][0]
 if c.get('finish_reason')!='stop':raise ValueError('incomplete:'+str(c.get('finish_reason')))
 d=json.loads(c['message']['content'])
 if d.get('action') not in ['accept','repair','quarantine'] or d.get('source_status') not in ['valid','uncertain','invalid'] or d.get('fidelity') not in ['pass','fail','uncertain'] or d.get('cleanup') not in ['pass','fail','uncertain'] or not isinstance(d.get('issues'),list):raise AuditValidationError('invalid schema',d)
 if d['action']=='accept' and (d['source_status']!='valid' or d['fidelity']!='pass' or d['cleanup']!='pass'):raise AuditValidationError('inconsistent accept',d)
 if d['action']!='accept' and not d['issues']:raise AuditValidationError('missing evidence',d)
 for issue in d['issues']:
  if not isinstance(issue,dict):raise AuditValidationError('invalid issue',d)
  for key,text in [('input_quote',row['input']),('target_quote',row['target'])]:
   quote=issue.get(key)
   if not isinstance(quote,str) or (quote and quote not in text):raise AuditValidationError('unverifiable quote:'+key,d)
 if 'checks' in system:
  checks=d.get('checks')
  if not isinstance(checks,list) or {x.get('check') for x in checks if isinstance(x,dict)}!={'source','corrections','coverage','literals','cleanup'}: raise AuditValidationError('missing evidence checklist',d)
 return d,r.get('usage')
async def main():
 a=argparse.ArgumentParser();a.add_argument('--sample',type=Path,required=True);a.add_argument('--prompt',type=Path,default=ROOT/'audit_system_v1.txt');a.add_argument('--output',type=Path,required=True);a.add_argument('--thinking',action='store_true');a.add_argument('--max-output',type=int,default=1100);args=a.parse_args()
 global THINKING,MAX_OUTPUT
 THINKING=args.thinking;MAX_OUTPUT=args.max_output
 system=args.prompt.read_text();rows=read(args.sample);done={r['audit_id'] for r in read(args.output)} if args.output.exists() else set()
 provenance={'model':'qwen38-teacher','endpoints':ENDPOINTS,'prompt_sha256':hashlib.sha256(system.encode()).hexdigest(),'sample_sha256':hashlib.sha256(args.sample.read_bytes()).hexdigest(),'temperature':0,'max_tokens':MAX_OUTPUT,'thinking':THINKING,'context_budget':'tokenize count; cap at model limit minus prompt minus16','concurrency_per_gpu':8,'reference_labels_sent':False,'scope':'calibration only; no training data changed'}
 f=args.output.with_suffix('.provenance.json')
 if f.exists() and json.loads(f.read_text())!=provenance:raise ValueError('provenance mismatch')
 f.write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
 sems=[asyncio.Semaphore(8),asyncio.Semaphore(8)];lock=asyncio.Lock();started=time.monotonic()
 async def one(row,index):
  async with sems[index%2]:
   result={'audit_id':row['audit_id'],'error':None}
   try:result['audit'],result['usage']=await asyncio.to_thread(request,row,system,index%2)
   except Exception as exc:
    result['error']=str(exc)[:500]
    if isinstance(exc,AuditValidationError):result['invalid_audit_for_manual_review']=exc.audit
   async with lock:
    with args.output.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    done.add(row['audit_id']);p={'completed':len(done),'target':len(rows),'wall_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    args.output.with_suffix('.progress.json').write_text(json.dumps(p));print(p,flush=True)
 await asyncio.gather(*(one(r,i) for i,r in enumerate(rows) if r['audit_id'] not in done))
if __name__=='__main__':asyncio.run(main())
