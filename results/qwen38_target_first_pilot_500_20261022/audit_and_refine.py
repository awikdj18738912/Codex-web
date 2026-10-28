"""Run frozen Refiner prompt and independent upstream semantic QC on target-first oral pairs."""
import argparse,asyncio,hashlib,importlib.util,json,os,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
os.environ.update({'VLLM_BASE_URL':'http://127.0.0.1:8012/v1','VLLM_MODEL_NAME':'qwen38-teacher','OPENROUTER_API_KEY':'','OPENROUTER_MODEL':'','LLM_MAX_CONCURRENT':'8'})
import sys
sys.path.insert(0,str(PROJECT/'pipeline'))
spec=importlib.util.spec_from_file_location('pilot_finalize',PROJECT/'pipeline/scripts/05_finalize.py')
finalize=importlib.util.module_from_spec(spec);sys.modules[spec.name]=finalize;spec.loader.exec_module(finalize)
from src.services.llm_client import AsyncLLMClient

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
async def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'oral_candidates_v2.jsonl');p.add_argument('--output',type=Path,default=ROOT/'refined_and_audited.jsonl');p.add_argument('--concurrency',type=int,default=6);p.add_argument('--limit',type=int);a=p.parse_args()
 rows=[r for r in read(a.input) if not r.get('error') and r.get('oral_text')]
 if a.limit:rows=rows[:a.limit]
 prompt=json.loads((PROJECT/'pipeline/configs/prompts_clean.json').read_text())
 prov={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'refiner_prompt_sha256':hashlib.sha256((PROJECT/'pipeline/configs/prompts_clean.json').read_bytes()).hexdigest(),'model':'Qwen3.8-27B-AWQ-INT4','concurrency':a.concurrency,'audit':'single-call full source+meaning+cleanup rubric, fail closed'}
 pp=a.output.with_suffix('.provenance.json')
 if pp.exists() and json.loads(pp.read_text())!=prov:raise ValueError('provenance mismatch')
 pp.write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
 done={r['case_id'] for r in read(a.output)} if a.output.exists() else set()
 client=AsyncLLMClient(base_url='http://127.0.0.1:8012/v1',model='qwen38-teacher',max_concurrent=a.concurrency,timeout=600,max_retries=2)
 async def no_think_generate(self,system_prompt,user_prompt,temperature=0.8,max_tokens=4096,response_format=None):
  kwargs={'model':self.model,'messages':[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}],
          'temperature':temperature,'max_tokens':max_tokens,
          'extra_body':{'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}}}
  if response_format=='json_object':kwargs['response_format']={'type':'json_object'}
  async with self.semaphore:
   last=None
   for attempt in range(3):
    try:
     response=await self.client.chat.completions.create(**kwargs)
     choice=response.choices[0]
     content=choice.message.content
     if not isinstance(content,str) or not content.strip() or choice.finish_reason=='length':
      raise ValueError('empty_or_truncated_response:'+str(choice.finish_reason))
     return content
    except Exception as exc:
     last=exc
     if attempt<2:await asyncio.sleep(2**attempt)
   raise last
 client.generate=types.MethodType(no_think_generate,client)
 sem=asyncio.Semaphore(a.concurrency);lock=asyncio.Lock();started=time.monotonic()
 async def one(r):
  oral=r['oral_text'];gold=r['clean_target'];gold_ok=False;gold_reason='';gold_audit={};candidate=None;candidate_error=None;candidate_ok=False;candidate_reason='';candidate_audit={}
  try:gold_ok,gold_reason,gold_audit=await finalize.audit_pair(oral,gold,client,recognition_repair=True)
  except Exception as e:gold_reason=type(e).__name__+':'+str(e)[:300]
  try:
   raw=await client.generate(system_prompt=prompt['system_prompts']['zh'],user_prompt=prompt['user_template']['zh'].format(oral_text=oral),temperature=0.0,max_tokens=768,response_format='json_object')
   parsed=json.loads(raw);candidate=parsed.get('clean_text')
   if not isinstance(candidate,str) or not candidate.strip():raise ValueError('empty clean_text')
  except Exception as e:candidate_error=type(e).__name__+':'+str(e)[:300]
  if candidate:
   try:candidate_ok,candidate_reason,candidate_audit=await finalize.audit_pair(oral,candidate,client,recognition_repair=True)
   except Exception as e:candidate_reason=type(e).__name__+':'+str(e)[:300]
  out={**r,'gold_audit_pass':gold_ok,'gold_audit_reason':gold_reason,'gold_audit':gold_audit,'refiner_output':candidate,'refiner_error':candidate_error,'refiner_audit_pass':candidate_ok,'refiner_audit_reason':candidate_reason,'refiner_audit':candidate_audit,'refiner_exact_matches_gold':candidate==gold}
  async with lock:
   with a.output.open('a') as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
   done.add(r['case_id']);allrows=read(a.output)
   prog={'completed':len(done),'target':len(rows),'gold_pass':sum(bool(x.get('gold_audit_pass')) for x in allrows),'refiner_pass':sum(bool(x.get('refiner_audit_pass')) for x in allrows),'refiner_errors':sum(bool(x.get('refiner_error')) for x in allrows),'wall_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
   a.output.with_suffix('.progress.json').write_text(json.dumps(prog,ensure_ascii=False,indent=2)+'\n');print(json.dumps(prog),flush=True)
 try:await asyncio.gather(*(one(r) for r in rows if r['case_id'] not in done))
 finally:await client.client.close()
if __name__=='__main__':asyncio.run(main())
