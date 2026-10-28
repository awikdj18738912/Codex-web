"""Generate current-prompt outputs and jointly review oral, gold, and candidate pair."""
import argparse,asyncio,hashlib,json,time
from pathlib import Path
from openai import AsyncOpenAI
ROOT=Path(__file__).resolve().parent
REVIEW_SYSTEM='''你审核中文口语ASR精修训练样本。输入是口语转写，金标是准备训练的精修文本，模型输出是当前生产精修提示词的结果。请分别基于完整口语源判断金标和模型输出；不要默认金标正确，也不要因为更短或更书面就判好。\nsource_valid：源文本像真实单一场景口语，最终意图可以可靠恢复；改口中的旧候选清楚撤回；没有编辑者注释、胡乱拼接或未解歧义。\ngold_pass/refiner_pass：分别保留口语源中的最终有效事实、实体、动作、关系、条件、理由、数字、英文名、否定、问句和不确定程度；清除无意义口癖、重复、已撤回候选和纯辨名提示。目标若新增事实、漏掉最终信息、把提问变断言或把推测变确定，判false。自然短句和口语多样性本身不扣分。原文有真实未解决矛盾则source_valid=false。\n只输出严格JSON，字段source_valid、gold_pass、refiner_pass为布尔值，source_reason、gold_reason、refiner_reason为简短中文。'''
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
async def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'audit_sample_60.jsonl');p.add_argument('--output',type=Path,default=ROOT/'joint_review_60.jsonl');p.add_argument('--concurrency',type=int,default=8);a=p.parse_args()
 rows=read(a.input);prompts=json.loads((ROOT.parents[1]/'pipeline/configs/prompts_clean.json').read_text())
 refiner=AsyncOpenAI(base_url='http://127.0.0.1:8012/v1',api_key='EMPTY',timeout=600)
 judge=AsyncOpenAI(base_url='http://127.0.0.1:8012/v1',api_key='EMPTY',timeout=600)
 prov={'sample_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'refiner_prompt_sha256':hashlib.sha256((ROOT.parents[1]/'pipeline/configs/prompts_clean.json').read_bytes()).hexdigest(),'review_prompt_sha256':hashlib.sha256(REVIEW_SYSTEM.encode()).hexdigest(),'model':'Qwen3.8-27B-AWQ-INT4','temperature_refiner':0,'temperature_judge':0,'max_tokens':768,'concurrency':a.concurrency,'purpose':'screen stratified pilot; target and candidate judged independently against source'}
 pp=a.output.with_suffix('.provenance.json')
 if pp.exists() and json.loads(pp.read_text())!=prov:raise ValueError('provenance mismatch')
 pp.write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n');a.output.parent.mkdir(parents=True,exist_ok=True)
 done={x['case_id'] for x in read(a.output)} if a.output.exists() else set();sem=asyncio.Semaphore(a.concurrency);lock=asyncio.Lock();start=time.monotonic()
 async def request(client,messages,seed):
  async with sem:
   r=await client.chat.completions.create(model='qwen38-teacher',messages=messages,temperature=0,max_tokens=768,seed=seed,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}})
   c=r.choices[0]
   if c.finish_reason!='stop' or not c.message.content:raise ValueError('empty_or_truncated')
   return c.message.content
 async def one(row):
  t=time.monotonic();error=None;candidate=None;review=None
  try:
   prompt=prompts['user_template']['zh'].format(oral_text=row['oral_text'])
   raw=await request(refiner,[{'role':'system','content':prompts['system_prompts']['zh']},{'role':'user','content':prompt}],int(hashlib.sha256((row['case_id']+'refiner').encode()).hexdigest()[:8],16))
   candidate=json.loads(raw).get('clean_text')
   if not isinstance(candidate,str) or not candidate.strip():raise ValueError('missing_clean_text')
   judge_user=json.dumps({'scene':row['scene'],'phenomenon':row['phenomenon'],'oral_source':row['oral_text'],'gold_target':row['clean_target'],'refiner_output':candidate},ensure_ascii=False)
   rawj=await request(judge,[{'role':'system','content':REVIEW_SYSTEM},{'role':'user','content':judge_user}],int(hashlib.sha256((row['case_id']+'judge').encode()).hexdigest()[:8],16))
   review=json.loads(rawj)
   for k in ['source_valid','gold_pass','refiner_pass']:
    if type(review.get(k)) is not bool:raise ValueError('invalid_'+k)
  except Exception as e:error=type(e).__name__+':'+str(e)[:250]
  out={**row,'refiner_output':candidate,'joint_review':review,'error':error,'exact_gold_match':candidate==row['clean_target'],'seconds':round(time.monotonic()-t,2)}
  async with lock:
   with a.output.open('a') as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
   done.add(row['case_id']);allrows=read(a.output);valid=[x for x in allrows if not x.get('error') and isinstance(x.get('joint_review'),dict)]
   prog={'completed':len(done),'target':len(rows),'errors':sum(bool(x.get('error')) for x in allrows),'source_valid':sum(x['joint_review']['source_valid'] for x in valid),'gold_pass':sum(x['joint_review']['gold_pass'] for x in valid),'refiner_pass':sum(x['joint_review']['refiner_pass'] for x in valid),'wall_seconds':round(time.monotonic()-start,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
   a.output.with_suffix('.progress.json').write_text(json.dumps(prog,ensure_ascii=False,indent=2)+'\n');print(json.dumps(prog),flush=True)
 try:await asyncio.gather(*(one(r) for r in rows if r['case_id'] not in done))
 finally:await refiner.close();await judge.close()
if __name__=='__main__':asyncio.run(main())
