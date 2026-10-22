import asyncio,importlib.util,json,os,random,sys
from pathlib import Path
repo=Path('/home/aim0/桌面/tan/ASR/Codex-web');out=repo/'results/refiner_spoken_qc_repair_20261020'
os.environ['ASR_DATA_DIR']='/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018'
os.environ['LLM_MAX_TOKENS_ORAL']='1024'
sys.path.insert(0,str(repo/'pipeline'))
from src.generators.oral_text_generator import OralTextGenerator
from src.services.llm_client import AsyncLLMClient
spec=importlib.util.spec_from_file_location('spoken_finalize',repo/'pipeline/scripts/05_finalize.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
async def main():
 client=AsyncLLMClient(base_url='http://127.0.0.1:8010/v1',model='refiner-teacher',max_concurrent=4,timeout=120)
 gen=OralTextGenerator(async_client=client,rng=random.Random(20261020))
 originals=[]
 with (Path(os.environ['ASR_DATA_DIR'])/'final/train.jsonl').open() as f:
  for line in f:
   r=json.loads(line)
   if r['id']=='academic_1':originals.append(r);break
 tests=[('editorial_alias',originals[0]['input'],originals[0]['output']['refined_text'],False),
 ('spoken_alias_kept','嗯，我们这次用困惑度，也就是Perplexity，来评估这个模型。','我们这次用困惑度，也就是Perplexity，来评估这个模型。',True),
 ('literal_code','把代码改成print("hello")，其他地方别动。','把代码改成print("hello")，其他地方别动。',True),
 ('filler','嗯，那个，明天下午开会。','明天下午开会。',True),
 ('correction','周二开会，啊不对，是周三。','周三开会。',True),
 ('stage_direction','明天下午开会（停顿两秒），大家记得来。','明天下午开会，大家记得来。',False)]
 sem=asyncio.Semaphore(4)
 async def judge(case):
  label,source,target,expected=case
  reviewed,passed,reason=await m._qc_upstream_teacher({'id':label},source,target,client,sem)
  return {'case':label,'expected':expected,'passed':passed,'matches_expected':passed==expected,'audit':reviewed['meta']['quality_audit']}
 judgments=await asyncio.gather(*(judge(c) for c in tests))
 (out/'pilot_qc.json').write_text(json.dumps(judgments,ensure_ascii=False,indent=2)+'\n')
 generation=await asyncio.gather(*(gen.generate_one(scene,i) for i,scene in enumerate(['daily_chat','academic','vibe_coding']*3)))
 (out/'pilot_oral.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in generation if x is not None))
 summary={'qc_cases':len(judgments),'qc_cases_matching_expected':sum(r['matches_expected'] for r in judgments),'generation_attempts':len(generation),'generation_valid':sum(r is not None for r in generation),'generation_with_parentheses':sum(r is not None and any(c in r['oral_text'] for c in '（）()') for r in generation),'scope':'Small diagnostic pilot only; no full-dataset semantic certification'}
 (out/'pilot_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary),flush=True)
 await client.client.close()
asyncio.run(main())
