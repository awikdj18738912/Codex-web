import asyncio,importlib.util,json,sys
from pathlib import Path
repo=Path('/home/aim0/桌面/tan/ASR/Codex-web');root=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018')
sys.path.insert(0,str(repo/'pipeline'))
spec=importlib.util.spec_from_file_location('rv',repo/'pipeline/scripts/05_finalize.py');fin=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fin;spec.loader.exec_module(fin)
sp=importlib.util.spec_from_file_location('review_runner',root/'review_repair_20261020/review_all.py');runner=importlib.util.module_from_spec(sp);sys.modules[sp.name]=runner;sp.loader.exec_module(runner)
rows={}
with (root/'final/train.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r['id'] in {'academic_1','academic_4','vibe_coding_2','daily_chat_0'}:rows[r['id']]=r
cases=[]
def add(name,source,expected,scene,language,record=None):
 item=dict(record or {});item.update({'id':name,'oral_text':source,'meta':{'scene':scene,'language':language}});cases.append((name,item,source,source,expected))
for key,expected in [('academic_1',False),('academic_4',False)]:
 r=rows[key];add(key,r['oral_text'],expected,r['meta']['scene'],r['meta']['language'],r)
add('english_only_zh_mix','The model performed well on the validation set, but latency remains high.',False,'academic','zh-en-mix')
add('spoken_term_gloss','我们用困惑度，也就是Perplexity，评估模型。',True,'academic','zh-en-mix')
add('dictated_code','把代码改成print("hello")，其他地方别动。',True,'vibe_coding','zh-en-mix')
add('stage_direction','明天下午开会（旁白：手机响起）。',False,'meeting','zh')
add('clean_daily','明天下午开会，大家记得来。',True,'daily_chat','zh')
add('related_metrics','我们比较准确率和召回率，再看综合的F1分数。',True,'academic','zh-en-mix')
async def main():
 client=runner.TwoGpuClient();sem=asyncio.Semaphore(4)
 async def one(c):
  name,item,source,target,expected=c;reviewed,passed,reason=await fin._qc_upstream_teacher(item,source,target,client,sem)
  return {'case':name,'expected':expected,'passed':passed,'matches':passed==expected,'reason':reason,'evidence':reviewed['meta']['quality_audit']['source_audit']}
 result=await asyncio.gather(*(one(c) for c in cases));out={'prompt_version':fin.UPSTREAM_SEMANTIC_QC_PROMPT_VERSION,'examples_in_prompt_not_reused':True,'results':result,'matched':sum(r['matches'] for r in result),'total':len(result)}
 Path('results/refiner_spoken_qc_repair_20261020/source_gate_v3_pilot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'matched':out['matched'],'total':out['total'],'results':[{k:r[k] for k in ['case','passed','matches','reason']} for r in result]},ensure_ascii=False))
 await client.close()
asyncio.run(main())
