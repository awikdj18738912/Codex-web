import asyncio,json,sys
from pathlib import Path
sys.path.insert(0,'pipeline')
from src.services.llm_client import AsyncLLMClient
from src.pipeline.semantic_verdict import parse_semantic_verdict
out=Path('results/refiner_spoken_qc_repair_20261020')
prompt='''你审核纯口语ASR文本训练数据的源文本，只看source，不考虑它能否被精修。不改写文本。
源文本应只包含说话人实际说出的词语，以及普通ASR标点。编者给读者加的括号术语对照、旁白和舞台动作标注不符合这个数据用途，即使很容易删除，也应判源文本不合格。
允许真实口头解释、流畅短句、自然改口和中英混用。不能只看到括号符号就拒绝：说话人明确谈论/口述的代码、字符串和公式等字面内容可以含括号。
边界示例（只用于说明审核原则，不要求说话人采用固定说法）：
不合格：我们要做文本分类（Text classification）。——括号是书面术语对照。
不合格：今天发布新产品（音效：掌声）。——括号是制作说明。
合格：把那行函数改为printf("hi")。——代码是讨论的字面内容。
合格：那个指标英文名我记得叫AUC，先用它算一下。——解释实际说出口了。
逐字检查当前source，不能凭空给它添加示例里才有的内容。仅返回严格JSON，pass为布尔值，reason必须是非空字符串。'''
source=None
with Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/final/train.jsonl').open() as f:
 for l in f:
  r=json.loads(l)
  if r['id']=='academic_1':source=r['oral_text'];break
cases=[('real_academic_1',source,False),('narrator','我明天下午去取快递（旁白：此时手机响起）。',False),('spoken_term','把这个指标叫CER，中文就是字错率，先看这一个。',True),('code','把代码改成sum(x)，其他地方别动。',True),('clean_question','你明天下午几点到上海？',True),('correction','学习率用0.7，不对，用0.9。',True)]
async def main():
 c=AsyncLLMClient(base_url='http://127.0.0.1:8010/v1',model='refiner-teacher',max_concurrent=4,timeout=120)
 async def one(case):
  key,text,expected=case;raw=await c.generate(system_prompt=prompt,user_prompt='source: '+text,temperature=0,max_tokens=400,response_format='json_object');v=parse_semantic_verdict(raw)
  return {'case':key,'source':text,'expected':expected,'passed':v.passed,'valid':v.valid,'reason':v.reason,'raw_response':raw,'matches_expected':v.valid and v.passed==expected}
 rows=await asyncio.gather(*(one(c) for c in cases))
 report={'prompt':prompt,'fewshot_cases_are_distinct_from_evaluation':True,'results':rows,'matches':sum(r['matches_expected'] for r in rows),'total':len(rows)}
 (out/'heldout_source_pilot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 for r in rows:print(r['case'],r['matches_expected'],r['reason'])
 await c.client.close()
asyncio.run(main())
