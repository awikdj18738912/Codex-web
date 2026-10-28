"""Oralize verified training targets for a source-first pilot; no production files touched."""
import argparse,asyncio,hashlib,json,re,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parent
SYSTEM='''将已确定的精修目标转换为一条符合指定口语现象的中文ASR转写。目标文本是说话者最后要表达的全部内容。完整保留目标中的事实、实体、动作、关系、条件、理由、语气和结论；不得增删未撤回的内容。只允许增加指定口语现象。按要求真的生成该现象，不能用“保持自然”作为省略要求的理由；同时句式和语气要自然多样，不套固定句型。
数字、日期、金额、英文实体、型号、代码、人名、地名原样保留，不用它们制造改口。错误候选只能是说话时未执行、且当场清楚撤回的选词或想法，不得描述为实际发生的经历、方案或结果。中文标点误断只可切断本应连续的短语，不得损坏代码、URL、小数、正常句界、问句。输出JSON：oral_text和speech_phenomena。'''
INSTRUCTIONS={
 'fillers':'必须在句中自然加入至少两个分散的口语停顿或语气词（从嗯、啊、那个、你看等自然选择），不改变事实。不要只调换标点。',
 'repetition_false_start':'必须包含一次词语或短语的自然重复，并包含一个没说完就重新组织的起头；不引入新的事实候选，也不只改标点。',
 'self_correction':'必须有一次可识别的即时自我纠正：先说一个尚未执行的普通措辞/想法，随后用自然但多样的说法明确撤回，并回到目标中的最终内容。改动仅限普通词语或计划措辞，禁止改动或替换数字、日期、金额、人名、地名、机构、产品、型号、英文术语和代码。错误说法不得写成已经发生的事件或有效操作。最终纠正后的关键短语须与目标一致。',
 'boundary_punctuation':'必须在一个中文短语内部插入且仅插入一处不合语义的标点，造成可恢复的误断。不得插在英文实体、URL、代码、小数、数字、正常句界或问句中。',
 'natural_mixed_code_switch':'若目标有英文/代码，必须保留原文并自然地在中文口语中复述至少一个已有英文实体；若目标无英文/代码，不得生造英文，改为使用自然中文口语表达。'
}
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def call(row,timeout=600):
 user='口语现象要求：'+INSTRUCTIONS[row['phenomenon']]+'\n场景：'+row['scene']+'\n最终精修目标：\n'+row['clean_target']+'\n只返回JSON。'
 body={'model':'qwen38-teacher','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}], 'temperature':0.7,'top_p':0.8,'max_tokens':768,'seed':int(hashlib.sha256(row['case_id'].encode()).hexdigest()[:8],16),'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}}
 req=Request('http://127.0.0.1:8012/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urlopen(req,timeout=timeout) as f:return json.load(f)
def validate(row,raw):
 resp=raw['choices'][0]
 if resp.get('finish_reason')!='stop':raise ValueError('finish_reason='+str(resp.get('finish_reason')))
 data=json.loads(resp['message']['content']);oral=data.get('oral_text')
 if not isinstance(oral,str) or len(oral.strip())<4:raise ValueError('missing oral_text')
 target=row['clean_target'];nums=re.findall(r'(?<!\w)\d+(?:\.\d+)?',target)
 missing=[n for n in nums if n not in oral]
 if missing:raise ValueError('changed_or_missing_numeric_surface:'+','.join(missing[:5]))
 spans=re.findall(r'[A-Za-z][A-Za-z0-9_./:+-]{1,}',target)
 missing_ascii=[s for s in spans if s not in oral]
 if missing_ascii:raise ValueError('changed_or_missing_ascii_span:'+','.join(missing_ascii[:5]))
 return oral,data.get('speech_phenomena',[])
async def main():
 p=argparse.ArgumentParser();p.add_argument('--sample',type=Path,default=ROOT/'sample_500.jsonl');p.add_argument('--output',type=Path,default=ROOT/'oral_candidates_v2.jsonl');p.add_argument('--concurrency',type=int,default=12);p.add_argument('--limit',type=int);a=p.parse_args()
 rows=read(a.sample);rows=rows[:a.limit] if a.limit else rows
 prov={'model':'Qwen3.8-27B-AWQ-INT4','revision':'e3a700906e6a73ba4290b1873dc2ba8198527ab5','sample_sha256':hashlib.sha256(a.sample.read_bytes()).hexdigest(),'prompt_version':'v3_no_editor_gloss','system_prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),'temperature':0.7,'top_p':0.8,'max_tokens':768,'concurrency':a.concurrency,'task':'gold-target-to-oral augmentation; target is deliberately provided as the conditioning source'}
 pp=a.output.with_suffix('.provenance.json')
 if pp.exists() and json.loads(pp.read_text())!=prov:raise ValueError('resume provenance mismatch')
 pp.write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n');a.output.parent.mkdir(parents=True,exist_ok=True)
 done={r['case_id'] for r in read(a.output)} if a.output.exists() else set();sem=asyncio.Semaphore(a.concurrency);lock=asyncio.Lock();start=time.monotonic()
 async def one(row):
  async with sem:
   t=time.monotonic();error=None;oral=None;phen=[];usage=None
   try:
    if row['phenomenon']=='clean_passthrough':oral=row['clean_target'];phen=['clean_passthrough']
    else:
     raw=await asyncio.to_thread(call,row);usage=raw.get('usage');oral,phen=validate(row,raw)
   except Exception as e:error=type(e).__name__+':'+str(e)[:350]
   out={**row,'oral_text':oral,'observed_phenomena':phen,'error':error,'usage':usage,'request_seconds':round(time.monotonic()-t,2)}
   async with lock:
    with a.output.open('a') as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
    done.add(row['case_id']);progress={'completed':len(done),'target':len(rows),'errors':sum(bool(x.get('error')) for x in read(a.output)),'wall_seconds':round(time.monotonic()-start,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    a.output.with_suffix('.progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n');print(json.dumps(progress),flush=True)
 await asyncio.gather(*(one(r) for r in rows if r['case_id'] not in done))
if __name__=='__main__':asyncio.run(main())
