"""Isolated teacher repair proposals; never updates the training dataset."""
import argparse, asyncio, hashlib, json, time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SYSTEM = '''你负责复核中文口语转写的精修目标。输入包括原始口语和已有目标。依据说话者最终意图判断。
先区分真实叙述与说话过程中的废弃候选。出现“我真正想说的是”“不是X是Y”“这个才对”等时，应解析最终说的是哪一件事。被明确撤回的方案、型号、数值不应写成经历或思考起点来保留；没有撤回的独立事情、请求和事实应完整保留。不要自行补出反思过程或因果。
已有目标准确且完成必要清理则 KEEP，不要为文风换词。只有能指出具体语义错误或未完成的必要清理时才 REPAIR，并给出最小修订的完整目标。原始口语含无法确定的矛盾或歧义则 UNRESOLVED，不猜测。不要引入新事实，不翻译名称、技术标识，不擅自改变人物关系、时间或语气。保持正常口语意思，删除明确撤回内容和无意义重复。
只输出JSON对象：{"decision":"KEEP|REPAIR|UNRESOLVED","reason":"50字以内说明已有目标的具体问题或正确之处","source_evidence":"支持判断的原文短片段，最多80字","clean_text":"完整目标；KEEP时逐字复制已有目标；UNRESOLVED时为空字符串"}。'''

def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def request(row):
    payload = {'input':row['input'], 'existing_target':row['existing_target']}
    body={'model':'refiner-teacher','messages':[{'role':'system','content':SYSTEM},
          {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
          'temperature':0, 'max_tokens':1000,'response_format':{'type':'json_object'}}
    req=Request('http://127.0.0.1:8011/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),
                headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=600) as response:
        result=json.load(response)
    raw=result['choices'][0]['message']['content']
    return raw,result.get('usage'),result['choices'][0].get('finish_reason')

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=100)
    args=parser.parse_args()
    rows=read(ROOT/'pilot_sample_100.jsonl')[:args.limit];output=ROOT/'teacher_proposals.jsonl'
    completed={r['id'] for r in read(output)} if output.exists() else set()
    semaphore=asyncio.Semaphore(4);lock=asyncio.Lock();started=time.monotonic()
    (ROOT/'repair_prompt.txt').write_text(SYSTEM)
    provenance={'sample_sha256':hashlib.sha256((ROOT/'pilot_sample_100.jsonl').read_bytes()).hexdigest(),
                'model':'Qwen3-30B-A3B-Instruct-2507-AWQ-4bit','gpu':0,'prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
                'purpose':'proposals only; independent review required; training files unchanged'}
    (ROOT/'repair_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    async def one(row):
        async with semaphore:
            t=time.monotonic();parsed=None;raw=None;usage=None;error=None;finish_reason=None
            try:
                raw,usage,finish_reason=await asyncio.to_thread(request,row)
                parsed=json.loads(raw)
                if parsed.get('decision') not in {'KEEP','REPAIR','UNRESOLVED'}:raise ValueError('invalid decision')
                if not isinstance(parsed.get('clean_text'),str):raise ValueError('invalid clean_text')
                if parsed['decision']=='KEEP' and parsed['clean_text']!=row['existing_target']:raise ValueError('KEEP changed target')
                if parsed['decision']=='REPAIR' and not parsed['clean_text'].strip():raise ValueError('empty repair')
                if parsed['decision']=='REPAIR' and parsed['clean_text']==row['existing_target']:
                    parsed['original_decision']='REPAIR';parsed['decision']='KEEP'
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            item={**row,'proposal':parsed,'raw_content':raw,'usage':usage,'error':error,'finish_reason':finish_reason,
                  'request_seconds':round(time.monotonic()-t,3)}
            async with lock:
                with output.open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                completed.add(row['id'])
                progress={'completed':len(completed),'total':len(rows),'wall_seconds':round(time.monotonic()-started,1)}
                (ROOT/'repair_progress.json').write_text(json.dumps(progress))
                if len(completed)%10==0:print(progress,flush=True)
    await asyncio.gather(*(one(r) for r in rows if r['id'] not in completed))

if __name__=='__main__':asyncio.run(main())
