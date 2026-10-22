"""Generate same-input teacher outputs via local OpenAI-compatible vLLM API."""
from __future__ import annotations
import argparse, asyncio, json, time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent
PROMPTS=Path('/home/aim0/桌面/tan/ASR/Codex-web/pipeline/configs/prompts_clean.json')
SAMPLE=ROOT/'sample_1000.jsonl'

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def request_one(base_url, model, system, user, max_tokens, timeout):
    body={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],
          'temperature':0.0,'top_p':1.0,'max_tokens':max_tokens,
          'response_format':{'type':'json_object'}}
    req=Request(base_url.rstrip('/')+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),
                headers={'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=timeout) as response:
        result=json.loads(response.read())
    choice=result['choices'][0]['message']
    return choice.get('content',''),choice.get('reasoning_content'),result.get('usage')

async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-label',required=True)
    p.add_argument('--prompt-json',type=Path,default=PROMPTS)
    p.add_argument('--api-model',default='refiner-teacher')
    p.add_argument('--base-url',default='http://127.0.0.1:8010/v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1000)
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args()
    prompts=json.loads(args.prompt_json.read_text())
    system=prompts['system_prompts']['zh'];template=prompts['user_template']['zh']
    rows=read_jsonl(SAMPLE)[:args.limit]
    completed=set()
    if args.output.exists():
        completed={r['id'] for r in read_jsonl(args.output)}
    pending=[r for r in rows if r['id'] not in completed]
    lock=asyncio.Lock();sem=asyncio.Semaphore(args.concurrency);done=len(completed)
    started=time.monotonic()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    async def one(row):
        nonlocal done
        user=template.format(oral_text=row['input'])
        t=time.monotonic();content='';reasoning=None;usage=None;error=None;parsed=None
        async with sem:
            try:
                content,reasoning,usage=await asyncio.to_thread(request_one,args.base_url,args.api_model,system,user,768,args.timeout)
                try:
                    value=json.loads(content)
                    parsed=value.get('clean_text') if isinstance(value,dict) else None
                    if not isinstance(parsed,str) or not parsed.strip():error='missing_clean_text'
                except Exception as exc:error='invalid_json:'+type(exc).__name__
            except Exception as exc:error=type(exc).__name__+':'+str(exc)[:500]
        record={'case_no':row['case_no'],'id':row['id'],'scene':row['scene'],'stratum':row['stratum'],
                'source_decision':row['source_decision'],'input':row['input'],'existing_target':row['existing_target'],
                'spoken_source':row['spoken_source'],'teacher_model':args.model_label,'clean_text':parsed,
                'raw_content':content,'reasoning_content':reasoning,'usage':usage,'error':error,
                'elapsed_seconds':round(time.monotonic()-t,3)}
        async with lock:
            with args.output.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            done+=1
            if done%10==0 or done==len(rows):
                progress={'model':args.model_label,'completed':done,'target':len(rows),'failed':sum(bool(x.get('error')) for x in read_jsonl(args.output)),'elapsed_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
                (ROOT/f'progress_{args.model_label}.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
                print(f"{args.model_label} {done}/{len(rows)} errors={progress['failed']} elapsed={progress['elapsed_seconds']}s",flush=True)
    await asyncio.gather(*(one(r) for r in pending))
    results=read_jsonl(args.output)
    final={'model':args.model_label,'records':len(results),'errors':sum(bool(r.get('error')) for r in results),
           'mean_elapsed_seconds':round(sum(r['elapsed_seconds'] for r in results)/max(1,len(results)),3),
           'total_request_seconds':round(sum(r['elapsed_seconds'] for r in results),1),'wall_seconds':round(time.monotonic()-started,1)}
    (ROOT/f'summary_{args.model_label}.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(final,ensure_ascii=False),flush=True)

if __name__=='__main__':asyncio.run(main())
