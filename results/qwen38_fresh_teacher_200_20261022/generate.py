"""Generate fresh local teacher proposals from source transcripts only."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parent
def read_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def request(row,prompt):
    body={'model':'qwen38-teacher','messages':[{'role':'system','content':prompt['system_prompt']},
      {'role':'user','content':prompt['user_template'].format(oral_text=row['input'])}],
      'temperature':0,'top_p':1,'max_tokens':768,'response_format':{'type':'json_object'},
      'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}}
    req=Request('http://127.0.0.1:8012/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=600) as f:return json.load(f)
async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=200);parser.add_argument('--concurrency',type=int,default=4);args=parser.parse_args()
    rows=read_jsonl(ROOT/'sample_200.jsonl')[:args.limit];prompt=json.loads((ROOT/'prompt_snapshot.json').read_text());out=ROOT/'teacher_candidates.jsonl'
    completed={r['id'] for r in read_jsonl(out)} if out.exists() else set();initial=len(completed)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate ids in sample')
    sem=asyncio.Semaphore(args.concurrency);lock=asyncio.Lock();started=time.monotonic()
    async def one(row):
        async with sem:
            start=time.monotonic();result=None;raw=None;clean=None;error=None;finish=None
            try:
                result=await asyncio.to_thread(request,row,prompt)
                choice=result['choices'][0];finish=choice.get('finish_reason');raw=choice['message'].get('content')
                if finish!='stop':raise ValueError('finish_reason='+str(finish))
                parsed=json.loads(raw);clean=parsed.get('clean_text')
                if not isinstance(clean,str) or not clean.strip():raise ValueError('missing clean_text')
            except Exception as exc:error=f'{type(exc).__name__}: {str(exc)[:400]}'
            record={'case_no':row['case_no'],'id':row['id'],'scene':row['scene'],'input':row['input'],
                    'reference_target_evidence':row['existing_target'],'teacher_target':clean,'raw_content':raw,
                    'error':error,'finish_reason':finish,'usage':result.get('usage') if result else None,
                    'request_seconds':round(time.monotonic()-start,3)}
            async with lock:
                with out.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                completed.add(row['id']);progress={'completed':len(completed),'target':len(rows),'this_run_completed':len(completed)-initial,
                   'this_run_wall_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
                (ROOT/'generation_progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2));print(progress,flush=True)
    await asyncio.gather(*(one(r) for r in rows if r['id'] not in completed))
    results=read_jsonl(out);valid=[r for r in results if not r['error']]
    summary={'records':len(results),'errors':sum(bool(r['error']) for r in results),'valid':len(valid),
       'changed_from_reference_target':sum(r['teacher_target']!=r['reference_target_evidence'] for r in valid),
       'warning':'The reference target is retained only for later audit and was not included in teacher requests.'}
    (ROOT/'generation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary,flush=True)
if __name__=='__main__':asyncio.run(main())
