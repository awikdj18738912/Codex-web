"""Run a frozen prompt on a frozen input set through the local teacher API."""
import argparse,asyncio,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def call(row,prompt,max_tokens):
    body={'model':'qwen38-teacher','messages':[{'role':'system','content':prompt['system_prompt']},
       {'role':'user','content':prompt['user_template'].format(oral_text=row['input'],draft_text=row.get('baseline_target',''))}],
       'temperature':0,'top_p':1,'max_tokens':max_tokens,'response_format':{'type':'json_object'},
       'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}}
    req=Request('http://127.0.0.1:8012/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=600) as f:return json.load(f)
async def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',type=Path,required=True);p.add_argument('--prompt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--max-tokens',type=int,default=768);p.add_argument('--limit',type=int);a=p.parse_args()
    rows=read(a.sample)
    if a.limit is not None:rows=rows[:a.limit]
    prompt=json.loads(a.prompt.read_text())
    if '{draft_text}' in prompt.get('user_template',''):
        missing=[r.get('id') for r in rows if not isinstance(r.get('baseline_target'),str) or not r['baseline_target'].strip()]
        if missing:raise ValueError(f'second-pass prompt requires a nonempty baseline_target; missing for {len(missing)} records')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    provenance={'sample_sha256':hashlib.sha256(a.sample.read_bytes()).hexdigest(),'prompt_sha256':hashlib.sha256(a.prompt.read_bytes()).hexdigest(),
       'model':'cyankiwi/Qwen3.8-27B-AWQ-INT4','temperature':0,'top_p':1,'max_tokens':a.max_tokens,'thinking':False}
    if '{draft_text}' in prompt['user_template']:provenance['second_pass_draft_field']='baseline_target'
    provpath=a.output.with_suffix('.provenance.json')
    if provpath.exists() and json.loads(provpath.read_text())!=provenance:raise ValueError('resume provenance mismatch')
    provpath.write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    done={r['id'] for r in read(a.output)} if a.output.exists() else set();initial=len(done)
    assert len({r['id'] for r in rows})==len(rows)
    sem=asyncio.Semaphore(a.concurrency);lock=asyncio.Lock();start=time.monotonic()
    async def one(row):
        async with sem:
            t=time.monotonic();raw=None;clean=None;result=None;error=None
            try:
                result=await asyncio.to_thread(call,row,prompt,a.max_tokens)
                c=result['choices'][0];raw=c['message'].get('content')
                if c.get('finish_reason')!='stop':raise ValueError('finish_reason='+str(c.get('finish_reason')))
                clean=json.loads(raw).get('clean_text')
                if not isinstance(clean,str) or not clean.strip():raise ValueError('empty clean_text')
            except Exception as exc:error=f'{type(exc).__name__}: {str(exc)[:400]}'
            record={**row,'clean_text':clean,'raw_content':raw,'error':error,'usage':result.get('usage') if result else None,
                   'request_seconds':round(time.monotonic()-t,3)}
            async with lock:
                with a.output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                done.add(row['id']);progress={'completed':len(done),'target':len(rows),'this_run_completed':len(done)-initial,'wall_seconds':round(time.monotonic()-start,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
                a.output.with_suffix('.progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2));print(progress,flush=True)
    await asyncio.gather(*(one(r) for r in rows if r['id'] not in done))
if __name__=='__main__':asyncio.run(main())
