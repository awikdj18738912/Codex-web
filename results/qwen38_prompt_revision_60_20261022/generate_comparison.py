"""Same-input, same-prompt teacher comparison. No training files are changed."""
import argparse,asyncio,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parent
def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def call(row,prompt):
    body={'model':'qwen38-teacher','messages':[{'role':'system','content':prompt['system_prompt']},
          {'role':'user','content':prompt['user_template'].format(oral_text=row['input'])}],
          'temperature':0,'top_p':1,'max_tokens':768,'response_format':{'type':'json_object'},
          'chat_template_kwargs':{'enable_thinking':False,'preserve_thinking':False}}
    req=Request('http://127.0.0.1:8012/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=600) as f:return json.load(f)
async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=60)
    parser.add_argument('--concurrency',type=int,default=2);args=parser.parse_args()
    rows=read(ROOT/'sample_60.jsonl')[:args.limit];output=ROOT/'predictions_qwen38.jsonl'
    done={r['id'] for r in read(output)} if output.exists() else set()
    prompt=json.loads((ROOT/'prompt_snapshot.json').read_text());sem=asyncio.Semaphore(args.concurrency);lock=asyncio.Lock()
    started=time.monotonic();initial_done=len(done)
    provenance={'sample_sha256':hashlib.sha256((ROOT/'sample_60.jsonl').read_bytes()).hexdigest(),
      'prompt_sha256':hashlib.sha256((ROOT/'prompt_snapshot.json').read_bytes()).hexdigest(),
      'candidate':'cyankiwi/Qwen3.8-27B-AWQ-INT4','revision':'e3a700906e6a73ba4290b1873dc2ba8198527ab5',
      'sampling':{'temperature':0,'top_p':1,'max_tokens':768,'enable_thinking':False,'preserve_thinking':False},
      'comparison':'Same Qwen3.8 AWQ model, same source and sampling; only Chinese system and user prompts changed.'}
    (ROOT/'generation_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    async def one(row):
        async with sem:
            t=time.monotonic();result=None;raw=None;clean=None;error=None;finish=None
            try:
                result=await asyncio.to_thread(call,row,prompt)
                choice=result['choices'][0];finish=choice['finish_reason'];raw=choice['message'].get('content')
                if finish!='stop':raise ValueError('finish_reason='+str(finish))
                parsed=json.loads(raw);clean=parsed.get('clean_text')
                if not isinstance(clean,str) or not clean.strip():raise ValueError('missing clean_text')
            except Exception as exc:error=type(exc).__name__+': '+str(exc)[:500]
            record={**row,'clean_text':clean,'raw_content':raw,'error':error,'finish_reason':finish,
                    'usage':result.get('usage') if result else None,'request_seconds':round(time.monotonic()-t,3)}
            async with lock:
                with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                done.add(row['id']);progress={'completed':len(done),'target':len(rows),'this_run_completed':len(done)-initial_done,
                    'this_run_wall_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
                (ROOT/'generation_progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2))
                print(progress,flush=True)
    await asyncio.gather(*(one(r) for r in rows if r['id'] not in done))
    results=read(output)
    summary={'records':len(results),'errors':sum(bool(r['error']) for r in results),
      'changed_vs_baseline':sum(r['clean_text']!=r['baseline_target'] for r in results if not r['error']),
      'note':'Generation success is not semantic acceptance. Independent blinded review is required.'}
    (ROOT/'generation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary,flush=True)
if __name__=='__main__':asyncio.run(main())
