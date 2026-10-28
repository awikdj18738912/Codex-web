import asyncio,json,sys,time
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parent
async def main():
    for _ in range(120):
        try:
            with urlopen('http://127.0.0.1:8012/health',timeout=2) as r0, urlopen('http://127.0.0.1:8013/health',timeout=2) as r1:
                if r0.status==200 and r1.status==200: break
        except Exception:
            await asyncio.sleep(5)
    else: raise RuntimeError('teacher startup timeout')
    async def run(variant):
        with (ROOT/f'{variant}.log').open('w') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'generate_comparison.py'),'--variant',variant,'--output',str(ROOT/f'{variant}.jsonl'),'--concurrency','8','--retries','0',stdout=log,stderr=asyncio.subprocess.STDOUT)
            code=await proc.wait()
            if code: raise RuntimeError(f'{variant} exited {code}')
    print('teacher ready; paired generation started',flush=True)
    await asyncio.gather(run('baseline'),run('revised'))
    proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'prepare_blind.py'))
    if await proc.wait(): raise RuntimeError('blind export failed')
    (ROOT/'generation_complete.json').write_text(json.dumps({'completed_at':time.strftime('%Y-%m-%d %H:%M:%S'),'status':'ready_for_blind_review'}))
    print('both arms complete; blind review ready',flush=True)
asyncio.run(main())
