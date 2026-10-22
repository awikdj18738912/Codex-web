"""Start local teacher, smoke test, generate 200 targets, then stop the service."""
import json,os,signal,subprocess,time
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parent;SERVER=None
def status(stage,**kwargs):
    x={'stage':stage,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs}
    (ROOT/'experiment_status.json').write_text(json.dumps(x,ensure_ascii=False,indent=2));print(x,flush=True)
def stop():
    global SERVER
    if SERVER is None:return
    try:os.killpg(SERVER.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:SERVER.wait(timeout=25)
    except subprocess.TimeoutExpired:
        try:os.killpg(SERVER.pid,signal.SIGKILL)
        except ProcessLookupError:pass
def main():
    global SERVER
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<18000:raise RuntimeError(f'GPU0 free memory below safe load threshold: {free} MiB')
    env=os.environ.copy();env['QWEN38_GPU_UTIL']='0.80';env['QWEN38_OFFLOAD_GB']='3'
    status('starting_teacher',gpu0_free_mib=free,offload_gb=3,model='Qwen3.8-27B-AWQ-INT4')
    with (ROOT/'server.log').open('a') as log:SERVER=subprocess.Popen(['bash',str(ROOT/'start_teacher.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    (ROOT/'server_pid.json').write_text(json.dumps({'pid':SERVER.pid,'port':8012,'gpu':0}))
    start=time.monotonic()
    while True:
        if SERVER.poll() is not None:raise RuntimeError('teacher server exited; inspect server.log')
        try:
            with urlopen('http://127.0.0.1:8012/health',timeout=2) as f:
                if f.status==200:break
        except Exception:pass
        if time.monotonic()-start>1200:raise TimeoutError('teacher startup timed out')
        time.sleep(5)
    status('smoke_generation',records=10)
    with (ROOT/'generation.log').open('a') as log:subprocess.run(['python',str(ROOT/'generate.py'),'--limit','10'],stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke=[r for r in map(json.loads,(ROOT/'teacher_candidates.jsonl').read_text().splitlines())]
    if len(smoke)!=10 or any(r['error'] for r in smoke):raise RuntimeError('smoke format validation failed')
    status('generating',records=200)
    with (ROOT/'generation.log').open('a') as log:subprocess.run(['python',str(ROOT/'generate.py'),'--limit','200'],stdout=log,stderr=subprocess.STDOUT,check=True)
    stop();SERVER=None
    summary=json.loads((ROOT/'generation_summary.json').read_text())
    status('generation_complete_service_stopped',**summary)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        stop();status('failed',error=type(e).__name__+': '+str(e));raise
