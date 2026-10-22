"""Durable local test runner: download readiness -> smoke -> 200 -> blind files."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parent
MODEL=Path('/home/aim0/data/ASR/models/Qwen3.8-27B-AWQ-INT4')
SERVER=None
def status(stage,**kwargs):
    item={'stage':stage,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs}
    (ROOT/'experiment_status.json').write_text(json.dumps(item,ensure_ascii=False,indent=2))
    print(json.dumps(item,ensure_ascii=False),flush=True)
def jsonl(name):return [json.loads(l) for l in (ROOT/name).read_text().splitlines() if l.strip()]
def cleanup():
    if SERVER is not None:
        try:os.killpg(SERVER.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:SERVER.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(SERVER.pid,signal.SIGKILL)
            except ProcessLookupError:pass
def main():
    global SERVER
    status('waiting_for_download',model_dir=str(MODEL))
    started=time.monotonic()
    while True:
        provenance=json.loads((ROOT/'download_provenance.json').read_text())
        if provenance.get('finished_epoch'):break
        pid_record=ROOT/'download_pid.json'
        if pid_record.exists():
            pid=json.loads(pid_record.read_text())['pid']
            cmd=Path(f'/proc/{pid}/cmdline')
            if not cmd.exists() or b'download_model.py' not in cmd.read_bytes():
                raise RuntimeError('download process exited before completion; inspect download.log')
        if time.monotonic()-started>24*3600:raise TimeoutError('download not ready after 24 hours')
        time.sleep(10)
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        if not (MODEL/name).is_file() or (MODEL/name).stat().st_size==0:raise ValueError('missing shard '+name)
    status('starting_server')
    gpu_line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    free_mib=int(gpu_line.strip());env=os.environ.copy()
    if free_mib<18000:
        raise RuntimeError(f'GPU0 has only {free_mib} MiB free; other services retained; defer comparison')
    if free_mib<22500:
        env['QWEN38_OFFLOAD_GB']='3'
        env['QWEN38_GPU_UTIL']='0.80'
    runtime={'gpu0_free_mib_before_launch':free_mib,'cpu_offload_gb':float(env.get('QWEN38_OFFLOAD_GB','0')),
             'gpu_memory_utilization':float(env.get('QWEN38_GPU_UTIL','0.90')),'speed_comparison_fair':False}
    (ROOT/'runtime_config.json').write_text(json.dumps(runtime,indent=2))
    with (ROOT/'server.log').open('a') as log:
        SERVER=subprocess.Popen(['bash',str(ROOT/'start_teacher.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    (ROOT/'server_pid.json').write_text(json.dumps({'pid':SERVER.pid,'pgid':SERVER.pid,'gpu':0,'port':8012}))
    t=time.monotonic()
    while True:
        if SERVER.poll() is not None:raise RuntimeError('server exited: inspect server.log')
        try:
            with urlopen('http://127.0.0.1:8012/health',timeout=2) as f:
                if f.status==200:break
        except Exception:pass
        if time.monotonic()-t>1200:raise TimeoutError('server readiness timeout')
        time.sleep(5)
    status('smoke_generation',records=10)
    with (ROOT/'generation.log').open('a') as log:
        subprocess.run(['python',str(ROOT/'generate_comparison.py'),'--limit','10','--concurrency','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke=jsonl('predictions_qwen38.jsonl')
    if len(smoke)!=10 or any(r['error'] for r in smoke):raise RuntimeError('smoke output/format failure; inspect predictions_qwen38.jsonl')
    status('generating_comparison',records=200)
    with (ROOT/'generation.log').open('a') as log:
        subprocess.run(['python',str(ROOT/'generate_comparison.py'),'--limit','200','--concurrency','4'],stdout=log,stderr=subprocess.STDOUT,check=True)
    results=jsonl('predictions_qwen38.jsonl');cases=[];keys=[]
    for row in sorted(results,key=lambda r:r['case_no']):
        if row['error']:continue
        flip=int(hashlib.sha256((row['id']+'qwen38comparison').encode()).hexdigest(),16)%2
        old,new=row['baseline_target'],row['clean_text'];a,b=(new,old) if flip else (old,new)
        cases.append({'case_no':row['case_no'],'input':row['input'],'candidate_a':a,'candidate_b':b})
        keys.append({'case_no':row['case_no'],'id':row['id'],'a_is':'qwen38' if flip else 'qwen30',
                     'b_is':'qwen30' if flip else 'qwen38','identical':old==new})
    for name,rows in [('blind_cases.jsonl',cases),('blind_key.jsonl',keys),
                      ('blind_cases_001_067.jsonl',cases[:67]),('blind_cases_068_134.jsonl',cases[67:134]),('blind_cases_135_200.jsonl',cases[134:])]:
        (ROOT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    cleanup();SERVER=None
    status('generation_complete_awaiting_independent_review',records=len(results),errors=sum(bool(r['error']) for r in results),blind_cases=len(cases))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        cleanup();status('failed',error=type(exc).__name__+': '+str(exc));raise
