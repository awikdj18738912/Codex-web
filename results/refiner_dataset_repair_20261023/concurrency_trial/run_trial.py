import os,signal,subprocess,time,json,sys,hashlib
from pathlib import Path
from urllib.request import urlopen
P=Path(__file__).resolve().parent;JOB=P.parent;PROJECT=JOB.parents[1]
SUPERVISOR=1482844
OLD_PIDS=[1482843,1482842]
LAUNCH=PROJECT/'results/qwen38_prompt_v4_3_controlled_20261022'
def health():
 try:
  for port in [8012,8013]:
   with urlopen(f'http://127.0.0.1:{port}/health',timeout=2)as r:
    if r.status!=200:return False
  return True
 except Exception:return False
def stop(pids):
 for pid in pids:
  try:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'api_server'in cmd and b'qwen38-teacher'in cmd:os.kill(pid,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
 for _ in range(60):
  if all(not Path(f'/proc/{pid}').exists()for pid in pids):return
  time.sleep(1)
 raise RuntimeError('teacher did not exit cleanly')
def start(seqs):
 processes=[]
 for gpu in [0,1]:
  script=(LAUNCH/f'start_teacher_gpu{gpu}.sh').read_text().replace('--max-num-seqs 8',f'--max-num-seqs {seqs}')
  if seqs==16:script=script.replace('--enforce-eager','').replace('QWEN38_GPU_UTIL:-0.90','QWEN38_GPU_UTIL:-0.95')
  path=P/f'serve_gpu{gpu}_seq{seqs}.sh';path.write_text(script)
  with (P/f'serve_gpu{gpu}_seq{seqs}.log').open('a')as log:
   processes.append(subprocess.Popen(['bash',str(path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
 pids=[x.pid for x in processes];(P/'current_teacher_pids.json').write_text(json.dumps(pids))
 for _ in range(120):
  if health():return pids
  if any(x.poll()is not None for x in processes):raise RuntimeError('teacher startup failed')
  time.sleep(5)
 raise RuntimeError('teacher startup timeout')
def bench(n):
 d=P/f'client_per_gpu_{n}';d.mkdir(exist_ok=True)
 s=(JOB/'run_specialists.py').read_text().replace('asyncio.Semaphore(8)',f'asyncio.Semaphore({n})').replace("'per_gpu_concurrency':8",f"'per_gpu_concurrency':{n}")
 (d/'run_specialists.py').write_text(s);(d/'prompts.json').write_bytes((JOB/'prompts.json').read_bytes())
 started=time.monotonic()
 with (d/'run.log').open('w')as log:
  subprocess.run([sys.executable,str(d/'run_specialists.py'),'--sample',str(P/'sample24.jsonl'),'--output',str(d/'results.jsonl')],stdout=log,stderr=subprocess.STDOUT,check=True)
 rows=[json.loads(x)for x in(d/'results.jsonl').read_text().splitlines()]
 return {'seconds':time.monotonic()-started,'records':len(rows),'errors':sum(bool(v.get('error'))for r in rows for v in r['checks'].values()),'signatures':{r['audit_id']:{k:v.get('result',{}).get('status',v.get('error'))for k,v in r['checks'].items()}for r in rows}}
pids=OLD_PIDS;paused=False
try:
 (P/'status.json').write_text(json.dumps({'stage':'starting_recovery_and_benchmark','existing_results_preserved':True}))
 pids=start(8)
 baseline=bench(8)
 stop(pids);pids=start(16)
 trials={12:bench(12),16:bench(16)}
 eligible={n:t for n,t in trials.items()if t['signatures']==baseline['signatures']and t['errors']<=baseline['errors']and t['records']==24 and t['seconds']<baseline['seconds']*.95}
 selected_n=min(eligible,key=lambda n:eligible[n]['seconds'])if eligible else 8
 if selected_n!=8:
  (JOB/'selected_scheduling.json').write_text(json.dumps({'worker':str(P/f'client_per_gpu_{selected_n}'/'run_specialists.py'),'total_concurrency':selected_n*2,'use_after_initial_batch':True}))
 else:stop(pids);pids=start(8)
 result={'stage':'complete','selected_total_client_concurrency':selected_n*2,'cuda_graph_enabled':selected_n!=8,'baseline':baseline,'trials':trials,'quality_scope':'same model precision prompts and token budgets; small-sample status comparison cannot prove universal bitwise equality or zero quality impact','teacher_pids':pids}
 (P/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result,flush=True)
except Exception as exc:
 (P/'error.json').write_text(json.dumps({'error':str(exc)}))
 if not health():
  try:
   if (P/'current_teacher_pids.json').exists():stop(json.loads((P/'current_teacher_pids.json').read_text()))
   pids=start(8)
  except Exception as fallback:(P/'fallback_error.json').write_text(json.dumps({'error':str(fallback)}))
 raise
finally:
 if health():
  with (JOB/'run.log').open('a') as log:
   subprocess.Popen([sys.executable,str(JOB/'run_batches.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with (JOB/'export.log').open('a') as log:
   subprocess.Popen([sys.executable,str(JOB/'export_when_done.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
