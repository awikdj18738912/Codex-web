import json,os,signal,subprocess,time
from pathlib import Path
from urllib.request import urlopen
P=Path(__file__).resolve().parent
SUP=1527965
OLD=[1524892,1524899]
def healthy():
 try:
  for port in (8012,8013):
   with urlopen(f'http://127.0.0.1:{port}/health',timeout=2)as r:
    if r.status!=200:return False
  return True
 except Exception:return False
def proc_state(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();return s[s.rfind(')')+2:].split()[0]
 except FileNotFoundError:return None
def stop_old():
 for pid in OLD:
  try:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'api_server' in cmd and b'Qwen3.8-27B-AWQ-INT4' in cmd:os.kill(pid,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
 deadline=time.time()+120
 while time.time()<deadline:
  ports=[]
  for port in (8012,8013):
   try:
    with urlopen(f'http://127.0.0.1:{port}/health',timeout=1):ports.append(True)
   except Exception:ports.append(False)
  if not any(ports) and all(proc_state(pid) in (None,'Z') for pid in OLD):return
  time.sleep(2)
 raise RuntimeError('eager services have not shut down')
def start_graph():
 procs=[]
 for gpu in (0,1):
  log=(P/f'teacher_graph_gpu{gpu}.log').open('a')
  procs.append(subprocess.Popen(['bash',str(P/f'start_teacher_graph_gpu{gpu}.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
 (P/'graph_teacher_pids.json').write_text(json.dumps([x.pid for x in procs]))
 deadline=time.time()+600
 while time.time()<deadline:
  if healthy():return procs
  if any(x.poll() is not None for x in procs):raise RuntimeError('a graph service exited while starting')
  time.sleep(5)
 raise RuntimeError('graph services did not become healthy')
def start_eager_fallback():
 procs=[]
 for gpu in (0,1):
  log=(P/f'teacher_gpu{gpu}_fallback.log').open('a')
  procs.append(subprocess.Popen(['bash',str(P/f'concurrency_trial/serve_gpu{gpu}_seq8.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
 deadline=time.time()+600
 while time.time()<deadline:
  if healthy():return
  time.sleep(5)
 raise RuntimeError('eager fallback startup failed')
try:
 while not(P/'outputs/batch_0000.complete.json').exists():time.sleep(10)
 (P/'graph_switch_status.json').write_text(json.dumps({'stage':'current_batch_complete','rows':500,'at':time.strftime('%Y-%m-%d %H:%M:%S')}))
 stop_old()
 (P/'graph_switch_status.json').write_text(json.dumps({'stage':'starting_cuda_graph_services','at':time.strftime('%Y-%m-%d %H:%M:%S')}))
 procs=start_graph()
 (P/'selected_scheduling.json').write_text(json.dumps({'worker':str(P/'concurrency_trial/client_per_gpu_12/run_specialists.py'),'client_concurrency_per_gpu':12,'total_client_concurrency':24,'server_max_num_seqs_per_gpu':16,'gpu_memory_utilization':0.95,'cudagraph':'vLLM default FULL_AND_PIECEWISE; enforce_eager disabled','model':'Qwen3.8-27B-AWQ-INT4'}))
 (P/'graph_switch_status.json').write_text(json.dumps({'stage':'cuda_graph_enabled','health':[8012,8013],'at':time.strftime('%Y-%m-%d %H:%M:%S')}))
 os.kill(SUP,signal.SIGCONT)
 (P/'graph_switch_status.json').write_text(json.dumps({'stage':'cuda_graph_enabled_scheduler_resumed','at':time.strftime('%Y-%m-%d %H:%M:%S')}))
except Exception as exc:
 (P/'graph_switch_error.json').write_text(json.dumps({'error':str(exc),'at':time.strftime('%Y-%m-%d %H:%M:%S')}))
 try:
  for pid in json.loads((P/'graph_teacher_pids.json').read_text()):
   try:os.killpg(pid,signal.SIGTERM)
   except ProcessLookupError:pass
  if not healthy():start_eager_fallback()
  os.kill(SUP,signal.SIGCONT)
  (P/'graph_switch_status.json').write_text(json.dumps({'stage':'eager_fallback_scheduler_resumed','at':time.strftime('%Y-%m-%d %H:%M:%S')}))
 except Exception as fallback:
  (P/'graph_fallback_error.json').write_text(json.dumps({'error':str(fallback)}))
