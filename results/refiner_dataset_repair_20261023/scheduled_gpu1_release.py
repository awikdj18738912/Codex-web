import argparse,json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import urlopen
P=Path(__file__).resolve().parent
TZ=ZoneInfo('Asia/Shanghai')
def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def health0():
 try:
  with urlopen('http://127.0.0.1:8012/health',timeout=5)as r:return r.status==200
 except Exception:return False
def gpu_status():
 return subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout

def matches_gpu1_server(pid):
 try:
  proc=Path('/proc')/str(pid);args=proc.joinpath('cmdline').read_bytes().split(b'\0')
  env=proc.joinpath('environ').read_bytes().split(b'\0')
  return (b'vllm.entrypoints.openai.api_server'in args and b'--port'in args and args[args.index(b'--port')+1]==b'8013' and b'/home/aim0/data/ASR/models/Qwen3.8-27B-AWQ-INT4'in args and b'CUDA_VISIBLE_DEVICES=1'in env)
 except (OSError,ValueError,IndexError):return False

def action(mode):
 now=datetime.now(TZ).isoformat();before=gpu_status();before0=health0()
 # Route all queued/new model calls to GPU0. This never stops the GPU0 service.
 policy=json.loads((P/'gpu_routing.json').read_text());policy.update(gpu1_enabled=False,drained_at=now)
 atomic(P/'gpu_routing.json',policy)
 record={'action':mode,'at':now,'gpu0_health_before':before0,'gpu_before':before,'stopped_pids':[]}
 if mode=='release':
  ids=[int(f.name)for f in Path('/proc').iterdir()if f.name.isdigit()and matches_gpu1_server(f.name)]
  groups=[]
  for pid in ids:
   # Match port, checkpoint and CUDA device immediately before sending a signal.
   if not matches_gpu1_server(pid):continue
   pgid=os.getpgid(pid)
   if pgid==pid:groups.append(pgid)
   os.kill(pid,signal.SIGTERM);record['stopped_pids'].append(pid)
  deadline=time.monotonic()+20
  while time.monotonic()<deadline and any(matches_gpu1_server(pid)for pid in ids):time.sleep(1)
  for pgid in groups:
   try:os.killpg(pgid,signal.SIGTERM)
   except ProcessLookupError:pass
  time.sleep(2)
  for pgid in groups:
   try:os.killpg(pgid,signal.SIGKILL)
   except ProcessLookupError:pass
  record['gpu0_health_after']=health0();record['gpu_after']=gpu_status()
  atomic(P/'gpu1_release_result.json',record)
  if not record['gpu0_health_after']:raise RuntimeError('GPU0 health check failed after GPU1 release; no signal was sent to GPU0')
 else:
  record['gpu0_health_after']=health0();atomic(P/'gpu1_drain_result.json',record)
 print(json.dumps(record,ensure_ascii=False),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['drain','release']);action(a.parse_args().mode)
