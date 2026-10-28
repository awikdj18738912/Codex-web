import sys,json,time,subprocess
from pathlib import Path
from urllib.request import urlopen
P=Path(__file__).resolve().parent
for attempt in range(120):
 try:
  enabled=json.loads((P/'gpu_routing.json').read_text()).get('gpu1_enabled') is True
  for port in ([8012,8013] if enabled else [8012]):
   with urlopen(f'http://127.0.0.1:{port}/health',timeout=2)as r:assert r.status==200
  break
 except Exception:time.sleep(5)
else:raise RuntimeError('teacher startup timeout')
m=json.loads((P/'manifest.json').read_text());started=time.monotonic()
for index,sample in enumerate(sorted((P/'inputs').glob('*.jsonl'))):
 out=P/'outputs'/sample.name
 if not out.with_suffix('.complete.json').exists():
  with out.with_suffix('.log').open('a')as f:
   worker=P/'run_specialists.py'
   if index>0 and (P/'selected_scheduling.json').exists():worker=Path(json.loads((P/'selected_scheduling.json').read_text())['worker'])
   proc=subprocess.run([sys.executable,str(worker),'--sample',str(sample),'--output',str(out),'--repair'],stdout=f,stderr=subprocess.STDOUT)
  if proc.returncode:raise RuntimeError(f'batch failed {sample.name}: {proc.returncode}')
 progress={'completed_batches':index+1,'total_batches':m['batches'],'reused_records':m['reused'],'completed_new_records':sum(json.loads(x.read_text())['completed']for x in(P/'outputs').glob('*.complete.json')),'elapsed_seconds':round(time.monotonic()-started,1),'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
 (P/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2));print(progress,flush=True)
(P/'all_batches_complete.json').write_text(json.dumps({'completed_at':time.strftime('%Y-%m-%d %H:%M:%S'),'next':'export new version and independently audit sample before training'}))

