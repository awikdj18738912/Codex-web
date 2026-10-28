import time,json,subprocess,sys
from pathlib import Path
from urllib.request import urlopen
P=Path(__file__).resolve().parent
for _ in range(180):
 try:
  for port in [8012,8013]:
   with urlopen(f'http://127.0.0.1:{port}/health',timeout=2)as r:assert r.status==200
  break
 except Exception:time.sleep(10)
else:raise RuntimeError('teacher readiness timeout')
(P/'concurrency_trial/status.json').write_text(json.dumps({'stage':'baseline_eager_servers_restored','servers_healthy':True,'prior_batch_records_preserved':165}))
with (P/'run.log').open('a')as f:subprocess.Popen([sys.executable,str(P/'run_batches.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
with (P/'export.log').open('a')as f:subprocess.Popen([sys.executable,str(P/'export_when_done.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(P/'recovery_started.json').write_text(json.dumps({'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'resumes_batch0':True,'prior_rows_preserved':165}))
