import json,os,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 except (OSError,UnicodeError):continue
 if 'system.web_app' in a and '--port' in a and a[a.index('--port')+1]=='8081':
  env={x.split(b'=',1)[0].decode():x.split(b'=',1)[1].decode() for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x};break
else:raise RuntimeError('live worker absent')
a=[a[0],str(OUT/'experiment_web.py'),*a[3:]]
a[a.index('--port')+1]='18187';a[a.index('--output')+1]=str(OUT/'service_logs/session.jsonl')
(OUT/'service_logs').mkdir(exist_ok=True)
proc=subprocess.Popen(a,cwd=ROOT,env=env,stdout=(OUT/'experiment_web.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'experiment_worker.json').write_text(json.dumps({'pid':proc.pid,'argv':a,'cuda_visible_devices':env.get('CUDA_VISIBLE_DEVICES')},ensure_ascii=False,indent=2)+'\n')
print('STARTED',proc.pid,flush=True)
for _ in range(180):
 if proc.poll() is not None:raise RuntimeError('worker exited')
 try:
  r=json.load(urllib.request.urlopen('http://127.0.0.1:18187/health',timeout=2))
  if r.get('ok'):print('READY',flush=True);break
 except Exception:pass
 time.sleep(.5)
else:raise RuntimeError('worker timeout')
