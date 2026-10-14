import hashlib,json,os,shutil,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
if not (OUT/'source_snapshot/system').exists():shutil.copytree(ROOT/'system',OUT/'source_snapshot/system',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 except (OSError,UnicodeError):continue
 if 'system.web_app' in a and '--port' in a and a[a.index('--port')+1]=='8081':
  env={x.split(b'=',1)[0].decode():x.split(b'=',1)[1].decode() for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x};break
else:raise RuntimeError('live Web not found')
original=a.copy();a=[a[0],str(OUT/'experiment_worker.py'),*a[3:]]
metadata=json.loads((OUT/'library_metadata.json').read_text())['libraries'];base=metadata[next(iter(metadata))]['snapshot_path']
a[a.index('--entity-db')+1]=base
a[a.index('--output')+1]=str(OUT/'service_logs/session.jsonl')
with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
a[a.index('--port')+1]=str(port);(OUT/'service_logs').mkdir(exist_ok=True)
env['PYTHONPATH']=str(OUT/'source_snapshot')
worker=subprocess.Popen(a,cwd=ROOT,env=env,stdout=(OUT/'worker.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':worker.pid,'port':port,'argv':a,'original_argv':original,'cuda_visible_devices':env.get('CUDA_VISIBLE_DEVICES'),'source_hashes':{str(p.relative_to(OUT/'source_snapshot')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'source_snapshot').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.html')}}
(OUT/'worker.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print('STARTED',worker.pid,'PORT',port,flush=True)
for _ in range(180):
 if worker.poll() is not None:raise RuntimeError('worker exited')
 try:
  h=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2))
  if h.get('ok'):
   record['health']=h;(OUT/'worker.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print('READY',flush=True);break
 except Exception:pass
 time.sleep(.5)
else:raise RuntimeError('worker timeout')
