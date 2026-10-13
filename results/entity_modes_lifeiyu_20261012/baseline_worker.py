"""Temporary no-dictionary Web worker cloned from live settings; no production edits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:args=[a.decode() for a in (entry/'cmdline').read_bytes().split(b'\0') if a]
 except (OSError,UnicodeError):continue
 if 'system.web_app' in args and '--port' in args and args[args.index('--port')+1]=='8081':
  original_args=args.copy()
  env={s.split(b'=',1)[0].decode():s.split(b'=',1)[1].decode() for s in (entry/'environ').read_bytes().split(b'\0') if b'=' in s}
  break
else:raise RuntimeError('live worker not found')
i=args.index('--entity-db');del args[i:i+2]
if '--enable-entity-audio-repair' in args:args.remove('--enable-entity-audio-repair')
if '--enable-entity-asr-context' in args:args.remove('--enable-entity-asr-context')
args[args.index('--entity-fuzzy-mode')+1]='off'
args[args.index('--port')+1]='18081'
args[args.index('--output')+1]=str(OUT/'baseline_service_logs/session.jsonl')
(OUT/'baseline_service_logs').mkdir(exist_ok=True)
log=(OUT/'baseline_worker.log').open('ab')
worker=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'baseline_worker.json').write_text(json.dumps({'pid':worker.pid,'port':18081,'argv':args,'original_argv':original_args,'cuda_visible_devices':env.get('CUDA_VISIBLE_DEVICES'),'only_setting_differences':'entity database absent, entity fuzzy off; separate port and experiment logs'},ensure_ascii=False,indent=2)+'\n')
print('STARTED',worker.pid,flush=True)
deadline=time.monotonic()+90
while time.monotonic()<deadline:
 if worker.poll() is not None:raise RuntimeError('baseline worker exited')
 try:
  health=json.load(urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=2))
  if health.get('ok'):
   assert not health['entity_db'];(OUT/'baseline_health.json').write_text(json.dumps(health,ensure_ascii=False,indent=2)+'\n');print('READY',flush=True);break
 except Exception:pass
 time.sleep(.5)
else:raise RuntimeError('baseline worker health timeout')
