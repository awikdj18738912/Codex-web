"""Load the validated entity change, preserving each existing Web argv/env."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('entity_web_reload_helpers',ROOT/'results/entity_local_replay_20261012/reload_web.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
records=[]
for port in (8081,8051):
 pid,args,env=helpers.process_for(port)
 old_health=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=5))
 os.kill(pid,signal.SIGTERM)
 deadline=time.monotonic()+15
 while Path(f'/proc/{pid}').exists() and time.monotonic()<deadline:
  try:
   if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
  except OSError:break
  time.sleep(.2)
 if Path(f'/proc/{pid}').exists():
  try:
   assert Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z','old worker failed to exit'
  except FileNotFoundError:pass
 log=(OUT/f'deployed_{port}.log').open('ab')
 process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  if process.poll() is not None:raise RuntimeError(f'worker exited: {port}')
  try:
   health=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2))
   if health.get('ok'):break
  except Exception:pass
  time.sleep(.5)
 else:raise RuntimeError(f'health timeout: {port}')
 record={'port':port,'previous_pid':pid,'pid':process.pid,'argv_preserved':True,'env_preserved':True,
         'previous_health':old_health,'health':health,'entity_audio_ready':helpers.probe(port),
         'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('system/entity_pipeline.py','system/entity_prose_projection.py')}}
 records.append(record)
 (OUT/'deployment.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 print('READY',port,process.pid,flush=True)
