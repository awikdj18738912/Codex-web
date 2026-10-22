"""Restart only the existing Web processes, preserving their argv/environment."""
import hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
worker=json.loads((OUT/'worker.json').read_text())
try:os.kill(worker['pid'],signal.SIGTERM)
except ProcessLookupError:pass
records=[]
for port in (8081,8051):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except (OSError,UnicodeError):continue
  if 'system.web_app' in argv and '--port' in argv and argv[argv.index('--port')+1]==str(port):
   env={x.split(b'=',1)[0].decode():x.split(b'=',1)[1].decode() for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x};old=int(p.name);break
 else:raise RuntimeError(f'original Web {port} not found')
 before=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3))
 os.kill(old,signal.SIGTERM)
 for _ in range(200):
  if not Path(f'/proc/{old}').exists():break
  time.sleep(.1)
 else:raise RuntimeError(f'Web {port} has not exited gracefully')
 new=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=(OUT/f'deployment_{port}.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
 new_pid=new.pid;print('RESTARTED',port,'PID',new_pid,flush=True)
 for _ in range(240):
  if new.poll() is not None:raise RuntimeError(f'Web {port} startup failed')
  try:
   after=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2))
   if after.get('ok'):break
  except Exception:pass
  time.sleep(.5)
 else:raise RuntimeError(f'Web {port} health timed out')
 for key in ('entity_db','entity_asr_context_enabled','entity_audio_repair_enabled','refinement_gate_mode','window_pipeline','final_refinement_mode','numeric_backend','zh_itn_rule_pack'):
  assert before.get(key)==after.get(key),(port,key)
 for key in ('mode','ready','model_dir','model_sha256','threshold','load_error','scope','calibrated'):
  assert before.get('learned_window_gate',{}).get(key)==after.get('learned_window_gate',{}).get(key),(port,key)
 assert after.get('confidence_supplement',{}).get('mode','off')=='off'
 records.append(dict(port=port,old_pid=old,new_pid=new_pid,argv=argv,health=after))
 (OUT/'web_rollout.json').write_text(json.dumps(dict(services=records,private_worker_stopped=worker['pid'],verification_note='Gate configuration compared separately from request counters, which reset after restart.',production_hashes={str(ROOT/'system'/n):hashlib.sha256((ROOT/'system'/n).read_bytes()).hexdigest() for n in ['entity_audio_repair.py','entity_streaming_live.py']}),ensure_ascii=False,indent=2)+'\n')
 print('HEALTHY',port,flush=True)
