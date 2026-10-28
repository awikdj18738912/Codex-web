from pathlib import Path
import json,time,os,signal
from urllib.request import urlopen,Request
P=Path(__file__).resolve().parent
info=json.loads((P/'io_wait_client_pause.json').read_text())
stable=0
while True:
 try:
  req=Request('http://127.0.0.1:8012/tokenize',data=json.dumps({'model':'qwen38-teacher','prompt':'你好'}).encode(),headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=5)as f:tokens=json.load(f)
  req=Request('http://127.0.0.1:8012/v1/chat/completions',data=json.dumps({'model':'qwen38-teacher','messages':[{'role':'user','content':'只输出好'}],'max_tokens':2,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode(),headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=15)as f:result=json.load(f)
  if not result.get('choices'):raise RuntimeError('no response')
  stable+=1
  if stable<3:
   time.sleep(15);continue
  for pid in info['paused_client_pids']:
   try:
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'run_batches.py'in cmd or b'run_specialists.py'in cmd:os.kill(pid,signal.SIGCONT)
   except FileNotFoundError:pass
  (P/'io_wait_resumed.json').write_text(json.dumps({'at':time.strftime('%Y-%m-%d %H:%M:%S'),'gpu0_service_restarted':False}));break
 except Exception:
  stable=0;time.sleep(30)
