from pathlib import Path
from urllib.request import urlopen,Request
import json,time
P=Path(__file__).resolve().parent
for _ in range(120):
 try:
  with urlopen('http://127.0.0.1:8013/health',timeout=3)as r:assert r.status==200
  req=Request('http://127.0.0.1:8013/v1/chat/completions',data=json.dumps({'model':'qwen38-teacher','messages':[{'role':'user','content':'只输出好'}],'temperature':0,'max_tokens':3,'chat_template_kwargs':{'enable_thinking':False}}).encode(),headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=30)as r:result=json.load(r)
  assert result.get('choices')
  policy=json.loads((P/'gpu_routing.json').read_text());policy.update(gpu1_enabled=True,reenabled_at=time.strftime('%Y-%m-%d %H:%M:%S'),release_schedule_completed_and_revoked=True)
  tmp=P/'gpu_routing.tmp';tmp.write_text(json.dumps(policy,ensure_ascii=False,indent=2));tmp.replace(P/'gpu_routing.json')
  (P/'gpu1_reenabled.json').write_text(json.dumps({'at':time.strftime('%Y-%m-%d %H:%M:%S'),'gpu1_health':True,'inference_probe_passed':True,'gpu0_service_restarted':False,'routing':'both GPUs, 16 concurrent model calls per GPU'},ensure_ascii=False,indent=2))
  print('GPU1 inference ready; dual-GPU routing enabled',flush=True);break
 except Exception:time.sleep(5)
else:raise RuntimeError('GPU1 startup timed out; routing retained GPU0-only')
