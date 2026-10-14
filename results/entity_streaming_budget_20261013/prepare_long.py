import math, hashlib, json, sqlite3
from pathlib import Path
import numpy as np
import soundfile as sf
out=Path(__file__).resolve().parent
meta=json.loads((out/'long_inputs.json').read_text());li=meta['li_feiyu_web_source']
wave,rate=sf.read(li['audio_path'],dtype='float32',always_2d=True);mono=wave.mean(axis=1,dtype=np.float32)
ratio=rate/16000;pos=np.arange(math.floor(len(mono)/ratio),dtype=np.float64)*ratio
left=np.floor(pos).astype(np.int64);right=np.minimum(left+1,len(mono)-1)
pcm=(mono[left]*(1-(pos-left))+mono[right]*(pos-left)).astype('<f4').tobytes()
assert hashlib.sha256(pcm).hexdigest()==li['pcm_sha256_expected']
path=out/'li_feiyu.f32le';path.write_bytes(pcm)
db=out/'library_snapshots/li_feiyu.db'
source=sqlite3.connect('file:'+li['library_db_path']+'?mode=ro',uri=True);target=sqlite3.connect(db);source.backup(target);target.close();source.close()
libraries=json.loads((out/'library_metadata.json').read_text());libraries['libraries']['li_feiyu']={'snapshot_path':str(db),'sha256':hashlib.sha256(db.read_bytes()).hexdigest()};(out/'library_metadata.json').write_text(json.dumps(libraries,ensure_ascii=False,indent=2)+'\n')
samples=[]
for row in meta['sparse_development_source']['inputs']:
 samples.append(dict(id=row['id'],corpus='concatenated_stress',stratum='long',split='long',library_key=row['library_key'],pcm_path=row['pcm_path'],pcm_sha256=row['pcm_sha256_verified'],duration_seconds=row['duration_seconds_from_pcm_bytes'],selection_hash=hashlib.sha256(row['id'].encode()).hexdigest()))
samples.append(dict(id=li['id'],corpus='web_development_no_full_reference',stratum='long',split='long',library_key='li_feiyu',pcm_path=str(path),pcm_sha256=li['pcm_sha256_expected'],duration_seconds=len(pcm)/64000,selection_hash=hashlib.sha256(li['id'].encode()).hexdigest()))
(out/'long_test_plan.json').write_text(json.dumps({'samples':samples},ensure_ascii=False,indent=2)+'\n')
code=(out/'collect.py').read_text().replace("frozen_test_plan.json","long_test_plan.json").replace("'max_outstanding':4","'max_outstanding':1").replace("'artificial_delay':0","'artificial_delay':.08").replace('max(0,n-4)','max(0,n-1)').replace('max_pending<=4','max_pending<=1').replace("choices=['development','confirmation'],default='development'","choices=['long'],default='long'")
code=code.replace('max_pending=max(max_pending,n-acked)','max_pending=max(max_pending,n-acked)\n                        await wait_ack(n);await asyncio.sleep(.08)')
(out/'collect_long.py').write_text(code)
print('PREPARED',len(samples),hashlib.sha256(pcm).hexdigest())
