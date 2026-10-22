"""Resume paragraph synthesis with four texts per GPU batch."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import soundfile as sf
import torch
from pipeline.src.speech.qwen_tts_service import QwenTTSService
root=Path(__file__).resolve().parent
config=json.loads((root/'configuration.json').read_text())
config['batch_size']=4
config['generation_note']='First completed paragraphs from single generation retained; remaining paragraphs use batched sampling with fixed batch seed.'
(root/'configuration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
paragraphs=[t.strip() for t in (root/'reference_text.txt').read_text().split('\n\n') if t.strip()]
spoken=[t.strip() for t in (root/'spoken_text.txt').read_text().split('\n\n') if t.strip()]
service=QwenTTSService(device='cuda:1',default_speaker='Vivian')
records={};pending=[];began=time.perf_counter()
for i,text in enumerate(spoken,1):
 meta=root/f'part_{i:02d}.json';path=root/f'part_{i:02d}.wav'
 if path.exists() and meta.exists():
  record=json.loads(meta.read_text());assert record['spoken_sha256']==hashlib.sha256(text.encode()).hexdigest()
  records[i]=record
 else:pending.append(i)
print('RESUME',len(records),'cached,',len(pending),'remaining',flush=True)
model=service._load_model()
for offset in range(0,len(pending),4):
 indexes=pending[offset:offset+4]
 torch.manual_seed(20261016+indexes[0])
 print('BATCH_START',indexes,flush=True);t=time.perf_counter()
 wavs,rate=model.generate_custom_voice(text=[spoken[i-1] for i in indexes],language='Chinese',speaker='Vivian',max_new_tokens=2048)
 elapsed=time.perf_counter()-t
 assert len(wavs)==len(indexes)
 for i,wave in zip(indexes,wavs):
  path=root/f'part_{i:02d}.wav';meta=root/f'part_{i:02d}.json'
  wave=np.asarray(wave,dtype=np.float32)
  assert len(wave)>rate and np.isfinite(wave).all() and np.max(np.abs(wave))>.001
  sf.write(path,wave,rate,subtype='PCM_16')
  record=dict(index=i,audio_path=str(path),sample_rate=rate,duration_seconds=len(wave)/rate,speaker='Vivian',backend=config['model'],device='cuda:1',batch_indexes=indexes,batch_elapsed_seconds=elapsed,reference=paragraphs[i-1],spoken=spoken[i-1],spoken_sha256=hashlib.sha256(spoken[i-1].encode()).hexdigest(),audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),peak=float(np.max(np.abs(wave))),rms=float(np.sqrt(np.mean(wave**2))))
  meta.write_text(json.dumps(record,ensure_ascii=False,indent=2));records[i]=record
  print('DONE',i,'seconds',round(record['duration_seconds'],2),'batch_compute',round(elapsed,2),flush=True)
 (root/'progress.json').write_text(json.dumps(dict(completed=len(records),total=len(spoken),audio_seconds=sum(x['duration_seconds'] for x in records.values()),wall_seconds=time.perf_counter()-began),indent=2))
joined=[];offset=0;rows=[]
for i in range(1,len(spoken)+1):
 r=records[i];wave,sr=sf.read(r['audio_path'],dtype='float32');assert sr==rate and wave.ndim==1
 r['start_seconds']=offset;r['end_seconds']=offset+len(wave)/rate;rows.append(r)
 joined.append(wave);offset=r['end_seconds']
 if i<len(spoken):
  gap=np.zeros(int(.5*rate),dtype=np.float32);joined.append(gap);offset+=len(gap)/rate
output=root/'immune_31_entities.wav';sf.write(output,np.concatenate(joined),rate,subtype='PCM_16')
summary=dict(configuration=config,paragraphs=rows,final_audio=str(output),sample_rate=rate,duration_seconds=offset,batch_runner_wall_seconds=time.perf_counter()-began,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),scope='Synthetic speech from authored text; pronunciations not human-adjudicated; not an independent natural-audio test')
(root/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('COMPLETE',output,'duration',offset,flush=True)
