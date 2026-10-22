"""Local paragraph synthesis, resume from verified per-paragraph metadata."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,time,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import soundfile as sf
from pipeline.src.speech.qwen_tts_service import QwenTTSService
root=Path(__file__).resolve().parent
source=Path('results/immune_generated_text_20261016/reading.txt')
original=source.read_text()
# Speak symbol names explicitly; keep the canonical reference separate.
pronunciations={'γδ T细胞':'伽马德尔塔T细胞','αβ T细胞':'阿尔法贝塔T细胞',
 'ILC1':'I L C 一','ILC2':'I L C 二','ILC3':'I L C 三',
 'B1细胞':'B一细胞','B2细胞':'B二细胞','FAS':'F A S',
 'ITAM':'I T A M','ITIM':'I T I M','HLDA':'H L D A','HLA':'H L A',
 'CD95':'C D 九十五','MAC':'M A C'}
paragraphs=[s.strip() for s in original.split('\n\n') if s.strip()]
spoken=[]
for s in paragraphs:
 for a,b in pronunciations.items():s=s.replace(a,b)
 spoken.append(s)
(root/'spoken_text.txt').write_text('\n\n'.join(spoken)+'\n')
(root/'reference_text.txt').write_text(original)
config=dict(model='Qwen3-TTS-12Hz-0.6B-CustomVoice',speaker='Vivian',device='cuda:1',seed_base=20261016,max_new_tokens=2048,source_sha256=hashlib.sha256(original.encode()).hexdigest(),pronunciations=pronunciations,paragraph_count=len(spoken),gap_seconds=.5)
(root/'configuration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
service=QwenTTSService(device='cuda:1',default_speaker='Vivian')
records=[];began=time.perf_counter()
for i,(reference,text) in enumerate(zip(paragraphs,spoken),1):
 path=root/f'part_{i:02d}.wav';meta=root/f'part_{i:02d}.json';digest=hashlib.sha256(text.encode()).hexdigest()
 if meta.exists() and path.exists():
  record=json.loads(meta.read_text());assert record['spoken_sha256']==digest
  print('CACHED',i,flush=True)
 else:
  import torch
  torch.manual_seed(config['seed_base']+i)
  print('START',i,'/',len(spoken),'chars',len(text),flush=True)
  record=service.synthesize(text=text,output_path=path,language='Chinese',max_new_tokens=2048)
  wave,rate=sf.read(path)
  assert np.isfinite(wave).all() and np.max(np.abs(wave))>0.001
  record.update(index=i,reference=reference,spoken=text,spoken_sha256=digest,audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),peak=float(np.max(np.abs(wave))),rms=float(np.sqrt(np.mean(wave**2))))
  meta.write_text(json.dumps(record,ensure_ascii=False,indent=2))
  print('DONE',i,'seconds',round(record['duration_seconds'],2),'compute',round(record['elapsed_seconds'],2),flush=True)
 records.append(record)
 rate=sf.info(records[0]['audio_path']).samplerate
 # File kept up to date for progress; final waveform assembled only at end.
 (root/'progress.json').write_text(json.dumps(dict(completed=len(records),total=len(spoken),audio_seconds=sum(x['duration_seconds'] for x in records),wall_seconds=time.perf_counter()-began),indent=2))
joined=[];offset=0
for i,r in enumerate(records):
 wave,sr=sf.read(r['audio_path'],dtype='float32');assert sr==rate and wave.ndim==1
 r['start_seconds']=offset;r['end_seconds']=offset+len(wave)/rate
 joined.append(wave);offset=r['end_seconds']
 if i<len(records)-1:
  gap=np.zeros(int(.5*rate),dtype=np.float32);joined.append(gap);offset+=len(gap)/rate
output=root/'immune_31_entities.wav';sf.write(output,np.concatenate(joined),rate,subtype='PCM_16')
summary=dict(configuration=config,paragraphs=records,final_audio=str(output),sample_rate=rate,duration_seconds=offset,wall_seconds=time.perf_counter()-began,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),scope='Synthetic speech from user-approved authored text, not independent natural audio evaluation; pronunciations not human-adjudicated')
(root/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('COMPLETE',str(output),'duration',offset,flush=True)
