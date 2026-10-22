import os,json,time
from pathlib import Path
from huggingface_hub import snapshot_download
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
root=Path(__file__).resolve().parent
model_dir=Path('/home/aim0/data/ASR/models/Qwen3.8-27B-AWQ-INT4')
started=time.time()
provenance={'repo_id':'cyankiwi/Qwen3.8-27B-AWQ-INT4','revision':'e3a700906e6a73ba4290b1873dc2ba8198527ab5','source_model':'Qwen/Qwen3.8-27B','local_dir':str(model_dir),'quantization':'community AWQ INT4 mixed precision','started_epoch':started}
(root/'download_provenance.json').write_text(json.dumps(provenance,indent=2))
print('Downloading pinned model to',model_dir,flush=True)
snapshot_download(provenance['repo_id'],revision=provenance['revision'],local_dir=str(model_dir),max_workers=4,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','README.md'])
provenance['finished_epoch']=time.time();provenance['wall_seconds']=round(time.time()-started,1)
(root/'download_provenance.json').write_text(json.dumps(provenance,indent=2))
print('Download complete',flush=True)
