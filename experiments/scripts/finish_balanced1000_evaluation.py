"""One-off local pipeline: wait for GPU0 timing, then rubric judge, then reports."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/vllm_synthetic_balanced1000_20261028'
BENCH_UNIT='vllm-balanced1000-20261028.service'
TEACHER_UNIT='balanced1000-rubric-teacher-20261028.service'
PYTHON=ROOT/'.runtime/llamafactory-env/bin/python'


def active(unit):
    return subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0


def status(phase,**extra):
    path=OUT/'evaluation_pipeline_progress.json'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase,updated_unix=time.time(),**extra),ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def main():
    status('waiting_for_timing_benchmark')
    while active(BENCH_UNIT):time.sleep(5)
    assert json.loads((OUT/'progress.json').read_text())['phase']=='complete', 'Timing benchmark did not finish'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    subprocess.run([str(PYTHON),'experiments/scripts/report_balanced1000_vllm.py'],cwd=ROOT,check=True)
    # A separate loopback port prevents previous data generators from using this judge.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8024))==0:
            raise RuntimeError('Judge port8024 already in use; refusing to replace any service')
    assert not active(TEACHER_UNIT)
    command=['systemd-run','--user','--unit='+TEACHER_UNIT,
        '--property=WorkingDirectory='+str(ROOT),
        '--property=StandardOutput=append:'+str(OUT/'judge_server.log'),
        '--property=StandardError=append:'+str(OUT/'judge_server.log'),
        '--setenv=CUDA_VISIBLE_DEVICES=0','--setenv=HF_HUB_OFFLINE=1',
        '--setenv=TOKENIZERS_PARALLELISM=false','--setenv=OMP_NUM_THREADS=8',
        '--setenv=VLLM_USE_FLASHINFER_SAMPLER=0',
        '--setenv=CUDA_HOME=/home/aim0/data/conda/envs/agentic-asr/lib/python3.12/site-packages/nvidia/cu13',
        '/home/aim0/data/conda/envs/agentic-asr/bin/python','-u','-m','vllm.entrypoints.openai.api_server',
        '--model','/home/aim0/data/ASR/models/gemma-4-31B-it-qat-w4a16-ct',
        '--served-model-name','aasr-rubric-gemma4','--host','127.0.0.1','--port','8024',
        '--chat-template','/home/aim0/data/ASR/models/gemma-4-31B-it-qat-w4a16-ct/chat_template.jinja',
        '--safetensors-load-strategy','prefetch','--language-model-only','--tensor-parallel-size','1',
        '--dtype','bfloat16','--max-model-len','3072','--max-num-seqs','16',
        '--max-num-batched-tokens','1024','--gpu-memory-utilization','0.95',
        '--kv-cache-memory-bytes','3221225472','--reasoning-parser','gemma4',
        '--enforce-eager','--disable-custom-all-reduce']
    try:
        status('starting_local_rubric_judge')
        subprocess.run(command,check=True)
        deadline=time.monotonic()+2400
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8024/health',timeout=3) as response:
                    if response.status==200:break
            except OSError:pass
            if not active(TEACHER_UNIT):raise RuntimeError('Judge server exited; see judge_server.log')
            time.sleep(5)
        else:raise RuntimeError('Judge server did not become ready within40minutes')
        status('judging_custom_rubric_with_local_quantized_model')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        with (OUT/'judge_run.log').open('a') as log:
            for attempt in range(3):
                result=subprocess.run(['/home/aim0/data/conda/envs/agentic-asr/bin/python','-u','experiments/scripts/judge_balanced1000_local.py'],
                    cwd=ROOT,env=env,stdout=log,stderr=log)
                if result.returncode==0:
                    break
                if attempt==2:
                    result.check_returncode()
                status('resuming_failed_rubric_requests',attempt=attempt+2)
                time.sleep(5)
    finally:
        subprocess.run(['systemctl','--user','stop',TEACHER_UNIT],check=False)
    subprocess.run([str(PYTHON),'experiments/scripts/report_balanced1000_vllm.py'],cwd=ROOT,check=True)
    status('complete',gpu0_service_stopped=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error))
        raise
