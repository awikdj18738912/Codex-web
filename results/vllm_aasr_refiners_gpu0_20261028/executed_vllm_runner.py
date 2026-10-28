"""Actual local vLLM serving benchmark on the frozen AASR Chinese oral inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'results/aasr_bench_refiners_gpu0_20261028'
OUT=ROOT/'results/vllm_aasr_refiners_gpu0_20261028'
PYTHON=ROOT/'.runtime/llamafactory-env/bin/python'
SERVER_PY='/home/aim0/data/conda/envs/agentic-asr/bin/python'
PORT=8026
GATE=Path('/home/aim0/data/ASR/training/refiner_noC02_gate_rbt3_20261027')


def read(path):return [json.loads(s) for s in Path(path).open()]
def write(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('text_manifest.jsonl','libzh_itn_bridge.so'):
        shutil.copy2(SOURCE/name,OUT/name)
    old=json.loads((SOURCE/'protocol.json').read_text())
    write('protocol.json',{'dataset':'AASR-Bench510 Chinese oral text only','gpu':0,
        'hardware':'RTX309024GB','source':str(SOURCE),'source_protocol':old,
        'models':dict(old['models'],lora=str(OUT/'merged_lora_bf16')),
        'base_model':old['base_model'],'prompts':old['prompts'],
        'gate':str(GATE),'gate_device':'GPU0 float32, serialized gate calls under mutex',
        'gate_threshold':.005,'rules':'frozen native v91; source normalization + numeric surface repair/guard; guard fallback included',
        'latency_ids':old['latency_subset_ids'],'concurrencies':[1,8,16],
        'sampling':'temperature0 top_p1 repetition_penalty1 max_tokens1024; same task prompts',
        'precision':'BF16 all three refiners; LoRA merged in BF16 matching prior Transformers run',
        'server':'vLLM OpenAI loopback HTTP, CUDA graphs/compile default ENABLED, no enforce_eager, prefix caching DISABLED',
        'latency':'32 same paired texts, closed-loop1 request; warm wall time rules+gate+HTTP tokenization/prefill/decode+guard, excludes model startup',
        'throughput':'all510 inputs closed-loop max8 and16 concurrent workers; per-request wall time starts after client worker begins; total sweep wall time includes scheduling and client overhead',
        'comparison_limit':'Transformers prior is in-process; vLLM includes local HTTP overhead. Prior batch8 is fixed batch, vLLM8/16 are continuous request concurrency. Input same, scheduling different.',
        'quality':'Compare outputs and raw/rule-normalized references; changed outputs do NOT inherit prior rubric scores. Exact same input+output can reuse prior judge rows for diagnostic only.',
        'manifest_sha256':sha(OUT/'text_manifest.jsonl'),'library_sha256':sha(OUT/'libzh_itn_bridge.so'),
        'script_sha256':sha(Path(__file__))})
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(8)
    model=AutoModelForCausalLM.from_pretrained(old['base_model'],dtype=torch.bfloat16,
        device_map='cuda:0',local_files_only=True).eval()
    model=PeftModel.from_pretrained(model,old['models']['lora'],is_trainable=False).merge_and_unload().eval()
    model.save_pretrained(OUT/'merged_lora_bf16',safe_serialization=True)
    AutoTokenizer.from_pretrained(old['base_model'],local_files_only=True).save_pretrained(OUT/'merged_lora_bf16')
    print('LoRA exported BF16; no original weights modified.',flush=True)


def client(name):
    import torch
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.refinement_guard import restore_rule_numeric_surfaces,reject_reasons
    torch.set_num_threads(8)
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=read(OUT/'text_manifest.jsonl')
    normalizer=ZhITNNormalizer(OUT/'libzh_itn_bridge.so')
    gate,gate_tok=None,None
    if name!='agentic1b':
        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=AutoModel.from_config(AutoConfig.from_pretrained(GATE/'encoder_config',local_files_only=True),add_pooling_layer=False)
                self.head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(self.encoder.config.hidden_size,2))
            def forward(self,**inputs):
                x=self.encoder(**inputs).last_hidden_state
                mask=inputs['attention_mask'].unsqueeze(-1).to(x.dtype)
                return self.head((x*mask).sum(1)/mask.sum(1).clamp_min(1))
        gate=Gate();gate.load_state_dict(torch.load(GATE/'gate.pt',map_location='cpu',weights_only=True))
        gate=gate.to('cuda:0').eval()
        gate_tok=AutoTokenizer.from_pretrained(GATE/'tokenizer',local_files_only=True)
    lock=threading.Lock()
    prompt=protocol['prompts']['agentic1b' if name=='agentic1b' else 'qwen']

    def one(sample,variant):
        start=time.perf_counter()
        raw=sample['text'];is_qwen=name!='agentic1b'
        stage=time.perf_counter()
        baseline=normalizer.normalize(raw).text if is_qwen else raw
        rule_s=time.perf_counter()-stage
        route,score=True,None
        gate_s=0.
        if variant.endswith('_gate'):
            stage=time.perf_counter()
            # Include waiting for the gate mutex in end-to-end request latency.
            with lock:
                enc=gate_tok(raw,return_tensors='pt',truncation=False)
                if enc['input_ids'].shape[1]<=512:
                    with torch.inference_mode():score=gate(**enc.to('cuda:0')).softmax(-1)[0,1].item()
                    route=score>=.005
            gate_s=time.perf_counter()-stage
        output=baseline;raw_output=baseline;model_output=baseline
        tokens=0;finish='gate_skip';request_s=0.;guard_s=0.;fallback=False;repair=False
        if route:
            payload={'model':'aasr-speed-refiner','messages':[{'role':'system','content':prompt},
                {'role':'user','content':baseline}], 'temperature':0,'top_p':1,'max_tokens':1024,
                'repetition_penalty':1.,'seed':20261028,'chat_template_kwargs':{'enable_thinking':False}}
            req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',
                data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'},method='POST')
            stage=time.perf_counter()
            with urllib.request.urlopen(req,timeout=180) as response:body=json.load(response)
            request_s=time.perf_counter()-stage
            raw_output=body['choices'][0]['message']['content'].strip()
            model_output=raw_output.split('<KEY>',1)[0].strip() if not is_qwen else raw_output
            output=model_output;finish=body['choices'][0]['finish_reason'];tokens=body['usage']['completion_tokens']
            if is_qwen:
                stage=time.perf_counter()
                repaired=restore_rule_numeric_surfaces(baseline,output)
                failures=() if repaired is None else tuple(x for x in reject_reasons(baseline,repaired,numeric_rule_mode=True)
                    if x in {'numeric_value_mismatch','numeric_rule_surface_mismatch','partial_numeric_range_conversion'})
                if repaired is None or failures:output=baseline;fallback=True
                else:repair=repaired!=output;output=repaired
                guard_s=time.perf_counter()-stage
        return dict(sample,variant=variant,output=output,model_output=model_output,raw_output=raw_output,
            baseline=baseline,route=route,gate_score=score,total_seconds=time.perf_counter()-start,
            rule_seconds=rule_s,gate_seconds=gate_s,request_seconds=request_s,numeric_guard_seconds=guard_s,
            generated_tokens=tokens,finish_reason=finish,numeric_fallback=fallback,numeric_repaired=repair)

    variants=[name] if name=='agentic1b' else [name,name+'_gate']
    warm=rows[:16]
    for variant in variants:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda sample:one(sample,variant),warm))
    order=[(c,v) for c in (1,8,16) for v in variants]
    random.Random(20261028).shuffle(order)
    with (OUT/(name+'.predictions.jsonl')).open('w') as stream:
        for concurrency,variant in order:
            selected=[r for r in rows if r['id'] in protocol['latency_ids']] if concurrency==1 else rows
            stage=time.perf_counter()
            results=[]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending={pool.submit(one,r,variant):r['id'] for r in selected}
                for future in as_completed(pending):results.append(future.result())
            elapsed=time.perf_counter()-stage
            for result in results:
                result['concurrency']=concurrency
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            with (OUT/'sweeps.jsonl').open('a') as summary:
                summary.write(json.dumps({'variant':variant,'concurrency':concurrency,'n':len(results),
                    'wall_seconds':elapsed,'inputs_per_second':len(results)/elapsed})+'\n')
            write('progress.json',{'phase':'benchmarking','model':name,'last_variant':variant,
                'last_concurrency':concurrency,'last_count':len(results)})
            print(name,variant,'concurrency',concurrency,'n',len(results),'seconds',round(elapsed,3),flush=True)


def orchestrate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    if not OUT.exists():
        subprocess.run([str(PYTHON),'-u',str(Path(__file__)),'prepare'],check=True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',PORT))==0:raise RuntimeError('Port8026occupied; refuse replacing other services')
    for name in ('lora','full','agentic1b'):
        path=protocol['models'][name]
        template=Path(path)/'chat_template.jinja'
        command=[SERVER_PY,'-u','-m','vllm.entrypoints.openai.api_server','--model',path,
            '--served-model-name','aasr-speed-refiner','--host','127.0.0.1','--port',str(PORT),
            '--dtype','bfloat16','--max-model-len','4096','--max-num-seqs','16',
            '--max-num-batched-tokens','4096','--gpu-memory-utilization','0.50',
            '--generation-config','vllm','--no-enable-prefix-caching','--safetensors-load-strategy','prefetch']
        if template.exists():command+=['--chat-template',str(template)]
        write(name+'.server_command.json',command)
        write('progress.json',{'phase':'starting_server','model':name})
        with (OUT/(name+'.server.log')).open('w') as log:
            server=subprocess.Popen(command,env=dict(os.environ,VLLM_USE_FLASHINFER_SAMPLER='0',
                CUDA_HOME='/home/aim0/data/conda/envs/agentic-asr/lib/python3.12/site-packages/nvidia/cu13'),
                stdout=log,stderr=log,start_new_session=True)
            try:
                deadline=time.monotonic()+1800
                while time.monotonic()<deadline:
                    if server.poll() is not None:raise RuntimeError(f'{name}serverexit{server.returncode}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2) as response:
                            if response.status==200:break
                    except OSError:pass
                    time.sleep(3)
                else:raise RuntimeError(name+'serverstartup timeout')
                subprocess.run([str(PYTHON),'-u',str(Path(__file__)),'client','--model-name',name],check=True)
            finally:
                import signal
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL);server.wait()
        time.sleep(3)
    write('progress.json',{'phase':'complete','models':3,'pipelines':5,'concurrencies':[1,8,16],
        'gpu0_servers_stopped':True})
    subprocess.run([str(PYTHON),str(ROOT/'experiments/scripts/report_vllm_aasr_benchmark.py')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','client','run']);p.add_argument('--model-name')
    args=p.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='client':client(args.model_name)
        else:orchestrate()
    except Exception as error:
        if OUT.exists():write('failure.json',{'error':str(error),'mode':args.mode,'model':args.model_name})
        raise
