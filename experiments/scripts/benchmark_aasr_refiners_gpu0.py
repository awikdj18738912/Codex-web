"""AASR-Bench Chinese ORAL text-only refiner benchmark, with original rubric retained."""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'results/aasr_bench_refiners_gpu0_20261028'
BASE = '/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct'
TRAIN = Path('/home/aim0/data/ASR/training')
GATE = TRAIN/'refiner_noC02_gate_rbt3_20261027'
MODELS = {
 'lora':TRAIN/'qwen25_05b_refiner_noC02_lora_20261027/checkpoint-990',
 'full':TRAIN/'qwen25_05b_refiner_noC02_full_20261027',
 'agentic1b':Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')}
VARIANTS=['lora','lora_gate','full','full_gate','agentic1b']

def read(path): return [json.loads(s) for s in Path(path).open()]
def write(name,value):
 p=OUT/name
 tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 tmp.replace(p)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
 from system.zh_itn_bridge import DEFAULT_LIBRARY, ZhITNNormalizer
 OUT.mkdir(parents=True,exist_ok=False)
 shutil.copy2(DEFAULT_LIBRARY,OUT/'libzh_itn_bridge.so')
 normalizer=ZhITNNormalizer(OUT/'libzh_itn_bridge.so')
 source=Path('/home/aim0/data/datasets/ASR/AASR-Bench/data/benchmark.jsonl')
 rubric=source.parent/'rubric.json'
 records=[r for r in read(source) if r['language']=='zh']
 assert len(records)==510 and len({r['source_record_id'] for r in records})==510
 random.Random(20261028).shuffle(records)
 shutil.copy2(source,OUT/'original_benchmark.jsonl')
 shutil.copy2(rubric,OUT/'original_rubric.json')
 rows=[]
 for r in records:
  rows.append({'id':r['source_record_id'],'source_record_id':r['source_record_id'],
   'source':'AASR-Bench','category':r['scene'],'scene':r['scene'],'text':r['oral'],
   'oral':r['oral'],'clean':r['clean'],'original_reference':r['clean'],
   'reference':normalizer.normalize(r['clean']).text,
   'rule_baseline':normalizer.normalize(r['oral']).text,'label':int(r['oral']!=r['clean'])})
 (OUT/'text_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 # Preserve approximately natural 85/15 modified/clean proportion for latency.
 latency=[r['id'] for r in rows if r['label']==0][:5]+[r['id'] for r in rows if r['label']==1][:27]
 previous=json.loads((ROOT/'results/refiner_rules_text_benchmark_20261028/protocol.json').read_text())
 previous.update({'mode':'AASR-Bench_Chinese_oral_text_only','quality_count':510,
  'dataset':'Andrew0425/AASR-Bench','benchmark_source':str(source),'rubric_source':str(rubric),
  'benchmark_sha256':sha(source),'rubric_sha256':sha(rubric),
  'latency_subset_ids':latency,'latency_count':32,'latency_label_balance':[5,27],
  'quality_label_balance':[sum(not r['label'] for r in rows),sum(r['label'] for r in rows)],
  'weak_label_balance':None,'numeric_rule_library':str(OUT/'libzh_itn_bridge.so'),
  'library_sha256':sha(OUT/'libzh_itn_bridge.so'),'manifest_sha256':sha(OUT/'text_manifest.jsonl'),
  'script_sha256':sha(Path(__file__)),
  'latency_mode':'32 paired AASR inputs (5 unchanged,27 changed) batch1, warm, full text pipeline latency',
  'throughput_mode':'All510 Chinese inputs with their natural distribution (76 unchanged,434 different), fixed batches8 (last batch6); batch completion time and total throughput, not individual latency',
  'quality':'Official AASR oral as input, clean as reference, original rubric frozen. Direct oral input bypasses ASR and is a text-only adaptation, not the paper full audio pipeline. References never enter model prompts. Raw and rule-normalized exact agreement are descriptive, not official rubric score.'})
 write('protocol.json',previous)
 print('Prepared AASR-Bench:510 Chinese,76unchanged/434different,rule',normalizer.rule_pack,flush=True)

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU0 only')
    import torch
    import transformers
    from transformers import AutoModel,AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.refinement_guard import restore_rule_numeric_surfaces,reject_reasons
    torch.set_num_threads(8)
    torch.manual_seed(20261028)
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'libzh_itn_bridge.so')==protocol['library_sha256']
    normalizer=ZhITNNormalizer(OUT/'libzh_itn_bridge.so')
    rows=read(OUT/'text_manifest.jsonl')
    loads={}
    models,tokenizers={},{}
    for name, path in MODELS.items():
        start = time.perf_counter()
        source = BASE if name == 'lora' else str(path)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16,
            device_map='cuda:0', local_files_only=True, attn_implementation='sdpa').eval()
        if name == 'lora':
            model = PeftModel.from_pretrained(model, str(path), is_trainable=False).eval()
            model = model.merge_and_unload().eval()
        model.config.use_cache = True
        models[name], tokenizers[name] = model, tokenizer
        loads[name] = {'load_seconds': time.perf_counter()-start,
                       'parameters': sum(p.numel() for p in model.parameters())}
        print('Loaded', name, loads[name], flush=True)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_config(
                    transformers.AutoConfig.from_pretrained(str(GATE/'encoder_config'), local_files_only=True),
                    add_pooling_layer=False)
            self.head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(self.encoder.config.hidden_size, 2))

        def forward(self, **inputs):
            x = self.encoder(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).to(x.dtype)
            return self.head((x*mask).sum(1)/mask.sum(1).clamp_min(1))

    gate = Gate()
    gate.load_state_dict(torch.load(GATE/'gate.pt', map_location='cpu', weights_only=True))
    gate = gate.to('cuda:0').eval()
    gate_tok = AutoTokenizer.from_pretrained(str(GATE/'tokenizer'), local_files_only=True)

    def process(texts, variant):
        torch.cuda.synchronize()
        start = time.perf_counter()
        is_qwen = variant != 'agentic1b'
        rule_start = time.perf_counter()
        normalizations = [normalizer.normalize(text) for text in texts] if is_qwen else []
        baselines = [n.text for n in normalizations] if is_qwen else list(texts)
        rule_s = time.perf_counter()-rule_start
        scores, routes = [None]*len(texts), [True]*len(texts)
        gate_s = 0.
        if variant.endswith('_gate'):
            gate_start = time.perf_counter()
            # Gate was trained on original text. Numeric-only changes do not require an LLM call.
            enc = gate_tok(texts, padding=True, return_tensors='pt', truncation=False)
            lengths = enc['attention_mask'].sum(-1).tolist()
            valid = [i for i,n in enumerate(lengths) if n <= 512]
            if valid:
                enc_valid = gate_tok([texts[i] for i in valid], padding=True,
                                     return_tensors='pt', truncation=False).to('cuda:0')
                with torch.inference_mode():
                    probabilities = gate(**enc_valid).softmax(-1)[:,1].tolist()
                for i,p in zip(valid, probabilities):
                    scores[i], routes[i] = p, p >= .005
            torch.cuda.synchronize()
            gate_s = time.perf_counter()-gate_start
        selected = [i for i,route in enumerate(routes) if route]
        outputs = list(baselines)
        model_outputs = list(baselines)
        raw_outputs = list(baselines)
        token_counts, ends = [0]*len(texts), [True]*len(texts)
        numeric_fallback = [False]*len(texts)
        numeric_repairs = [False]*len(texts)
        guard_s, refiner_s = 0., 0.
        if selected:
            name = variant.removesuffix('_gate')
            model, tokenizer = models[name], tokenizers[name]
            prompt = protocol['prompts']['qwen' if is_qwen else 'agentic1b']
            template_kwargs = {'enable_thinking':False} if 'enable_thinking' in tokenizer.get_chat_template() else {}
            refiner_start = time.perf_counter()
            chats = [tokenizer.apply_chat_template([{'role':'system','content':prompt},
                {'role':'user','content':baselines[i]}], tokenize=False, add_generation_prompt=True,
                **template_kwargs) for i in selected]
            inputs = tokenizer(chats, return_tensors='pt', padding=True,
                               add_special_tokens=False).to('cuda:0')
            with torch.inference_mode():
                seq = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                    use_cache=True, repetition_penalty=1., pad_token_id=tokenizer.pad_token_id)
            generated = seq[:,inputs['input_ids'].shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos,list) else [eos]
            for i,tokens in zip(selected, generated):
                stop = next((j+1 for j,t in enumerate(tokens) if t in eos),len(tokens))
                tokens = tokens[:stop]
                token_counts[i], ends[i] = len(tokens), bool(tokens and tokens[-1] in eos)
                raw = tokenizer.decode(tokens,skip_special_tokens=True).strip()
                model_outputs[i] = raw.split('<KEY>',1)[0].strip() if not is_qwen else raw
                outputs[i], raw_outputs[i] = model_outputs[i], raw
            torch.cuda.synchronize()
            refiner_s = time.perf_counter()-refiner_start
            if is_qwen:
                guard_start = time.perf_counter()
                for i in selected:
                    repaired = restore_rule_numeric_surfaces(baselines[i], outputs[i])
                    failures = () if repaired is None else tuple(reason for reason in
                        reject_reasons(baselines[i], repaired, numeric_rule_mode=True) if reason in
                        {'numeric_value_mismatch','numeric_rule_surface_mismatch','partial_numeric_range_conversion'})
                    if repaired is None or failures:
                        numeric_fallback[i] = True
                        outputs[i] = baselines[i]
                    else:
                        numeric_repairs[i] = repaired != outputs[i]
                        outputs[i] = repaired
                guard_s = time.perf_counter()-guard_start
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        predictions = [{'output':outputs[i], 'model_output':model_outputs[i], 'raw_output':raw_outputs[i],
            'model_input':baselines[i] if routes[i] else None, 'route':routes[i], 'gate_score':scores[i],
            'generated_tokens':token_counts[i], 'eos':ends[i], 'numeric_fallback':numeric_fallback[i],
            'numeric_surface_repaired':numeric_repairs[i]} for i in range(len(texts))]
        return predictions, {'batch_size':len(texts), 'batch_seconds':elapsed,
            'rule_seconds':rule_s, 'gate_seconds':gate_s, 'refiner_seconds':refiner_s,
            'numeric_guard_seconds':guard_s, 'refiner_calls':len(selected)}

    write('runtime.json',{'loads':loads, 'torch':torch.__version__,
        'transformers':transformers.__version__, 'gpu':torch.cuda.get_device_name(0),
        'allocated_bytes':torch.cuda.memory_allocated()})
    print('Warmup batch1 and batch8',flush=True)
    for v in VARIANTS:
        process(['嗯，周五下午三点开会，不对，改成周六下午三点。'],v)
        process([r['text'] for r in rows[:8]],v)
    rng = random.Random(20261028)
    completed = 0
    with (OUT/'predictions.jsonl').open('w') as pred_stream, (OUT/'batch_timings.jsonl').open('w') as time_stream:
        for mode,samples,batch_size in (
            ('latency',[r for r in rows if r['id'] in protocol['latency_subset_ids']],1),
            ('throughput',rows,8)):
            for batch_index,offset in enumerate(range(0,len(samples),batch_size)):
                batch = samples[offset:offset+batch_size]
                variants = VARIANTS.copy(); rng.shuffle(variants)
                for v in variants:
                    predictions,timing = process([r['text'] for r in batch],v)
                    time_stream.write(json.dumps(dict(timing,mode=mode,batch_index=batch_index,variant=v))+'\n')
                    time_stream.flush()
                    for sample,prediction in zip(batch,predictions):
                        row = dict(sample,**prediction,mode=mode,variant=v,batch_index=batch_index)
                        if mode == 'latency': row['total_seconds'] = timing['batch_seconds']
                        pred_stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    pred_stream.flush()
                    completed += len(batch)
                write('progress.json',{'phase':mode,'completed_text_requests':completed,
                    'total_text_requests':len(rows)*5+32*5,'batch_index':batch_index,'batch_size':batch_size})
                print(mode,batch_index,completed,flush=True)
    write('progress.json',{'phase':'complete','completed_text_requests':completed,'unique_inputs':len(rows)})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().mode]()

