"""Frozen gate and LoRA evaluation with an exactly balanced edit/keep mix."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_refiner13_pregate import Gate, metrics, ENCODER, DATA, ROOT
from evaluate_refiner13_qwen import BASE, metric_text, levenshtein

MODEL = Path('/home/aim0/data/ASR/training/refiner13_pregate_tiny_gpu0_20261027')
ADAPTER = Path('/home/aim0/data/ASR/training/qwen25_05b_refiner13_lora_20261026_retry3/checkpoint-1072')
OUT = ROOT/'results/refiner13_pregate_balanced_20261027'


def readlines(path):
    return [json.loads(s) for s in path.open()]


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=False)
    original = readlines(MODEL/'test.jsonl')
    positives = [r for r in original if r['label']==1]
    clean = [r for r in original if r['label']==0 and not r['derived_clean']]
    derived = sorted([r for r in original if r['derived_clean']],
        key=lambda r:hashlib.sha256(('balanced20261027:'+r['id']).encode()).hexdigest())
    rows = positives+clean+derived[:len(positives)-len(clean)]
    assert len(rows)==3346 and sum(r['label'] for r in rows)==1673
    texts = {r['text'] for r in rows}
    assert len(texts)==len(rows)
    for split in ('train', 'dev'):
        assert not texts.intersection(r['text'] for r in readlines(MODEL/(split+'.jsonl')))
    (OUT/'test.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    threshold = json.loads((MODEL/'summary.json').read_text())['threshold_selected_on_dev']
    write('manifest.json', {'need_edit':len(positives),'keep':len(rows)-len(positives),
        'original_keep':len(clean),'added_reference_keep':len(rows)-len(positives)-len(clean),
        'gate':str(MODEL),'adapter':str(ADAPTER),'threshold':threshold,
        'threshold_policy':'Frozen from prior dev; no retuning or training',
        'selection':'Keep all original test inputs; add deterministically hash-sorted test reference texts',
        'exact_train_dev_text_overlap':0,'label_status':'Weak candidate references, not human gold',
        'test_status':'Previously inspected test and paired clean-reference challenge; not new independent ASR data',
        'gate_sha256':hashlib.sha256((MODEL/'gate.pt').read_bytes()).hexdigest()})
    tokenizer=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True)
    gate=Gate().cuda().eval()
    gate.load_state_dict(torch.load(MODEL/'gate.pt',map_location='cpu',weights_only=True))
    probs=np.ones(len(rows))
    encoded=tokenizer([r['text'] for r in rows],truncation=False)
    indices=[i for i,t in enumerate(encoded['input_ids']) if len(t)<=512]
    with torch.inference_mode():
        for start in range(0,len(indices),64):
            ids=indices[start:start+64]
            batch=tokenizer.pad([{k:v[i] for k,v in encoded.items()} for i in ids],return_tensors='pt')
            probs[ids]=gate(**{k:v.cuda() for k,v in batch.items()}).softmax(-1)[:,1].cpu().numpy()
    records=[dict(r,probability=float(p),route_to_refiner=bool(p>=threshold)) for r,p in zip(rows,probs)]
    (OUT/'gate_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write('gate_metrics.json', metrics(rows,probs,threshold))
    print('GATE',json.dumps(metrics(rows,probs,threshold),ensure_ascii=False),flush=True)
    del gate
    torch.cuda.empty_cache()
    cached={r['id']:r for r in readlines(ROOT/'results/refiner13_qwen25_lora_20261026/evaluation/qwen25_lora.jsonl')}
    missing=[r for r in records if r['id'] not in cached]
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,
        device_map='cuda:0',local_files_only=True).eval()
    model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False).eval()
    prompt=json.loads((DATA/'manifest.json').read_text())['system_prompt']
    eos=model.generation_config.eos_token_id
    eos_set=set(eos if isinstance(eos,list) else [eos])
    started=time.perf_counter()
    with (OUT/'added_clean_lora.jsonl').open('w') as stream, torch.inference_mode():
        for start in range(0,len(missing),16):
            group=missing[start:start+16]
            rendered=[tokenizer.apply_chat_template([{'role':'system','content':prompt},
                {'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True) for r in group]
            inputs=tokenizer(rendered,padding=True,return_tensors='pt').to('cuda:0')
            outputs=model.generate(**inputs,max_new_tokens=512,do_sample=False,
                repetition_penalty=1.,pad_token_id=tokenizer.pad_token_id)
            tokens=outputs[:,inputs['input_ids'].shape[1]:].tolist()
            for r,t in zip(group,tokens):
                raw=tokenizer.decode(t,skip_special_tokens=True).strip()
                record={'id':r['id'],'input':r['text'],'reference':r['text'],
                    'prediction':raw.split('<KEY>',1)[0].strip(),'raw_output':raw,
                    'completed_with_eos':any(v in eos_set for v in t)}
                cached[r['id']]=record
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stream.flush()
            if start%160==0 or start+16>=len(missing):
                print('CLEAN_LORA',min(start+16,len(missing)),'/',len(missing),flush=True)
    outcomes=[]
    for r in records:
        base=cached[r['id']]
        assert base['input']==r['text']
        ref=base['reference'] if r['label'] else r['text']
        outcomes.append(dict(r,reference=ref,lora_prediction=base['prediction'],
            gated_prediction=base['prediction'] if r['route_to_refiner'] else r['text']))
    (OUT/'comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outcomes))
    def result(key):
        negative=[r for r in outcomes if not r['label']]
        total=sum(len(metric_text(r['reference'])) for r in outcomes)
        return {'n':len(outcomes),'exact_count':sum(r[key]==r['reference'] for r in outcomes),
            'exact_rate':sum(r[key]==r['reference'] for r in outcomes)/len(outcomes),
            'clean_false_edit_count':sum(r[key]!=r['text'] for r in negative),
            'clean_n':len(negative),'normalized_char_difference_rate':sum(levenshtein(metric_text(r[key]),
                metric_text(r['reference'])) for r in outcomes)/total}
    summary={'status':'complete','threshold':threshold,'gate':metrics(rows,probs,threshold),
        'lora_only':result('lora_prediction'),'gated_lora':result('gated_prediction'),
        'refiner_calls':sum(r['route_to_refiner'] for r in records),
        'saved_calls':sum(not r['route_to_refiner'] for r in records),
        'lost_exact':sum(r['lora_prediction']==r['reference'] and r['gated_prediction']!=r['reference'] for r in outcomes),
        'gained_exact':sum(r['lora_prediction']!=r['reference'] and r['gated_prediction']==r['reference'] for r in outcomes),
        'added_clean_inference_and_analysis_seconds':time.perf_counter()-started,
        'speed_note':'Cached original outputs plus batched added-clean inference; not an end-to-end latency benchmark',
        'labels':'Weak supervision, not human gold; paired clean references are diagnostic only'}
    write('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
