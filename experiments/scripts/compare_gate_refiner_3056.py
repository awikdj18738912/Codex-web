"""Run every balanced gate-test input through the no-C02 LoRA, then compare."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_refiner13_qwen import BASE, metric_text, levenshtein, ENGLISH_RE, PLACEHOLDER_RE

ROOT=Path(__file__).resolve().parents[2]
GATE=Path('/home/aim0/data/ASR/training/refiner_noC02_pregate_gpu0_20261027')
DATA=Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
ADAPTER=Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_noC02_lora_20261027/checkpoint-990')
OUT=ROOT/'results/refiner13_no_C02_gate_20261027/all3056_refiner'
BATCH=8


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def read(path): return [json.loads(s) for s in path.open()]


def matrix(rows,flag):
    return {'gate_edit_lora_edit':sum(r['route_to_refiner'] and r[flag] for r in rows),
        'gate_edit_lora_keep':sum(r['route_to_refiner'] and not r[flag] for r in rows),
        'gate_keep_lora_edit':sum(not r['route_to_refiner'] and r[flag] for r in rows),
        'gate_keep_lora_keep':sum(not r['route_to_refiner'] and not r[flag] for r in rows)}


def score(rows,key):
    clean=[r for r in rows if r['label']==0]
    return {'n':len(rows),'exact_count':sum(r[key]==r['reference'] for r in rows),
        'exact_rate':sum(r[key]==r['reference'] for r in rows)/len(rows),
        'clean_text_changed':sum(r[key]!=r['text'] for r in clean),
        'normalized_char_difference_rate':sum(levenshtein(metric_text(r[key]),metric_text(r['reference'])) for r in rows)
            /max(1,sum(len(metric_text(r['reference'])) for r in rows))}


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.22,0)
    source=GATE/'balanced_test_predictions.jsonl'
    rows=read(source)
    assert len(rows)==3056 and len({r['id'] for r in rows})==3056
    assert Counter(r['label'] for r in rows)=={0:1528,1:1528}
    assert all(r['category']!='C02' for r in rows)
    threshold=json.loads((GATE/'summary.json').read_text())['threshold_selected_on_dev']
    assert all(r['route_to_refiner']==(r['probability']>=threshold) for r in rows)
    audit={r['id']:r for r in read(DATA/'split_audit.jsonl') if r['split']=='test'}
    for r in rows:
        original_id=r['id'].removesuffix(':clean_reference') if r['derived_clean'] else r['id']
        original=audit[original_id]
        assert r['text']==(original['reference'] if r['derived_clean'] else original['input'])
        r['reference']=original['reference'] if r['label']==1 else r['text']
    OUT.mkdir(parents=True,exist_ok=False)
    prompt=json.loads((DATA/'manifest.json').read_text())['system_prompt']
    write('run.json',{'status':'running','input':str(source),
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'base':str(BASE),'adapter':str(ADAPTER),
        'adapter_sha256':hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'gate_threshold':threshold,'batch_size':BATCH,'gpu':0,
        'decode':'BF16 greedy max_new_tokens512 repetition_penalty1; same system prompt as training',
        'full_inference':True,'reused_predictions':False,'gate_controls_inference':False,
        'labels':'Candidate weak labels, not human gold; Refiner edits are not correctness labels',
        'concurrent_full_training':True,'latency_benchmark':False})
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,
        device_map='cuda:0',local_files_only=True).eval()
    model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False).eval()
    eos=model.generation_config.eos_token_id
    eos_set=set(eos if isinstance(eos,list) else [eos])
    results=[];started=time.perf_counter()
    with (OUT/'predictions.jsonl').open('x') as f, torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            batch=rows[start:start+BATCH]
            prompts=[tokenizer.apply_chat_template([{'role':'system','content':prompt},
                {'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True) for r in batch]
            inputs=tokenizer(prompts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda:0')
            outputs=model.generate(**inputs,max_new_tokens=512,do_sample=False,repetition_penalty=1.,
                pad_token_id=tokenizer.pad_token_id)
            tokens=outputs[:,inputs['input_ids'].shape[1]:].tolist()
            for r,t in zip(batch,tokens):
                raw=tokenizer.decode(t,skip_special_tokens=True)
                prediction=raw.strip().split('<KEY>',1)[0].strip()
                trim_changed=prediction.strip()!=r['text'].strip()
                whitespace_changed=''.join(prediction.split())!=''.join(r['text'].split())
                record=dict(r,prediction=prediction,raw_output=raw,
                    changed_exact=prediction!=r['text'],changed_after_outer_whitespace=trim_changed,
                    changed_after_all_whitespace=whitespace_changed,
                    whitespace_only_change=(prediction!=r['text'] and not whitespace_changed),
                    completed_with_eos=any(v in eos_set for v in t),
                    gated_prediction=prediction if r['route_to_refiner'] else r['text'],
                    english_tokens_match_reference=ENGLISH_RE.findall(prediction)==ENGLISH_RE.findall(r['reference']),
                    placeholder_sequence_matches_reference=PLACEHOLDER_RE.findall(prediction)==PLACEHOLDER_RE.findall(r['reference']))
                results.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
            if start%64==0 or start+BATCH>=len(rows):
                print(f'REFINER {min(start+BATCH,len(rows))}/{len(rows)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    summary={'status':'complete','n':len(results),'label_counts':dict(Counter(r['label'] for r in results)),
        'gate_threshold':threshold,'gate_edit_count':sum(r['route_to_refiner'] for r in results),
        'lora_edit_exact':sum(r['changed_exact'] for r in results),
        'lora_edit_excluding_outer_whitespace':sum(r['changed_after_outer_whitespace'] for r in results),
        'lora_edit_excluding_all_whitespace':sum(r['changed_after_all_whitespace'] for r in results),
        'whitespace_only_changes':sum(r['whitespace_only_change'] for r in results),
        'matrix_exact':matrix(results,'changed_exact'),
        'matrix_excluding_outer_whitespace':matrix(results,'changed_after_outer_whitespace'),
        'by_weak_label':{str(label):{'n':sum(r['label']==label for r in results),
            'gate_edit_count':sum(r['label']==label and r['route_to_refiner'] for r in results),
            'lora_edit_count':sum(r['label']==label and r['changed_exact'] for r in results),
            'matrix':matrix([r for r in results if r['label']==label],'changed_exact')}
            for label in (0,1)},
        'lora_only':score(results,'prediction'),'gated_lora':score(results,'gated_prediction'),
        'lost_exact_from_gate':sum(r['prediction']==r['reference'] and r['gated_prediction']!=r['reference'] for r in results),
        'gained_exact_from_gate':sum(r['prediction']!=r['reference'] and r['gated_prediction']==r['reference'] for r in results),
        'eos_count':sum(r['completed_with_eos'] for r in results),
        'elapsed_seconds':time.perf_counter()-started,
        'qualification':'Actual full batched LoRA inference, not cached. Output changes != objective need to refine. References are weak candidates. Concurrent GPU work means this is not an end-to-end timing benchmark.'}
    old70=ROOT/'results/refiner13_no_C02_gate_20261027/sample70_refiner/predictions.jsonl'
    old={r['id']:r for r in read(old70)}
    summary['serial70_crosscheck']={'n':len(old),'different_outputs':sum(r['id'] in old and
        r['prediction']!=old[r['id']]['prediction'] for r in results),
        'note':'Batch BF16 numerical differences may change greedy output; full-run comparison uses only this run.'}
    write('summary.json',summary)
    for name,subset in [('gate_edit_lora_keep',[r for r in results if r['route_to_refiner'] and not r['changed_after_outer_whitespace']]),
        ('gate_keep_lora_edit',[r for r in results if not r['route_to_refiner'] and r['changed_after_outer_whitespace']]),
        ('weak_clean_lora_changed',[r for r in results if not r['label'] and r['changed_after_outer_whitespace']])]:
        (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset))
    m=summary['matrix_excluding_outer_whitespace']
    lines=['# 3,056 条 LoRA 实际精修与门控对照','',
        '两类各 1,528 条。所有输入都实际调用去 C02 两轮 LoRA checkpoint-990；门控不阻止本次生成，阈值仍为验证集选定的 0.02。','',
        '**模型是否改写，不等于句子客观上是否需要精修。** 参考仍是候选弱标签。','',
        '## 门控与实际改写（忽略首尾空白）','',
        '| | LoRA 改写 | LoRA 原样返回 |','| --- | ---: | ---: |',
        f'| 门控送精修 | {m["gate_edit_lora_edit"]} | {m["gate_edit_lora_keep"]} |',
        f'| 门控直接放行 | {m["gate_keep_lora_edit"]} | {m["gate_keep_lora_keep"]} |','',
        f'严格文本比较改写 {summary["lora_edit_exact"]} 条；忽略首尾空白改写 {summary["lora_edit_excluding_outer_whitespace"]} 条。只调整空白 {summary["whitespace_only_changes"]} 条。原始严格比较矩阵保存在 summary.json。',
        '', '## 最终答案对照','',
        '| 结果 | 全部调用 LoRA | 门控后调用 LoRA |','| --- | ---: | ---: |',
        f'| 匹配候选参考 | {summary["lora_only"]["exact_count"]} | {summary["gated_lora"]["exact_count"]} |',
        f'| 弱标签干净句被改动 | {summary["lora_only"]["clean_text_changed"]} | {summary["gated_lora"]["clean_text_changed"]} |','',
        f'门控损失原本匹配参考的答案 {summary["lost_exact_from_gate"]} 条，同时新增匹配参考的答案 {summary["gained_exact_from_gate"]} 条。改变候选参考不自动视为有害误改，删除口癖或压缩重复可以是合理处理。',
        '', '## 可复查文件','',
        '- predictions.jsonl：全部3,056条门控输出、LoRA输出、弱标签参考及精修路由结果。',
        '- gate_edit_lora_keep.jsonl：门控送精修但实际原样返回（忽略首尾空白）。',
        '- gate_keep_lora_edit.jsonl：门控放行但LoRA实际会改写（忽略首尾空白）。',
        '- weak_clean_lora_changed.jsonl：标为干净、但LoRA实际改写的样本。',
        '- summary.json：严格／忽略空白的矩阵、两种流程答案指标与70条串行推理交叉对照。',
        '- run.json：模型、输入哈希及生成参数。','',
        'BF16批量推理可能与之前逐条推理有少量数值差异；全部比较使用本轮同一组实际输出。与其他GPU任务并行，不用此耗时声称端到端提速。']
    (OUT/'全部3056条精修与门控对照.md').write_text('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
