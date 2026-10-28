"""Reproduce a single real batch-size-dependent LoRA output, using frozen inputs."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/refiner_rules_text_benchmark_20261028'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(8)
    protocol=json.loads((OUT/'protocol.json').read_text())
    records=[json.loads(s) for s in (OUT/'predictions.jsonl').open()]
    target='A02-002006043-1'
    example=next(r for r in records if r['id']==target and r['mode']=='throughput' and r['variant']=='lora')
    batch=[r for r in records if r['mode']=='throughput' and r['variant']=='lora_gate' and r['batch_index']==example['batch_index']]
    tok=AutoTokenizer.from_pretrained(protocol['base_model'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(protocol['base_model'],dtype=torch.bfloat16,
        device_map='cuda:0',local_files_only=True,attn_implementation='sdpa').eval()
    model=PeftModel.from_pretrained(model,protocol['models']['lora'],is_trainable=False).merge_and_unload().eval()
    result=[]
    for repeat in range(2):
        for label,items in [('original_batch8',batch),('gate_routed_subbatch',[r for r in batch if r['route']]),
                           ('single',[r for r in batch if r['id']==target])]:
            chats=[tok.apply_chat_template([{'role':'system','content':protocol['prompts']['qwen']},
                   {'role':'user','content':r['rule_baseline']}],tokenize=False,add_generation_prompt=True) for r in items]
            enc=tok(chats,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda:0')
            with torch.inference_mode():
                seq=model.generate(**enc,max_new_tokens=1024,do_sample=False,use_cache=True,
                                   repetition_penalty=1.,pad_token_id=tok.pad_token_id)
            texts=tok.batch_decode(seq[:,enc['input_ids'].shape[1]:],skip_special_tokens=True)
            index=next(i for i,r in enumerate(items) if r['id']==target)
            row={'repeat':repeat,'group':label,'batch_size':len(items),'output':texts[index].strip()}
            result.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    (OUT/'lora_batch_difference_replay.json').write_text(json.dumps({'id':target,'reference':example['reference'],
        'observed':result,'note':'Same weights, BF16 SDPA, greedy; tests batch shape, not different gate classification'},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
