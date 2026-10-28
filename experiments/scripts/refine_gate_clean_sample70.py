"""Run the actual no-C02 LoRA Refiner on the fixed 70 gate-selected texts."""
import hashlib
import json
from pathlib import Path
from evaluate_refiner13_qwen import BASE, run_model

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/refiner13_no_C02_gate_20261027'
DATA=Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
ADAPTER=Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_noC02_lora_20261027/checkpoint-990')


def main():
    source=RUN/'clean_routed_sample70.jsonl'
    sample=[json.loads(s) for s in source.open()]
    assert len(sample)==70 and len({r['id'] for r in sample})==70
    rows=[{'id':r['id'],'category':r['category'],'source':r['source'],
        'input':r['text'],'reference':r['text']} for r in sample]
    out=RUN/'sample70_refiner'
    out.mkdir(exist_ok=False)
    (out/'run.json').write_text(json.dumps({'input_file':str(source),
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'base':str(BASE),'adapter':str(ADAPTER),'gpu':0,'sample_count':70,
        'weak_label_note':'Input-as-reference measures text preservation, not human correctness.',
        'decode':'greedy; max_new_tokens=512; repetition_penalty=1; same training system prompt'},ensure_ascii=False,indent=2)+'\n')
    prompt=json.loads((DATA/'manifest.json').read_text())['system_prompt']
    run_model('no_C02_lora_gate_clean70',BASE,ADAPTER,rows,prompt,out/'predictions.jsonl')


if __name__=='__main__':main()
