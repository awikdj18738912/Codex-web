"""Serial warm TEXT ONLY gate/refiner benchmark. References never enter model prompts."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/refiner_text_benchmark_20261028'
BASE = '/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct'
TRAIN = Path('/home/aim0/data/ASR/training')
GATE = TRAIN / 'refiner_noC02_gate_rbt3_20261027'
MODELS = {
    'lora': TRAIN / 'qwen25_05b_refiner_noC02_lora_20261027/checkpoint-990',
    'full': TRAIN / 'qwen25_05b_refiner_noC02_full_20261027',
    'agentic1b': Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'),
}
VARIANTS = ['lora', 'lora_gate', 'full', 'full_gate', 'agentic1b']


def write(name, obj):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def read(path):
    return [json.loads(x) for x in Path(path).open()]


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    # Freeze selection before any new inference. Balanced by weak need-refine label.
    pairs = read(ROOT/'results/refiner13_no_C02_gate_20261027/all3056_refiner/predictions.jsonl')
    rng = random.Random(20261028)
    text = []
    for label in (0, 1):
        group = [r for r in pairs if r['label'] == label]
        rng.shuffle(group)
        text.extend(group[:128])
    rng.shuffle(text)
    (OUT / 'text_manifest.jsonl').write_text(''.join(json.dumps({k:r[k] for k in
        ('id','text','reference','label','category','source')}, ensure_ascii=False)+'\n' for r in text))
    prompt = json.loads(Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02/manifest.json').read_text())['system_prompt']
    tree = ast.parse((ROOT / 'experiments/scripts/postprocess_asr.py').read_text())
    old_prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets))
    write('protocol.json', {'gpu': 0, 'mode':'text_only_no_audio_no_ASR',
        'models': {k:str(v) for k,v in MODELS.items()}, 'base_model':BASE,
        'gate': str(GATE), 'threshold': .005, 'dtype': 'bfloat16', 'gate_dtype': 'float32',
        'lora_deployment': 'merge_and_unload in memory; no checkpoint files changed',
        'batch_size': 1, 'max_new_tokens': 1024, 'do_sample': False, 'use_cache': True,
        'prompts': {'qwen': prompt, 'agentic1b': old_prompt},
        'text_manifest_sha256': hashlib.sha256((OUT/'text_manifest.jsonl').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sample_count':256, 'weak_clean_count':128, 'weak_need_refine_count':128,
        'repeat_second_round_first_n':64,
        'quality_limitations':['References are candidate labels, not human gold',
          'Small pilot from previously inspected heldout corpus, not accuracy guarantee'],
        'timing':'Warm per-request text wall time includes gate, tokenization, host/device copy, autoregressive generation and output decode; excludes model startup, disk logging, ASR and network',
        'order':'seeded shuffled pipeline order per sample; 256 first round plus fixed64 second round',
        'extras':'No streaming, entity protection, ITN, web/network transport, or vLLM; direct Transformers SDPA'})
    print('Prepared', len(text), 'text samples', flush=True)


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Physical GPU0 only')
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(8)
    torch.manual_seed(20261028)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    text_rows = read(OUT/'text_manifest.jsonl')
    loads = {}
    models, tokenizers = {}, {}
    for name, path in MODELS.items():
        start = time.perf_counter()
        source = BASE if name == 'lora' else str(path)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
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

    def refinement(raw, variant):
        torch.cuda.synchronize()
        start = time.perf_counter()
        route, score, overflow, gate_s = True, None, False, 0.
        if variant.endswith('_gate'):
            enc = gate_tok(raw, return_tensors='pt', truncation=False)
            overflow = enc['input_ids'].shape[1] > 512
            if not overflow:
                with torch.inference_mode():
                    score = gate(**enc.to('cuda:0')).softmax(-1)[0, 1].item()
                route = score >= .005
            torch.cuda.synchronize()
            gate_s = time.perf_counter()-start
        raw_output, output, tokens, eos = raw, raw, [], True
        refine_s = 0.
        if route:
            name = variant.removesuffix('_gate')
            model, tokenizer = models[name], tokenizers[name]
            prompt = protocol['prompts']['agentic1b' if name == 'agentic1b' else 'qwen']
            kwargs = {'enable_thinking': False} if 'enable_thinking' in tokenizer.get_chat_template() else {}
            refine_start = time.perf_counter()
            inputs = tokenizer.apply_chat_template([{'role':'system','content':prompt},
                {'role':'user','content':raw}], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt', **kwargs).to('cuda:0')
            with torch.inference_mode():
                seq = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                    use_cache=True, repetition_penalty=1., pad_token_id=tokenizer.pad_token_id)
            tokens = seq[0, inputs['input_ids'].shape[1]:].tolist()
            raw_output = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            output = raw_output.split('<KEY>', 1)[0].strip() if name == 'agentic1b' else raw_output
            ends = model.generation_config.eos_token_id
            eos = bool(tokens and tokens[-1] in (ends if isinstance(ends, list) else [ends]))
            torch.cuda.synchronize()
            refine_s = time.perf_counter()-refine_start
        return {'output': output, 'raw_output': raw_output, 'route': route,
            'gate_score': score, 'overflow': overflow, 'gate_seconds': gate_s,
            'refiner_seconds': refine_s, 'postprocess_seconds': time.perf_counter()-start,
            'generated_tokens': len(tokens), 'eos': eos}

    write('runtime.json', {'loads':loads, 'torch':torch.__version__,
        'transformers':transformers.__version__, 'gpu':torch.cuda.get_device_name(0),
        'memory_allocated_bytes':torch.cuda.memory_allocated()})
    print('Warmup', flush=True)
    for raw in ('嗯，那个，我们周五开会，不对，周六开会。', '我们周六上午十点开会。'):
        for variant in VARIANTS:
            refinement(raw, variant)
    rng = random.Random(20261028)
    with (OUT/'text_predictions.jsonl').open('w') as stream:
        completed = 0
        for repeat, samples in enumerate((text_rows, text_rows[:64])):
            for i, sample in enumerate(samples, 1):
                variants = VARIANTS.copy(); rng.shuffle(variants)
                for variant in variants:
                    result = refinement(sample['text'], variant)
                    row = dict(sample, variant=variant, repeat=repeat, **result)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                    completed += 1
                write('progress.json', {'phase':'text','repeat':repeat,'completed_requests':completed,
                    'total_requests':1600,'last_id':sample['id']})
                print('Text', repeat, i, completed, flush=True)
    write('progress.json', {'phase':'complete','completed_requests':completed,'unique_text_inputs':len(text_rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    mode = parser.parse_args().mode
    {'prepare':prepare, 'run':run}[mode]()
