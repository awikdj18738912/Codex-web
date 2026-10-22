"""Generate a fixed, scene-balanced validation sample for checkpoint comparison."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
DATA = Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, prev[j] + 1, prev[j - 1] + (left != right)))
        prev = current
    return prev[-1]


def select_rows(limit):
    rows = json.loads((DATA / 'val_sft.json').read_text())
    by_scene = defaultdict(list)
    with (DATA / 'split_manifest.jsonl').open() as f:
        for line in f:
            item = json.loads(line)
            if item['split'] == 'validation':
                by_scene[item['scene']].append(item)
    rng = random.Random(20261021)
    groups = []
    for scene in sorted(by_scene):
        group = sorted(by_scene[scene], key=lambda r: r['source_id'])
        rng.shuffle(group)
        groups.append(group)
    selected = []
    depth = 0
    while len(selected) < min(limit, len(rows)):
        for group in groups:
            if depth < len(group) and len(selected) < limit:
                selected.append(group[depth])
        depth += 1
    return [(manifest, rows[manifest['output_index']]) for manifest in selected]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        p.error('--limit and --batch-size must be positive')
    if args.output.exists():
        p.error('Refusing to overwrite an existing evaluation output')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    selected = select_rows(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(int)
    scenes = defaultdict(lambda: defaultdict(int))
    def record(manifest, row, new_tokens, maximum, index, stream):
        _, source, reference = [m['value'] for m in row['conversations']]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        clean = source == reference
        metrics = dict(samples=1, reference_characters=len(reference),
            character_errors=edit_distance(reference, prediction),
            input_character_errors=edit_distance(reference, source),
            exact_matches=int(prediction == reference), clean_samples=int(clean),
            clean_overedits=int(clean and prediction != reference), changed_samples=int(not clean),
            changed_exact_matches=int(not clean and prediction == reference),
            generation_limit_hits=int(len(new_tokens) >= maximum and tokenizer.eos_token_id not in new_tokens))
        for key, value in metrics.items():
            totals[key] += value
            scenes[manifest['scene']][key] += value
        result = dict(id=manifest['source_id'], scene=manifest['scene'], input=source,
            reference=reference, prediction=prediction, metrics=metrics)
        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        stream.flush()
        print(f'{index}/{len(selected)} {manifest["source_id"]}', flush=True)

    with args.output.open('x', encoding='utf-8') as f:
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            prompts = []
            for _, row in batch:
                system, source, _ = [m['value'] for m in row['conversations']]
                prompts.append(tokenizer.apply_chat_template(
                    [dict(role='system', content=system), dict(role='user', content=source)],
                    add_generation_prompt=True, tokenize=False))
            inputs = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(model.device)
            # The generation budget depends only on inputs, equally for both models.
            maximum = min(2048, max(256, max(len(tokenizer.encode(r['conversations'][1]['value'], add_special_tokens=False))
                for _, r in batch) + 128))
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=maximum, do_sample=False,
                    repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            prompt_width = inputs['input_ids'].shape[-1]
            for offset, (manifest, row) in enumerate(batch):
                new_tokens = generated[offset, prompt_width:].tolist()
                if tokenizer.eos_token_id in new_tokens:
                    new_tokens = new_tokens[:new_tokens.index(tokenizer.eos_token_id) + 1]
                record(manifest, row, new_tokens, maximum, start + offset + 1, f)

    def summarize(values):
        values = dict(values)
        values['character_error_rate'] = values['character_errors'] / max(1, values['reference_characters'])
        values['input_character_error_rate'] = values['input_character_errors'] / max(1, values['reference_characters'])
        values['exact_match_rate'] = values['exact_matches'] / max(1, values['samples'])
        values['clean_overedit_rate'] = (values['clean_overedits'] / values['clean_samples']) if values['clean_samples'] else None
        return values
    report = dict(checkpoint=str(args.checkpoint.resolve()), seed=20261021,
        selection='round-robin scene-balanced sample from frozen validation split',
        decoding=dict(do_sample=False, repetition_penalty=1.0, batch_size=args.batch_size,
            max_new_tokens='min(2048,max(256,longest_source_tokens_in_batch+128))'),
        metrics=summarize(totals), scenes={s: summarize(v) for s, v in scenes.items()},
        note='Reference similarity metrics are not semantic correctness guarantees. Inspect predictions and compare the same samples against the base model.')
    report_path = args.output.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))



if __name__ == '__main__':
    main()
