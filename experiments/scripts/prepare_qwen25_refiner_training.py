"""Prepare the reviewed Refiner corpus for Qwen2.5 without loading GPU weights."""
import hashlib
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')
MODEL = Path('/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct')
TRAIN_OUTPUT = Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_reviewed_v5_full')
PREP = ROOT / 'results/qwen25_05b_refiner_preparation_20261021'
PREPARATION_STORAGE = TRAIN_OUTPUT.parent / PREP.name
CONFIG = ROOT / 'experiments/llamafactory/qwen25_05b_refiner_reviewed_v5.yaml'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import yaml
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args

    PREPARATION_STORAGE.mkdir(parents=True, exist_ok=True)
    PREP.parent.mkdir(parents=True, exist_ok=True)
    if not PREP.exists() and not PREP.is_symlink():
        PREP.symlink_to(PREPARATION_STORAGE, target_is_directory=True)
    if PREP.resolve() != PREPARATION_STORAGE.resolve():
        raise RuntimeError('Preparation directory must point to the data disk: ' + str(PREPARATION_STORAGE))
    export = json.loads((DATA / 'export_report.json').read_text())
    for name, info in export['files'].items():
        if sha256(DATA / name) != info['sha256']:
            raise RuntimeError('Export file hash changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    template = TEMPLATES['qwen']
    stats = {}
    expected_signatures = {}
    for split, expected in [('train', 84875), ('val', 14977)]:
        rows = json.loads((DATA / f'{split}_sft.json').read_text())
        if len(rows) != expected:
            raise RuntimeError(f'{split} count changed: {len(rows)}')
        lengths = []
        targets = []
        longest = None
        for start in range(0, len(rows), 512):
            batch = rows[start:start + 512]
            texts = []
            answers = []
            for row in batch:
                conv = row['conversations']
                if [m['from'] for m in conv] != ['system', 'human', 'gpt']:
                    raise RuntimeError('Unexpected conversation roles')
                system, source, target = [m['value'] for m in conv]
                if not all(isinstance(x, str) and x.strip() for x in (system, source, target)):
                    raise RuntimeError('Empty training field')
                messages = [dict(role='system', content=system), dict(role='user', content=source)]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                answer = target + '<|im_end|>\n'
                texts.append(prompt + answer)
                answers.append(answer)
                # Check the actual installed LLaMA Factory Qwen template against runtime formatting.
                if start == 0:
                    source_ids, target_ids = template.encode_oneturn(tokenizer,
                        [dict(role='user', content=source), dict(role='assistant', content=target)], system=system)
                    if source_ids != tokenizer.encode(prompt, add_special_tokens=False):
                        raise RuntimeError('LLaMA Factory/runtime prompt mismatch')
                    if target_ids != tokenizer.encode(answer, add_special_tokens=False):
                        raise RuntimeError('LLaMA Factory/runtime answer mismatch')
            sizes = tokenizer(texts, add_special_tokens=False, truncation=False,
                return_length=True, return_attention_mask=False)['length']
            answer_sizes = tokenizer(answers, add_special_tokens=False, truncation=False,
                return_length=True, return_attention_mask=False)['length']
            for offset, n in enumerate(sizes):
                if longest is None or n > longest['tokens']:
                    longest = dict(index=start + offset, tokens=n)
            lengths.extend(sizes)
            targets.extend(answer_sizes)
            if start % 10240 == 0:
                print(f'{split}: {min(start + len(batch), len(rows))}/{len(rows)} tokenized', flush=True)
        ordered = sorted(lengths)
        expected_signatures[split] = Counter(zip(lengths, targets))
        stats[split] = dict(records=len(rows), max_tokens=max(lengths), min_tokens=min(lengths),
            p50=ordered[int((len(ordered)-1)*.5)], p95=ordered[int((len(ordered)-1)*.95)],
            p99=ordered[int((len(ordered)-1)*.99)], max_answer_tokens=max(targets), longest=longest)
    cutoff = math.ceil(max(s['max_tokens'] for s in stats.values()) / 256) * 256
    if cutoff > 32768:
        raise RuntimeError('Example exceeds student model context')
    fingerprint = hashlib.sha256(json.dumps(dict(
        train=export['files']['train_sft.json']['sha256'],
        validation=export['files']['val_sft.json']['sha256'],
        tokenizer=sha256(MODEL / 'tokenizer.json'), cutoff=cutoff, template='qwen'
    ), sort_keys=True).encode()).hexdigest()[:16]
    config = dict(
        model_name_or_path=str(MODEL), trust_remote_code=False, flash_attn='sdpa',
        cache_dir=str(PREP / 'hf_cache'), stage='sft', do_train=True, do_eval=True,
        finetuning_type='full', dataset_dir=str(DATA), dataset='refiner_zh_reviewed_v5_train',
        eval_dataset='refiner_zh_reviewed_v5_val', template='qwen', cutoff_len=cutoff,
        packing=False, train_on_prompt=False, preprocessing_num_workers=8,
        tokenized_path=str(PREP / f'tokenized_{fingerprint}'),
        dataloader_num_workers=2, overwrite_cache=False,
        output_dir=str(TRAIN_OUTPUT),
        overwrite_output_dir=False, logging_steps=10, logging_first_step=True,
        save_strategy='epoch', eval_strategy='epoch', save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        prediction_loss_only=True, eval_accumulation_steps=1, plot_loss=True,
        save_only_model=False, report_to='none', push_to_hub=False,
        per_device_train_batch_size=2, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, num_train_epochs=5,
        learning_rate=2e-5, lr_scheduler_type='cosine', warmup_ratio=.05,
        weight_decay=.01, max_grad_norm=1., optim='adamw_torch',
        bf16=True, fp16=False, gradient_checkpointing=True,
        seed=20261021, data_seed=20261021,
    )
    # Parse the installed framework's full configuration on CPU; no weights/optimizer are loaded.
    cpu = dict(config, use_cpu=True, bf16=False, dataloader_num_workers=0)
    model_args, data_args, training_args, _, _ = get_train_args(cpu)
    fixed_template = get_template_and_fix_tokenizer(tokenizer, data_args)
    module = get_dataset(fixed_template, model_args, data_args, training_args, 'sft', tokenizer)
    mask_checks = {}
    for split, name in [('train', 'train_dataset'), ('val', 'eval_dataset')]:
        dataset = module[name]
        observed = Counter()
        checked = 0
        for batch in dataset.iter(batch_size=512):
            for ids, labels in zip(batch['input_ids'], batch['labels']):
                first = next((i for i, n in enumerate(labels) if n != -100), None)
                if first is None or first == 0 or labels[first:] != ids[first:]:
                    raise RuntimeError('Invalid assistant-only labels')
                observed[(len(ids), len(ids) - first)] += 1
                checked += 1
        if observed != expected_signatures[split]:
            raise RuntimeError(f'{split}: preprocessed lengths/labels differ from the untruncated Qwen examples')
        mask_checks[split] = checked
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text('# Qwen2.5-0.5B-Instruct full SFT on reviewed v5 pairs.\n' +
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    report = dict(status='prepared', model=str(MODEL), data=str(DATA), config=str(CONFIG),
        approved_records=sum(s['records'] for s in stats.values()),
        excluded_needs_manual=133, excluded_qc_error=15, token_lengths=stats,
        cutoff_len=cutoff, truncated_examples=0, training_started=False,
        tokenized_path=config['tokenized_path'], assistant_masks_checked=mask_checks,
        tokenized_files={str(p.relative_to(config['tokenized_path'])): sha256(p)
            for p in Path(config['tokenized_path']).rglob('*') if p.is_file()},
        training_output=str(TRAIN_OUTPUT),
        preparation_storage=str(PREPARATION_STORAGE),
        template='qwen', template_runtime_checks=1024, assistant_only_loss=True,
        effective_batch_size_single_gpu=32, evaluation='full validation loss and checkpoint each epoch; best eval_loss restored',
        semantic_quality_guaranteed=False, config_parser_validated=True,
        packages={p: importlib.metadata.version(p) for p in ['torch','transformers','llamafactory','datasets','accelerate']},
        source_export_report_sha256=sha256(DATA / 'export_report.json'),
        config_sha256=sha256(CONFIG))
    (PREP / 'preparation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
