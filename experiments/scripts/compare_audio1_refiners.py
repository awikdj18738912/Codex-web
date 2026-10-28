"""Fresh local ASR for audio1, then compare frozen Refiner checkpoints."""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/audio1_refiner_comparison_20261027'
AUDIO = ROOT / '测试音频1.mp3'
ASR = Path('/home/aim0/data/models/ASR/Qwen3-ASR-1.7B')
BASE = Path('/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct')


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def transcribe():
    import torch
    from qwen_asr import Qwen3ASRModel
    OUT.mkdir(parents=True, exist_ok=False)
    config = {'audio': str(AUDIO), 'audio_sha256': hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
        'asr_model': str(ASR), 'gpu': 0, 'dtype': 'bfloat16', 'language': 'Chinese',
        'max_new_tokens': 1024, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'offline_whole_audio', 'human_reference_provided': False}
    write('config.json', config)
    model = Qwen3ASRModel.from_pretrained(str(ASR), dtype=torch.bfloat16,
        device_map='cuda:0', max_inference_batch_size=1, max_new_tokens=1024)
    start = time.perf_counter()
    result = model.transcribe(audio=str(AUDIO), language='Chinese')[0]
    torch.cuda.synchronize()
    raw = {'text': result.text, 'language': result.language,
        'elapsed_seconds': time.perf_counter() - start}
    write('asr_raw.json', raw)
    (OUT / 'asr_raw.txt').write_text(result.text + '\n')
    print(json.dumps(raw, ensure_ascii=False), flush=True)


def refine():
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    raw = json.loads((OUT / 'asr_raw.json').read_text())['text']
    manifest = json.loads(Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional/manifest.json').read_text())
    prompt = manifest['system_prompt']
    specs = [
        ('LoRA两轮', BASE, Path('/home/aim0/data/ASR/training/qwen25_05b_refiner13_lora_20261026_retry3/checkpoint-1072')),
        ('LoRA追加一轮最佳验证版', BASE, Path('/home/aim0/data/ASR/training/qwen25_05b_refiner13_lora_epoch3_20261027')),
        ('全量两轮最佳验证版', Path('/home/aim0/data/ASR/training/qwen25_05b_refiner13_full_20261027_provisional'), None),
    ]
    results = []
    for name, model_path, adapter in specs:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
            device_map='cuda:0', local_files_only=True).eval()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': raw}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True)
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        row = {'name': name, 'model': str(model_path), 'adapter': str(adapter) if adapter else None,
            'input': raw, 'output': text, 'input_tokens': inputs['input_ids'].shape[1],
            'output_tokens': len(tokens), 'finished_with_eos': bool(tokens and tokens[-1] in eos),
            'elapsed_seconds': time.perf_counter() - start}
        results.append(row)
        write('refiner_outputs.json', {'system_prompt': prompt,
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'peft_version': peft.__version__, 'results': results})
        print(name, text, flush=True)
        del model
        torch.cuda.empty_cache()
    md = ['# 测试音频1：原始ASR与精修对比', '',
        '音频：'+str(AUDIO)+'；约40.85秒。重新使用本地Qwen3-ASR-1.7B离线整段识别，再对同一转写使用相同训练提示词、BF16与贪心解码。无人工逐字参考，不能报告识别准确率或CER。未添加实体库、数字归一化或规则回退，以下为各模型直接输出。', '',
        '## 原始ASR', '', raw, '']
    for row in results:
        md += ['## '+row['name'], '', row['output'], '',
            f"正常EOS结束：{row['finished_with_eos']}；输入{row['input_tokens']} tokens，生成{row['output_tokens']} tokens。", '',
            '相对原始ASR的字符改动：', '']
        for op, a, b, c, d in difflib.SequenceMatcher(None, raw, row['output'], autojunk=False).get_opcodes():
            if op != 'equal':
                md.append('- '+json.dumps({'operation':op,'before':raw[a:b],'after':row['output'][c:d]},ensure_ascii=False))
        md.append('')
    (OUT / '测试音频1精修对比.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['asr', 'refine'])
    args = p.parse_args()
    transcribe() if args.stage == 'asr' else refine()
