"""Run a frozen review queue through the existing local Refiner, without labels in its prompt."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
OUT = ROOT/'results/review21_existing_refiner1b_20261026'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def normalize_punctuation(text):
    import unicodedata
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('This comparison must use physical GPU0 only')
    OUT.mkdir(exist_ok=False)
    current = json.loads((ROOT/'results/current_refiner_dataset.json').read_text())
    queue = Path(current['review_queue_path'])
    frozen = OUT/'input_21.jsonl'
    frozen.write_bytes(queue.read_bytes())
    rows = [json.loads(line) for line in frozen.open()]
    if len(rows) != 21 or len({r['id'] for r in rows}) != 21:
        raise ValueError('expected_current_21_unique_review_samples')
    prompt_path = ROOT/'experiments/scripts/postprocess_asr.py'
    tree = ast.parse(prompt_path.read_text())
    prompt = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets))
    config = {'started_utc':datetime.now(timezone.utc).isoformat(), 'model_path':str(MODEL),
              'model_config':json.loads((MODEL/'config.json').read_text()),
              'model_config_sha256':sha(MODEL/'config.json'),
              'weight_file_size':(MODEL/'model.safetensors').stat().st_size,
              'source_queue':str(queue), 'source_queue_sha256':sha(queue),
              'sample_count':len(rows), 'by_source':dict(Counter(r['dataset'] for r in rows)),
              'gpu_ids':[0], 'dtype':'bfloat16', 'batch_size':1, 'do_sample':False,
              'max_new_tokens':512, 'prompt_source':str(prompt_path),
              'prompt_source_sha256':sha(prompt_path), 'system_prompt':prompt,
              'script_sha256':sha(Path(__file__)),
              'purpose':'User-requested inference comparison on all21 records including public; not machine quality audit',
              'reference_in_model_input':False, 'category_in_model_input':False,
              'postprocessing':'Remove optional <KEY> annotation only, retain raw output separately',
              'production_candidates_modified':False, 'training_started':False}
    write(OUT/'config.json', config)
    (OUT/'run_script.py').write_bytes(Path(__file__).read_bytes())
    print('Loading existing local Refiner on GPU0', flush=True)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20261026)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                device_map='cuda:0', local_files_only=True).eval()
    template = tokenizer.get_chat_template()
    kwargs = {'enable_thinking':False} if 'enable_thinking' in template else {}
    config.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                  model_parameter_count=sum(p.numel() for p in model.parameters()),
                  device_name=torch.cuda.get_device_name(0), template_kwargs=kwargs,
                  model_weight_sha256=sha(MODEL/'model.safetensors'))
    write(OUT/'config.json', config)
    print('Model loaded; starting21 inputs', flush=True)
    completed = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with (OUT/'results.jsonl').open('w') as stream:
        for index, row in enumerate(rows, 1):
            conversation = [{'role':'system', 'content':prompt}, {'role':'user', 'content':row['input']}]
            inputs = tokenizer.apply_chat_template(conversation, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt', **kwargs).to('cuda:0')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-started)*1000
            tokens = generated.sequences[0, inputs['input_ids'].shape[1]:].tolist()
            raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            output = raw.split('<KEY>', 1)[0].strip()
            result = {'index':index, 'id':row['id'], 'category':row['category'], 'dataset':row['dataset'],
                      'input':row['input'], 'candidate_reference':row['reference'],
                      'refiner_1b_raw_output':raw, 'refiner_1b_output':output,
                      'auxiliary_key':raw.split('<KEY>',1)[1].strip() if '<KEY>' in raw else None,
                      'completed_with_eos':bool(tokens and tokens[-1] in eos),
                      'generated_tokens':len(tokens), 'wall_ms':round(elapsed,2),
                      'equals_reference':output == row['reference'],
                      'equals_input':output == row['input'],
                      'equals_reference_ignoring_punctuation':normalize_punctuation(output)==normalize_punctuation(row['reference']),
                      'old_review_reason':row.get('judge',{}).get('reason'),
                      'primary_agent_finding':row.get('primary_agent_finding'),
                      'placeholder_sequence_preserved':re.findall(r'__ENTITY_\d+__',row['input'])==re.findall(r'__ENTITY_\d+__',output)}
            completed.append(result)
            stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
            write(OUT/'progress.json', {'status':'running', 'completed':index, 'total':len(rows)})
            print(index,row['id'],output,flush=True)
    summary = {'status':'complete', 'sample_count':len(completed),
               'exact_reference_matches':sum(r['equals_reference'] for r in completed),
               'matches_ignoring_punctuation':sum(r['equals_reference_ignoring_punctuation'] for r in completed),
               'unchanged_from_input':sum(r['equals_input'] for r in completed),
               'completed_with_eos':sum(r['completed_with_eos'] for r in completed),
               'placeholder_sequence_violations':[r['id'] for r in completed if not r['placeholder_sequence_preserved']],
               'source_queue_unchanged':sha(queue)==config['source_queue_sha256'],
               'note':'Agreement is not accuracy: candidate references are disputed;21 selected cases are not representative.'}
    write(OUT/'summary.json', summary)
    write(OUT/'progress.json', {'status':'complete','completed':len(completed),'total':len(rows)})
    md=['# 待复核21条：现有1B精修模型输出对比','',
        '模型：'+str(MODEL)+'；GPU0、BF16、逐条贪心解码，使用项目现有离线精修提示词。模型只看到输入，不提供候选答案或类别。',
        '', '全部21条均包含在内，包括公开来源。这是用户指定的推理对比，不是自动质量审核。候选答案存在争议，不能用一致率表示正确率。未修改原数据。','',
        f"逐字一致{summary['exact_reference_matches']}/21；忽略标点与空白后内容一致{summary['matches_ignoring_punctuation']}/21；输出与输入完全相同{summary['unchanged_from_input']}/21。",'']
    for row in completed:
        md += [f"## {row['index']:02d} · {row['category']} · {row['id']}",'',
               '**原输入**','',row['input'],'','**数据集候选答案**','',row['candidate_reference'],'',
               '**现有1B模型输出**','',row['refiner_1b_output'],'',
               f"逐字匹配候选：{row['equals_reference']}；生成耗时：{row['wall_ms']} ms。",'']
        if row['auxiliary_key'] is not None:
            md += ['**模型附加KEY（单独显示，不计正文）**','',row['auxiliary_key'],'']
        md += ['**原待复核理由（可能有误）**','',row['old_review_reason'] or '未提供','']
    (OUT/'1B精修21条逐条对比.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
