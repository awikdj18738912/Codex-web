import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE_URL = 'http://127.0.0.1:8012/v1/chat/completions'
MODEL = 'qwen38-teacher'

INSTRUCTIONS = {
    'clean_passthrough': '保持原文不变，不增加任何内容。',
    'fillers': '自然加入一到两个短填充词或停顿词，放在合适的句中位置；不改变语气强度和事实。',
    'repetition_false_start': '自然地重复一个短语一次，并出现一次未说完就重新组织的短起头；不重复人名、数字、地点、型号或代码，不新增错误事实。',
    'self_correction': '出现一次轻微、自然的措辞自我修正，只能改普通措辞或句子组织；被撤回片段不得引入新实体、新事件、新数值或新方案。表达最终意图时必须与目标一致。',
    'boundary_punctuation': '只在一个中文短语内部错误插入一个标点，形成可恢复的意外断句；不得改动任何文字，也不得把标点放进代码、URL、英文实体或数字内部。',
    'natural_mixed_code_switch': '目标含有英文或代码。保持原样并自然放在中文表达中；不要重复拼读，不添加无关英文或实体。'
}

SYSTEM = (ROOT / 'prompt_v4.txt').read_text(encoding='utf-8')

NUMBER_RE = re.compile(r'(?<!\w)\d+(?:\.\d+)?')
ASCII_SPAN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_./:+-]{1,}')
QUOTED_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
URL_RE = re.compile(r'https?://[^\s，。；！？]+')
ASSIGNMENT_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^\s，。；！？]+')
OPERATOR_RE = re.compile(r'\|\||&&|(?<!\|)\|(?!\|)|==|!=|<=|>=|->|=>|::')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def protected_surface_errors(target, oral):
    for label, pattern in [
        ('number', NUMBER_RE),
        ('ascii_span', ASCII_SPAN_RE),
        ('url', URL_RE),
        ('assignment', ASSIGNMENT_RE),
        ('operator', OPERATOR_RE),
    ]:
        for span in pattern.findall(target):
            if span not in oral:
                return f'missing_or_changed_{label}:{span}'
    for literal in QUOTED_LITERAL_RE.findall(target):
        if re.search(r'[A-Za-z0-9_$./:+\\-]', literal) and literal not in oral:
            return f'missing_or_changed_quoted_literal:{literal}'
    for ch in '()（）[]【】':
        if oral.count(ch) > target.count(ch):
            return f'new_editorial_bracket:{ch}'
    return None


def call(row, timeout, attempt=0):
    user = (
        f"口语现象要求：{INSTRUCTIONS[row['phenomenon']]}\n"
        f"场景：{row['scene']}\n"
        f"最终干净目标：\n{row['clean_target']}\n"
        '请仅返回 JSON。'
    )
    body = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user},
        ],
        'temperature': 0.65,
        'top_p': 0.8,
        'max_tokens': 768,
        'seed': int(hashlib.sha256(f"{row['case_id']}:{attempt}".encode()).hexdigest()[:8], 16),
        'response_format': {'type': 'json_object'},
        'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False},
    }
    req = Request(BASE_URL, data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parse_response(raw):
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError(f"finish_reason={choice.get('finish_reason')}")
    result = json.loads(choice['message']['content'])
    if result.get('valid') is not True:
        raise ValueError('teacher_declined_oralization')
    oral = result.get('oral_text')
    if not isinstance(oral, str) or len(oral.strip()) < 4:
        raise ValueError('missing_oral_text')
    return oral, result.get('speech_phenomena', [])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, default=ROOT / 'seed_pool_100.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'oral_candidates_v4.jsonl')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args()
    rows = read_jsonl(args.sample)
    provenance = {
        'model': MODEL,
        'sample_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        'system_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'validator': 'numbers+ASCII+URLs+assignments+operators+quoted ASCII literals+new brackets',
        'temperature': 0.65,
        'top_p': 0.8,
        'max_tokens': 768,
        'concurrency': args.concurrency,
        'no_production_data_modified': True,
    }
    prov_path = args.output.with_suffix('.provenance.json')
    if prov_path.exists() and json.loads(prov_path.read_text(encoding='utf-8')) != provenance:
        raise ValueError('resume provenance mismatch')
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    done_rows = read_jsonl(args.output) if args.output.exists() else []
    done = {r['case_id'] for r in done_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    started = time.monotonic()

    async def process(row):
        async with semaphore:
            request_started = time.monotonic()
            error = None
            oral = None
            phenomena = []
            usage = None
            if row['phenomenon'] == 'clean_passthrough':
                oral = row['clean_target']
                phenomena = ['clean_passthrough']
            else:
                for attempt in range(args.retries + 1):
                    try:
                        raw = await asyncio.to_thread(call, row, args.timeout, attempt)
                        usage = raw.get('usage')
                        oral, phenomena = parse_response(raw)
                        error = protected_surface_errors(row['clean_target'], oral)
                        if error:
                            oral = None
                            raise ValueError(error)
                        error = None
                        break
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {str(exc)[:300]}'
            result = {
                **row,
                'oral_text': oral,
                'observed_phenomena': phenomena,
                'error': error,
                'usage': usage,
                'request_seconds': round(time.monotonic() - request_started, 2),
            }
            async with lock:
                with args.output.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                done.add(row['case_id'])
                existing = read_jsonl(args.output)
                progress = {
                    'completed': len(done),
                    'target': len(rows),
                    'errors': sum(bool(x.get('error')) for x in existing),
                    'wall_seconds': round(time.monotonic() - started, 1),
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                args.output.with_suffix('.progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(json.dumps(progress), flush=True)

    await asyncio.gather(*(process(row) for row in rows if row['case_id'] not in done))


if __name__ == '__main__':
    asyncio.run(main())
