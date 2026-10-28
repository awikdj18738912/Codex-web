import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE_URLS = ['http://127.0.0.1:8012/v1/chat/completions', 'http://127.0.0.1:8013/v1/chat/completions']

def endpoint(row):
    return BASE_URLS[int(hashlib.sha256(row['case_id'].encode()).hexdigest()[:8], 16) % len(BASE_URLS)]
MODEL = 'qwen38-teacher'

INSTRUCTIONS = {}
SYSTEM = ''

NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')
NUMBER_WITH_UNIT_RE = re.compile(
    r'\d+(?:\.\d+)?(?:\s?(?:帧每秒|个百分点|百分比|毫秒|分钟|小时|个月|厘米|毫米|公里|千米|千克|公斤|毫升|摄氏度|万元|亿元|\u0025|％|°C|°F|fps|kHz|MHz|GHz|mAh|GB|TB|MB|KB|Wh|kW|MP|px|ms|Hz|秒|天|周|月|年|号|日|米|克|吨|升|万|亿|元|次|个|台|件|岁|倍|路|层|楼|点|分|名|人|组|项|步|期|K|V|W))?'
)
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
        ('number_unit_surface', NUMBER_WITH_UNIT_RE),
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
    req = Request(endpoint(row), data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
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
    parser.add_argument('--variant', choices=['baseline', 'revised'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args()
    global SYSTEM, INSTRUCTIONS
    SYSTEM = (ROOT / f'{args.variant}_system.txt').read_text(encoding='utf-8')
    INSTRUCTIONS = json.loads((ROOT / f'{args.variant}_instructions.json').read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.sample)
    provenance = {
        'model': MODEL,
        'variant': args.variant,
        'endpoints': BASE_URLS,
        'routing': 'sha256(case_id) modulo 2; identical paired case uses identical GPU',
        'server_max_num_seqs_per_gpu': 8,
        'instructions_sha256': hashlib.sha256(json.dumps(INSTRUCTIONS, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'retries': args.retries,
        'seed_rule': 'sha256(case_id:attempt) first 8 hex digits, identical across variants',
        'passthrough_policy': 'deterministic copy, excluded from generation effectiveness estimates',
        'sample_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        'system_prompt_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'validator': 'all-digit-surfaces+number-unit-surfaces+ASCII+URLs+assignments+operators+quoted ASCII literals+new brackets',
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
                'endpoint': endpoint(row),
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
