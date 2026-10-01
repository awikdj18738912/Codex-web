from __future__ import annotations
import argparse, json, pathlib, statistics, time, urllib.request, uuid

parser = argparse.ArgumentParser()
parser.add_argument('--url', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--vllm-api', action='store_true')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()

ROOT = pathlib.Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'chunks_manifest.json').read_text())
audio_paths = [(item, ROOT / 'chunks' / item['file']) for item in manifest]
audio_bytes = [(item, path, path.read_bytes()) for item, path in audio_paths]

def post(item: dict, path: pathlib.Path, payload: bytes) -> tuple[float, dict]:
    boundary = '----codexaudit' + uuid.uuid4().hex
    fields = [('model', args.model)]
    if args.vllm_api:
        # vLLM's Qwen3-ASR transcription adapter consumes `to_language` when
        # constructing the Qwen language-prefixed ASR prompt.
        fields += [('language', 'zh'), ('to_language', 'zh'), ('prompt', ''),
                   ('response_format', 'json'), ('temperature', '0'),
                   ('max_completion_tokens', '512')]
    else:
        fields += [('language', 'Chinese'), ('prompt', '')]
    parts = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + payload + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    body = b''.join(parts)
    request = urllib.request.Request(args.url, data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}, method='POST')
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    return elapsed, result

warm_item, warm_path, warm_bytes = audio_bytes[0]
warm_elapsed, warm_result = post(warm_item, warm_path, warm_bytes)
warm_record = {'file': warm_path.name, 'client_elapsed_s': warm_elapsed,
               'server_wall_ms': (warm_result.get('timing') or {}).get('wall_ms'),
               'text': warm_result.get('text', '')}
(ROOT / f'{args.label}_warmup.json').write_text(json.dumps(warm_record, ensure_ascii=False, indent=2) + '\n')
print(f'{args.label} API warmup: {warm_elapsed:.3f}s', flush=True)

records = []
partial = ROOT / f'{args.label}_api_results.jsonl'
partial.write_text('')
for run in range(1, args.repeats + 1):
    pass_records = []
    for index, (item, path, payload) in enumerate(audio_bytes, 1):
        elapsed, result = post(item, path, payload)
        record = {**item, 'run': run, 'client_elapsed_s': elapsed,
                  'server_wall_ms': (result.get('timing') or {}).get('wall_ms'),
                  'text': result.get('text', '')}
        if 'text' not in result:
            raise RuntimeError(f'{args.label} response omitted text for {path.name}: {result}')
        pass_records.append(record)
        records.append(record)
        with partial.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        if index % 20 == 0 or index == len(audio_bytes):
            elapsed_sum = sum(r['client_elapsed_s'] for r in pass_records)
            print(f'{args.label} pass {run}/{args.repeats}: {index}/{len(audio_bytes)}, pass_client_s={elapsed_sum:.2f}', flush=True)

audio_s = sum(r['duration_s'] for r in manifest)
per_pass = []
for run in range(1, args.repeats + 1):
    group = [r for r in records if r['run'] == run]
    client_s = sum(r['client_elapsed_s'] for r in group)
    server_s = sum((r['server_wall_ms'] or 0) / 1000 for r in group)
    per_pass.append({'run': run, 'chunks': len(group), 'audio_s': audio_s,
                     'client_elapsed_s': client_s, 'server_wall_s': server_s,
                     'client_rtf': client_s / audio_s,
                     'transcript_chars': sum(len(r['text']) for r in group)})
client_values = [r['client_elapsed_s'] for r in per_pass]
summary = {
    'runtime': args.label,
    'endpoint': args.url,
    'model': args.model,
    'weight_dtype': 'BF16 safetensors',
    'vllm_api': args.vllm_api,
    'repeats': args.repeats,
    'measured_chunks_per_pass': len(manifest),
    'total_audio_s_per_pass': audio_s,
    'warmup': warm_record,
    'per_pass': per_pass,
    'client_elapsed_mean_s': statistics.mean(client_values),
    'client_elapsed_median_s': statistics.median(client_values),
    'server_wall_mean_s': statistics.mean(p['server_wall_s'] for p in per_pass),
    'rtf_client_median': statistics.median(p['client_rtf'] for p in per_pass),
}
(ROOT / f'{args.label}_api_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
