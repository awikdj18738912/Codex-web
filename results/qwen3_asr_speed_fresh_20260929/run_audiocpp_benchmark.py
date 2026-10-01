from __future__ import annotations
import json, pathlib, time, urllib.request, uuid, wave

ROOT = pathlib.Path(__file__).resolve().parent
URL = 'http://127.0.0.1:38761/v1/audio/transcriptions'
chunks = json.loads((ROOT / 'chunks_manifest.json').read_text())
records = []
partial = ROOT / 'audiocpp_results.jsonl'
partial.write_text('')

def post(path: pathlib.Path) -> tuple[float, dict]:
    boundary = '----codexbench' + uuid.uuid4().hex
    started = time.perf_counter()
    parts = []
    for name, value in [('model', 'qwen3-asr-1.7b'), ('language', 'Chinese'), ('prompt', '')]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + path.read_bytes() + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    request = urllib.request.Request(URL, data=b''.join(parts), headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}, method='POST')
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read())
    return time.perf_counter() - started, result

for i, item in enumerate(chunks, 1):
    path = ROOT / 'chunks' / item['file']
    elapsed, result = post(path)
    record = {
        **item,
        'client_elapsed_s': elapsed,
        'server_wall_ms': (result.get('timing') or {}).get('wall_ms'),
        'server_rtf': (result.get('timing') or {}).get('rtf'),
        'text': result.get('text', ''),
    }
    records.append(record)
    with partial.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    if i % 10 == 0 or i == len(chunks):
        print(f'AudioCpp: {i}/{len(chunks)} chunks, cumulative_client_s={sum(r["client_elapsed_s"] for r in records):.2f}', flush=True)

audio_s = sum(r['duration_s'] for r in records)
client_s = sum(r['client_elapsed_s'] for r in records)
server_s = sum((r['server_wall_ms'] or 0) / 1000 for r in records)
summary = {
    'runtime': 'AudioCpp server multipart transcription',
    'model': '/home/aim0/data/models/ASR/Qwen3-ASR-1.7B',
    'weight_dtype': 'BF16 safetensors',
    'gpu': 'physical GPU 0, NVIDIA GeForce RTX 3090, CUDA_VISIBLE_DEVICES=0',
    'audio_format': '16 kHz mono PCM16 WAV; external non-overlapping 30 s chunks',
    'language': 'Chinese',
    'max_tokens': 512,
    'warmup': json.loads((ROOT / 'audiocpp_warmup.json').read_text()),
    'measured_chunks': len(records),
    'total_audio_s': audio_s,
    'total_client_elapsed_s': client_s,
    'client_rtf': client_s / audio_s,
    'client_speed_x_realtime': audio_s / client_s,
    'total_server_wall_s': server_s,
    'server_rtf': server_s / audio_s,
    'server_speed_x_realtime': audio_s / server_s,
    'per_source': {},
}
for source in sorted({r['source'] for r in records}):
    group = [r for r in records if r['source'] == source]
    duration = sum(r['duration_s'] for r in group)
    client = sum(r['client_elapsed_s'] for r in group)
    server = sum((r['server_wall_ms'] or 0) / 1000 for r in group)
    summary['per_source'][source] = {
        'chunks': len(group), 'audio_s': duration,
        'client_elapsed_s': client, 'client_rtf': client/duration,
        'server_wall_s': server, 'server_rtf': server/duration,
        'transcript': ''.join(r['text'] for r in group),
    }
(ROOT / 'audiocpp_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('measured_chunks','total_audio_s','total_client_elapsed_s','client_rtf','client_speed_x_realtime','total_server_wall_s','server_rtf','server_speed_x_realtime')}, indent=2))
