"""Measure the Web file-streaming path with explicit transport/decode intervals."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import websockets
from test_offline_streaming import CHUNK_BYTES, decode_audio

ROOT = Path(__file__).resolve().parents[1]


def health():
    with urlopen('http://127.0.0.1:8081/health', timeout=5) as response:
        return json.load(response)


def configuration():
    return dict(health=health(), hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/'system/live_qwen_refiner.py', ROOT/'system/web/index.html', ROOT/'system/web_app.py',
                     ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so']})


async def benchmark(path, transport, decode, output, compression='deflate'):
    initial_config = configuration()
    audio = decode_audio(path)
    chunk_bytes = round(CHUNK_BYTES * transport)
    count = (len(audio) + chunk_bytes - 1) // chunk_bytes
    trace = ROOT/'results/web/session.stream_trace.jsonl'
    trace_offset = trace.stat().st_size
    url = 'ws://127.0.0.1:8081/ws/stream?' + urlencode(dict(language='Chinese', mode='streaming',
        domain='general', asr_chunk_seconds=decode))
    event_counts = Counter()
    updates = []
    first_text = first_refined = None
    started = time.perf_counter()
    last_progress = started
    print(f'START {path.name}: audio={len(audio)/CHUNK_BYTES:.1f}s transport={transport}s', flush=True)
    async with websockets.connect(url, max_size=None, ping_timeout=None, compression=compression) as socket:
        ready = json.loads(await asyncio.wait_for(socket.recv(), 120))
        if ready.get('event') != 'ready' or ready.get('asr_chunk_seconds') != decode:
            raise RuntimeError(ready)
        trace_id = ready['trace_id']
        protocol = getattr(socket, 'protocol', socket)
        negotiated_extensions = [type(extension).__name__ for extension in protocol.extensions]

        def observe(event):
            nonlocal first_text, first_refined
            name = event.get('event', '')
            event_counts[name] += 1
            elapsed = time.perf_counter() - started
            if name == 'error': raise RuntimeError(event)
            if first_text is None and event.get('raw_text'): first_text = elapsed
            if name == 'update':
                if first_refined is None: first_refined = elapsed
                updates.append(dict(elapsed_s=round(elapsed, 3), revision=event.get('refinement_revision'),
                    refined_chars=len(event.get('display_refined_text', event.get('clean_text', ''))),
                    pending_chars=len(event.get('pending_raw_text', ''))))

        for index in range(count):
            await socket.send(audio[index*chunk_bytes:(index+1)*chunk_bytes])
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), 330))
                observe(event)
                if event.get('event') == 'chunk_ack': break
            now = time.perf_counter()
            if now-last_progress > 25 or index+1 == count:
                print(f'FEED {path.name} {index+1}/{count} {now-started:.1f}s', flush=True)
                last_progress = now
        feed = time.perf_counter() - started
        await socket.send(json.dumps(dict(event='finish')))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), 1200))
            observe(event)
            if event.get('event') == 'final':
                result = event
                break
    elapsed = time.perf_counter() - started
    if configuration() != initial_config:
        raise RuntimeError('Runtime or measured source changed during the benchmark')
    timings = []
    other_traces = set()
    with trace.open('rb') as handle:
        handle.seek(trace_offset)
        for line in handle:
            row = json.loads(line)
            if row.get('trace_id') != trace_id:
                if row.get('event') in ('asr_chunk_timing', 'asr_session_started'): other_traces.add(row.get('trace_id'))
                continue
            if row.get('event') == 'asr_chunk_timing': timings.append(row)
    if other_traces: raise RuntimeError(f'Concurrent transcription: {other_traces}')
    metrics = dict(trace_id=trace_id, audio=str(path.resolve()),
        captured_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        audio_duration_s=len(audio)/CHUNK_BYTES, transport_interval_s=transport, decode_interval_s=decode,
        elapsed_s=round(elapsed, 3), feed_s=round(feed, 3), finish_s=round(elapsed-feed, 3),
        first_text_s=round(first_text, 3) if first_text is not None else None,
        first_refined_s=round(first_refined, 3) if first_refined is not None else None,
        chunks=count, events=dict(event_counts), configuration=initial_config,
        compression=compression, negotiated_extensions=negotiated_extensions)
    for key in ('model_ms', 'vad_ms', 'request_ms', 'lock_wait_ms'):
        metrics[key.replace('_ms', '_sum_s')] = round(sum(r.get('server_timing', {}).get(key, 0) for r in timings)/1000, 3)
    metrics['web_round_trip_sum_s'] = round(sum(r.get('web_round_trip_ms', 0) for r in timings)/1000, 3)
    result['_speed_benchmark'] = metrics
    result['_stream_updates'] = updates
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('DONE '+json.dumps(dict(metrics, configuration=None, calls=result.get('session_refiner_call_count')),
                           ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--transport', type=float, default=1)
    parser.add_argument('--decode', type=float, default=2)
    parser.add_argument('--compression', choices=['deflate', 'none'], default='deflate')
    args = parser.parse_args()
    if args.transport <= 0 or args.decode <= 0: parser.error('Intervals must be positive')
    for path in args.audio:
        asyncio.run(benchmark(path, args.transport, args.decode, args.output/f'{args.label}_{path.stem}.json',
                              None if args.compression=='none' else args.compression))


if __name__ == '__main__': main()
