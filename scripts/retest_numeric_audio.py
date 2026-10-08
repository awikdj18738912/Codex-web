"""Run actual audio through the existing Web offline-streaming pipeline."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import websockets
from test_offline_streaming import CHUNK_BYTES, decode_audio


async def run_audio(path, output, base_url, interval):
    audio = decode_audio(path)
    count = (len(audio) + CHUNK_BYTES - 1) // CHUNK_BYTES
    started = time.perf_counter()
    first_text = None
    event_counts = Counter()
    url = base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/stream?'
    url += urlencode(dict(language='Chinese', mode='streaming', domain='general',
        asr_chunk_seconds=interval))
    print(f'START {path.name}: {len(audio)/CHUNK_BYTES:.1f}s audio', flush=True)
    last_progress = started
    latest_raw = latest_clean = ''
    async with websockets.connect(url, max_size=None, ping_timeout=None) as socket:
        ready = json.loads(await asyncio.wait_for(socket.recv(), 120))
        if ready.get('event') != 'ready' or ready.get('asr_chunk_seconds') != interval:
            raise RuntimeError(f'Unexpected ready/configuration: {ready}')
        trace_id = ready['trace_id']
        for index in range(count):
            await socket.send(audio[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES])
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), 330))
                name = event.get('event', '')
                event_counts[name] += 1
                if name == 'error':
                    raise RuntimeError(f'{path.name} chunk {index + 1}: {event}')
                if 'raw_text' in event:
                    latest_raw = event['raw_text']
                    if latest_raw and first_text is None:
                        first_text = time.perf_counter() - started
                if 'clean_text' in event:
                    latest_clean = event['clean_text']
                if name == 'chunk_ack':
                    break
            now = time.perf_counter()
            if now - last_progress >= 25 or index + 1 == count:
                print(f'FEED {path.name} {index+1}/{count} {now-started:.1f}s '
                    f'raw={len(latest_raw)} clean={len(latest_clean)} trace={trace_id}', flush=True)
                last_progress = now
        feed_seconds = time.perf_counter() - started
        await socket.send(json.dumps({'event': 'finish'}))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), 1200))
            name = event.get('event', '')
            event_counts[name] += 1
            if name == 'error':
                raise RuntimeError(f'{path.name} finalization: {event}')
            if time.perf_counter() - last_progress >= 25:
                print(f'DRAIN {path.name} {time.perf_counter()-started:.1f}s '
                    f'event={name} clean={len(event.get("clean_text", ""))}', flush=True)
                last_progress = time.perf_counter()
            if name == 'final':
                result = event
                break
    elapsed = time.perf_counter() - started
    result['_test'] = dict(audio_path=str(path.resolve()), trace_id=trace_id,
        captured_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        audio_duration_s=len(audio)/CHUNK_BYTES, transport_interval_s=1,
        decode_interval_s=interval, elapsed_s=round(elapsed, 3),
        feed_s=round(feed_seconds, 3), finish_s=round(elapsed-feed_seconds, 3),
        first_text_s=round(first_text, 3) if first_text is not None else None,
        event_counts=dict(event_counts))
    output.mkdir(parents=True, exist_ok=True)
    target = output / (path.stem + '.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('DONE ' + json.dumps(dict(result['_test'],
        raw_chars=len(result['raw_text']), clean_chars=len(result['clean_text']),
        refiner_calls=result.get('session_refiner_call_count'),
        numeric_changes=len(result.get('numeric_normalizations', []))), ensure_ascii=False), flush=True)
    return str(target)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8081')
    parser.add_argument('--decode-interval', type=float, default=2)
    args = parser.parse_args()
    with urlopen(args.base_url + '/health', timeout=5) as response:
        health = json.load(response)
    if not health.get('zh_itn_enabled') or health.get('zh_itn_rule_pack') != 'zh-itn-strict-conservative-v90':
        raise RuntimeError(f'Expected live v90 numeric rules: {health}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'configuration.json').write_text(json.dumps(health, ensure_ascii=False, indent=2) + '\n')
    for path in args.audio:
        await run_audio(path, args.output, args.base_url, args.decode_interval)


if __name__ == '__main__':
    asyncio.run(main())
