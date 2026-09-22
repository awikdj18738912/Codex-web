"""Replay an audio file through the browser's offline-streaming WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

import websockets


SAMPLE_RATE = 16_000
CHUNK_BYTES = SAMPLE_RATE * 4  # One second of mono float32, as in the UI.


def decode_audio(path: Path) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar",
            str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


async def replay(audio: bytes, url: str) -> dict[str, object]:
    total_chunks = (len(audio) + CHUNK_BYTES - 1) // CHUNK_BYTES
    started = time.monotonic()
    async with websockets.connect(url, max_size=None, ping_timeout=None) as socket:
        first = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
        if first.get("event") != "ready":
            raise RuntimeError(f"Expected ready event, got {first!r}")
        for index in range(total_chunks):
            await socket.send(audio[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES])
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=330))
                if event.get("event") == "error":
                    raise RuntimeError(f"Chunk {index + 1}: {event.get('detail')}")
                if event.get("event") == "chunk_ack":
                    break
            if (index + 1) % 60 == 0 or index + 1 == total_chunks:
                print(
                    f"{index + 1}/{total_chunks} chunks, "
                    f"{time.monotonic() - started:.0f}s elapsed",
                    flush=True,
                )
        await socket.send(json.dumps({"event": "finish"}))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=1200))
            if event.get("event") == "error":
                raise RuntimeError(f"Finalization: {event.get('detail')}")
            if event.get("event") == "final":
                return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8081/ws/stream?language=Chinese&mode=streaming&domain=general",
    )
    args = parser.parse_args()
    audio = decode_audio(args.audio)
    print(f"Decoded {len(audio) / CHUNK_BYTES:.1f}s: {args.audio}", flush=True)
    result = asyncio.run(replay(audio, args.url))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved final transcript: {args.output}", flush=True)


if __name__ == "__main__":
    main()
