#!/usr/bin/env python3
"""Repeatable eager/CUDA Graph benchmark against a local OpenAI-compatible API.

The benchmark uses the frozen requests.json beside this script. It performs
eight untimed warmup calls, then times one concurrent pass over all 16 inputs.
Responses are benchmark artifacts only and must not be added to training data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


BASE_URL = "http://127.0.0.1:8010/v1"
MODEL = "refiner-teacher"
CONCURRENCY = 8
COMPLETION_TOKENS = 256
TIMEOUT_SECONDS = 180.0
SEED = 20261018
REQUESTS_PATH = Path(__file__).with_name("requests.json")


def load_requests() -> dict[str, Any]:
    with REQUESTS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != 16:
        raise ValueError(f"{REQUESTS_PATH} must contain exactly 16 frozen requests")
    if any(not isinstance(item.get("oral_text"), str) for item in requests):
        raise ValueError("Every frozen request must contain string oral_text")
    return payload


async def call_one(
    client: AsyncOpenAI, semaphore: asyncio.Semaphore, item: dict[str, Any], phase: str
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "phase": phase,
        "id": item["id"],
        "latency_seconds": None,
        "usage": None,
        "response": None,
        "error": None,
    }
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["oral_text"]},
                ],
                max_tokens=COMPLETION_TOKENS,
                temperature=0,
                seed=SEED,
                extra_body={"ignore_eos": True},
            )
            result["usage"] = (
                response.usage.model_dump()
                if response.usage is not None
                else None
            )
            result["response"] = {
                "content": response.choices[0].message.content if response.choices else None,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "model": response.model,
                "request_id": getattr(response, "_request_id", None),
            }
        except Exception as exc:  # Preserve API errors, including unsupported request parameters.
            result["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            result["latency_seconds"] = time.perf_counter() - started
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen = load_requests()
    samples = [
        {"id": item["id"], "oral_text": item["oral_text"], "system": frozen["system"]}
        for item in frozen["requests"]
    ]
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(CONCURRENCY)
    try:
        warmup_results = await asyncio.gather(
            *(call_one(client, semaphore, item, "warmup") for item in samples[:8])
        )
        timed_started = time.perf_counter()
        timed_results = await asyncio.gather(
            *(call_one(client, semaphore, item, "timed") for item in samples)
        )
        walltime = time.perf_counter() - timed_started
    finally:
        await client.close()

    completion_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0)
        for row in timed_results
    )
    latencies = [float(row["latency_seconds"]) for row in timed_results]
    errors = [row for row in timed_results if row["error"]]
    warmup_errors = [row for row in warmup_results if row["error"]]
    report = {
        "label": args.label,
        "endpoint": BASE_URL,
        "model": MODEL,
        "request_parameters": {
            "concurrency": CONCURRENCY,
            "max_tokens": COMPLETION_TOKENS,
            "extra_body": {"ignore_eos": True},
            "temperature": 0,
            "seed": SEED,
            "timeout_seconds": TIMEOUT_SECONDS,
            "openai_max_retries": 0,
        },
        "frozen_requests": str(REQUESTS_PATH),
        "training_use": "benchmark responses only; do not add to any training dataset",
        "warmup": {"count": len(warmup_results), "results": warmup_results},
        "timed": {
            "count": len(timed_results),
            "walltime_seconds": walltime,
            "sum_completion_tokens": completion_tokens,
            "completion_tokens_per_second": completion_tokens / walltime if walltime else None,
            "latency_seconds": {
                "min": min(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "error_count": len(errors),
            "errors": errors,
            "usage_details": [
                {"id": row["id"], "usage": row["usage"], "error": row["error"]}
                for row in timed_results
            ],
            "results": timed_results,
        },
        "warmup_error_count": len(warmup_errors),
        "warmup_errors": warmup_errors,
        "note": "No parameter fallback is attempted; unsupported seed/ignore_eos errors are recorded verbatim.",
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "label": args.label,
        "output": str(output),
        "walltime_seconds": report["timed"]["walltime_seconds"],
        "sum_completion_tokens": completion_tokens,
        "completion_tokens_per_second": report["timed"]["completion_tokens_per_second"],
        "latency_seconds": report["timed"]["latency_seconds"],
        "error_count": len(errors),
        "warmup_error_count": len(warmup_errors),
    }, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("eager", "graph"), required=True)
    parser.add_argument("--output", required=True, help="Path for the JSON report and responses")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
