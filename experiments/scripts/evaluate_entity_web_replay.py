"""Replay saved *real* ASR responses through the real Web + Refiner pipeline.

ASR is held fixed across dictionaries. Only its HTTP response transport is
replayed; the WebSocket handler, window scheduler, gate, local GPU Refiner,
protection and validator execute normally. These are quality comparisons,
not measurements of end-to-end online latency.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from evaluate_entity_library import ROOT, append_jsonl, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/entity_library_20261003")
    parser.add_argument("--model", type=Path, default=Path("/home/aim0/data/models/ASR/AgenticASR-Refiner"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", nargs="+", help="optional diagnostic subset; primary evaluation uses all samples")
    parser.add_argument("--conditions", nargs="+", default=["no_dictionary", "current_snapshot_auto", "typed_shadow", "typed_hint", "typed_auto", "all_term_auto"])
    parser.add_argument("--mode", choices=["offline", "streaming", "online"], default="offline")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--cache-model-responses", action="store_true",
                        help="quality-only evaluation: reuse identical greedy Refiner requests; log every hit")
    args = parser.parse_args()
    import torch
    import transformers
    from fastapi.testclient import TestClient
    import system.web_app as web
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    from system.zh_itn_bridge import DEFAULT_LIBRARY

    # A separate process and model on the available GPU keeps the user's
    # running Web service and SQLite database unchanged.
    model = TransformersRefiner(args.model, args.device, 256)
    calls = []
    response_cache = {}

    class AuditedRefiner:
        def __getattr__(self, name):
            value = getattr(model, name)
            if not callable(value) or not name.startswith("refine"):
                return value

            def invoke(*positional, **kwargs):
                key = json.dumps([name,positional,kwargs],ensure_ascii=False,sort_keys=True)
                if args.cache_model_responses and key in response_cache:
                    result = response_cache[key]
                    calls.append(dict(method=name,input=list(positional),kwargs=kwargs,
                                      output=result,wall_seconds=0.0,cache_hit=True))
                    return result
                started = time.perf_counter()
                result = value(*positional, **kwargs)
                if args.cache_model_responses:
                    response_cache[key] = result
                calls.append(dict(method=name, input=list(positional), kwargs=kwargs,
                                  output=result, wall_seconds=time.perf_counter()-started,cache_hit=False))
                return result
            return invoke

    audited_model = AuditedRefiner()
    web.TransformersRefiner = lambda *_args, **_kwargs: audited_model
    plausibility = RepetitionPlausibility(args.device)
    web.RepetitionPlausibility = lambda *_args, **_kwargs: plausibility
    context = {}

    async def replay_http(_url, endpoint, session_id=None, data=b"", params=None, **_kwargs):
        if endpoint == "/stream/start":
            context["chunk_index"] = 0
            return dict(session_id="evaluation-replay", chunk_seconds=0.5)
        if endpoint == "/stream/chunk":
            index = context["chunk_index"]
            context["chunk_index"] += 1
            return context["row"]["chunk_payloads"][index]
        if endpoint == "/stream/finish":
            return context["row"]["final_payload"]
        if endpoint == "/stream/cancel":
            return dict(cancelled=True)
        raise ValueError(endpoint)

    web._stream_request_async = replay_http
    apps = {}
    for condition in args.conditions:
        if condition == "no_dictionary":
            database, mode = None, "off"
        else:
            name, mode = condition.rsplit("_", 1)
            database = args.output/f"{name}.db"
        apps[condition] = web.create_app(args.model, args.device, "http://replayed-asr.invalid",
            "Chinese", 256, None, entity_db=database, entity_fuzzy_mode=mode,
            refinement_gate_mode="tri_state", final_refinement_mode="off",
            zh_itn_library=DEFAULT_LIBRARY)
    config_suffix = "" if "no_dictionary" in args.conditions else "_" + "_".join(args.conditions)
    write_json(args.output/f"web_{args.mode}_configuration{config_suffix}.json", dict(
        model=str(args.model), device=args.device, max_new_tokens=256,
        gate="tri_state", final_refinement="off", numeric_backend="zh-itn",
        conditions=args.conditions, torch=torch.__version__, transformers=transformers.__version__,
        model_config=json.loads((args.model/"config.json").read_text()),
        domain=args.domain,
        cache_model_responses=args.cache_model_responses,
        method="Saved real Qwen3-ASR HTTP payloads replayed into unmodified WebSocket handler; real local GPU Refiner"))
    rows = read_jsonl(args.output/"asr.jsonl")
    if args.sample_ids:
        rows = [row for row in rows if row["id"] in set(args.sample_ids)]
    if args.limit:
        rows = rows[:args.limit]
    destination = args.output/f"web_{args.mode}.jsonl"
    completed = {(r["id"],r["variant"],r["condition"]) for r in read_jsonl(destination)}
    for index, row in enumerate(rows):
        for condition, app in apps.items():
            if (row["id"], row["variant"], condition) in completed:
                continue
            context["row"] = row
            calls.clear()
            events = []
            started = time.perf_counter()
            with TestClient(app) as client:
                from urllib.parse import urlencode
                query = urlencode(dict(mode=args.mode,domain=args.domain,language="Chinese"))
                with client.websocket_connect(f"/ws/stream?{query}") as socket:
                    ready = socket.receive_json()
                    if ready.get("event") != "ready":
                        raise RuntimeError(ready)
                    for _payload in row["chunk_payloads"]:
                        # Transport token only: replay_http returns the exact
                        # response already obtained from this real audio chunk.
                        socket.send_bytes(b"\0\0\0\0")
                        while True:
                            event = socket.receive_json()
                            events.append(event)
                            if event.get("event") == "error":
                                raise RuntimeError(event)
                            if event.get("event") == "chunk_ack":
                                break
                    socket.send_json(dict(event="finish"))
                    while True:
                        event = socket.receive_json()
                        events.append(event)
                        if event.get("event") == "error":
                            raise RuntimeError(event)
                        if event.get("event") == "final":
                            final = event
                            break
            if final.get("raw_text") != row["raw_text"]:
                raise ValueError("ASR replay changed the authoritative transcript")
            append_jsonl(destination, dict(id=row["id"], variant=row["variant"], condition=condition,
                result=final, elapsed_seconds=time.perf_counter()-started, model_calls=list(calls),
                actual_inference_calls=sum(not c.get("cache_hit",False) for c in calls),
                event_counts=dict(Counter(e.get("event") for e in events)),
                intermediate_updates=[e for e in events if e.get("event") == "update"]))
            print(f'WEB {index+1}/{len(rows)} {condition} {row["id"]} {row["variant"]}: {final["clean_text"]} [calls={len(calls)}]', flush=True)


if __name__ == "__main__":
    main()
