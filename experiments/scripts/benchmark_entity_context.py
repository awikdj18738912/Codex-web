"""Paired real-audio experiment for canonical-only Qwen ASR context.

Uses the installed medical glossary; the prompt never contains reference
transcripts, observed ASR errors or per-record answer-derived candidate lists.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
from evaluate_entity_library import (
    ROOT,
    read_jsonl,
    write_json,
    append_jsonl,
    audio_bytes,
    normalized,
)
from system.entity_store import EntityStore
from system.entity_asr_context import build_entity_asr_context, context_digest
from system.zh_itn_bridge import ZhITNNormalizer
from rapidfuzz.distance import Levenshtein


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/entity_optimization_20261003"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT.parent / "AgenticASR_entity_e5bedb3/data/entities.db",
    )
    args = parser.parse_args()
    directory = ROOT / "results/medical_entity_20261003"
    rows = [
        r
        for r in read_jsonl(directory / "development/asr.jsonl")
        if r["id"] == "MultiMed_Chinese_eval_0003"
    ]
    negatives = {"MultiMed_Chinese_test_" + str(n).zfill(4) for n in [13, 15, 101, 118]}
    rows += [r for r in read_jsonl(directory / "asr.jsonl") if r["id"] in negatives]
    ordinary = [r for r in read_jsonl(directory / "controls/asr.jsonl")]
    ordinary_ids = sorted({r["id"] for r in ordinary})[:4]
    rows += [r for r in ordinary if r["id"] in ordinary_ids]
    glossary = build_entity_asr_context(
        EntityStore(args.db).list_entities(domain="medical"), "medical"
    )
    args.output.mkdir(exist_ok=True, parents=True)
    write_json(
        args.output / "context_protocol.json",
        dict(
            terms=glossary.terms,
            context=glossary.text,
            digest=context_digest(glossary.text),
            sample_ids=[dict(id=r["id"], variant=r["variant"]) for r in rows],
            role="Development functional sample plus previously observed negative regressions; not independent generalization benchmark",
            prompt_source="Installed medical entity database; canonical-only, identical across all audio",
        ),
    )
    destination = args.output / "context_asr.jsonl"
    complete = {(r["id"], r["variant"], r["arm"]) for r in read_jsonl(destination)}
    with httpx.Client(timeout=180) as client:
        health = client.get(args.url + "/health").raise_for_status().json()
        if not health.get("context_supported"):
            raise RuntimeError("ASR service must advertise context support")
        write_json(args.output / "context_asr_health.json", health)
        for index, row in enumerate(rows):
            pcm, duration = audio_bytes(
                row, row["variant"], SimpleNamespace(seed=20261003, noise_snr=10.0)
            )
            if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
                raise ValueError("original audio transport hash changed")
            for arm in ["off", "on"] if index % 2 == 0 else ["on", "off"]:
                if (row["id"], row["variant"], arm) in complete:
                    continue
                params = dict(language="Chinese", chunk_seconds=0.5)
                if arm == "on":
                    params["context"] = glossary.text
                started = time.perf_counter()
                ready = (
                    client.post(args.url + "/stream/start", params=params)
                    .raise_for_status()
                    .json()
                )
                if ready.get("context_applied") != (arm == "on"):
                    raise ValueError("ASR context acknowledgement mismatch")
                if arm == "on" and ready.get("context_digest") != context_digest(
                    glossary.text
                ):
                    raise ValueError("context digest mismatch")
                sid = ready["session_id"]
                try:
                    for offset in range(0, len(pcm), 64000):
                        client.post(
                            args.url + "/stream/chunk",
                            params=dict(session_id=sid),
                            content=pcm[offset : offset + 64000],
                        ).raise_for_status()
                    final = (
                        client.post(
                            args.url + "/stream/finish", params=dict(session_id=sid)
                        )
                        .raise_for_status()
                        .json()
                    )
                except BaseException:
                    client.post(
                        args.url + "/stream/cancel", params=dict(session_id=sid)
                    )
                    raise
                append_jsonl(
                    destination,
                    dict(
                        id=row["id"],
                        variant=row["variant"],
                        arm=arm,
                        reference=row["reference"],
                        text=final["text"],
                        duration_seconds=duration,
                        elapsed_seconds=time.perf_counter() - started,
                        original_audio_hash=row["pcm_sha256"],
                        ready=ready,
                        result=final,
                        tavi_expected="tavi" in normalized(row["reference"]),
                    ),
                )
                print(
                    "ASR-CONTEXT",
                    index + 1,
                    "/",
                    len(rows),
                    arm,
                    row["id"],
                    row["variant"],
                    final["text"],
                    flush=True,
                )
    normalizer = ZhITNNormalizer()
    summary = {}
    for arm in ["off", "on"]:
        selected = [r for r in read_jsonl(destination) if r["arm"] == arm]
        target = [r for r in selected if r["tavi_expected"]]
        negative = [r for r in selected if not r["tavi_expected"]]
        errors = sum(
            Levenshtein.distance(
                normalized(normalizer.normalize(r["reference"]).text),
                normalized(normalizer.normalize(r["text"]).text),
            )
            for r in selected
        )
        chars = sum(
            len(normalized(normalizer.normalize(r["reference"]).text)) for r in selected
        )
        summary[arm] = dict(
            samples=len(selected),
            itn_cer_percent=100 * errors / chars,
            target_samples=len(target),
            tavi_recovered=sum("tavi" in normalized(r["text"]) for r in target),
            tavi_inserted_in_negative=sum(
                "tavi" in normalized(r["text"]) for r in negative
            ),
        )
    write_json(args.output / "context_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
