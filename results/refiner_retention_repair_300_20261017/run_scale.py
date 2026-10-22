"""Frozen-pipeline scale audit: 1000 attempted pairs, independent blind sample."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.environ.update(VLLM_BASE_URL="http://127.0.0.1:8010/v1", VLLM_MODEL_NAME="refiner-teacher",
                  OPENROUTER_API_KEY="", OPENROUTER_MODEL="")
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))
from configs import settings

for key, subdir in (("SEEDS_DIR", "seeds"), ("RAW_DIR", "raw"),
                    ("PROCESSED_DIR", "processed"), ("FINAL_DIR", "final")):
    path = OUT / subdir
    path.mkdir(exist_ok=True)
    setattr(settings, key, str(path))

from src.generators.seed_generator import SeedGenerator
from src.generators.oral_text_generator import OralTextGenerator
from src.generators.clean_text_generator import CleanTextGenerator
from src.generators.asr_simulator import ASRSimulator, _natural_prefixes
from src.pipeline.dedup import deduplicate
from src.services.llm_client import AsyncLLMClient


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline/scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


assemble = module("scale_assemble", "04_assemble.py")
finalize = module("scale_finalize", "05_finalize.py")
exporter = module("scale_export", "export_sft.py")
CONTEXT = ContextVar("request_context", default={})
CALL_STATS = Counter()
CALL_SECONDS = Counter()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


class RecordedClient(AsyncLLMClient):
    async def generate(self, **kwargs):
        started = time.monotonic()
        row = {**CONTEXT.get(), "endpoint": str(self.client.base_url),
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "request": kwargs}
        try:
            result = await super().generate(**kwargs)
            row["response"] = result
            return result
        except Exception as exc:
            row["error"] = type(exc).__name__ + ":" + str(exc)
            raise
        finally:
            row["seconds"] = round(time.monotonic() - started, 3)
            stage = row.get("stage", "unknown")
            CALL_STATS[stage] += 1
            CALL_SECONDS[stage] += row["seconds"]
            append(OUT / "api_calls.jsonl", row)


async def tagged(stage, case_id, coroutine):
    token = CONTEXT.set({"stage": stage, "id": case_id})
    try:
        return await coroutine
    finally:
        CONTEXT.reset(token)


async def ready(url, seconds=900):
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        try:
            def fetch():
                with urllib.request.urlopen(url + "/models", timeout=2) as response:
                    return json.load(response)
            value = await asyncio.to_thread(fetch)
            if any(x.get("id") == "refiner-teacher" for x in value.get("data", [])):
                return
        except Exception:
            pass
        await asyncio.sleep(3)
    raise RuntimeError("Teacher service unavailable: " + url)


def source_files():
    return [str(p.relative_to(ROOT)) for p in sorted((ROOT / "pipeline").rglob("*.py"))
            if "__pycache__" not in str(p)] + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "pipeline/configs").glob("*.json"))]


def make_manifest():
    files = source_files()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    for name in files:
        target = OUT / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(ROOT / name, target)
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(), candidate_target=300,
                  api_model="refiner-teacher", source_sha256=hashes,
                  seed_entries_requested_per_category=16, case_rng_seed=2026101709,
                  review_rng_seed=2026101717, review_sample_target=200,
                  comparison="Matched first 300 jobs, seed pools and RNG from scale1000 baseline; only production code changed.",
                  urls=["http://127.0.0.1:8010/v1", "http://127.0.0.1:8014/v1"],
                  note="Fixed pipeline; candidate failures count toward the denominator. No student training or audio-ASR evaluation.")
    if not (OUT / "manifest.json").exists():
        dump(OUT / "manifest.json", record)
    return hashes


def jobs_for_run():
    weights = settings.SCENE_DISTRIBUTION
    quotas = {scene: int(1000 * ratio) for scene, ratio in weights.items()}
    order = sorted(weights, key=lambda scene: (1000 * weights[scene] - quotas[scene]), reverse=True)
    for scene in order[:1000 - sum(quotas.values())]:
        quotas[scene] += 1
    jobs = [(scene, number) for scene, count in quotas.items() for number in range(1, count + 1)]
    random.Random(2026101709).shuffle(jobs)
    return jobs, quotas


def make_review_sample(rows):
    rng = random.Random(2026101717)
    groups = defaultdict(list)
    for row in rows:
        groups[row["meta"]["scene"]].append(row)
    selected = {}
    for scene, records in sorted(groups.items()):
        rng.shuffle(records)
        # Cover rare branches before filling each scene's initial quota.
        traits = ["clean", "correction:none", "correction:single", "correction:rollback",
                  "correction:multiple", "boundary_punctuation", "explanation", "fragment"]
        for trait in traits:
            matches = [x for x in records if (x["meta"].get("is_fragment") if trait == "fragment"
                       else trait in x["meta"].get("speech_phenomena", []))]
            if matches:
                selected[matches[0]["id"]] = matches[0]
        current = sum(x["meta"]["scene"] == scene for x in selected.values())
        for row in records:
            if current >= min(20, len(records)):
                break
            if row["id"] not in selected:
                selected[row["id"]] = row
                current += 1
    remaining = [row for row in rows if row["id"] not in selected]
    rng.shuffle(remaining)
    for row in remaining[:max(0, min(200, len(rows)) - len(selected))]:
        selected[row["id"]] = row
    sample = list(selected.values())
    rng.shuffle(sample)
    for reviewer in range(2):
        blind = []
        for row in sample[reviewer::2]:
            item = dict(id=row["id"], scene=row["meta"]["scene"], input=row["input"],
                        target=row["output"]["refined_text"], is_fragment=bool(row["meta"].get("is_fragment")))
            if item["is_fragment"]:
                item.update(full_source=row.get("oral_text"), full_target_reference=row["meta"].get("fragment_reference_clean"))
            blind.append(item)
        dump(OUT / f"blind_review_{reviewer + 1}.json", blind)
    dump(OUT / "review_sample_manifest.json", dict(total=len(sample), ids=[x["id"] for x in sample],
         selection="Seeded scene-balanced sample, with rare controls/fragments covered; not a uniform random sample.",
         hidden_from_reviewers=["generation_plan", "expected_clean", "automatic quality_audit", "teacher identity"]))


async def main():
    started = time.monotonic()
    initial_hashes = make_manifest()
    all_jobs, _ = jobs_for_run()
    jobs = all_jobs[:300]
    quotas = dict(Counter(scene for scene, _ in jobs))
    old = []
    if (OUT / "attempts.jsonl").exists():
        old = [json.loads(line) for line in (OUT / "attempts.jsonl").read_text().splitlines()]
    completed_ids = {row["id"] for row in old}
    statuses = Counter(row["status"] for row in old)
    scenes_done = Counter(row["scene"] for row in old)
    accepted = []
    if (OUT / "accepted_candidates.jsonl").exists():
        accepted = [json.loads(line) for line in (OUT / "accepted_candidates.jsonl").read_text().splitlines()]
    dump(OUT / "progress.json", dict(stage="seeds", completed=len(old), target=300, accepted=len(accepted), quotas=quotas))
    clients = [RecordedClient(base_url=url, model="refiner-teacher", max_concurrent=4,
                              timeout=600, max_retries=1) for url in (
                              "http://127.0.0.1:8010/v1", "http://127.0.0.1:8014/v1")]
    second_ready = asyncio.create_task(ready("http://127.0.0.1:8014/v1"))
    await ready("http://127.0.0.1:8010/v1")
    seed_generator = SeedGenerator(clients[0])
    for config in seed_generator.loader._seed_config["seed_generation"].values():
        config["user_template"] = re.sub(r"Produce(?: at least)? \d+", "Produce 16", config["user_template"])
    await asyncio.gather(*(tagged("seed", scene, seed_generator.generate_scene_seeds(scene)) for scene in quotas))
    await second_ready
    print("Both teachers ready; fresh seed pools complete.", flush=True)
    queue = asyncio.Queue()
    for serial, (scene, number) in enumerate(jobs):
        case_id = f"scale_{scene}_{number:04d}"
        if case_id not in completed_ids:
            queue.put_nowait((serial, scene, number, case_id))
    complete = len(old)
    sem = asyncio.Semaphore(8)

    def progress(stage="generation"):
        elapsed = time.monotonic() - started
        record = dict(stage=stage, completed=complete, target=300, accepted=len(accepted),
                      statuses=dict(statuses), scene_completed=dict(scenes_done), quotas=quotas,
                      api_calls=dict(CALL_STATS), elapsed_seconds=round(elapsed, 1),
                      candidates_per_hour=round((complete - len(old)) * 3600 / max(elapsed, 1), 1),
                      updated_utc=datetime.now(timezone.utc).isoformat())
        dump(OUT / "progress.json", record)
        (OUT / "progress.md").write_text(f"# 1000条候选测试\n\n阶段：{stage}\n\n完成：{complete}/300\n\n自动接收（去重前）：{len(accepted)}\n\n拒收：{dict(statuses)}\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    async def worker(worker_id):
        nonlocal complete
        client = clients[worker_id % 2]
        cleaner = CleanTextGenerator(client)
        cleaner.qc.llm = client
        while True:
            try:
                serial, scene, number, case_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            row = dict(id=case_id, scene=scene, worker=worker_id, teacher_endpoint=str(client.client.base_url))
            t0 = time.monotonic()
            try:
                rng = random.Random(2026101709 + serial * 1009)
                generator = OralTextGenerator(client, rng=rng)
                oral = await tagged("oral", case_id, generator.generate_one(scene, number))
                if not oral:
                    row["status"] = "oral_rejected"
                else:
                    oral["id"] = case_id
                    row["oral"] = oral
                    clean = await tagged("clean", case_id, cleaner.generate_one(scene, oral))
                    row["clean"] = clean
                    if not clean:
                        row["status"] = "clean_rejected"
                    else:
                        keywords_ok, reason, _ = await cleaner.qc.validate_keywords_one(clean)
                        if not keywords_ok:
                            row.update(status="keyword_rejected", reason=reason)
                        else:
                            asr_seed = 2026101723 + serial * 401
                            passthrough = "clean" in clean.get("speech_phenomena", [])
                            row["asr_selected"] = not passthrough and random.Random(asr_seed).random() < settings.ASR_SIMULATION_RATIO
                            row["natural_fragment_eligible"] = bool(_natural_prefixes(clean["clean_text"], 8))
                            simulator = ASRSimulator(client, mode="llm_sim", rng=random.Random(asr_seed))
                            asr = await tagged("asr", case_id, simulator.simulate_one(scene, clean))
                            row["asr"] = asr
                            if not asr:
                                row["status"] = "asr_rejected"
                            else:
                                records, stats = assemble._assemble_from_asr(scene, [asr])
                                row["assembly_stats"] = stats
                                if not records:
                                    row["status"] = "assembly_rejected"
                                else:
                                    checked, passed, reason = await tagged("audit", case_id,
                                            finalize.qc_one(records[0], client, sem))
                                    row["audit"] = checked.get("meta", {}).get("quality_audit")
                                    row["audit_reason"] = reason
                                    if not passed:
                                        row["status"] = "audit_rejected"
                                    elif exporter.convert_one(checked) is None:
                                        row["status"] = "export_rejected"
                                    else:
                                        row["status"] = "accepted"
                                        accepted.append(checked)
                                        append(OUT / "accepted_candidates.jsonl", checked)
            except Exception as exc:
                row.update(status="exception", error=type(exc).__name__ + ":" + str(exc))
                logging.exception("Case failed: %s", case_id)
            finally:
                row["seconds"] = round(time.monotonic() - t0, 3)
                append(OUT / "attempts.jsonl", row)
                statuses[row["status"]] += 1
                scenes_done[scene] += 1
                complete += 1
                queue.task_done()
                if complete % 25 == 0 or complete == 300:
                    progress()

    progress()
    await asyncio.gather(*(worker(index) for index in range(8)))
    progress("finalizing")
    # Make dedup deterministic despite concurrent completion order.
    accepted.sort(key=lambda row: row["id"])
    unique = deduplicate(accepted, threshold=settings.DEDUP_THRESHOLD)
    sft = [exporter.convert_one(row) for row in unique]
    if any(row is None for row in sft):
        raise RuntimeError("A previously accepted pair failed final export")
    (OUT / "final/train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique), encoding="utf-8")
    dump(OUT / "final/pilot_sft.json", sft)
    make_review_sample(unique)
    current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in initial_hashes}
    attempts = [json.loads(line) for line in (OUT / "attempts.jsonl").read_text().splitlines()]
    accepted_by_scene = Counter(row["meta"]["scene"] for row in accepted)
    unique_by_scene = Counter(row["meta"]["scene"] for row in unique)
    summary = dict(candidate_target=300, candidates_completed=complete, statuses=dict(statuses),
                   accepted_before_dedup=len(accepted), unique_retained=len(unique),
                   near_duplicates_removed=len(accepted) - len(unique),
                   acceptance_rate=len(accepted) / complete, retained_rate=len(unique) / complete,
                   scene_quotas=quotas, accepted_by_scene=dict(accepted_by_scene), unique_by_scene=dict(unique_by_scene),
                   controls=dict(Counter(tag for row in unique for tag in row["meta"].get("speech_phenomena", [])
                       if tag.startswith("correction:") or tag.startswith("oral:") or tag in {"clean", "explanation", "boundary_punctuation"})),
                   real_fragments=sum(bool(row["meta"].get("is_fragment")) for row in unique),
                   asr_selected=sum(bool(row.get("asr_selected")) for row in attempts),
                   fragment_eligible=sum(bool(row.get("natural_fragment_eligible")) for row in attempts),
                   elapsed_seconds=round(time.monotonic() - started, 2), api_calls=dict(CALL_STATS),
                   source_files_changed=[name for name in initial_hashes if initial_hashes[name] != current_hashes[name]],
                   independent_review_status="pending",
                   limitations=["Synthetic text only; no audio-ASR or student training evaluation", "Teacher also judges semantics",
                       "Blind review is stratified toward scene/rare-branch coverage, not a uniform random error-rate estimate"])
    dump(OUT / "summary.json", summary)
    progress("awaiting_independent_review")
    for client in clients:
        await client.client.close()
    print("GENERATION_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
