"""Generate a retained-size corpus with durable checkpoints and final acceptance.

The default invocation freezes the implementation and native numeric library
under the output directory, then runs that frozen copy. Rejected and duplicate
candidates are replaced automatically until each scene's retained quota is met.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CTX = ContextVar("corpus_request", default={})
LOG = logging.getLogger("corpus")


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def scene_quotas(total, weights):
    counts = {scene: int(total * weight) for scene, weight in weights.items()}
    order = sorted(weights, key=lambda s: (total * weights[s] - counts[s]), reverse=True)
    for scene in order[:total - sum(counts.values())]:
        counts[scene] += 1
    if sum(counts.values()) != total:
        raise ValueError("Scene weights do not sum to one")
    return counts


def plan_batch(quotas, retained, size, first_serial, seed):
    """Allocate no more candidates than any scene's outstanding retained quota."""
    remaining = {s: n - retained.get(s, 0) for s, n in quotas.items()}
    selected = []
    planned = Counter()
    for _ in range(min(size, sum(remaining.values()))):
        scene = max(remaining, key=lambda s: (remaining[s] - planned[s]) / max(1, quotas[s]))
        if planned[scene] >= remaining[scene]:
            raise RuntimeError("Invalid batch allocation")
        selected.append(scene)
        planned[scene] += 1
    random.Random(seed + first_serial * 1009).shuffle(selected)
    return [(first_serial + i, scene) for i, scene in enumerate(selected)]


class CorpusStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts(serial INTEGER PRIMARY KEY, scene TEXT NOT NULL,
            status TEXT NOT NULL, reason TEXT, record TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS retained(id TEXT PRIMARY KEY, serial INTEGER UNIQUE,
            scene TEXT NOT NULL, input_hash TEXT NOT NULL, target_hash TEXT NOT NULL,
            intent_key TEXT NOT NULL, record TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS retained_scene ON retained(scene);
          CREATE TABLE IF NOT EXISTS responses(request_hash TEXT PRIMARY KEY, case_id TEXT,
            stage TEXT, endpoint TEXT, request TEXT, response TEXT NOT NULL, seconds REAL);
          CREATE TABLE IF NOT EXISTS request_errors(id INTEGER PRIMARY KEY, case_id TEXT,
            stage TEXT, endpoint TEXT, error TEXT, seconds REAL, utc TEXT);
        """)
        self.db.commit()

    def metadata(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_metadata(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value)))
        self.db.commit()

    def response(self, key):
        row = self.db.execute("SELECT response FROM responses WHERE request_hash=?", (key,)).fetchone()
        return row[0] if row else None

    def save_response(self, key, context, endpoint, kwargs, response, seconds):
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?)",
            (key, context.get("id"), context.get("stage"), endpoint,
             canonical(kwargs), response, seconds))
        self.db.commit()

    def save_error(self, context, endpoint, error, seconds):
        self.db.execute("INSERT INTO request_errors(case_id,stage,endpoint,error,seconds,utc) VALUES (?,?,?,?,?,?)",
            (context.get("id"), context.get("stage"), endpoint, error, seconds, utc()))
        self.db.commit()

    def retained_rows(self):
        for (text,) in self.db.execute("SELECT record FROM retained ORDER BY serial"):
            yield json.loads(text)

    def next_serial(self):
        return self.db.execute("SELECT COALESCE(MAX(serial),-1)+1 FROM attempts").fetchone()[0]

    def commit_batch(self, rows, retained):
        with self.db:
            self.db.executemany("INSERT INTO attempts VALUES (?,?,?,?,?)",
                [(r["serial"], r["scene"], r["status"], r.get("reason", ""), canonical(r)) for r in rows])
            for serial, record, intent_key in retained:
                self.db.execute("INSERT INTO retained VALUES (?,?,?,?,?,?,?)",
                    (record["id"], serial, record["meta"]["scene"], digest(record["input"]),
                     digest(record["output"]["refined_text"]), intent_key, canonical(record)))


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline/scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def freeze(args):
    snapshot = args.output / "source_snapshot"
    manifest_path = args.output / "source_manifest.json"
    if not manifest_path.exists():
        if snapshot.exists():
            raise RuntimeError("Incomplete source snapshot exists; inspect it before resuming")
        snapshot.mkdir()
        hashes = {}
        files = sorted((ROOT / "pipeline").rglob("*.py")) + sorted((ROOT / "pipeline/configs").glob("*.json"))
        files += sorted((ROOT / "system").rglob("*.py"))
        files += [ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so"]
        for source in files:
            rel = source.relative_to(ROOT)
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
        atomic_json(manifest_path, {"created_utc": utc(), "source_root": str(ROOT), "sha256": hashes})
    saved = json.loads(manifest_path.read_text())
    for name, expected in saved["sha256"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen source changed: " + name)
    command = [sys.executable, str(snapshot / "pipeline/scripts/generate_corpus.py"), *sys.argv[1:], "--frozen"]
    return subprocess.call(command, env=os.environ.copy())


async def generate(args):
    sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT)]
    os.environ.update(VLLM_BASE_URL=args.base_url, VLLM_MODEL_NAME=args.model,
                      OPENROUTER_API_KEY="", OPENROUTER_MODEL="", CLEAN_TARGET_MODE="construction",
                      REFINER_TRAIN_ZH_ITN_LIBRARY=str(ROOT/"zh-itn/zh-itn/build/libzh_itn_bridge.so"))
    from configs import settings
    for key, subdir in [("SEEDS_DIR", "seeds"), ("RAW_DIR", "raw"),
                        ("PROCESSED_DIR", "processed"), ("FINAL_DIR", "final")]:
        folder = args.output / subdir
        folder.mkdir(exist_ok=True)
        setattr(settings, key, str(folder))
    from src.generators.seed_generator import SeedGenerator
    from src.generators.oral_text_generator import OralTextGenerator, _flatten_seed_values
    from src.generators.clean_text_generator import CleanTextGenerator
    from src.generators.asr_simulator import ASRSimulator
    from src.pipeline.dedup import StreamingDeduplicator
    from src.services.llm_client import AsyncLLMClient
    from src.utils.numeric_policy import normalize_source

    assemble = load_script("bulk_assemble", "04_assemble.py")
    finalize = load_script("bulk_finalize", "05_finalize.py")
    exporter = load_script("bulk_export", "export_sft.py")
    audit = load_script("bulk_acceptance", "audit_dataset.py")
    store = CorpusStore(args.output / "state.sqlite3")
    quotas = scene_quotas(args.target, settings.SCENE_DISTRIBUTION)
    configuration = {"target": args.target, "quotas": quotas, "seed": args.seed,
                     "batch_size": args.batch_size, "model": args.model, "base_url": args.base_url,
                     "dedup_threshold": settings.DEDUP_THRESHOLD, "max_same_intent": args.max_same_intent,
                     "seed_target": args.seed_target, "bootstrap_seeds": args.bootstrap_seeds,
                     "seed_source": str(args.seed_source) if args.seed_source else None,
                     "readiness_target": args.readiness_target}
    saved = store.metadata("configuration")
    if saved is not None and saved != configuration:
        raise RuntimeError("Resume configuration differs from the recorded dataset configuration")
    store.set_metadata("configuration", configuration)
    if store.metadata("created_utc") is None:
        store.set_metadata("created_utc", utc())
    source_manifest = json.loads((args.output / "source_manifest.json").read_text())
    fingerprint = digest(canonical(source_manifest["sha256"]))
    if store.metadata("source_fingerprint") not in (None, fingerprint):
        raise RuntimeError("Frozen implementation differs from the corpus database fingerprint")
    store.set_metadata("source_fingerprint", fingerprint)
    started = time.monotonic()
    accumulated = store.metadata("active_seconds", 0.0)
    retained_by_scene = Counter()
    intent_counts = Counter()
    input_targets = {}
    deduper = StreamingDeduplicator(settings.DEDUP_THRESHOLD)
    for item in store.retained_rows():
        if not deduper.add(item):
            raise RuntimeError("Persisted corpus contains a duplicate")
        retained_by_scene[item["meta"]["scene"]] += 1
        input_targets[digest(item["input"])] = digest(item["output"]["refined_text"])
        intent_counts[exporter.intent_group_key(item)] += 1
    statuses = Counter(dict(store.db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")))
    attempted_by_scene = Counter(dict(store.db.execute("SELECT scene,COUNT(*) FROM attempts GROUP BY scene")))
    first_serial = store.next_serial()
    first_retained = sum(retained_by_scene.values())
    stage = "preparing"
    runtime = {"live_requests": Counter(), "cached_requests": Counter(), "inflight": 0}
    (args.output / "worker.pid").write_text(str(os.getpid()) + "\n")
    model_info = await asyncio.to_thread(fetch_models, args.base_url, args.model)
    model_identity = {key: model_info.get(key) for key in ("id", "root", "parent", "max_model_len")}
    if store.metadata("model_identity") not in (None, model_identity):
        raise RuntimeError("Teacher identity changed since this corpus was started")
    store.set_metadata("model_identity", model_identity)
    store.set_metadata("model_info", model_info)

    class RecordedClient(AsyncLLMClient):
        async def generate(self, **kwargs):
            context = dict(CTX.get())
            key = digest(canonical({"context": context, "request": kwargs}))
            previous = store.response(key)
            if previous is not None:
                runtime["cached_requests"][context.get("stage", "unknown")] += 1
                return previous
            request_started = time.monotonic()
            runtime["inflight"] += 1
            try:
                response = await super().generate(**kwargs)
                store.save_response(key, context, str(self.client.base_url), kwargs, response,
                                    time.monotonic() - request_started)
                runtime["live_requests"][context.get("stage", "unknown")] += 1
                return response
            except Exception as exc:
                store.save_error(context, str(self.client.base_url), type(exc).__name__ + ":" + str(exc),
                                 time.monotonic() - request_started)
                raise
            finally:
                runtime["inflight"] -= 1

    client = RecordedClient(base_url=args.base_url, model=args.model,
                            max_concurrent=args.concurrency, timeout=600, max_retries=2)
    cleaner = CleanTextGenerator(client, target_mode="construction")
    cleaner.qc.llm = client
    seedgen = SeedGenerator(client)
    sem = asyncio.Semaphore(args.concurrency)
    seed_locks = {s: asyncio.Lock() for s in quotas}
    seed_refresh_count = Counter(store.metadata("seed_refresh_count", {}))
    seed_last_attempt = Counter(store.metadata("seed_last_attempt", {}))
    seed_requests = store.metadata("seed_request_serial", 0)
    seed_hashes = store.metadata("seed_pool_sha256", {})

    def remember_seed(scene):
        path = Path(settings.SEEDS_DIR)/f"{scene}.json"
        seed_hashes[scene] = hashlib.sha256(path.read_bytes()).hexdigest()
        store.set_metadata("seed_pool_sha256", seed_hashes)

    def seed_size(scene):
        path = Path(settings.SEEDS_DIR)/f"{scene}.json"
        return len(_flatten_seed_values(json.loads(path.read_text()))) if path.is_file() else 0

    async def tagged(name, case_id, coroutine):
        token = CTX.set({"stage": name, "id": case_id})
        try:
            return await coroutine
        finally:
            CTX.reset(token)

    def progress(error=None):
        elapsed = time.monotonic() - started
        gained = sum(retained_by_scene.values()) - first_retained
        hourly = gained * 3600 / max(1.0, elapsed)
        record = {"stage": stage, "target": args.target, "retained": sum(retained_by_scene.values()),
                  "retained_by_scene": dict(retained_by_scene), "scene_quotas": quotas,
                  "attempts": sum(statuses.values()), "statuses": dict(statuses),
                  "active_seconds": round(accumulated + elapsed, 1), "updated_utc": utc(),
                  "worker_pid": os.getpid(), "live_requests_this_run": dict(runtime["live_requests"]),
                  "cached_requests_this_run": dict(runtime["cached_requests"]),
                  "inflight_requests": runtime["inflight"], "retained_per_hour_this_run": round(hourly, 1),
                  "estimated_remaining_hours": round((args.target-sum(retained_by_scene.values()))/hourly, 2)
                       if hourly and elapsed > 120 else None,
                  "seed_counts": {s: seed_size(s) for s in quotas}}
        if error:
            record["error"] = error
        atomic_json(args.output / "progress.json", record)
        return record

    async def heartbeat():
        while True:
            progress()
            await asyncio.sleep(20)

    async def refresh_seeds(scene):
        nonlocal seed_requests
        async with seed_locks[scene]:
            for retry in range(3):
                number = seed_refresh_count[scene]
                seed_requests += 1
                store.set_metadata("seed_request_serial", seed_requests)
                try:
                    await tagged("seed", f"seed_{scene}_{seed_requests}",
                                 seedgen.generate_incremental(scene, number))
                    seed_refresh_count[scene] += 1
                    remember_seed(scene)
                    seed_last_attempt[scene] = attempted_by_scene[scene]
                    store.set_metadata("seed_refresh_count", dict(seed_refresh_count))
                    store.set_metadata("seed_last_attempt", dict(seed_last_attempt))
                    return
                except Exception as exc:
                    LOG.warning("Seed refresh %s failed (%s): %s", scene, retry+1, exc)
            raise RuntimeError("Seed generation failed repeatedly for " + scene)

    async def bootstrap_scene(scene):
        path = Path(settings.SEEDS_DIR) / f"{scene}.json"
        if scene in seed_hashes and (not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != seed_hashes[scene]):
            raise RuntimeError("Recorded seed pool was removed or changed outside this run: " + scene)
        if not path.exists() and args.seed_source:
            source = args.seed_source / path.name
            if source.is_file():
                shutil.copy2(source, path)
        previous = 0
        for _ in range(20):
            pool = seedgen.load_seeds(scene)
            if path.is_file():
                remember_seed(scene)
            count = len(set(_flatten_seed_values(pool)))
            if count >= min(args.bootstrap_seeds, args.seed_target):
                return
            await refresh_seeds(scene)
            current = len(set(_flatten_seed_values(seedgen.load_seeds(scene))))
            if current <= count and count <= previous:
                raise RuntimeError("Seed pool is not growing for " + scene)
            previous = count
        raise RuntimeError("Bootstrap seed round limit reached for " + scene)

    async def candidate(serial, scene):
        cid = f"{scene}_{serial:08d}"
        row = {"serial": serial, "id": cid, "scene": scene}
        t0 = time.monotonic()
        try:
            generator = OralTextGenerator(client, rng=random.Random(args.seed + serial * 1009))
            oral = await tagged("oral", cid, generator.generate_one(scene, serial))
            if not oral:
                row["status"] = "oral_rejected"
                return row, None
            oral["id"] = cid
            row["oral"] = oral
            clean = await tagged("clean", cid, cleaner.generate_one(scene, oral))
            if not clean:
                row["status"] = "clean_rejected"
                return row, None
            row["clean"] = clean
            ok, reason, _ = await cleaner.qc.validate_keywords_one(clean)
            if not ok:
                row.update(status="keyword_rejected", reason=reason)
                return row, None
            simulator = ASRSimulator(client, mode="llm_sim", rng=random.Random(args.seed + serial * 401 + 23))
            asr = await tagged("asr", cid, simulator.simulate_one(scene, clean))
            if not asr:
                row["status"] = "asr_rejected"
                return row, None
            row["asr"] = asr
            records, stats = assemble._assemble_from_asr(scene, [asr])
            row["assembly_stats"] = stats
            if not records:
                row["status"] = "assembly_rejected"
                return row, None
            checked, passed, reason = await tagged("audit", cid, finalize.qc_one(records[0], client, sem))
            row["quality_audit"] = checked.get("meta", {}).get("quality_audit")
            if not passed:
                row.update(status="audit_rejected", reason=reason)
                return row, None
            if exporter.convert_one(checked) is None:
                row["status"] = "export_rejected"
                return row, None
            row["status"] = "quality_pass"
            return row, checked
        except Exception as exc:
            row.update(status="exception", reason=type(exc).__name__+":"+str(exc))
            LOG.exception("Candidate failed: %s", cid)
            return row, None
        finally:
            row["seconds"] = round(time.monotonic() - t0, 3)

    heart = asyncio.create_task(heartbeat())
    try:
        # Load native numeric rules before any expensive generation.
        normalize_source("明天下午三点开会。")
        stage = "preparing_seeds"
        await asyncio.gather(*(bootstrap_scene(s) for s, n in quotas.items() if n))
        stage = "generating"
        stalled = 0
        while sum(retained_by_scene.values()) < args.target:
            if (args.readiness_target and not store.metadata("readiness_released", False)
                    and sum(retained_by_scene.values()) >= args.readiness_target):
                stage = "awaiting_readiness_release"
                folder = args.output/"readiness"
                folder.mkdir(exist_ok=True)
                path = folder/"train.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    for item in store.retained_rows():
                        stream.write(json.dumps(item, ensure_ascii=False)+"\n")
                check, sample = audit.audit(path, folder/"acceptance_report.json", None,
                                            args.review_sample, args.seed, min_records=args.readiness_target)
                atomic_json(folder/"acceptance_report.json", check)
                with (folder/"review_sample.jsonl").open("w", encoding="utf-8") as stream:
                    for item in sample:
                        stream.write(json.dumps(item, ensure_ascii=False)+"\n")
                if not check["passed"]:
                    raise RuntimeError("Readiness subset failed automatic acceptance")
                progress()
                release_path = args.output/"readiness_release.json"
                while True:
                    if release_path.is_file():
                        release = json.loads(release_path.read_text())
                        if (release.get("allow") is True and release.get("source_fingerprint") == fingerprint
                                and release.get("readiness_sha256") == check["input_sha256"]):
                            store.set_metadata("readiness_released", {"utc": utc(), "evidence": release})
                            break
                        raise RuntimeError("Readiness release does not match this frozen corpus subset")
                    await asyncio.sleep(10)
                stage = "generating"
            serial = store.next_serial()
            if serial >= args.max_candidates:
                raise RuntimeError("Candidate budget exhausted before retained target")
            if shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB free in corpus output filesystem")
            batch_size = min(args.batch_size, args.max_candidates-serial)
            if args.readiness_target and not store.metadata("readiness_released", False):
                batch_size = min(batch_size, args.readiness_target-sum(retained_by_scene.values()))
            jobs = plan_batch(quotas, retained_by_scene, batch_size, serial, args.seed)
            # At most concurrency workers, not hundreds of queued OpenAI calls.
            queue = asyncio.Queue()
            for job in jobs:
                queue.put_nowait(job)
            results = []
            async def worker():
                while not queue.empty():
                    number, scene = queue.get_nowait()
                    results.append(await candidate(number, scene))
                    queue.task_done()
            await asyncio.gather(*(worker() for _ in range(min(args.concurrency, len(jobs)))))
            results.sort(key=lambda result: result[0]["serial"])
            before = sum(retained_by_scene.values())
            committed = []
            for row, item in results:
                if item is not None:
                    key = exporter.intent_group_key(item)
                    ih, th = digest(item["input"]), digest(item["output"]["refined_text"])
                    if retained_by_scene[row["scene"]] >= quotas[row["scene"]]:
                        row["status"] = "quota_excess"
                    elif ih in input_targets and input_targets[ih] != th:
                        row["status"] = "input_target_conflict"
                    elif intent_counts[key] >= args.max_same_intent:
                        row["status"] = "intent_frequency_limit"
                    elif not deduper.add(item):
                        row["status"] = "near_duplicate"
                    else:
                        row["status"] = "retained"
                        retained_by_scene[row["scene"]] += 1
                        intent_counts[key] += 1
                        input_targets[ih] = th
                        committed.append((row["serial"], item, key))
                statuses[row["status"]] += 1
                attempted_by_scene[row["scene"]] += 1
            store.commit_batch([r for r, _ in results], committed)
            store.set_metadata("active_seconds", accumulated + time.monotonic() - started)
            state = progress()
            LOG.info("retained=%s/%s attempts=%s batch_retained=%s", state["retained"], args.target,
                     state["attempts"], state["retained"]-before)
            stalled = stalled+1 if state["retained"] == before else 0
            if stalled >= 12:
                raise RuntimeError("Twelve batches retained no data; inspect rejects before resuming")
            refresh = []
            for scene, needed in quotas.items():
                if retained_by_scene[scene] >= needed:
                    continue
                count = seed_size(scene)
                if count < args.seed_target and (attempted_by_scene[scene] - seed_last_attempt[scene] >= 1000 or stalled >= 2):
                    refresh.append(scene)
            if refresh:
                stage = "refreshing_seeds"
                await asyncio.gather(*(refresh_seeds(s) for s in refresh))
                stage = "generating"
        stage = "final_acceptance"
        final_path = args.output / "final/train.jsonl"
        temporary = final_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for item in store.retained_rows():
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary.replace(final_path)
        report, samples = audit.audit(final_path, args.output/"acceptance_report.json", None,
                                      args.review_sample, args.seed, min_records=args.target)
        atomic_json(args.output / "acceptance_report.json", report)
        with (args.output/"review_sample.jsonl").open("w", encoding="utf-8") as stream:
            for item in samples:
                stream.write(json.dumps(item, ensure_ascii=False)+"\n")
        if not report["passed"] or report["counts"]["records"] != args.target:
            raise RuntimeError("Final full-corpus acceptance failed")
        stage = "exporting"
        sft = []
        for item in store.retained_rows():
            converted = exporter.convert_one(item)
            if converted is None:
                raise RuntimeError("Previously retained record failed final SFT export")
            converted.update(_scene=item["meta"]["scene"], _group_key=exporter.intent_group_key(item),
                _length_bucket=exporter.get_length_bucket(item["output"]["refined_text"]))
            sft.append(converted)
        train, val = exporter.grouped_split(sft, 0.15, random.Random(args.seed), stratified=True)
        train_keys = {r["_group_key"] for r in train}
        val_keys = {r["_group_key"] for r in val}
        if train_keys & val_keys:
            raise RuntimeError("Intent leakage between training and validation")
        for subset in (train, val):
            for item in subset:
                for key in ("_scene", "_group_key", "_length_bucket"):
                    item.pop(key, None)
        atomic_json(args.output/"final/train_sft.json", train)
        atomic_json(args.output/"final/val_sft.json", val)
        atomic_json(args.output/"final/dataset_info.json", {
            "refiner_train": {"file_name": "train_sft.json", "formatting": "sharegpt",
                "columns": {"messages": "conversations"}},
            "refiner_val": {"file_name": "val_sft.json", "formatting": "sharegpt",
                "columns": {"messages": "conversations"}}})
        stage = "complete"
        summary = progress()
        summary.update(train_records=len(train), validation_records=len(val), unique_intents=len(intent_counts),
                       grouped_split_overlap=0, acceptance_passed=True, completed_utc=utc(),
                       final_sha256=report["input_sha256"], numeric_owner="native_rules")
        atomic_json(args.output/"summary.json", summary)
        store.set_metadata("completed", True)
        return summary
    except BaseException as exc:
        stage = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        progress(type(exc).__name__ + ":" + str(exc))
        raise
    finally:
        heart.cancel()
        await asyncio.gather(heart, return_exceptions=True)
        store.set_metadata("active_seconds", accumulated + time.monotonic() - started)
        await client.client.close()
        store.db.close()


def fetch_models(base_url, model):
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
        value = json.load(response)
    match = next((row for row in value.get("data", []) if row.get("id") == model), None)
    if match is None:
        raise RuntimeError("Teacher model is not served: " + model)
    return match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="refiner-teacher")
    parser.add_argument("--seed", type=int, default=2026101717)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=500000)
    parser.add_argument("--seed-source", type=Path)
    parser.add_argument("--bootstrap-seeds", type=int, default=500)
    parser.add_argument("--seed-target", type=int, default=3000)
    parser.add_argument("--max-same-intent", type=int, default=3)
    parser.add_argument("--review-sample", type=int, default=300)
    parser.add_argument("--readiness-target", type=int, default=0,
                        help="pause at this many retained pairs until a matching readiness_release.json exists")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.target, args.concurrency, args.batch_size, args.bootstrap_seeds,
           args.seed_target, args.max_same_intent) <= 0 or args.max_candidates < args.target or args.review_sample < 0:
        parser.error("Positive sizes required; max-candidates must be at least target")
    if args.readiness_target < 0 or args.readiness_target >= args.target:
        parser.error("readiness-target must be zero or positive and smaller than target")
    args.output = args.output.expanduser().resolve()
    if args.seed_source:
        args.seed_source = args.seed_source.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    lock = (args.output / ("worker.lock" if args.frozen else "launcher.lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This corpus is already running")
    if not args.frozen:
        return freeze(args)
    result = asyncio.run(generate(args))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
