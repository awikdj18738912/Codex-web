"""Fresh nine-scene pilot with balanced controls and full provenance of rejections."""
import asyncio
from collections import Counter
import json
from pathlib import Path
import random
import re
import time

import run_pilot as p
from src.generators.oral_text_generator import (CORRECTION_STRUCTURES, ORAL_LEVELS,
    BOUNDARY_PUNCTUATION_CONTROL, EXPLANATION_CONTROL, PASSTHROUGH_CONTROL)
from src.pipeline.training_quality import audit_pair

OUT = Path(__file__).resolve().parent


async def main():
    started = time.monotonic()
    client = p.AuditedClient(base_url=p.settings.VLLM_BASE_URL, model="refiner-teacher",
                             timeout=600, max_concurrent=2, max_retries=1)
    seeds = p.SeedGenerator(client)
    # Small fresh pools generated with the current production seed prompts.
    for config in seeds.loader._seed_config["seed_generation"].values():
        config["user_template"] = re.sub(r"Produce(?: at least)? \d+", "Produce 4", config["user_template"])
    scenes = list(p.settings.SCENE_DISTRIBUTION)
    await asyncio.gather(*(p.tagged("seed", scene, seeds.generate_scene_seeds(scene)) for scene in scenes))
    clean = p.CleanTextGenerator(client)
    clean.qc.llm = client
    attempts = []
    accepted = []
    fixed = []
    sem = asyncio.Semaphore(2)
    extra = [
        ("novel_single", "navigation", "嗯，寄到南京，不对，寄到苏州。", "寄到苏州。"),
        ("novel_spelling", "explanation", "联系人是许砚，允许的许，砚台的砚。", "联系人是许砚。"),
        ("novel_english", "vibe_coding", "嗯，用Polars处理Parquet文件，那个，保留session_id。", None),
        ("novel_condition", "meeting", "如果接口超时，就不要重试付款请求。", "如果接口超时，就不要重试付款请求。"),
    ]

    async def do_fixed(entry):
        name, scene, text, expected = entry
        labels = ["boundary_punctuation"] if name.startswith("boundary_") else []
        result = await p.tagged("fixed_clean", name, clean.generate_one(scene, {
            "id": "fixed_" + name, "oral_text": text, "speech_phenomena": labels}))
        calls = [x for x in p.CALLS if x.get("stage") == "fixed_clean" and x.get("id") == name]
        raw = None
        if calls:
            try:
                raw = p.LLMClient._parse_json(calls[-1]["response"]).get("clean_text")
            except (ValueError, KeyError, AttributeError):
                pass
        row = dict(id="fixed_" + name, source=text, expected=expected, raw_target=raw,
                   accepted=result is not None, target=result["clean_text"] if result else None)
        fixed.append(row)
        p.append_jsonl(OUT / "fixed_checks.jsonl", row)

    await asyncio.gather(*(do_fixed(entry) for entry in p.FIXED + extra))
    p.dump_json(OUT / "fixed_checks.json", fixed)

    counts = Counter()
    serials = Counter()

    async def one(scene, scene_index):
        serials[scene] += 1
        number = serials[scene]
        case_id = f"{scene}_{number:02d}"
        kind = (scene_index + number - 1) % 6
        correction = ("none", "single", "rollback", "multiple", "none", "none")[kind]
        level = ("low", "moderate", "high")[number % 3]
        boundary = kind == 4
        passthrough = kind == 5
        generator = p.OralTextGenerator(client, rng=random.Random(61000 + scene_index * 100 + number))

        def controls(_scene):
            if passthrough:
                return PASSTHROUGH_CONTROL, PASSTHROUGH_CONTROL, "None", ["clean"]
            labels = ["correction:" + correction, "oral:" + level]
            additions = []
            if scene == "explanation":
                labels.append("explanation")
                additions.append(EXPLANATION_CONTROL)
            if boundary:
                labels.append("boundary_punctuation")
                additions.append(BOUNDARY_PUNCTUATION_CONTROL)
            return CORRECTION_STRUCTURES[correction], ORAL_LEVELS[level], "\n".join(additions) or "None", labels

        generator._controls = controls
        row = dict(id=case_id, scene=scene, requested_correction=correction,
                   requested_boundary=boundary, requested_passthrough=passthrough)
        item = await p.tagged("oral", case_id, generator.generate_one(scene, number))
        if not item:
            row["status"] = "oral_rejected"
        else:
            item["id"] = case_id
            row["oral"] = item
            result = await p.tagged("clean", case_id, clean.generate_one(scene, item))
            row["clean"] = result
            if result is None:
                row["status"] = "clean_rejected"
            else:
                # Force two selected cases through real ASR simulation; other
                # cases use full aligned pairs. The production defaults stay unchanged.
                simulate = scene in {"daily_chat", "academic"} and number == 1 and not passthrough
                simulator = p.ASRSimulator(client, mode="llm_sim", simulation_ratio=1 if simulate else 0)
                asr = await p.tagged("asr", case_id, simulator.simulate_one(scene, result))
                row["asr"] = asr
                if not asr:
                    row["status"] = "asr_rejected"
                else:
                    records, stats = p.assemble._assemble_from_asr(scene, [asr])
                    row["assembly_stats"] = stats
                    if not records:
                        row["status"] = "assembly_rejected"
                    else:
                        checked, passed, reason = await p.tagged("audit", case_id,
                            p.finalize.qc_one(records[0], client, sem))
                        row["audit"] = checked.get("meta", {}).get("quality_audit")
                        row["audit_reason"] = reason
                        if not passed:
                            row["status"] = "audit_rejected"
                        elif p.exporter.convert_one(checked) is None:
                            row["status"] = "export_rejected"
                        else:
                            row["status"] = "accepted"
                            accepted.append(checked)
                            counts[scene] += 1
                            p.append_jsonl(OUT / "accepted_candidates.jsonl", checked)
        attempts.append(row)
        p.append_jsonl(OUT / "attempts.jsonl", row)
        print(case_id, row["status"], dict(counts), flush=True)

    # Target a nonempty balanced sample rather than accepting an empty set.
    # Failed attempts remain in the report, including any exhausted scenes.
    for round_number in range(10):
        pending = [(scene, index) for index, scene in enumerate(scenes) if counts[scene] < 2]
        if not pending:
            break
        print("ROUND", round_number + 1, "pending", [scene for scene, _ in pending], flush=True)
        await asyncio.gather(*(one(scene, index) for scene, index in pending))

    unique = p.deduplicate(accepted)
    for record in unique:
        p.append_jsonl(OUT / "final/train.jsonl", record)
    p.dump_json(OUT / "final/pilot_sft.json", [p.exporter.convert_one(x) for x in unique])
    summary = dict(total_attempts=len(attempts), statuses=dict(Counter(x["status"] for x in attempts)),
                   accepted=len(accepted), unique=len(unique), scene_counts=dict(counts),
                   scenes_missing_target=[s for s in scenes if counts[s] < 2],
                   fixed_count=len(fixed), fixed_accepted=sum(x["accepted"] for x in fixed),
                   actual_asr_fragments=sum(bool(x["meta"].get("is_fragment")) for x in unique),
                   controls=dict(Counter(tag for x in unique for tag in x["meta"].get("speech_phenomena", [])
                                         if tag.startswith("correction:") or tag in {"clean", "boundary_punctuation", "explanation"})),
                   elapsed_seconds=round(time.monotonic() - started, 2), api_calls=len(p.CALLS),
                   warning="Automatic admissions require independent manual review before declaring this pilot clean.")
    p.dump_json(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    await client.client.close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
