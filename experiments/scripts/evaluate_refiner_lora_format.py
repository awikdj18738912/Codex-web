"""Evaluate the strict-local LoRA adapter on its untouched synthetic format test."""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts.prepare_local_edit_sft import placeholder_variant
from experiments.scripts.synthetic_gate_data import digest, file_hash, read_rows, write_json
from system.refinement_protocol import STRICT_LOCAL_PROTOCOL_VERSION, apply_structured_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cuda:0")
    parser.add_argument("--adapter", type=Path, default=ROOT / "results/refiner_lora_20261010")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "experiments/llamafactory/data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/refiner_lora_20261010/format_eval_20261010")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != ROOT / ".runtime/llamafactory-env":
        raise RuntimeError("Use .runtime/llamafactory-env/bin/python")
    os.chdir(ROOT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device.split(":")[1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    adapter = args.adapter.resolve()
    data = args.data_dir.resolve()
    model = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")
    tests = read_rows(data / "test.jsonl")
    audit = {row["seed_id"]: row for row in read_rows(data / "data_audit.jsonl")}
    seeds = {row["seed_id"]: row for row in read_rows(ROOT / "results/gate_synthetic_20261009/texts.jsonl")}
    if len(tests) != 250 or len(audit) != 1000 or len(seeds) != 1000:
        raise RuntimeError("Unexpected frozen synthetic data coverage")
    registered = json.loads((data / "dataset_info.json").read_text())
    if registered["refiner_local_test"]["file_name"] != "test.jsonl":
        raise RuntimeError("Wrong test data registry")
    test_audit = [row for row in read_rows(data / "data_audit.jsonl") if row["exported_split"] == "test"]
    if len(tests) != len(test_audit):
        raise RuntimeError("Registered test data and its audit differ")
    experiment = dict(protocol=STRICT_LOCAL_PROTOCOL_VERSION, test_sha256=file_hash(data / "test.jsonl"),
        adapter_sha256=file_hash(adapter / "adapter_model.safetensors"),
        base_config_sha256=file_hash(model / "config.json"))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    predfile = out / "predictions.jsonl"
    existing = {row["seed_id"]: row for row in read_rows(predfile)}
    if any(row["experiment"] != experiment for row in existing.values()):
        raise RuntimeError("Old predictions use a different adapter or dataset")
    selected = list(zip(tests, test_audit))[:args.limit] if args.limit else list(zip(tests, test_audit))
    def sample_metadata(row, meta):
        if digest(row["messages"]) != meta["messages_sha256"]:
            raise RuntimeError("Registered test rows differ from their preparation audit")
        return meta
    pending = [(row, meta) for row, meta in selected if meta["seed_id"] not in existing]
    if pending:
        import torch
        from peft import PeftModel
        from system.live_qwen_refiner import TransformersRefiner
        torch.set_num_threads(4)
        refiner = TransformersRefiner(model, args.device, 256, window_edit_policy="strict_local")
        refiner._model = PeftModel.from_pretrained(refiner._model, adapter, is_trainable=False).eval()
        for i, (row, audit_meta) in enumerate(pending, 1):
            meta = sample_metadata(row, audit_meta)
            seed = seeds[meta["seed_id"]]
            source, target = seed["spoken_text"], seed["intended_text"]
            if meta["input_variant"] == "synthetic_placeholders":
                source, target = placeholder_variant(source, target)
            if source != row["messages"][1]["content"] or digest(target) != meta["target_text_sha256"]:
                raise RuntimeError("Test input/target identity mismatch")
            started = time.perf_counter()
            response, latency = refiner.refine(source)
            predicted, payload, issue = apply_structured_patch(source, response, policy="strict_local")
            result = dict(experiment=experiment, seed_id=meta["seed_id"], category=meta["category"],
                action=payload.get("action") if isinstance(payload, dict) else None, issue=issue,
                model_response=response, predicted=predicted, expected=target, exact_target_match=predicted == target,
                source_preserved=predicted == source, model_latency_ms=round(latency, 2),
                wall_ms=round((time.perf_counter()-started)*1000, 2), human_gold=False, real_asr_audio=False)
            with predfile.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0 or i == len(pending):
                print(f"EVALUATED {i}/{len(pending)} {meta['seed_id']} valid={issue is None}", flush=True)
        del refiner
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    selected_ids = {meta["seed_id"] for row, meta in selected}
    predictions = {row["seed_id"]: row for row in read_rows(predfile) if row["seed_id"] in selected_ids}
    rows = [predictions[meta["seed_id"]] for row, meta in selected]
    by_category = defaultdict(Counter)
    issue_counts = Counter()
    for row in rows:
        group = by_category[row["category"]]
        group["n"] += 1
        group["valid_json"] += int(row["issue"] is None)
        group["exact_target"] += int(row["exact_target_match"])
        group["source_preserved"] += int(row["source_preserved"])
        group[row["action"] or "invalid"] += 1
        issue_counts[row["issue"] or "accepted"] += 1
    summary = dict(examples=len(rows), by_category={k:dict(v) for k,v in sorted(by_category.items())},
        valid_json_rate=sum(row["issue"] is None for row in rows)/len(rows),
        exact_target_rate=sum(row["exact_target_match"] for row in rows)/len(rows),
        accepted_replace_count=sum(row["issue"] is None and row["action"] == "replace" for row in rows),
        response_outcome_counts=dict(issue_counts),
        mean_model_latency_ms=round(sum(row["model_latency_ms"] for row in rows)/len(rows),2),
        test_sha256=experiment["test_sha256"], adapter_sha256=experiment["adapter_sha256"],
        human_gold=False, real_asr_audio=False,
        scope="synthetic text-format and controlled-target evaluation; not ASR/WER or live-speech accuracy")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
