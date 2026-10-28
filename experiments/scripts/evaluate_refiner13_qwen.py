"""Evaluate base and LoRA Refiner models on the frozen provisional test split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path("/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional")
BASE = Path("/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct")
DEFAULT_OUTPUT = Path("/home/aim0/桌面/tan/ASR/Codex-web/results/refiner13_qwen25_lora_20261026")
PLACEHOLDER_RE = re.compile(r"__ENTITY_\d+__")
ENGLISH_RE = re.compile(r"[A-Za-z][A-Za-z0-9._/+:#-]*")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def metric_text(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value)
        if not char.isspace() and not unicodedata.category(char).startswith("P"))


def get_template_kwargs(tokenizer):
    template = getattr(tokenizer, "chat_template", "") or ""
    return {"enable_thinking": False} if "enable_thinking" in template else {}


def run_model(name: str, model_path: Path, adapter_path: Path | None,
              rows: list[dict], system_prompt: str, output_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
        padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False).eval()
    kwargs = get_template_kwargs(tokenizer)
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    predictions = []
    started_all = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["input"]},
            ]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt", **kwargs)
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=512,
                    do_sample=False, repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True)
            torch.cuda.synchronize()
            tokens = generated.sequences[0, encoded["input_ids"].shape[1]:].tolist()
            raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            body = raw.split("<KEY>", 1)[0].strip()
            record = {"id": row["id"], "category": row["category"], "source": row["source"],
                "input": row["input"], "reference": row["reference"], "raw_output": raw,
                "prediction": body, "equals_input": body == row["input"],
                "equals_reference": body == row["reference"],
                "placeholder_sequence_matches_reference": PLACEHOLDER_RE.findall(body) == PLACEHOLDER_RE.findall(row["reference"]),
                "english_tokens_match_reference": ENGLISH_RE.findall(body) == ENGLISH_RE.findall(row["reference"]),
                "completed_with_eos": bool(tokens and tokens[-1] in eos_set),
                "generated_tokens": len(tokens),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
            predictions.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(rows):
                stream.flush()
                print(f"{name}: {index}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started_all

    by_class = defaultdict(list)
    for row in predictions:
        by_class[row["category"]].append(row)
    class_metrics = {}
    for category, group in sorted(by_class.items()):
        ref_total = sum(len(metric_text(row["reference"])) for row in group)
        pred_edits = sum(levenshtein(metric_text(row["reference"]), metric_text(row["prediction"])) for row in group)
        input_edits = sum(levenshtein(metric_text(row["reference"]), metric_text(row["input"])) for row in group)
        identity = [row for row in group if row["input"] == row["reference"]]
        class_metrics[category] = {
            "n": len(group),
            "exact_match_rate": sum(row["equals_reference"] for row in group) / len(group),
            "normalized_char_error_rate": pred_edits / max(ref_total, 1),
            "input_normalized_char_error_rate": input_edits / max(ref_total, 1),
            "changed_output_rate": sum(not row["equals_input"] for row in group) / len(group),
            "identity_case_count": len(identity),
            "identity_false_edit_rate": sum(not row["equals_input"] for row in identity) / len(identity) if identity else None,
            "placeholder_mismatch_count": sum(not row["placeholder_sequence_matches_reference"] for row in group),
            "english_token_mismatch_count": sum(not row["english_tokens_match_reference"] for row in group),
            "eos_rate": sum(row["completed_with_eos"] for row in group) / len(group),
        }
    ref_total = sum(sum(len(metric_text(row["reference"])) for row in group) for group in by_class.values())
    errors = sum(levenshtein(metric_text(row["reference"]), metric_text(row["prediction"])) for row in predictions)
    input_errors = sum(levenshtein(metric_text(row["reference"]), metric_text(row["input"])) for row in predictions)
    summary = {
        "model": name, "base_model": str(model_path), "adapter": str(adapter_path) if adapter_path else None,
        "test_label_status": "provisional candidate references; not human gold",
        "sample_count": len(predictions),
        "exact_match_rate": sum(row["equals_reference"] for row in predictions) / len(predictions),
        "normalized_char_error_rate": errors / max(ref_total, 1),
        "input_normalized_char_error_rate": input_errors / max(ref_total, 1),
        "macro_13class_normalized_char_error_rate": sum(v["normalized_char_error_rate"] for v in class_metrics.values()) / len(class_metrics),
        "placeholder_mismatch_count": sum(not row["placeholder_sequence_matches_reference"] for row in predictions),
        "english_token_mismatch_count": sum(not row["english_tokens_match_reference"] for row in predictions),
        "eos_rate": sum(row["completed_with_eos"] for row in predictions) / len(predictions),
        "elapsed_seconds": round(elapsed, 2), "class_metrics": class_metrics,
        "prediction_file": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, help="Best LoRA adapter/checkpoint directory")
    parser.add_argument("--full-model", type=Path, help="Completed full-SFT checkpoint directory")
    parser.add_argument("--skip-base", action="store_true", help="Reuse an existing base evaluation; run only the supplied adapter")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-existing-1b", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=0,
        help="Optional deterministic preview size within each class; zero evaluates all rows")
    args = parser.parse_args()
    if bool(args.adapter) == bool(args.full_model):
        parser.error("Provide exactly one of --adapter or --full-model")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires the training GPU to be free")
    manifest = read_json(DATA / "manifest.json")
    if not manifest.get("warning"):
        raise RuntimeError("Expected provisional dataset status in its manifest")
    rows = read_json(DATA / "test.json")
    sidecar = [json.loads(line) for line in (DATA / "split_audit.jsonl").open(encoding="utf-8")]
    test_meta = [row for row in sidecar if row["split"] == "test"]
    if len(rows) != len(test_meta) or len(rows) != manifest["split_counts"]["test"]:
        raise RuntimeError("Frozen test split does not match its sidecar/manifest")
    joined = []
    for message, meta in zip(rows, test_meta):
        user = next(m["content"] for m in message["messages"] if m["role"] == "user")
        target = next(m["content"] for m in message["messages"] if m["role"] == "assistant")
        if user != meta["input"] or target != meta["reference"]:
            raise RuntimeError("Test conversation and audit sidecar are misaligned")
        joined.append(meta)
    if args.limit_per_class:
        if args.limit_per_class < 1:
            raise ValueError("limit-per-class must be positive")
        grouped = defaultdict(list)
        for row in joined:
            grouped[row["category"]].append(row)
        rng = random.Random(20261027)
        joined = [row for category in sorted(grouped)
            for row in rng.sample(grouped[category], args.limit_per_class)]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    system_prompt = manifest["system_prompt"]
    if args.adapter:
        if not args.skip_base:
            run_model("qwen25_05b_base", BASE, None, joined, system_prompt, output / "qwen25_base.jsonl")
        run_model("qwen25_05b_lora", BASE, args.adapter, joined, system_prompt, output / "qwen25_lora.jsonl")
    else:
        run_model("qwen25_05b_full", args.full_model, None, joined, system_prompt, output / "qwen25_full.jsonl")
    if args.include_existing_1b:
        model_path = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")
        # Reuse the deployed 1B instruction, then remove its optional KEY metadata.
        prompt_source = Path(__file__).resolve().parents[2] / "experiments/scripts/postprocess_asr.py"
        import ast
        tree = ast.parse(prompt_source.read_text(encoding="utf-8"))
        one_b_prompt = next(ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets))
        run_model("agenticasr_existing_1b", model_path, None, joined, one_b_prompt, output / "existing_1b.jsonl")


if __name__ == "__main__":
    main()
