"""Validate the provisional no-C02 Refiner split and loss masks on CPU."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/llamafactory/refiner13_qwen25_lora_no_C02_20261027.yaml"
REPORT = ROOT / "results/refiner13_no_C02_20261027/preflight.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if Path(sys.prefix).resolve() != (ROOT / ".runtime/llamafactory-env").resolve():
        raise RuntimeError("Use .runtime/llamafactory-env/bin/python for the installed LLaMA-Factory environment")
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".runtime/llamafactory-datasets"))
    import torch
    torch.set_num_threads(8)
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_dir = Path(config["dataset_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise RuntimeError(f"Dataset file hash changed: {name}")

    cpu_config = dict(config, use_cpu=True, bf16=False, fp16=False,
        output_dir=str(ROOT / ".runtime/refiner13_lora_argument_check"),
        dataloader_num_workers=0, preprocessing_num_workers=4)
    model_args, data_args, train_args, _finetuning_args, _generating_args = get_train_args(cpu_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    datasets = get_dataset(template, model_args, data_args, train_args, stage="sft", **tokenizer_module)
    expected = manifest["split_counts"]
    result = {"dataset_counts": {}, "examples_checked": 0, "max_tokens": 0,
        "max_target_tokens": 0, "assistant_only_loss": True, "truncation": False}
    for split_key, split_name in (("train_dataset", "train"), ("eval_dataset", "dev")):
        dataset = datasets.get(split_key)
        if dataset is None:
            raise RuntimeError(f"LLaMA-Factory did not load {split_name}")
        result["dataset_counts"][split_name] = len(dataset)
        if len(dataset) != expected[split_name]:
            raise RuntimeError(f"Unexpected {split_name} rows: {len(dataset)}")
        for batch in dataset.iter(batch_size=256):
            for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
                first = next((i for i, token in enumerate(labels) if token != -100), None)
                if first is None or first == 0 or labels[first:] != input_ids[first:]:
                    raise RuntimeError("Invalid assistant-only loss mask")
                if len(input_ids) > config["cutoff_len"]:
                    raise RuntimeError("Configured cutoff truncates an example")
                result["max_tokens"] = max(result["max_tokens"], len(input_ids))
                result["max_target_tokens"] = max(result["max_target_tokens"], len(labels) - first)
                result["examples_checked"] += 1

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed", "model": config["model_name_or_path"],
        "config": str(CONFIG), "config_sha256": sha256(CONFIG),
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "candidate_data_are_provisional_not_human_gold": True,
        "template": config["template"], "lora_target": config["lora_target"], **result}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
