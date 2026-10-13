"""Use official LLaMA-Factory with a template matching Refiner inference.

preflight validates every prompt and label mask without loading model weights.
smoke runs one real CPU optimizer step in a separate disposable output folder.
train uses the declared LoRA configuration and a GPU memory preflight.
"""
import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts.synthetic_gate_data import file_hash, read_rows, write_json

TEMPLATE = "agenticasr_local_nothink"
NO_THINK_PREFIX = "<think>\n\n</think>\n\n"


def register_runtime_template():
    from llamafactory.data.formatter import EmptyFormatter, StringFormatter
    from llamafactory.data.template import TEMPLATES, register_template
    if TEMPLATE not in TEMPLATES:
        register_template(name=TEMPLATE,
            format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n" + NO_THINK_PREFIX]),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
            format_prefix=EmptyFormatter(slots=[{"bos_token"}]), stop_words=["<|im_end|>"])


def environment_info():
    packages = {name: metadata.version(name) for name in ("llamafactory", "torch", "transformers", "peft", "datasets", "accelerate", "trl")}
    revision = subprocess.check_output(["git", "-C", str(ROOT / "LLaMA-Factory"), "rev-parse", "HEAD"], text=True).strip()
    return dict(python=sys.executable, packages=packages, factory_revision=revision)


def verify_data(config: dict):
    data = ROOT / config["dataset_dir"]
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["exported_sha256"].items():
        if file_hash(data / name) != expected:
            raise RuntimeError("Registered training data changed: " + name)
    return manifest


def cpu_config(config: dict, output: Path) -> dict:
    config = dict(config)
    config.update(use_cpu=True, bf16=False, fp16=False, output_dir=str(output))
    return config


def preflight(config: dict, output: Path):
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    register_runtime_template()
    output.mkdir(parents=True, exist_ok=True)
    parsed = get_train_args(cpu_config(config, output / "argument_check"))
    model_args, data_args, training_args, _finetuning, _generating = parsed
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    data = ROOT / config["dataset_dir"]
    lengths, prompt_lengths = [], []
    for split in ("train", "dev", "test"):
        for row in read_rows(data / f"{split}.jsonl"):
            messages = row["messages"]
            system, user, answer = [message["content"] for message in messages]
            source_ids, target_ids = template.encode_oneturn(tokenizer,
                [dict(role="user", content=user), dict(role="assistant", content=answer)], system=system)
            runtime_ids = tokenizer.apply_chat_template(messages[:2], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=True)["input_ids"]
            if runtime_ids and isinstance(runtime_ids[0], list):
                runtime_ids = runtime_ids[0]
            if source_ids != runtime_ids:
                raise RuntimeError("Training/inference prompt mismatch")
            expected_target = tokenizer.encode(answer + "<|im_end|>\n", add_special_tokens=False)
            if target_ids != expected_target:
                raise RuntimeError("Assistant target encoding mismatch")
            lengths.append(len(source_ids) + len(target_ids))
            prompt_lengths.append(len(source_ids))
    if max(lengths) > config["cutoff_len"]:
        raise RuntimeError("Configured cutoff truncates the prepared examples")
    datasets = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    counts = {}
    checked = 0
    for name, dataset in datasets.items():
        if dataset is None:
            continue
        if isinstance(dataset, dict):
            entries = dataset.items()
        else:
            entries = [(name, dataset)]
        for key, records in entries:
            counts[key] = len(records)
            for row in records:
                labels, inputs = row["labels"], row["input_ids"]
                start = next((i for i, token in enumerate(labels) if token != -100), None)
                if start is None or any(token != -100 for token in labels[:start]) or labels[start:] != inputs[start:]:
                    raise RuntimeError("Invalid assistant-only loss mask")
                # The supervised suffix must decode to exactly the JSON answer.
                answer = tokenizer.decode(labels[start:], skip_special_tokens=True).strip()
                payload = json.loads(answer)
                if set(payload) != {"action", "source", "target", "reason"}:
                    raise RuntimeError("Non-answer content included in supervision")
                checked += 1
    expected_counts = verify_data(config)["split_counts"]
    if counts.get("train_dataset") != expected_counts["train"] or sum(v for k, v in counts.items() if k != "train_dataset") != expected_counts["dev"]:
        raise RuntimeError("Unexpected train/dev coverage; test data must stay out of training")
    report = dict(**environment_info(), dataset_counts=counts,
        prompt_and_target_pairs_checked=len(lengths), assistant_masks_checked=checked,
        longest_example_tokens=max(lengths), longest_prompt_tokens=max(prompt_lengths),
        runtime_prompt_exact_match=True, assistant_only_supervision=True, test_used_for_training=False,
        tokenizer_class=type(tokenizer).__name__, template=TEMPLATE,
        real_model_weights_loaded=False, full_training_started=False,
        data_manifest_sha256=file_hash(data / "manifest.json"))
    write_json(output / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(config: dict, *, smoke: bool, output: Path):
    from llamafactory.train.tuner import run_exp
    register_runtime_template()
    run_config = dict(config)
    if smoke:
        run_config = cpu_config(run_config, output)
        run_config.update(max_steps=1, max_samples=1, do_eval=False,
            gradient_accumulation_steps=1, per_device_train_batch_size=1,
            logging_steps=1, save_strategy="no", eval_strategy="no",
            load_best_model_at_end=False, plot_loss=False, lr_scheduler_type="constant", warmup_ratio=0)
        run_config.pop("eval_dataset", None)
    elif not run_config.get("use_cpu", False):
        import torch
        free, _total = torch.cuda.mem_get_info(0)
        minimum = 6 * 1024**3
        if free < minimum:
            raise RuntimeError(f"Free GPU memory {free / 1024**3:.2f} GiB is below the conservative 6 GiB start threshold")
    destination = Path(run_config["output_dir"])
    if not destination.is_absolute():
        destination = ROOT / destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Training output already exists; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "requested_run.json", dict(config=run_config, environment=environment_info(),
        purpose="one_step_cpu_integration_check_not_candidate_model" if smoke else "controlled_format_lora_training"))
    started = time.perf_counter()
    run_exp(args=run_config)
    state = json.loads((destination / "trainer_state.json").read_text())
    metrics = json.loads((destination / "train_results.json").read_text())
    if not math.isfinite(metrics["train_loss"]) or state["global_step"] < 1:
        raise RuntimeError("Training did not produce a finite loss and optimizer step")
    if smoke and state["global_step"] != 1:
        raise RuntimeError("Smoke validation must execute exactly one optimizer step")
    report = dict(real_model_forward_backward=True, optimizer_steps=state["global_step"],
                  elapsed_seconds=time.perf_counter() - started, is_smoke_adapter=smoke,
                  full_format_training=not smoke, **environment_info())
    write_json(destination / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "smoke", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/llamafactory/refiner_local_lora.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0", "cuda:1"])
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if Path(sys.prefix).resolve() != (ROOT / ".runtime/llamafactory-env").resolve():
        raise RuntimeError("Use .runtime/llamafactory-env/bin/python")
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.stage != "train" or args.device == "cpu" else args.device.split(":")[1]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_CACHE"] = str(ROOT / ".runtime/llamafactory-datasets")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".runtime/llamafactory-hub")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    import torch
    torch.set_num_threads(args.cpu_threads)
    config = yaml.safe_load(args.config.read_text())
    verify_data(config)
    if args.stage == "preflight":
        preflight(config, args.output or ROOT / "results/llamafactory_setup_20261010")
    elif args.stage == "smoke":
        train(config, smoke=True, output=args.output or ROOT / ".runtime/llamafactory-smoke")
    else:
        if args.device == "cpu":
            config = cpu_config(config, Path(config["output_dir"]))
        if args.output:
            config["output_dir"] = str(args.output.resolve())
        train(config, smoke=False, output=Path(config["output_dir"]))


if __name__ == "__main__":
    main()
