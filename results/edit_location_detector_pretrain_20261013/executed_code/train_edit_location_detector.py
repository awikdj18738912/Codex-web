"""Offline grouped text pretraining, with all task-definition probes in train.

The internal text test is opened only after checkpoint/threshold selection.
No ASR review proposals, model outputs, or existing source dev/test are fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import TASKS, digest
from system.edit_location_detector import (LABELS, INPUT_VERSION, build_network, make_tokenizer,
                                         encode_input, collate, target_options, candidate_spans)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def split_rows(rows, seed):
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    def join(a, b):
        parent[find(b)] = find(a)
    for row in rows:
        group = "t:" + row["split_group"]
        join(group, "r:" + row["recording_group"])
        join(group, "b:" + digest(row["baseline_text"]))
    components = defaultdict(list)
    for row in rows:
        components[find("t:" + row["split_group"])].append(row)
    units = list(components.values())
    probes = [unit for unit in units if any(r["label_status"] == "controlled_task_definition" for r in unit)]
    ordinary = [unit for unit in units if unit not in probes]
    counts = Counter("positive" if r["edits"] else "keep" for unit in ordinary for r in unit)
    rng, best = random.Random(seed), None
    for _ in range(3000):
        shuffled = list(ordinary)
        rng.shuffle(shuffled)
        ntrain, nval = round(len(shuffled) * .6), max(1, round(len(shuffled) * .2))
        choices = [shuffled[:ntrain], shuffled[ntrain:ntrain + nval], shuffled[ntrain + nval:]]
        score, valid = 0., True
        for selected, ratio in zip(choices, (.6, .2, .2)):
            values = [r for unit in selected for r in unit]
            sizes = Counter("positive" if r["edits"] else "keep" for r in values)
            tasks = {edit["task"] for row in values for edit in row["edits"]}
            valid &= bool(sizes["positive"] and sizes["keep"] and {"disfluency", "repetition"} <= tasks)
            score += sum((sizes[key] / counts[key] - ratio) ** 2 for key in counts)
        if valid and (best is None or score < best[0]):
            best = score, choices
    if best is None:
        raise ValueError("Cannot split independent template/recording/identical-baseline components")
    result = {name: [row for unit in selected for row in unit]
              for name, selected in zip(("train", "validation", "test"), best[1])}
    result["train"].extend(row for unit in probes for row in unit)
    for key in ("split_group", "recording_group", "baseline_text"):
        values = [{row[key] for row in result[name]} for name in result]
        if any(values[i] & values[j] for i in range(3) for j in range(i)):
            raise ValueError("Split leakage: " + key)
    return result, dict(seed=seed, independent_components=len(units), ordinary_components=len(ordinary),
                        task_probe_components_train_only=len(probes), searches=3000,
                        split_uses="labels_and_group_sizes_before_fit_only",
                        constraints=["template", "recording", "identical_baseline"],
                        counts={name: dict(rows=len(values), keeps=sum(not r["edits"] for r in values),
                                          tasks=dict(Counter(edit["task"] for row in values for edit in row["edits"])))
                                for name, values in result.items()})


def predict(model, tokenizer, rows, device, max_tokens):
    import torch

    results = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            subset = rows[start:start + 32]
            items = [encode_input(tokenizer, row, max_tokens) for row in subset]
            chars, gaps = model(**{key: value.to(device) for key, value in collate(tokenizer, items).items()})
            for index, row in enumerate(subset):
                length = len(row["baseline_text"])
                results.append(dict(id=row["id"], chars=chars[index, :length].softmax(-1).cpu().tolist(),
                                    gaps=gaps[index, :length + 1].softmax(-1).cpu().tolist()))
    return results


def metrics(rows, predictions, threshold, insertion_supported):
    counts = Counter()
    tasks = {task: Counter() for task in TASKS}
    for row, prediction in zip(rows, predictions):
        candidates = candidate_spans(prediction["chars"], prediction["gaps"] if insertion_supported else [], threshold)
        needs, selected = bool(row["edits"]), bool(candidates)
        counts["rows"] += 1
        counts["needs_review"] += needs
        counts["true_positive_windows"] += needs and selected
        counts["keep_windows"] += not needs
        counts["false_positive_keep_windows"] += not needs and selected
        reference_chars = [(set(i for i, label in enumerate(chars) if label), set(i for i, label in enumerate(gaps) if label))
                           for chars, gaps in target_options(row)]
        chosen_chars = {i for span in candidates for i in range(span["start"], span["end"])}
        chosen_gaps = {span["start"] for span in candidates if span["start"] == span["end"]}
        best = max(reference_chars, key=lambda pair: len(pair[0] & chosen_chars) + len(pair[1] & chosen_gaps))
        counts["reference_locations"] += len(best[0]) + len(best[1])
        counts["predicted_locations"] += len(chosen_chars) + len(chosen_gaps)
        counts["matched_locations"] += len(best[0] & chosen_chars) + len(best[1] & chosen_gaps)
        for edit in row["edits"]:
            alternatives = row.get("offset_alternatives", []) if len(row["edits"]) == 1 else []
            intervals = alternatives or [edit]
            hit = any((location["start"] in chosen_gaps if location["start"] == location["end"] else
                       bool(set(range(location["start"], location["end"])) & chosen_chars)) for location in intervals)
            tasks[edit["task"]]["edits"] += 1
            tasks[edit["task"]]["located_edits"] += hit
    output = dict(counts)
    output.update(window_recall=counts["true_positive_windows"] / counts["needs_review"] if counts["needs_review"] else None,
                  keep_false_positive_rate=counts["false_positive_keep_windows"] / counts["keep_windows"] if counts["keep_windows"] else None,
                  location_recall=counts["matched_locations"] / counts["reference_locations"] if counts["reference_locations"] else None,
                  location_precision=counts["matched_locations"] / counts["predicted_locations"] if counts["predicted_locations"] else None,
                  by_task={task: dict(values, recall=values["located_edits"] / values["edits"] if values["edits"] else None)
                           for task, values in tasks.items()}, insertion_supported=insertion_supported)
    return output


def fit(args):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise ValueError("This training run requires the authorized GPU")
    rows = [json.loads(line) for line in args.data.open()]
    if any(not r["training_eligible"] or r["split"] != "train" or r["human_audio_gold"]
           or r["allowed_usage"] != "text_pretraining_only" for r in rows):
        raise ValueError("Only original controlled text training rows are accepted")
    partitions, split_info = split_rows(rows, args.seed)
    for name, values in partitions.items():
        write_rows(output / f"{name}.jsonl", values)
    write(output / "split_manifest.json", split_info)
    configuration = dict(data_sha256=sha(args.data), encoder_sha256={file.name: sha(file) for file in args.encoder.iterdir() if file.is_file()},
                         source=str(args.data), encoder=str(args.encoder), device=args.device,
                         seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                         scope="controlled_text_group_holdout_only_not_audio_gold", online_enabled=False,
                         human_audio_gold=0, no_external_downloads=True)
    write(output / "configuration.json", configuration)
    (output / "executed_code").mkdir()
    for file in (Path(__file__), ROOT / "system/edit_location_detector.py"):
        (output / "executed_code" / file.name).write_bytes(file.read_bytes())
    tokenizer = make_tokenizer(args.encoder)
    encoded = {row["id"]: encode_input(tokenizer, row, args.max_tokens) for row in partitions["train"] + partitions["validation"]}
    options = {row["id"]: target_options(row) for row in partitions["train"] + partitions["validation"]}
    class_counts = Counter(label for row in partitions["train"] for label in options[row["id"]][0][0])
    gap_positive = sum(any(gaps) for row in partitions["train"] for _, gaps in options[row["id"]][:1])
    insertion_supported = bool(gap_positive)
    class_weights = torch.tensor([1.] + [min(6., (class_counts[0] / max(1, class_counts[i])) ** .5) for i in (1, 2)], device=args.device)
    model = build_network(args.encoder).to(args.device)
    optimizer = torch.optim.AdamW([dict(params=model.encoder.parameters(), lr=3e-5),
                                  dict(params=list(model.char_head.parameters()) + list(model.gap_head.parameters()), lr=5e-4)], weight_decay=.01)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    gap_criterion = torch.nn.CrossEntropyLoss()
    def loss_for(batch_rows, training):
        chars, gaps = model(**{key: value.to(args.device) for key, value in collate(tokenizer, [encoded[row["id"]] for row in batch_rows]).items()})
        losses = []
        for index, row in enumerate(batch_rows):
            length = len(row["baseline_text"])
            alternatives = []
            for target_chars, target_gaps in options[row["id"]]:
                alternatives.append(criterion(chars[index, :length], torch.tensor(target_chars, device=args.device)) +
                                    .2 * gap_criterion(gaps[index, :length + 1], torch.tensor(target_gaps, device=args.device)))
            losses.append(torch.stack(alternatives).min())
        return torch.stack(losses).mean()
    best, history, started = None, [], time.perf_counter()
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = list(partitions["train"])
        rng.shuffle(order)
        model.train()
        total = 0.
        for start in range(0, len(order), args.batch_size):
            subset = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(subset, True)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(subset)
        model.eval()
        validation_loss = 0.
        with torch.inference_mode():
            for start in range(0, len(partitions["validation"]), 32):
                subset = partitions["validation"][start:start + 32]
                validation_loss += loss_for(subset, False).item() * len(subset)
        validation_loss /= len(partitions["validation"])
        record = dict(epoch=epoch, train_loss=total / len(order), validation_loss=validation_loss,
                      elapsed_seconds=time.perf_counter() - started)
        history.append(record)
        print("EPOCH " + json.dumps(record), flush=True)
        if best is None or validation_loss < best[0]:
            best = validation_loss, epoch
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "detector.pt")
    model.load_state_dict(torch.load(output / "detector.pt", map_location=args.device, weights_only=True))
    validation = predict(model, tokenizer, partitions["validation"], args.device, args.max_tokens)
    choices = []
    for threshold in (0., .01, .02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, .95):
        result = metrics(partitions["validation"], validation, threshold, insertion_supported)
        if (result["window_recall"] >= .95 and result["location_recall"] >= .95
                and all(item["recall"] >= .95 for item in result["by_task"].values() if item.get("edits"))):
            choices.append(((-result["false_positive_keep_windows"], result["location_precision"] or 0, threshold), threshold, result))
    if not choices:
        raise RuntimeError("No high-recall validation policy")
    _, threshold, validation_metrics = max(choices)
    write(output / "validation_threshold_search.json", [dict(threshold=value, metrics=metrics(partitions["validation"], validation, value, insertion_supported))
                                                        for value in (0., .01, .02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, .95)])
    (output / "encoder").mkdir()
    model.encoder.config.save_pretrained(output / "encoder")
    tokenizer.save_pretrained(output / "encoder")
    metadata = dict(input_version=INPUT_VERSION, labels=list(LABELS), max_tokens=args.max_tokens,
                    threshold=threshold, calibrated=False, online_enabled=False,
                    production_skip_allowed=False, weights_sha256=sha(output / "detector.pt"),
                    parameters=sum(param.numel() for param in model.parameters()), best_epoch=best[1],
                    training_seconds=time.perf_counter() - started, training_history=history,
                    class_counts=dict(class_counts), class_weights=class_weights.cpu().tolist(),
                    gap_positive_training_rows=gap_positive, insertion_detection_supported=insertion_supported,
                    task_probes_train_only=True, validation_selection=">=95% window/location/per-observed-task edit recall, then fewer KEEP false positives",
                    human_audio_gold=0, scope=configuration["scope"])
    write(output / "metadata.json", metadata)
    # The internal test remains untouched by checkpoint/threshold selection.
    test = predict(model, tokenizer, partitions["test"], args.device, args.max_tokens)
    probes = [row for row in partitions["train"] if row["label_status"] == "controlled_task_definition"]
    probe_predictions = predict(model, tokenizer, probes, args.device, args.max_tokens)
    summary = dict(validation=validation_metrics, internal_text_test=metrics(partitions["test"], test, threshold, insertion_supported),
                   trained_task_probe_fit=metrics(probes, probe_predictions, threshold, insertion_supported),
                   parameters=metadata["parameters"], threshold=threshold,
                   unsupported_validation_or_test=0,
                   limitations=["Internal heldout synthetic templates, not independent natural audio",
                                "Last four tasks have only two trained probes each; no heldout generalization evidence",
                                "No positive insertion examples; insertion detection is disabled" if not insertion_supported else "Insertion head only text-supervised",
                                "Scores uncalibrated; no production skip permission"])
    for name, subset, predictions in (("validation", partitions["validation"], validation), ("test", partitions["test"], test), ("trained_probes", probes, probe_predictions)):
        write_rows(output / f"{name}_predictions.jsonl", [dict(id=row["id"], baseline_text=row["baseline_text"], desired_text=row["desired_text"],
                    candidates=candidate_spans(prediction["chars"], prediction["gaps"] if insertion_supported else [], threshold),
                    edits=row["edits"], label_status=row["label_status"]) for row, prediction in zip(subset, predictions)])
    write(output / "summary.json", summary)
    # Time the real single-window CPU interface, including input preparation.
    del model, optimizer
    torch.cuda.empty_cache()
    from system.edit_location_detector import EditLocationDetector
    detector = EditLocationDetector(output, "cpu")
    for row in partitions["train"][:5]:
        detector.detect(row)
    latency = []
    for row in partitions["validation"]:
        start = time.perf_counter()
        result = detector.detect(row)
        latency.append((time.perf_counter() - start) * 1000)
        if result["status"] == "unsupported":
            raise RuntimeError("Unsupported benchmark input")
    write(output / "cpu_latency.json", dict(rows=len(latency), threads=torch.get_num_threads(),
        mean_ms=statistics.mean(latency), median_ms=statistics.median(latency),
        p95_ms=sorted(latency)[int(.95 * (len(latency) - 1))], total_ms=sum(latency),
        scope="warm single-window CPU interface including tokenization; not whole ASR file timing"))
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/edit_detection_data_20261013_v3/controlled_text_train.jsonl")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/entity_ranker/encoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    fit(args)
