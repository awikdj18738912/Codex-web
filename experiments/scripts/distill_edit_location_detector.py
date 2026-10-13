"""Derived four-layer student: hard labels plus frozen teacher soft targets."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_location_detector import (EditLocationDetector, build_network, initialize_from_teacher,
    collate, encode_input, target_options, LABELS, INPUT_VERSION)
from experiments.scripts.train_edit_location_detector import (
    load_predefined_partitions, predict, choose_policy, metrics, sha, write, write_rows)


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from threadpoolctl import threadpool_limits

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Distillation requires the authorized GPU")
    partitions, split = load_predefined_partitions(args.partitions)
    for name, rows in partitions.items():
        write_rows(output / f"{name}.jsonl", rows)
    write(output / "split_manifest.json", split)
    teacher = EditLocationDetector(args.teacher, args.device)
    teacher.model.eval()
    for parameter in teacher.model.parameters():
        parameter.requires_grad_(False)
    tokenizer = teacher.tokenizer
    mapping = [0, 3, 7, 11]
    student = build_network(args.teacher / "encoder", pretrained=False, num_hidden_layers=4).to(args.device)
    initialize_from_teacher(student, teacher.model, mapping)
    max_tokens = teacher.metadata["max_tokens"]
    insertion_supported = teacher.metadata["insertion_detection_supported"]
    encoded = {row["id"]: encode_input(tokenizer, row, max_tokens) for row in partitions["train"] + partitions["validation"]}
    options = {row["id"]: target_options(row) for row in partitions["train"] + partitions["validation"]}
    # Teacher scores are computed only on fit data. No test soft labels or
    # post-test tuning can affect the selected student checkpoint/threshold.
    teacher_predictions = predict(teacher.model, tokenizer, partitions["train"], args.device, max_tokens)
    soft_targets = {item["id"]: (torch.tensor(item["chars"], device=args.device),
                                torch.tensor(item["gaps"], device=args.device)) for item in teacher_predictions}
    counts = Counter(label for row in partitions["train"] for label in options[row["id"]][0][0])
    weights = torch.tensor([1.] + [min(6., (counts[0] / max(1, counts[index])) ** .5) for index in (1, 2)], device=args.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    gap_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([dict(params=student.encoder.parameters(), lr=3e-5),
        dict(params=list(student.char_head.parameters()) + list(student.gap_head.parameters()), lr=5e-4)], weight_decay=.01)
    configuration = dict(scope="controlled_text_student_distillation_not_audio_acceptance", seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, temperature=args.temperature, kd_weight=args.kd_weight,
        teacher_path=str(args.teacher), teacher_weights_sha256=sha(args.teacher / "detector.pt"),
        teacher_encoder_provenance=teacher.metadata.get("encoder_provenance"),
        initialization="teacher embeddings + selected layers + heads, then supervised/soft-target distillation",
        teacher_layer_indices=mapping, full_four_layer_pretraining_not_claimed=True,
        partition_sha256={name: sha(args.partitions / f"{name}.jsonl") for name in partitions},
        teacher_targets_from_train_only=True, selection_from_new_validation_only=True,
        test_status="consumed_authored_regression", no_online_changes=True)
    write(output / "configuration.json", configuration)
    (output / "executed_code").mkdir()
    for file in (Path(__file__), ROOT / "system/edit_location_detector.py"):
        (output / "executed_code" / file.name).write_bytes(file.read_bytes())

    def losses(rows, with_distillation):
        char_logits, gap_logits = student(**{key: value.to(args.device) for key, value in
            collate(tokenizer, [encoded[row["id"]] for row in rows]).items()})
        hard, soft = [], []
        for index, row in enumerate(rows):
            length = len(row["baseline_text"])
            alternatives = [criterion(char_logits[index, :length], torch.tensor(chars, device=args.device)) +
                .2 * gap_criterion(gap_logits[index, :length + 1], torch.tensor(gaps, device=args.device))
                for chars, gaps in options[row["id"]]]
            hard.append(torch.stack(alternatives).min())
            if with_distillation:
                char_target, gap_target = soft_targets[row["id"]]
                temperature = args.temperature
                def kl(logits, probabilities):
                    target = probabilities.clamp_min(1e-8).pow(1 / temperature)
                    target = target / target.sum(-1, keepdim=True)
                    return functional.kl_div(functional.log_softmax(logits / temperature, -1), target,
                                             reduction="batchmean") * temperature ** 2
                soft.append(kl(char_logits[index, :length], char_target) +
                            .2 * kl(gap_logits[index, :length + 1], gap_target))
        hard_loss = torch.stack(hard).mean()
        return hard_loss + args.kd_weight * torch.stack(soft).mean() if with_distillation else hard_loss

    best, history, rng, started = None, [], random.Random(args.seed), time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        student.train()
        order = list(partitions["train"])
        rng.shuffle(order)
        total = 0.
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = losses(rows, True)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite student loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(rows)
        student.eval()
        validation_loss = 0.
        with torch.inference_mode():
            for start in range(0, len(partitions["validation"]), 32):
                rows = partitions["validation"][start:start + 32]
                validation_loss += losses(rows, False).item() * len(rows)
        validation_loss /= len(partitions["validation"])
        predictions = predict(student, tokenizer, partitions["validation"], args.device, max_tokens)
        (rank, threshold, result), _ = choose_policy(partitions["validation"], predictions, insertion_supported, False)
        rank = (*rank, -validation_loss)
        record = dict(epoch=epoch, train_loss=total / len(order), validation_loss=validation_loss,
            validation_threshold=threshold, keep_false_positives=result["false_positive_keep_windows"],
            location_precision=result["location_precision"], elapsed_seconds=time.perf_counter() - started)
        history.append(record)
        print("STUDENT " + json.dumps(record), flush=True)
        if best is None or rank > best[0]:
            best = rank, epoch
            torch.save({key: value.detach().cpu() for key, value in student.state_dict().items()}, output / "detector.pt")
    student.load_state_dict(torch.load(output / "detector.pt", map_location=args.device, weights_only=True))
    validation_predictions = predict(student, tokenizer, partitions["validation"], args.device, max_tokens)
    (_, threshold, validation_metrics), search = choose_policy(partitions["validation"], validation_predictions, insertion_supported, False)
    write(output / "validation_threshold_search.json", search)
    (output / "encoder").mkdir()
    student.encoder.config.save_pretrained(output / "encoder")
    tokenizer.save_pretrained(output / "encoder")
    metadata = dict(input_version=INPUT_VERSION, labels=list(LABELS), max_tokens=max_tokens, threshold=threshold,
        weights_sha256=sha(output / "detector.pt"), parameters=sum(parameter.numel() for parameter in student.parameters()),
        student_layers=4, hidden_size=student.encoder.config.hidden_size, best_epoch=best[1], history=history,
        calibrated=False, online_enabled=False, production_skip_allowed=False, insertion_detection_supported=insertion_supported,
        teacher_weights_sha256=configuration["teacher_weights_sha256"], derivation=configuration["initialization"],
        teacher_layer_indices=mapping, teacher_encoder_provenance=configuration["teacher_encoder_provenance"],
        full_four_layer_pretraining_not_claimed=True, human_audio_gold=0, training_seconds=time.perf_counter() - started)
    write(output / "metadata.json", metadata)
    test_predictions = predict(student, tokenizer, partitions["test"], args.device, max_tokens)
    summary = dict(validation=validation_metrics,
        authored_test_regression=metrics(partitions["test"], test_predictions, threshold, insertion_supported),
        threshold=threshold, parameters=metadata["parameters"], test_status="consumed_authored_regression",
        natural_audio_gold=0, production_eligible=False)
    write(output / "summary.json", summary)
    del student, optimizer, teacher
    torch.cuda.empty_cache()
    detector = EditLocationDetector(output, "cpu")
    for row in partitions["validation"][:5]:
        detector.detect(row)
    latencies = []
    for row in partitions["validation"]:
        start = time.perf_counter()
        result = detector.detect(row)
        latencies.append((time.perf_counter() - start) * 1000)
        if result["status"] == "unsupported":
            raise RuntimeError("Student unsupported validation input")
    write(output / "cpu_latency.json", dict(rows=len(latencies), threads=2, mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies), p95_ms=sorted(latencies)[int(.95 * (len(latencies) - 1))],
        scope="warm single-window CPU interface; not file/ASR timing"))
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.)
    parser.add_argument("--kd-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.temperature <= 0 or args.kd_weight < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("Invalid distillation hyperparameters")
    run(args)
