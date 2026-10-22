# -*- coding: utf-8 -*-
"""Export final records to stratified LLaMAFactory ShareGPT datasets."""
import argparse
import json
import random
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from src.utils.io_utils import read_jsonl
from configs.scenes import get_enabled_scenes
from src.utils.numeric_policy import (
    TRAINING_SYSTEM_PROMPT,
    prepare_fragment_training_pair,
    prepare_training_pair,
)

# Keep the SFT protocol aligned with the deployed Refiner contract.  The
# numeric normalizer remains a deterministic runtime stage; the model is
# trained to preserve numeric values and remove oral disfluencies instead of
# learning a second, potentially inconsistent number conversion policy.
DEFAULT_VALIDATION_RATIO = 0.15

SYSTEM_PROMPT = TRAINING_SYSTEM_PROMPT


def convert_one(item: dict) -> dict | None:
    """Convert one finalized record to ShareGPT format."""
    input_text = item.get("input", "")
    output = item.get("output", {})
    refined_text = output.get("refined_text", "")

    if not input_text or not refined_text:
        return None
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    scene = meta.get("scene") or item.get("scene")
    language = meta.get("language") or item.get("language")
    expected = (meta.get("generation_plan") or {}).get("expected_clean")
    if meta.get("is_fragment"):
        reference = meta.get("fragment_reference_clean") or expected
        if not isinstance(reference, str) or not reference:
            return None
        pair = prepare_fragment_training_pair(
            input_text, refined_text, item.get("oral_text", ""), reference,
            meta.get("generation_plan") if isinstance(meta.get("generation_plan"), dict) else None,
            observed_prefix=meta.get("fragment_observed_prefix"),
        )
    else:
        pair = prepare_training_pair(input_text, refined_text, meta.get("generation_plan"))
    if expected and not meta.get("is_fragment") and refined_text != expected:
        return None
    if (scene and scene not in get_enabled_scenes()) or language == "en":
        return None
    if pair is None:
        return None
    input_text, refined_text, _numeric_meta = pair

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": input_text},
            {"from": "gpt", "value": refined_text},
        ]
    }


def get_length_bucket(text: str) -> str:
    """Assign a length bucket from the refined target."""
    length = len(text)
    if length <= 20:
        return "short"
    elif length <= 50:
        return "medium"
    elif length <= 100:
        return "long"
    else:
        return "very_long"


def stratified_split(
    samples: list[dict],
    val_split: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Split by scene and target length while retaining group coverage."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        scene = sample.get("_scene", "unknown")
        length_bucket = sample.get("_length_bucket", "medium")
        groups[(scene, length_bucket)].append(sample)

    train_samples = []
    val_samples = []

    for (scene, length_bucket), group_samples in groups.items():
        rng.shuffle(group_samples)
        split_n = int(len(group_samples) * val_split)

        # Put at least one sufficiently represented group item in validation.
        if len(group_samples) >= 5 and split_n == 0:
            split_n = 1

        group_val = group_samples[:split_n]
        group_train = group_samples[split_n:]

        val_samples.extend(group_val)
        train_samples.extend(group_train)

    # Shuffle after group-wise splitting.
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


_TRAILING_ZH_STOP = re.compile(r"[。\s]+$")


def intent_group_key(item: dict) -> str:
    """Return the final-intent identity used to keep variants in one split."""
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    plan = meta.get("generation_plan")
    expected = plan.get("expected_clean") if isinstance(plan, dict) else None
    value = expected or meta.get("fragment_reference_clean")
    if not value:
        output = item.get("output")
        value = output.get("refined_text") if isinstance(output, dict) else None
    # Only normalize whitespace and the Chinese sentence-final full stop. Other
    # punctuation (including ASCII periods, code punctuation, and digits) matters.
    return _TRAILING_ZH_STOP.sub("", str(value or "").strip())


def grouped_split(
    samples: list[dict], val_split: float, rng: random.Random, stratified: bool = True
) -> tuple[list[dict], list[dict]]:
    """Split complete intent groups; greedily minimize per-stratum count error.

    Each group's members are indivisible. With stratification, the greedy pass
    compares the squared distance from the requested validation count in every
    scene/length stratum after assigning the next shuffled group. Without it,
    the same rule is applied to the overall validation count only.
    """
    if not isinstance(val_split, (int, float)) or not math.isfinite(val_split) or not 0 <= val_split <= 1:
        raise ValueError("val_split must be a finite number between 0 and 1")
    by_key: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        key = sample.get("_group_key")
        if key is None:
            raise ValueError("each sample must contain _group_key")
        by_key[str(key)].append(sample)

    if stratified:
        def strata(sample):
            return (sample.get("_scene", "unknown"), sample.get("_length_bucket", "medium"))
        group_counts: dict[str, dict[tuple[str, str], int]] = {}
        totals: dict[tuple[str, str], int] = defaultdict(int)
        for key, members in by_key.items():
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for member in members:
                counts[strata(member)] += 1
                totals[strata(member)] += 1
            group_counts[key] = counts
        targets = {s: n * val_split for s, n in totals.items()}
        current: dict[tuple[str, str], int] = defaultdict(int)
        def cost(candidate=None):
            result = 0.0
            for s, target in targets.items():
                n = current[s] + (group_counts[candidate].get(s, 0) if candidate else 0)
                result += (n - target) ** 2
            return result
        order = list(by_key)
        rng.shuffle(order)
        # Larger groups first reduces avoidable overshoot; random order resolves ties.
        order.sort(key=lambda k: -len(by_key[k]))
        val_keys = set()
        for key in order:
            with_group = cost(key)
            without_group = cost()
            if with_group < without_group:
                val_keys.add(key)
                for s, n in group_counts[key].items():
                    current[s] += n
    else:
        order = list(by_key)
        rng.shuffle(order)
        order.sort(key=lambda k: -len(by_key[k]))
        target = len(samples) * val_split
        current_n = 0
        val_keys = set()
        for key in order:
            if abs(current_n + len(by_key[key]) - target) < abs(current_n - target):
                val_keys.add(key)
                current_n += len(by_key[key])

    train_samples, val_samples = [], []
    for key, members in by_key.items():
        (val_samples if key in val_keys else train_samples).extend(members)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="Export LLaMAFactory ShareGPT datasets")
    parser.add_argument("--inputs", type=str, nargs="+",
                        default=["data/final/train.jsonl"],
                        help="one or more finalized JSONL files")
    parser.add_argument("--train-output", type=str, default="data/final/train_sft.json")
    parser.add_argument("--val-output", type=str, default="data/final/val_sft.json")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VALIDATION_RATIO,
                        help=f"validation ratio (default: {DEFAULT_VALIDATION_RATIO})")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no-system", action="store_true",
                        help="omit the system prompt")
    parser.add_argument("--no-stratified", action="store_true",
                        help="use a simple random split")
    args = parser.parse_args()

    if not math.isfinite(args.val_split) or not 0 <= args.val_split <= 1:
        parser.error("--val-split must be a finite number between 0 and 1")

    # Load all requested datasets.
    all_data = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"[WARN] Input file not found, skipping: {input_path}")
            continue

        data = read_jsonl(input_path)
        print(f"Loaded {len(data)} samples from {input_path}")
        all_data.extend(data)

    if not all_data:
        print("[ERROR] No data loaded from any input file")
        sys.exit(1)

    print(f"Total samples: {len(all_data)}")

    # Convert records.
    samples = []
    skipped = 0
    for item in all_data:
        conv = convert_one(item)
        if conv is None:
            skipped += 1
            continue
        if args.no_system:
            conv["conversations"] = conv["conversations"][1:]  # remove system
        # Temporary metadata supports stratification.
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        conv["_scene"] = meta.get("scene") or item.get("scene") or "unknown"
        conv["_length_bucket"] = get_length_bucket(conv["conversations"][-1]["value"])
        conv["_group_key"] = intent_group_key(item)

        samples.append(conv)

    print(f"Converted: {len(samples)}, skipped: {skipped}")

    rng = random.Random(args.seed)

    train_samples, val_samples = grouped_split(
        samples, args.val_split, rng, stratified=not args.no_stratified
    )

    # Remove temporary metadata before export.
    for sample in train_samples:
        sample.pop("_scene", None)
        sample.pop("_length_bucket", None)
        sample.pop("_group_key", None)
    for sample in val_samples:
        sample.pop("_scene", None)
        sample.pop("_length_bucket", None)
        sample.pop("_group_key", None)

    # Print summary statistics.
    print("\n=== Dataset Statistics ===")
    print(f"Train: {len(train_samples)} samples")
    print(f"Val:   {len(val_samples)} samples")
    print(f"Split: {args.val_split:.0%} validation")

    # Write datasets.
    for path, subset in [
        (Path(args.train_output), train_samples),
        (Path(args.val_output), val_samples),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  {path}: {len(subset)} samples")

    print(f"\nDone. train={len(train_samples)}, val={len(val_samples)}")


if __name__ == "__main__":
    main()
