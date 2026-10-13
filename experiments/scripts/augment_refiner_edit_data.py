"""Create a train-only synthetic corruption round while freezing the original holdouts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.prepare_local_edit_sft import local_patch
from experiments.scripts.synthetic_gate_data import digest, file_hash, read_rows, write_json
from system.refinement_protocol import (
    STRICT_LOCAL_PROTOCOL_VERSION,
    apply_structured_patch,
)

LEXICAL_SPANS = (
    "现在", "今天", "明天", "昨天", "之前", "以后", "开始", "进行", "需要",
    "可以", "应该", "可能", "这个", "那个", "我们", "你们", "他们", "如果",
    "因为", "所以", "就是", "然后", "首先", "目前", "已经", "继续", "告诉",
    "知道", "觉得", "感觉", "出现", "发生", "恢复", "改变", "考虑", "选择",
    "复查", "电话", "时间", "地方", "城市", "工作", "安排", "会议", "开会",
    "回家", "出门", "吃饭", "休息", "检查", "结果", "问题", "饮食", "睡觉",
    "早上", "晚上", "上午", "下午", "每天", "有些", "一些", "其实",
)


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in rows).encode("utf-8")


def make_variants(text: str) -> list[tuple[str, str, str]]:
    """Return (corrupted input, reason, augmentation type), validated by caller."""
    variants = []
    for span in LEXICAL_SPANS:
        if text.count(span) != 1:
            continue
        pos = text.index(span)
        repeated = text[:pos] + span + text[pos:]
        variants.append((repeated, "repetition", "word_repeat"))
        restarted = text[:pos] + span[0] + text[pos:]
        if restarted != repeated:
            variants.append((restarted, "repetition", "character_restart"))
    return variants


def build(output: Path, source: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refuse to overwrite existing v2 dataset")

    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["exported_sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError("Source dataset hash mismatch: " + name)

    train_rows = read_rows(source / "train.jsonl")
    dev_rows = read_rows(source / "dev.jsonl")
    test_rows = read_rows(source / "test.jsonl")
    audits = read_rows(source / "data_audit.jsonl")
    audit_by_id = {row["seed_id"]: row for row in audits}
    if len(audit_by_id) != len(audits):
        raise ValueError("Duplicate source audit identities")
    if (len(train_rows), len(dev_rows), len(test_rows)) != (628, 122, 250):
        raise ValueError("Unexpected frozen source split sizes")

    group_splits: dict[str, set[str]] = {}
    for audit in audits:
        group_splits.setdefault(audit["split_group"], set()).add(audit["exported_split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Source template group crosses data splits")

    augmentable_categories = {"clean", "protected_content", "syntax_boundary"}
    additions: list[dict] = []
    addition_audits: list[dict] = []
    skip_counts: Counter[str] = Counter()
    used_ids = set(audit_by_id)
    train_audits = [audit for audit in audits if audit["exported_split"] == "train"]
    if len(train_audits) != len(train_rows):
        raise ValueError("Training rows and audit rows have different lengths")
    for row, audit in zip(train_rows, train_audits):
        messages = row["messages"]
        if digest(messages) != audit["messages_sha256"]:
            raise ValueError("Training row differs from audited source")
        if audit["action"] != "keep" or audit["category"] not in augmentable_categories:
            continue

        original = messages[1]["content"]
        candidates = make_variants(original)
        seed = int(hashlib.sha256(audit["seed_id"].encode("utf-8")).hexdigest()[:8], 16)
        if not candidates:
            if audit["category"] == "clean":
                candidates = [("嗯，" + original, "disfluency", "filler_prefix")]
            else:
                skip_counts["no_safe_lexical_span"] += 1
                continue
        available_types = list(dict.fromkeys(candidate[2] for candidate in candidates))
        preferred_type = available_types[seed % len(available_types)]
        candidates.sort(key=lambda candidate: (
            candidate[2] != preferred_type,
            hashlib.sha256((audit["seed_id"] + candidate[0]).encode("utf-8")).hexdigest(),
        ))
        chosen = None
        errors = []
        for corrupted, reason, aug_type in candidates:
            try:
                payload = local_patch(corrupted, original, reason)
                response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                applied, _, issue = apply_structured_patch(
                    corrupted, response, policy="strict_local")
                if issue or applied != original:
                    raise ValueError(str(issue or "target mismatch"))
                chosen = (corrupted, payload, response, aug_type)
                break
            except ValueError as exc:
                errors.append(str(exc))
        if chosen is None:
            skip_counts["no_valid_local_patch"] += 1
            continue

        corrupted, payload, response, aug_type = chosen
        new_id = audit["seed_id"] + "__v2_" + aug_type
        if new_id in used_ids:
            raise ValueError("Duplicate generated identity: " + new_id)
        used_ids.add(new_id)
        addition = {
            "messages": [
                messages[0],
                {"role": "user", "content": corrupted},
                {"role": "assistant", "content": response},
            ]
        }
        additions.append(addition)
        addition_audits.append({
            "seed_id": new_id,
            "base_seed_id": audit["seed_id"],
            "base_seed_sha256": audit["seed_sha256"],
            "category": audit["category"],
            "augmentation_type": aug_type,
            "split": "train",
            "split_group": audit["split_group"],
            "action": payload["action"],
            "reason": payload["reason"],
            "source_sha256": digest(corrupted),
            "target_sha256": digest(original),
            "messages_sha256": digest(addition["messages"]),
            "supervision_status": "train_only_synthetic_corruption",
            "real_asr_pair": False,
            "human_audio_gold": False,
            "protocol_version": STRICT_LOCAL_PROTOCOL_VERSION,
        })

    output.mkdir(parents=True, exist_ok=True)
    original_train = (source / "train.jsonl").read_bytes()
    if original_train and not original_train.endswith(b"\n"):
        original_train += b"\n"
    (output / "train.jsonl").write_bytes(original_train + jsonl_bytes(additions))
    shutil.copyfile(source / "dev.jsonl", output / "dev.jsonl")
    shutil.copyfile(source / "test.jsonl", output / "test.jsonl")
    shutil.copyfile(source / "data_audit.jsonl", output / "data_audit.jsonl")
    (output / "augmentation_audit.jsonl").write_bytes(jsonl_bytes(addition_audits))
    registry = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
    registry = {
        "refiner_local_v2_train": registry["refiner_local_train"],
        "refiner_local_v2_dev": registry["refiner_local_dev"],
        "refiner_local_v2_test": registry["refiner_local_test"],
    }
    (output / "dataset_info.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = Counter(item["augmentation_type"] for item in addition_audits)
    manifest_v2 = {
        "version": 2,
        "source_manifest_sha256": file_hash(manifest_path),
        "source_train_sha256": file_hash(source / "train.jsonl"),
        "frozen_dev_sha256": file_hash(source / "dev.jsonl"),
        "frozen_test_sha256": file_hash(source / "test.jsonl"),
        "frozen_test_records": len(test_rows),
        "original_split_counts": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "added_train_records": len(additions),
        "augmentation_type_counts": dict(counts),
        "skipped_counts": dict(skip_counts),
        "result_split_counts": {
            "train": len(train_rows) + len(additions), "dev": len(dev_rows), "test": len(test_rows)
        },
        "split_counts": {
            "train": len(train_rows) + len(additions), "dev": len(dev_rows), "test": len(test_rows)
        },
        "result_action_counts": {
            "added_train": dict(Counter(item["action"] for item in addition_audits)),
        },
        "output_sha256": {
            name: file_hash(output / name)
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "data_audit.jsonl",
                         "augmentation_audit.jsonl", "dataset_info.json")
        },
        "exported_sha256": {
            name: file_hash(output / name)
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "data_audit.jsonl",
                         "dataset_info.json")
        },
        "strict_local_roundtrip_validated": True,
        "human_audio_gold": False,
        "real_asr_pair": False,
        "scope": "synthetic_train_only_format_augmentation; frozen original dev and test",
    }
    # Verify the holdouts are byte-identical after copying.
    if manifest_v2["frozen_dev_sha256"] != file_hash(output / "dev.jsonl"):
        raise ValueError("Development split changed")
    if manifest_v2["frozen_test_sha256"] != file_hash(output / "test.jsonl"):
        raise ValueError("Test split changed")
    train_actions = Counter(audit["action"] for audit in audits if audit["exported_split"] == "train")
    train_actions.update(item["action"] for item in addition_audits)
    manifest_v2["result_action_counts"]["train"] = dict(train_actions)
    write_json(output / "manifest.json", manifest_v2)
    return manifest_v2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/llamafactory/data")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/llamafactory/data_v2")
    args = parser.parse_args()
    result = build(args.output, args.source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
