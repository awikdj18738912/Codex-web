"""Freeze a provisional 13-class Refiner corpus for a Qwen2.5 LoRA run.

The source ledger is never edited. Unreviewed candidates remain explicitly
marked provisional in the output manifest; the existing human-review queue and
the complete 130-row audit sample are isolated from train/dev/test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/aim0/data/datasets/ASR/refiner_balanced_13classes_20261026_no_A04_B03_C03_C04")
AUDIT = Path("/home/aim0/data/datasets/ASR/refiner_13class_audit_sample_20261026")
OUTPUT = Path("/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261026_provisional")
CLASSES = (
    "A01", "A02", "A03", "B01", "B02", "C01", "C02",
    "D01", "D02", "E03", "E05", "E06", "E07",
)
SEED = 20261027
SYSTEM_PROMPT = (
    "你是中文ASR转写精修助手。只做最小必要修改：清理口癖和无意义重复，"
    "处理明确的自我修正，修正明显错字或误词，修复语法赘余、必要标点和句界。"
    "说话者明确改口时保留最终意图；忽略强调表达，重复表达可以压缩。"
    "不得总结、扩写、补全或推测输入没有的信息；不得删掉不同事实、实体、数量、"
    "否定、条件、立场或情态。保留数字和日期的数值及输入写法，保留英文名称、"
    "大小写、空格、连字符和代码符号。__ENTITY_NNN__形式的实体占位符须逐字保留，"
    "每个出现一次且顺序不变。无需修改时原样输出。只输出精修正文，不要类别、"
    "解释、实体列表、<KEY>或JSON。"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    source_path = SOURCE / "candidates.pending_review.jsonl"
    queue_path = SOURCE / "review_queue.pending.jsonl"
    sample_path = AUDIT / "sample.jsonl"
    review_path = AUDIT / "codex_first_review_130.jsonl"
    rows = read_jsonl(source_path)
    queue = read_jsonl(queue_path)
    audit_sample = read_jsonl(sample_path)
    audit_review = read_jsonl(review_path)
    if len(rows) != 38233 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("The active source candidate file changed or contains duplicate IDs")
    if len(queue) != 21 or len(audit_sample) != 130 or len(audit_review) != 130:
        raise ValueError("Expected the current 21-row queue and complete 130-row audit sample")
    ids = {row["id"] for row in rows}
    queue_ids = {row["id"] for row in queue}
    sample_ids = {row["id"] for row in audit_sample}
    if queue_ids & sample_ids:
        raise ValueError("Review queue and audit sample overlap; inspect before splitting")
    if not queue_ids <= ids or not sample_ids <= ids:
        raise ValueError("Review/audit IDs no longer match active source candidates")
    issue_status = {row["id"]: row.get("codex_review_status") for row in audit_review}
    if any(row["id"] not in sample_ids for row in audit_review):
        raise ValueError("Audit findings do not align with the frozen 130-row sample")
    for row in rows:
        if row.get("category") not in CLASSES:
            raise ValueError(f"Unexpected or disabled class: {row.get('category')}")
        if row.get("dataset") not in {"Gemma4", "FCGEC", "CS2W"}:
            raise ValueError(f"Unknown source: {row.get('dataset')}")
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            raise ValueError(f"Missing input: {row['id']}")
        if not isinstance(row.get("reference"), str) or not row["reference"].strip():
            raise ValueError(f"Missing reference: {row['id']}")

    isolated_ids = queue_ids | sample_ids
    provisional = [row for row in rows if row["id"] not in isolated_ids]
    if len(provisional) != 38082:
        raise ValueError(f"Unexpected provisional pool size: {len(provisional)}")

    # Keep exact normalized inputs and references together. The source ledger
    # currently has no duplicates, but the grouping also protects future edits.
    pair_groups: dict[str, list[dict]] = {}
    for row in provisional:
        pair = hashlib.sha256((norm(row["input"]) + "\0" + norm(row["reference"])).encode()).hexdigest()
        pair_groups.setdefault(pair, []).append(row)

    rng = random.Random(SEED)
    strata: dict[tuple[str, str], list[list[dict]]] = {}
    for group in pair_groups.values():
        categories = {row["category"] for row in group}
        sources = {row["dataset"] for row in group}
        if len(categories) != 1 or len(sources) != 1:
            # A duplicate pair across labels/sources cannot be split safely;
            # isolate every member for human/data-leakage review.
            continue
        key = (next(iter(categories)), next(iter(sources)))
        strata.setdefault(key, []).append(group)

    split_rows: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for key in sorted(strata):
        groups = list(strata[key])
        rng.shuffle(groups)
        n = sum(len(group) for group in groups)
        target_dev = max(1, round(n * 0.05))
        target_test = max(1, round(n * 0.05))
        dev_n = test_n = 0
        train_groups = []
        for group in groups:
            if dev_n < target_dev:
                split_rows["dev"].extend(group)
                dev_n += len(group)
            elif test_n < target_test:
                split_rows["test"].extend(group)
                test_n += len(group)
            else:
                train_groups.extend(group)
        split_rows["train"].extend(train_groups)

    assigned = {row["id"] for split in split_rows.values() for row in split}
    if len(assigned) != len(provisional):
        raise ValueError("Some provisional rows could not be assigned to a split")
    if any(len(split_rows[name]) == 0 for name in split_rows):
        raise ValueError("Empty data split")

    def training_row(row: dict) -> dict:
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["reference"]},
        ]}

    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, split in split_rows.items():
        write_json(OUTPUT / f"{name}.json", [training_row(row) for row in split])
    sidecars = []
    for name, split in split_rows.items():
        for row in split:
            sidecars.append({"id": row["id"], "split": name, "category": row["category"],
                "source": row["dataset"], "group_id": row.get("group_id"),
                "source_id": row.get("source_id"), "input": row["input"],
                "reference": row["reference"], "human_review_status": row.get("human_review_status"),
                "training_ready_in_source": row.get("training_ready")})
    (OUTPUT / "split_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sidecars), encoding="utf-8")
    write_json(OUTPUT / "dataset_info.json", {
        "refiner13_train": {"file_name": "train.json", "formatting": "sharegpt",
            "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
        "refiner13_dev": {"file_name": "dev.json", "formatting": "sharegpt",
            "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}},
    })
    for name in ("train", "dev", "test"):
        split_rows[name].sort(key=lambda row: row["id"])
    class_counts = {name: dict(sorted(Counter(row["category"] for row in split).items()))
        for name, split in split_rows.items()}
    source_counts = {name: dict(sorted(Counter(row["dataset"] for row in split).items()))
        for name, split in split_rows.items()}
    manifest = {
        "status": "provisional_training_experiment",
        "authorized_by_user": True,
        "warning": "Source ledger remains pending human review. Metrics are preliminary and are not gold-label claims.",
        "source_candidate": str(source_path), "source_sha256": sha256(source_path),
        "source_candidate_count": len(rows), "active_classes": list(CLASSES),
        "excluded_review_queue_count": len(queue_ids), "excluded_audit_sample_count": len(sample_ids),
        "audit_sample_codex_status_counts": dict(Counter(issue_status.values())),
        "provisional_pool_count": len(provisional), "split_counts": {key: len(value) for key, value in split_rows.items()},
        "class_counts": class_counts, "source_counts": source_counts,
        "split_seed": SEED, "stratify_by": ["category", "dataset"],
        "grouping": "Exact NFKC+whitespace-normalized input/reference pair; source records show no duplicates.",
        "system_prompt": SYSTEM_PROMPT,
        "files_sha256": {name: sha256(OUTPUT / name) for name in
            ("train.json", "dev.json", "test.json", "split_audit.jsonl", "dataset_info.json")},
    }
    write_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
