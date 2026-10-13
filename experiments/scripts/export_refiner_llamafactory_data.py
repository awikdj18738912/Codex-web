"""Register the already grouped strict-local warmup data for LLaMA-Factory."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.synthetic_gate_data import digest, file_hash, read_rows, write_json
from system.refinement_protocol import apply_structured_patch


def export(source: Path, output: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    groups, ids, files, audit = defaultdict(set), set(), {}, []
    for split in ("train", "dev", "test"):
        path = source / f"{split}.jsonl"
        if file_hash(path) != manifest["output_sha256"][split]:
            raise ValueError("Prepared SFT evidence changed")
        records = []
        for row in read_rows(path):
            messages, meta = row["messages"], row["metadata"]
            if meta["seed_id"] in ids or meta["split"] != split:
                raise ValueError("Duplicate seed or incorrect split")
            ids.add(meta["seed_id"])
            groups[meta["split_group"]].add(split)
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError("Expected one system/user/assistant turn")
            target, _, issue = apply_structured_patch(messages[1]["content"], messages[2]["content"], policy="strict_local")
            if issue or digest(target) != meta["target_text_sha256"]:
                raise ValueError("SFT answer fails the current runtime contract")
            records.append(dict(messages=messages))
            audit.append(dict(**meta, exported_split=split, messages_sha256=digest(messages)))
        files[split] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Template group crosses splits")
    registry = {}
    for split in files:
        registry[f"refiner_local_{split}"] = dict(file_name=f"{split}.jsonl", formatting="sharegpt",
            columns={"messages": "messages"}, tags=dict(role_tag="role", content_tag="content",
            user_tag="user", assistant_tag="assistant", system_tag="system"))
    all_files = {f"{split}.jsonl": text for split, text in files.items()}
    all_files["dataset_info.json"] = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    all_files["data_audit.jsonl"] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit)
    if output.exists() and any(output.iterdir()):
        if all((output / name).is_file() and (output / name).read_text() == text for name, text in all_files.items()):
            return json.loads((output / "manifest.json").read_text())
        raise FileExistsError("Existing LLaMA-Factory dataset differs; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, text in all_files.items():
        (output / name).write_text(text, encoding="utf-8")
    summary = dict(records=len(ids), groups=len(groups), split_counts={split: len(read_rows(output / f"{split}.jsonl")) for split in files},
        source=str(source.resolve()), source_manifest_sha256=file_hash(source / "manifest.json"),
        exported_sha256={name: file_hash(output / name) for name in all_files},
        supervision="controlled_text_format_warmup_not_real_asr_gold", training_test_split_kept_separate=True)
    write_json(output / "manifest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/refiner_local_edit_20261009/format_sft")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/llamafactory/data")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
