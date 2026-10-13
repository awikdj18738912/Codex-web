"""Prepare controlled KEEP/local-patch format warmup, not real ASR supervision."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.synthetic_gate_data import NAMES, PLACES, digest, file_hash, read_rows, write_json
from system.refinement_protocol import (
    STRICT_LOCAL_MAX_SPAN_CHARS, STRICT_LOCAL_PROTOCOL_VERSION,
    apply_structured_patch, strict_local_system_prompt,
)


def local_patch(source: str, target: str, reason: str) -> dict:
    if source == target:
        return dict(action="keep", source="", target="", reason="no_change")
    edits = [(i, j, a, b) for tag, i, j, a, b in
             SequenceMatcher(None, source, target, autojunk=False).get_opcodes() if tag != "equal"]
    if len(edits) != 1:
        raise ValueError("Multiple edits require separate source-owned examples")
    i, j, a, b = edits[0]
    left, right = i, j
    while True:
        patch_source = source[left:right]
        patch_target = source[left:i] + target[a:b] + source[j:right]
        if max(len(patch_source), len(patch_target)) > STRICT_LOCAL_MAX_SPAN_CHARS:
            raise ValueError("A unique local source cannot fit the span budget")
        starts = [index for index in range(len(source))
                  if patch_source and source.startswith(patch_source, index)]
        if len(starts) == 1:
            payload = dict(action="replace", source=patch_source, target=patch_target, reason=reason)
            applied, _, issue = apply_structured_patch(source, json.dumps(payload, ensure_ascii=False), policy="strict_local")
            if issue or applied != target:
                raise ValueError(f"Training patch is not executable: {issue}")
            return payload
        # Add unchanged anchors when the deletion itself occurs repeatedly.
        if right < len(source):
            right = min(len(source), right + max(1, j - i))
        elif left:
            left -= 1
        else:
            raise ValueError("Source span cannot be uniquely anchored")


def placeholder_variant(source: str, target: str) -> tuple[str, str]:
    surfaces = [text for text in (*NAMES, *PLACES, "奥希替尼", "吉非替尼") if text in source]
    surfaces.sort(key=lambda text: (source.index(text), -len(text)))
    for index, text in enumerate(surfaces):
        marker = f"__ENTITY_{index:03d}__"
        source, target = source.replace(text, marker), target.replace(text, marker)
    return source, target


def build_records(seeds: list[dict]) -> list[dict]:
    if len({row["seed_id"] for row in seeds}) != len(seeds):
        raise ValueError("Duplicate seed identities")
    split_groups = defaultdict(set)
    records = []
    for index, row in enumerate(seeds):
        expected = digest({key: value for key, value in row.items() if key != "seed_sha256"})
        if row["seed_sha256"] != expected:
            raise ValueError("Seed metadata changed")
        split_groups[row["template_group"]].add(row["synthetic_split"])
        if row["synthetic_split"] not in {"train", "dev", "test"}:
            raise ValueError("Invalid split")
        source, target = row["spoken_text"], row["intended_text"]
        variant = "plain"
        if index % 2:
            source, target = placeholder_variant(source, target)
            variant = "synthetic_placeholders"
        reason = {"filler": "disfluency", "stutter": "repetition"}.get(row["category"], "no_change")
        if source != target and reason == "no_change":
            raise ValueError("No intended local edit has been specified for this category")
        payload = local_patch(source, target, reason)
        response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        applied, _, issue = apply_structured_patch(source, response, policy="strict_local")
        if issue or applied != target:
            raise ValueError("Training response fails runtime validation")
        records.append(dict(
            messages=[dict(role="system", content=strict_local_system_prompt(numeric_rule_mode=True)),
                      dict(role="user", content=source), dict(role="assistant", content=response)],
            metadata=dict(seed_id=row["seed_id"], seed_sha256=row["seed_sha256"],
                category=row["category"], domain=row["domain"], split=row["synthetic_split"],
                split_group=row["template_group"], input_variant=variant,
                protocol_version=STRICT_LOCAL_PROTOCOL_VERSION, action=payload["action"],
                target_text_sha256=digest(target), supervision_status="controlled_text_format_warmup",
                real_asr_pair=False, human_audio_gold=False)))
    if any(len(splits) != 1 for splits in split_groups.values()):
        raise ValueError("Template family crosses data splits")
    return records


def prepare(source: Path, output: Path) -> dict:
    rows = read_rows(source)
    if not rows:
        raise ValueError("No seed texts")
    records = build_records(rows)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refuse to overwrite existing SFT data")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        selected = [row for row in records if row["metadata"]["split"] == split]
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    summary = dict(records=len(records), actions=dict(Counter(row["metadata"]["action"] for row in records)),
        splits={split: dict(Counter(row["metadata"]["action"] for row in records if row["metadata"]["split"] == split))
                for split in ("train", "dev", "test")},
        categories=dict(Counter(row["category"] for row in rows)),
        unique_template_groups=len({row["metadata"]["split_group"] for row in records}),
        protocol_version=STRICT_LOCAL_PROTOCOL_VERSION, all_targets_roundtrip_validated=True,
        source=str(source.resolve()), source_sha256=file_hash(source),
        source_code_sha256=file_hash(Path(__file__)),
        output_sha256={split: file_hash(output / f"{split}.jsonl") for split in ("train", "dev", "test")},
        scope="synthetic_format_and_preservation_warmup_not_real_asr_quality_evaluation")
    write_json(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/gate_synthetic_20261009/texts.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/refiner_local_edit_20261009/format_sft")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
