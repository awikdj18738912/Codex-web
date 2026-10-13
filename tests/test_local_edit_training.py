from collections import defaultdict
import json

import pytest

from experiments.scripts.prepare_local_edit_sft import build_records, local_patch, prepare
from experiments.scripts.synthetic_gate_data import digest, make_seeds
from system.refinement_protocol import apply_structured_patch


def test_format_warmup_covers_keep_and_edit_without_erasing_meaningful_repeat():
    seeds = make_seeds(1000, 20261009)
    records = build_records(seeds)
    assert len(records) == 1000
    assert sum(row["metadata"]["action"] == "keep" for row in records) == 666
    assert sum(row["metadata"]["action"] == "replace" for row in records) == 334
    groups = defaultdict(set)
    for row in records:
        groups[row["metadata"]["split_group"]].add(row["metadata"]["split"])
        assert row["metadata"]["human_audio_gold"] is False
        assert row["metadata"]["real_asr_pair"] is False
        assert len(row["messages"]) == 3
        source = row["messages"][1]["content"]
        refined, payload, issue = apply_structured_patch(source, row["messages"][2]["content"], policy="strict_local")
        assert issue is None
        assert digest(refined) == row["metadata"]["target_text_sha256"]
        if row["metadata"]["category"] == "meaningful_repeat":
            assert payload["action"] == "keep" and refined == source
    assert all(len(value) == 1 for value in groups.values())
    assert any("__ENTITY_000__" in row["messages"][1]["content"] for row in records)


def test_deletion_is_uniquely_anchored_when_deleted_word_appears_twice():
    source = "我们我们明天开会。"
    target = "我们明天开会。"
    payload = local_patch(source, target, "repetition")
    assert payload["source"] == "我们我们"
    assert source.count(payload["source"]) == 1
    applied, _, issue = apply_structured_patch(source, json.dumps(payload, ensure_ascii=False), policy="strict_local")
    assert applied == target and issue is None


def test_two_distant_changes_are_not_packaged_as_one_training_patch():
    with pytest.raises(ValueError, match="Multiple edits"):
        local_patch("我们我们明天讨讨论安排。", "我们明天讨论安排。", "repetition")


def test_seed_hash_and_template_split_must_be_consistent():
    seeds = make_seeds(49, 20261009)
    seeds[-1]["synthetic_split"] = "dev" if seeds[0]["synthetic_split"] != "dev" else "test"
    seeds[-1]["seed_sha256"] = digest({key: value for key, value in seeds[-1].items() if key != "seed_sha256"})
    with pytest.raises(ValueError, match="crosses data splits"):
        build_records(seeds)
    seeds = make_seeds(1, 20261009)
    seeds[0]["intended_text"] = "被改写的目标"
    with pytest.raises(ValueError, match="metadata changed"):
        build_records(seeds)


def test_existing_training_dataset_is_not_overwritten(tmp_path):
    source = tmp_path / "texts.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in make_seeds(48, 20261009)))
    output = tmp_path / "sft"
    summary = prepare(source, output)
    assert summary["all_targets_roundtrip_validated"] is True
    before = (output / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source, output)
    assert (output / "train.jsonl").read_bytes() == before
