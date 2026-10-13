import json

import pytest

from experiments.scripts import export_refiner_llamafactory_data as exporter
from experiments.scripts.prepare_local_edit_sft import prepare
from experiments.scripts.synthetic_gate_data import make_seeds


def source_fixture(tmp_path):
    seed_file = tmp_path / "texts.jsonl"
    seed_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in make_seeds(48, 20261009)))
    source = tmp_path / "sft"
    prepare(seed_file, source)
    return source


def test_registration_preserves_splits_and_keeps_audit_metadata_out_of_messages(tmp_path):
    source = source_fixture(tmp_path)
    output = tmp_path / "registered"
    manifest = exporter.export(source, output)
    assert sum(manifest["split_counts"].values()) == 48
    assert manifest["training_test_split_kept_separate"] is True
    registry = json.loads((output / "dataset_info.json").read_text())
    assert set(registry) == {"refiner_local_train", "refiner_local_dev", "refiner_local_test"}
    assert registry["refiner_local_train"]["columns"] == {"messages": "messages"}
    assert registry["refiner_local_train"]["tags"]["assistant_tag"] == "assistant"
    for split in ("train", "dev", "test"):
        original = exporter.read_rows(source / f"{split}.jsonl")
        actual = exporter.read_rows(output / f"{split}.jsonl")
        assert [row["messages"] for row in actual] == [row["messages"] for row in original]
        assert all(set(row) == {"messages"} for row in actual)
    before = (output / "train.jsonl").read_bytes()
    assert exporter.export(source, output) == manifest
    assert (output / "train.jsonl").read_bytes() == before


def test_changed_source_or_existing_user_data_is_never_silently_overwritten(tmp_path):
    source = source_fixture(tmp_path)
    output = tmp_path / "registered"
    exporter.export(source, output)
    (output / "train.jsonl").write_text("user edited\n")
    with pytest.raises(FileExistsError):
        exporter.export(source, output)
    assert (output / "train.jsonl").read_text() == "user edited\n"
    (source / "train.jsonl").write_text("source changed\n")
    new_output = tmp_path / "new_registered"
    with pytest.raises(ValueError, match="evidence changed"):
        exporter.export(source, new_output)
    assert not new_output.exists()
