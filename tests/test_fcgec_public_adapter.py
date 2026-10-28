import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scripts"))
from fcgec_public_adapter import iter_fcgec_candidates  # noqa: E402


def _write(tmp_path, rows):
    path = tmp_path / "fcgec.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _one(tmp_path, sentence, native, operations, flag=1):
    p = _write(tmp_path, {"row": {
        "sentence": sentence, "error_type": native, "error_flag": flag,
        "operation": json.dumps(operations, ensure_ascii=False),
    }})
    return list(iter_fcgec_candidates(p, "rev-test", "train"))


def test_official_switch_then_delete_uses_switched_character_indices(tmp_path):
    # Position 0 is not fixed by Switch; operations address the switched text.
    (candidate,) = _one(tmp_path, "ABCDE", "IWO", [
        {"Switch": [2, 0, 3, 1, 4], "Delete": [0]},
    ])
    assert candidate["reference"] == "ADBE"
    assert candidate["category"] == "C04"
    assert candidate["operation"] == {"Switch": [2, 0, 3, 1, 4], "Delete": [0]}


def test_insert_pos_minus_one_and_index_boundaries(tmp_path):
    (candidate,) = _one(tmp_path, "你好！", "CM", [
        {"Insert": [{"pos": -1, "tag": "INS_1", "label": ["啊"]}]},
    ])
    assert candidate["reference"] == "啊你好！"
    assert candidate["category"] == "C03"

    bad = _one(tmp_path, "你好", "CM", [
        {"Delete": [2]},
    ])[0]
    assert bad["valid"] is False
    assert "outside switched sentence length" in bad["diagnostic"]
    assert bad["reference"] is None


def test_multiple_references_keep_same_group_and_distinct_targets(tmp_path):
    rows = _one(tmp_path, "甲乙丙。", "CR", [
        {"Delete": [1]},
        {"Modify": [{"pos": 1, "tag": "MOD_1", "label": ["丁"]}]},
    ])
    assert [row["reference"] for row in rows] == ["甲丙。", "甲丁丙。"]
    assert [row["reference_index"] for row in rows] == [0, 1]
    assert {row["group_id"] for row in rows} == {rows[0]["source_id"]}
    assert all(row["category"] == "C02" for row in rows)


def test_modify_can_change_length_and_preserves_chinese_punctuation(tmp_path):
    sentence = "我去学校。"
    operations = [{"Modify": [{"pos": 1, "tag": "MOD_2+INS_1", "label": "已经去吧"}]}]
    # The tag requests three output characters, but this label has four.
    (bad,) = _one(tmp_path, sentence, "CM", operations)
    assert bad["valid"] is False
    assert bad["reference"] is None

    (good,) = _one(tmp_path, sentence, "CM", [
        {"Modify": [{"pos": 1, "tag": "MOD_2+INS_2", "label": "已经去吧"}]},
    ])
    assert good["valid"] is True
    assert good["reference"] == "我已经去吧校。"


def test_correct_sentence_category_and_input_are_not_mutated(tmp_path):
    sentence = "原句，保留！"
    source = {"sentence": sentence, "error_flag": 0, "error_type": "*", "operation": "[]"}
    rows = _write(tmp_path, {"ok": source})
    (candidate,) = list(iter_fcgec_candidates(rows, "r1", "valid"))
    assert candidate["category"] == "E07"
    assert candidate["reference"] == sentence
    assert source["sentence"] == sentence


def test_mixed_native_type_is_pending(tmp_path):
    (candidate,) = _one(tmp_path, "甲乙", "CR;CM", [{"Delete": [0]}])
    assert candidate["category"] == "pending"


def test_conflicting_correct_flag_and_type_is_quarantined(tmp_path):
    path = _write(tmp_path, {
        "conflict": {"sentence": "保留。", "error_type": "*", "error_flag": 1, "operation": "[]"},
    })
    (candidate,) = list(iter_fcgec_candidates(path, "r1", "train"))
    assert candidate["category"] == "pending"
    assert candidate["valid"] is False
    assert "conflicting metadata" in candidate["diagnostic"]


@pytest.mark.parametrize("operation, diagnostic", [
    ({"Modify": [{"pos": 0, "tag": "MOD_0", "label": ""}]}, "span must be positive"),
    ({"Modify": [{"pos": 0, "tag": "MOD_2", "label": "甲乙"}],
      "Insert": [{"pos": 0, "tag": "INS_1", "label": "丙"}]}, "falls inside a Modify span"),
])
def test_rejects_zero_modify_and_insert_skipped_inside_modify(tmp_path, operation, diagnostic):
    (candidate,) = _one(tmp_path, "甲乙丙", "CM", [operation])
    assert candidate["valid"] is False
    assert diagnostic in candidate["diagnostic"]
