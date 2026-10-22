import random

import pytest

from pipeline.scripts.export_sft import grouped_split, intent_group_key


def sample(key, scene="a", bucket="short", variant=0):
    return {"id": f"{key}-{variant}", "_group_key": key,
            "_scene": scene, "_length_bucket": bucket}


def test_intent_key_prefers_final_plan_and_only_normalizes_chinese_terminal_stop():
    row = {"meta": {"generation_plan": {"expected_clean": "方案A。  "},
                    "fragment_reference_clean": "ignored"},
           "output": {"refined_text": "fallback"}}
    assert intent_group_key(row) == "方案A"
    assert intent_group_key({"meta": {"fragment_reference_clean": "片段。"}}) == "片段"
    assert intent_group_key({"output": {"refined_text": "v1.2; "}}) == "v1.2;"
    assert intent_group_key({"output": {"refined_text": "代码()"}}) != intent_group_key(
        {"output": {"refined_text": "代码"}}
    )


@pytest.mark.parametrize("stratified", [True, False])
def test_groups_never_cross_splits_and_are_reproducible(stratified):
    rows = [sample("same", "scene1", "short", 0),
            sample("same", "scene2", "long", 1)]
    rows += [sample(f"unique{i}", "scene1", "short") for i in range(20)]
    train, val = grouped_split(rows, .25, random.Random(17), stratified)
    train2, val2 = grouped_split(rows, .25, random.Random(17), stratified)
    assert [x["id"] for x in train] == [x["id"] for x in train2]
    assert [x["id"] for x in val] == [x["id"] for x in val2]
    assert not ({x["_group_key"] for x in train} & {x["_group_key"] for x in val})
    assert len(train) + len(val) == len(rows)


@pytest.mark.parametrize("ratio,want_val", [(0, 0), (1, 5)])
def test_ratio_edges_and_empty_samples(ratio, want_val):
    rows = [sample(str(i)) for i in range(5)]
    train, val = grouped_split(rows, ratio, random.Random(1))
    assert len(val) == want_val
    assert grouped_split([], ratio, random.Random(1)) == ([], [])


@pytest.mark.parametrize("ratio", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratio_fails_early(ratio):
    with pytest.raises(ValueError, match="val_split"):
        grouped_split([], ratio, random.Random(1))


def test_missing_group_key_is_rejected():
    with pytest.raises(ValueError, match="_group_key"):
        grouped_split([{}], .15, random.Random(1))
