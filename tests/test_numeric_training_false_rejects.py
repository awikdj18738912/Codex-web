from __future__ import annotations

from pipeline.src.utils.numeric_policy import lock_numeric_target


def test_weekday_self_correction_can_drop_superseded_weekday_only():
    source = "周一开会，不对周二，最后定周三开会。"
    assert lock_numeric_target(source, "最后定周三开会。") == "最后定周三开会。"
    assert lock_numeric_target("周一开会，最后定周三开会。", "最后定周三开会。") is None
    assert lock_numeric_target(
        "周一开会，不对周二，最后定周三开会，周五还有另一场。",
        "最后定周三开会。",
    ) is None


def test_weekday_correction_can_drop_marker_but_keep_final_clause_verbatim():
    source = "嗯，周一，不对，周二，最后改成，周三开会。"
    assert lock_numeric_target(source, "周三开会。") == "周三开会。"
    assert lock_numeric_target("周一开会，最后定周三开会。", "周三开会。") is None


def test_lexical_yixia_is_not_treated_as_a_numeric_literal():
    source = "呃，这个Kotlin函数，我重新改一下。"
    assert lock_numeric_target(source, "这个Kotlin函数，我重新改。") == "这个Kotlin函数，我重新改。"


def test_code_sentence_cleanup_keeps_comparison_and_api_numbers_owned():
    source = "我刚想用NestJS写一下接口，while循环里status判断=== true，不然Twilio API发不了短信。"
    target = "我刚想用NestJS写接口，while循环里status判断=== true，不然Twilio API发不了短信。"
    assert lock_numeric_target(source, target) == target
    assert lock_numeric_target(source, target.replace("=== true", "=== false")) is not None


def test_numeric_drift_and_unsupported_chinese_to_arabic_itn_stay_rejected():
    assert lock_numeric_target("有3个苹果。", "有4个苹果。") is None
    assert lock_numeric_target("按一到三的范围处理。", "按1到3的范围处理。") is None


def test_leading_zero_and_version_rewrites_stay_rejected():
    source = "编号0012，版本1.10。"
    # Runtime restoration must undo this surface rewrite before it can become a label.
    assert lock_numeric_target(source, "编号12，版本1.1。") == source
