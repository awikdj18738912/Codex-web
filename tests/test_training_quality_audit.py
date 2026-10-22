"""Focused admission regressions from the two real-teacher pilots."""
import asyncio
import json

from pipeline.src.pipeline.training_quality import audit_pair, surface_issues, unsupported_edits, validate_audit


def good(target):
    return dict(source_valid=True, meaning_preserved=True, cleanup_complete=True,
                issues=[], cleaned_target=target)


def test_old_bad_targets_get_surface_vetoes():
    assert surface_issues("预算300元，不对，500元。", "预算300元，不对，500元。")
    assert surface_issues("收件人钟宁，钟表的钟，宁静的宁。", "收件人钟宁，钟表的钟，宁静的宁。")
    assert surface_issues("南方电网，念作nán fāng。", "南方电网，念作nán fāng。")
    assert surface_issues("小米，‘小’加‘米’。", "小米，‘小’加‘米’。")
    assert surface_issues("支付宝，bǎo zhàng fú。", "支付宝，bǎo zhàng fú。")
    assert not surface_issues("公司的车，需要保养。", "公司的车需要保养。")
    assert not surface_issues("请保留拼音说明，念作nán fāng。", "念作nán fāng。")


def test_generic_true_missing_fields_or_string_booleans_cannot_pass():
    assert not validate_audit({"pass": True}, "源", "目标")[0]
    value = good("目标")
    value["cleanup_complete"] = "true"
    assert not validate_audit(value, "源", "目标")[0]


def test_claimed_cleanup_requires_an_actually_unchanged_clean_target():
    value = good("红色。")
    assert not validate_audit(value, "红色，不蓝色，还是红色。", "红色，不蓝色，还是红色。")[0]
    assert validate_audit(good("红色。"), "红色，不蓝色，还是红色。", "红色。")[0]
    value = good("红色。")
    value["issues"] = [{"kind": "cleanup", "quote": "蓝色"}]
    assert not validate_audit(value, "源", "红色。")[0]


def test_only_one_final_chinese_separator_may_differ():
    assert validate_audit(good("正文"), "源", "正文，") == (True, "typed_audit_passed_terminal_punctuation_equivalent")
    assert validate_audit(good("正文"), "源", "正文。")[0]
    assert validate_audit(good("正文，"), "源", "正文。")[0]
    for target, cleaned in (
        ("正文?", "正文。"), ("正文!", "正文。"), ('正文“吗？”', '正文“吗。”'),
        ("正文。第二句", "正文。"), ("版本1.2.", "版本1.2"),
        ("docker build -t myapp .", "docker build -t myapp"),
        ("预算300元。", "预算500元。"), ("不要购买。", "要购买。"),
        ("如果下雨就取消。", "下雨就取消。"),
    ):
        assert not validate_audit(good(cleaned), "源", target)[0], (target, cleaned)


def test_plain_cleanup_cannot_invent_purchase_or_acronym_claims():
    assert unsupported_edits("三顿半的咖啡豆，手冲一下。", "买三顿半的咖啡豆，手冲一下。")
    assert unsupported_edits("国家电网，G-D-W。", "国家电网，简称G-D-W。")
    assert not unsupported_edits("嗯真。让我感到惊讶。", "真让我感到惊讶。")


def test_upstream_asr_pair_uses_semantic_review_for_character_replacements():
    class Judge:
        async def generate(self, **kwargs):
            target = json.loads(kwargs["user_prompt"])["target"]
            return json.dumps(good(target), ensure_ascii=False)

    passed, _, _ = asyncio.run(audit_pair(
        "收件人是钟宁。", "收件人是钟玲。", Judge(),
        recognition_repair=True, generation_plan=None,
    ))
    assert passed


def test_upstream_teacher_pair_sends_surface_residue_to_semantic_review():
    class Judge:
        def __init__(self): self.calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            target = json.loads(kwargs["user_prompt"])["target"]
            return json.dumps(good(target), ensure_ascii=False)

    judge = Judge()
    source = "真。让我有点意外.。"
    plan = {"generation_mode": "upstream-teacher-v1", "expected_clean": source}
    passed, _, _ = asyncio.run(audit_pair(source, source, judge, generation_plan=plan))
    assert passed and judge.calls == 1


def test_audit_fails_closed_on_call_error_or_old_boolean_response():
    class Error:
        async def generate(self, **kwargs):
            raise RuntimeError("test error")
    class Old:
        async def generate(self, **kwargs):
            return json.dumps({"pass": True})
    assert not asyncio.run(audit_pair("源", "目标", Error()))[0]
    assert not asyncio.run(audit_pair("源", "目标", Old()))[0]


def test_terminal_punctuation_equivalence_does_not_admit_doubled_punctuation():
    assert not validate_audit(good("Run docker build ."), "源", "Run docker build .。")[0]
    assert surface_issues("源", "Run docker build .。") == ["duplicated_terminal_punctuation"]
