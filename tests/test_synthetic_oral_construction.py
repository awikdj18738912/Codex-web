from pipeline.src.utils.synthetic_oral import build_oral


def test_single_rollback_and_multi_keep_an_explicit_trace():
    assert build_oral("请寄到苏州。", ["南京", "苏州"], [], ["oral:low"]) == "嗯，请寄到南京，不对，苏州。"
    result = build_oral("我想用红色。", ["红色", "蓝色", "红色"], [], ["oral:high"])
    assert "红色，不对，蓝色，最后改成，红色" in result
    result = build_oral("周三开会。", ["周一", "周二", "周三"], [], ["oral:moderate"])
    assert "周一，不对，周二，最后改成，周三开会" in result


def test_spelling_boundary_and_passthrough_are_known_insertions():
    cue = [{"word": "允许", "character": "许"}, {"word": "砚台", "character": "砚"}]
    assert build_oral("联系人是许砚。", [], cue, ["oral:low"]).endswith("允许的许，砚台的砚。")
    value = build_oral("我想明天出发。", [], [], ["boundary_punctuation", "oral:low"])
    assert value.count("。") == 2
    assert build_oral("我到了。你在哪里？", [], [], ["clean"]) == "我到了。你在哪里？"


def test_question_force_is_preserved():
    assert build_oral("能查到订单吗？", [], [], ["oral:low"]).endswith("？")


def test_ascii_period_is_retained_without_appending_chinese_period():
    assert build_oral("Run docker build.", [], [], ["oral:low"]) == "嗯，Run docker build."
    # Existing v1 plans normalize an ASCII ending to ".。" before proofing.
    assert build_oral("Run docker build.。", [], [], ["oral:low"]) == "嗯，Run docker build.。"
