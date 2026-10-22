import pytest

from pipeline.src.utils.synthetic_oral import build_oral, validate_correction_slots


def test_rejects_vibe_coding_clause_candidates_with_comma():
    intent = "在React应用中使用Redux管理状态。"
    values = [
        "使用Vue.js替代React来管理状态，不对，用Spring Boot替代ASP.NET Core实现后端服务",
        "在React应用中使用Redux管理状态",
    ]
    with pytest.raises(ValueError):
        validate_correction_slots(intent, values)


def test_rejects_a_final_value_that_is_the_entire_intent():
    intent = "在React应用中使用Redux管理状态。"
    with pytest.raises(ValueError):
        validate_correction_slots(intent, ["React", "在React应用中使用Redux管理状态"])


def test_rejects_a_clause_without_punctuation_as_a_slot():
    intent = "请在React应用中使用Redux管理状态。"
    values = ["用Spring Boot替代ASP.NET Core实现后端服务", "Redux"]
    with pytest.raises(ValueError, match="whole clause"):
        validate_correction_slots(intent, values)


def test_allows_grouped_numbers_and_english_decimal_version_or_model_values():
    assert validate_correction_slots("合计348,700元需要复核。", ["200,000", "348,700"]) is None
    assert validate_correction_slots("请使用Python 3.12.2运行服务。", ["Python 3.11.8", "3.12.2"]) is None
    assert validate_correction_slots("请加载config.yaml后启动服务。", ["old.yaml", "config.yaml"]) is None


def test_allows_normal_entity_names_starting_with_yong_or_shiyong():
    assert validate_correction_slots("我在用友网络提交账单。", ["金蝶软件", "用友网络"]) is None
    assert validate_correction_slots("请查阅使用指南中的部署步骤。", ["操作手册", "使用指南"]) is None


def test_terminal_ascii_period_preserves_code_and_slot_dots():
    assert validate_correction_slots("Please use build-1.2.", ["build-1.1", "build-1.2"]) is None
    assert validate_correction_slots("请运行foo.bar.", ["old.", "bar."]) is None
    with pytest.raises(ValueError):
        validate_correction_slots("deploy v1.2.", ["deploy v1.1", "deploy v1.2"])


def test_rejects_non_numeric_ascii_comma_and_repeated_final_slot():
    with pytest.raises(ValueError):
        validate_correction_slots("采用Redux方案。", ["Vue, React", "Redux"])
    with pytest.raises(ValueError):
        validate_correction_slots("先选Redux方案，再确认Redux方案。", ["MobX", "Redux"])


def test_allows_short_action_command_as_a_proper_substring():
    intent = "请执行 kubectl apply -f prod.yaml 完成服务部署。"
    values = ["kubectl apply -f staging.yaml", "kubectl apply -f prod.yaml"]
    assert validate_correction_slots(intent, values) is None


def test_legacy_build_oral_reconstruction_remains_permissive():
    intent = "在React应用中使用Redux管理状态。"
    legacy_values = ["Vue.js替代React来管理状态", "在React应用中使用Redux管理状态"]
    with pytest.raises(ValueError):
        validate_correction_slots(intent, legacy_values)
    # Historical construction-v1 decoding still reconstructs the same source.
    assert build_oral(intent, legacy_values, [], ["oral:low"]) == (
        "嗯，Vue.js替代React来管理状态，不对，在React应用中使用Redux管理状态。"
    )
