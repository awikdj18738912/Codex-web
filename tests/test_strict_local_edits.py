import json
from pathlib import Path

import pytest

from system.refinement_protocol import apply_structured_patch, strict_local_system_prompt


def patch(source, target, reason="repetition", action="replace"):
    return json.dumps(dict(action=action, source=source, target=target, reason=reason), ensure_ascii=False)


@pytest.mark.parametrize("response,expected_issue", [
    ("我们明天开会。", "strict_local_json_required"),
    ('```json\n{"action":"keep"}\n```', "strict_local_json_required"),
    ('{"action":"keep","action":"replace","source":"","target":"","reason":"no_change"}', "strict_local_invalid_json"),
    ('{"action":"keep","source":"","target":"","reason":"no_change","extra":1}', "strict_local_invalid_fields"),
    ('{"action":"keep","source":"","target":null,"reason":"no_change"}', "strict_local_invalid_fields"),
    (patch("我们", "", "no_change", "keep"), "strict_local_invalid_keep"),
    (patch("我们我们", "我们", "guess"), "strict_local_invalid_action_or_reason"),
    (patch("我们", ""), "structured_patch_source_not_unique"),
    (patch("甲" * 33, "甲"), "strict_local_span_too_long"),
])
def test_strict_schema_rejects_unbounded_or_ambiguous_responses(response, expected_issue):
    source = "我们我们明天开会。"
    refined, _payload, issue = apply_structured_patch(source, response, policy="strict_local")
    assert refined == source and issue == expected_issue


def test_strict_patch_accepts_one_unique_edit_and_preserves_outside_span():
    source = "请我们我们明天去开会。后面保持原样，"
    refined, _, issue = apply_structured_patch(source, patch("我们我们", "我们"), policy="strict_local")
    assert issue is None
    assert refined == "请我们明天去开会。后面保持原样，"
    keep, _, issue = apply_structured_patch(source, patch("", "", "no_change", "keep"), policy="strict_local")
    assert issue is None and keep == source


def test_multiple_edits_and_overlapping_source_are_rejected():
    source = "请我们我们现在讨讨论，"
    _, _, issue = apply_structured_patch(source, patch("我们我们现在讨讨论", "我们现在讨论"), policy="strict_local")
    assert issue == "strict_local_multiple_edit_regions"
    _, _, issue = apply_structured_patch("哈哈哈，好。", patch("哈哈", "哈"), policy="strict_local")
    assert issue == "structured_patch_source_not_unique"


def test_window_terminal_comma_cannot_be_changed_to_period():
    source = "如果脚气一直不好，"
    refined, _, issue = apply_structured_patch(source, patch("不好，", "不好。", "boundary_punctuation"), policy="strict_local")
    assert refined == source and issue == "strict_local_terminal_boundary_changed"
    source = "开发商均面临。不均衡的复苏前景。"
    refined, _, issue = apply_structured_patch(source, patch("面临。不均衡", "面临不均衡", "boundary_punctuation"), policy="strict_local")
    assert issue is None and refined == "开发商均面临不均衡的复苏前景。"


def test_strict_patch_cannot_replace_all_utterance_content_with_punctuation():
    source = "我们呢，将在中山大学讨论安排。"
    whole = source[:-1]
    refined, _, issue = apply_structured_patch(
        source, patch(whole, "", "disfluency"), policy="strict_local")
    assert refined == source
    assert issue == "strict_local_edit_consumes_entire_window"


def test_schema_acceptance_does_not_claim_semantic_correctness():
    source = "他特别特别重视这个问题。"
    _text, _, issue = apply_structured_patch(source, patch("特别特别", "特别"), policy="strict_local")
    assert issue is None  # This is syntactically local, but may delete emphasis.
    assert "特别特别" in strict_local_system_prompt(numeric_rule_mode=True)


def runtime(monkeypatch, responses):
    import system.web_app as web
    from experiments.scripts.collect_gate_pairs import closure_value
    class FakeRefiner:
        def __init__(self, *args, **kwargs):
            self.policy = kwargs.get("window_edit_policy")
            self.calls = []
        def refine(self, text, **kwargs):
            self.calls.append(kwargs)
            return responses[min(len(self.calls) - 1, len(responses) - 1)], 1.0
    instance = FakeRefiner()
    monkeypatch.setattr(web, "TransformersRefiner", lambda *a, **k: (setattr(instance, "policy", k.get("window_edit_policy")) or instance))
    app = web.create_app(Path("/unused"), "cpu", "http://unused", "Chinese", 256, None,
                         window_edit_policy="strict_local", refinement_gate_mode="off")
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/ws/stream")
    refine = closure_value(endpoint, "refine_update")
    protector = web.EntityProtector((), enable_rule_protection=False)
    return refine, protector, instance, app


def test_web_rejects_legacy_edit_without_salvage_or_extra_model_calls(monkeypatch):
    refine, protector, model, app = runtime(monkeypatch, ["我们明天开会。"])
    result = refine("我们我们明天开会。", "Chinese", True, protector, None, single_window=True)
    assert result["clean_text"] == "我们我们明天开会。"
    assert result["refiner_accepted"] is False
    assert len(model.calls) == 1
    assert model.policy == "strict_local"
    assert result["safe_repetition_repairs"] == []
    assert result["local_deletion_reviews"] == []
    assert result["structured_patch_audits"][0]["issue"] == "strict_local_json_required"
    health = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/health")()
    assert health["window_edit_policy"] == "strict_local"


def test_web_applies_valid_patch_and_keeps_existing_numeric_safety(monkeypatch):
    refine, protector, _model, _app = runtime(monkeypatch, [patch("我们我们", "我们")])
    result = refine("我们我们明天开会。", "Chinese", True, protector, None, single_window=True)
    assert result["clean_text"] == "我们明天开会。"
    assert result["refiner_accepted"] is True
    refine, protector, model, _app = runtime(monkeypatch, [patch("2片", "20片", "local_typo"), "服用20片。"])
    result = refine("每天服用2片。", "Chinese", True, protector, None, single_window=True)
    assert result["clean_text"] == "每天服用2片。"
    assert result["refiner_accepted"] is False
    assert len(model.calls) == 2  # Existing numeric-integrity retry, then strict rejection.
    assert result["structured_patch_audits"][-1]["issue"] == "strict_local_json_required"
    assert result["safe_numeric_repairs"] == []


def test_window_prompt_selection_does_not_change_candidate_review_prompt(monkeypatch):
    from system.live_qwen_refiner import TransformersRefiner
    refiner = TransformersRefiner.__new__(TransformersRefiner)
    refiner.window_edit_policy = "strict_local"
    prompts = []
    monkeypatch.setattr(refiner, "_refine_with_system_prompt", lambda text, prompt, **kwargs: (prompts.append(prompt) or ("ok", 0)))
    refiner.refine("我们我们开会。")
    refiner.refine_with_numeric_rules("我们我们开会。")
    assert prompts == [strict_local_system_prompt(numeric_rule_mode=False), strict_local_system_prompt(numeric_rule_mode=True)]
