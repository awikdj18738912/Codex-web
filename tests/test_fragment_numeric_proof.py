"""Construction provenance permits only exact no-noise ASR prefix slices."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
sys.modules.setdefault("openai", types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object))

from src.utils.numeric_policy import prepare_fragment_training_pair
from src.utils.synthetic_oral import build_oral
from src.generators.asr_simulator import _align_observed_prefix


def _plan(intent):
    return {
        "construction_version": "synthetic-oral-v1",
        "raw_intent": intent,
        "expected_clean": intent,
        "construction_phenomena": ["oral:low"],
        "correction_values": [],
        "spelling_parts": [],
        "slot_policy": "single-slot-v1",
    }


def _pair(intent, target):
    plan = _plan(intent)
    oral = build_oral(intent, [], [], plan["construction_phenomena"])
    source = _align_observed_prefix(oral, target, plan, intent)
    assert source is not None
    return prepare_fragment_training_pair(source, target, oral, intent, plan, observed_prefix=source)


def test_six_no_noise_number_and_english_fragments_use_construction_proof():
    examples = [
        ("预算是890,000元，明天汇款。", "预算是890,000元，"),
        ("使用Nuxt.js搭建页面，然后部署。", "使用Nuxt.js搭建页面，"),
        ("设备型号是HX6730，今晚测试。", "设备型号是HX6730，"),
        ("他说Kris Wu负责设计，明天交稿。", "他说Kris Wu负责设计，"),
        ("模型达到了85%，继续观察。", "模型达到了85%，"),
        ("用MacBook Air M3剪辑视频，然后导出。", "用MacBook Air M3剪辑视频，"),
    ]
    for intent, target in examples:
        result = _pair(intent, target)
        assert result is not None, (intent, target)
        assert result[1] == target
        assert result[2]["fragment_training_proof"]["version"] == "construction-prefix-v1"


def test_source_number_or_english_edits_do_not_use_construction_exception():
    intent = "预算是890,000元，明天汇款。"
    target = "预算是890,000元，"
    plan = _plan(intent)
    oral = build_oral(intent, [], [], plan["construction_phenomena"])
    aligned = _align_observed_prefix(oral, target, plan, intent)
    assert aligned is not None
    assert prepare_fragment_training_pair(aligned.replace("890", "980"), target, oral, intent, plan,
                                         observed_prefix=aligned) is None

    english_intent = "使用Nuxt.js搭建页面，然后部署。"
    english_target = "使用Nuxt.js搭建页面，"
    english_plan = _plan(english_intent)
    english_oral = build_oral(english_intent, [], [], english_plan["construction_phenomena"])
    english_source = _align_observed_prefix(english_oral, english_target, english_plan, english_intent)
    assert english_source is not None
    assert prepare_fragment_training_pair(
        english_source.replace("Nuxt", "nuxt"), english_target,
        english_oral, english_intent, english_plan,
        observed_prefix=english_source,
    ) is None


def test_forged_plan_nonprefix_and_future_target_are_rejected():
    intent = "预算是890,000元，明天汇款。"
    plan = _plan(intent)
    oral = build_oral(intent, [], [], plan["construction_phenomena"])
    target = "预算是890,000元，"
    source = _align_observed_prefix(oral, target, plan, intent)
    assert source is not None

    forged = dict(plan, expected_clean="预算是980,000元，明天汇款。")
    assert prepare_fragment_training_pair(source, target, oral, intent, forged, observed_prefix=source) is None
    assert prepare_fragment_training_pair(source, "明天汇款。", oral, intent, plan, observed_prefix=source) is None
    assert prepare_fragment_training_pair(source, intent, oral, intent, plan, observed_prefix=source) is None


def test_ordinary_chinese_asr_omission_keeps_legacy_numeric_checks():
    full_oral = "嗯，请尽快导出日报，然后发给财务组。"
    full_clean = "请尽快导出日报，然后发给财务组。"
    source = "嗯，请尽快导出日，"
    target = "请尽快导出日报，"
    pair = prepare_fragment_training_pair(source, target, full_oral, full_clean, None)
    assert pair is not None
    assert "fragment_training_proof" not in pair[2]

    numeric_source = "嗯，预算是890,000元，"
    numeric_target = "预算是980,000元，"
    numeric_full_oral = "嗯，预算是890,000元，然后明天汇款。"
    numeric_full_clean = "预算是890,000元，然后明天汇款。"
    assert prepare_fragment_training_pair(
        numeric_source, numeric_target, numeric_full_oral, numeric_full_clean, None,
    ) is None


def test_final_candidate_alignment_cannot_stitch_earlier_candidates():
    intent = "我想用React管理路由，然后发布。"
    values = ["React管理状态", "Angular管理路由", "React管理路由"]
    plan = {
        "construction_version": "synthetic-oral-v1", "raw_intent": intent,
        "expected_clean": intent, "construction_phenomena": ["correction:multiple", "oral:low"],
        "correction_values": values, "spelling_parts": [], "slot_policy": "single-slot-v1",
    }
    oral = build_oral(intent, values, [], plan["construction_phenomena"])
    target = "我想用React管理路由，"
    legacy_alignment = _align_observed_prefix(oral, target)
    safe_alignment = _align_observed_prefix(oral, target, plan, intent)
    assert legacy_alignment is not None and "Angular管理路由" in legacy_alignment
    assert safe_alignment is not None and "最后改成" in safe_alignment
    assert safe_alignment.endswith("React管理路由")
    assert safe_alignment != legacy_alignment
    pair = prepare_fragment_training_pair(
        safe_alignment, target, oral, intent, plan, observed_prefix=safe_alignment,
    )
    assert pair is not None
    assert pair[2]["fragment_training_proof"]["source_alignment"] == "exact_normalized_observed_prefix"
    assert prepare_fragment_training_pair(
        safe_alignment, target, oral, intent, plan, observed_prefix=legacy_alignment,
    ) is None

    import asyncio
    from src.generators.asr_simulator import ASRSimulator
    class FixedClient:
        async def generate(self, **kwargs):
            prompt = kwargs["user_prompt"]
            assert safe_alignment in prompt
            assert "然后发布" not in prompt
            return __import__("json").dumps({"input": safe_alignment, "target": target}, ensure_ascii=False)
    class PickTarget:
        def choice(self, choices): return target
    simulator = ASRSimulator(async_client=FixedClient(), mode="llm_sim", simulation_ratio=1, rng=PickTarget())
    generated = asyncio.run(simulator._simulate_with_llm(
        oral, intent, plan, return_observed_prefix=True,
    ))
    assert generated == (safe_alignment, target, safe_alignment)


def test_prefix_before_correction_slot_stops_before_all_candidates():
    intent = "先说前缀，再使用React管理路由，然后发布。"
    values = ["Vue管理状态", "Angular管理路由", "React管理路由"]
    plan = {
        "construction_version": "synthetic-oral-v1", "raw_intent": intent,
        "expected_clean": intent, "construction_phenomena": ["correction:multiple", "oral:low"],
        "correction_values": values, "spelling_parts": [], "slot_policy": "single-slot-v1",
    }
    oral = build_oral(intent, values, [], plan["construction_phenomena"])
    target = "先说前缀，"
    safe_alignment = _align_observed_prefix(oral, target, plan, intent)
    assert safe_alignment == "嗯，先说前缀"
    assert "Vue管理状态" not in safe_alignment and "Angular管理路由" not in safe_alignment


def test_broken_marker_punctuation_fails_closed_and_uses_full_pair():
    intent = "我想用React管理路由，然后发布。"
    values = ["React管理状态", "Angular管理路由", "React管理路由"]
    plan = {
        "construction_version": "synthetic-oral-v1", "raw_intent": intent,
        "expected_clean": intent, "construction_phenomena": ["correction:multiple", "oral:low"],
        "correction_values": values, "spelling_parts": [], "slot_policy": "single-slot-v1",
    }
    oral = build_oral(intent, values, [], plan["construction_phenomena"])
    target = "我想用React管理路由，"
    broken = oral.replace("最后改成", "最后。改成")
    assert _align_observed_prefix(broken, target, plan, intent) is None

    import asyncio
    from src.generators.asr_simulator import ASRSimulator
    class NeverCalled:
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            raise AssertionError("unsafe fragment should not be sent to the teacher")
    class FirstChoice:
        def random(self): return 0
        def choice(self, values): return values[0]
    client = NeverCalled()
    simulator = ASRSimulator(async_client=client, mode="llm_sim", simulation_ratio=1, rng=FirstChoice())
    pair = asyncio.run(simulator._generate_pair(broken, intent, generation_plan=plan))
    assert pair == (broken, intent, False, False, None)
    assert client.calls == 0
