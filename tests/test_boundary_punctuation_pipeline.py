import asyncio
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_assemble():
    spec = importlib.util.spec_from_file_location(
        "assemble_pipeline_module", ROOT / "pipeline/scripts/04_assemble.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedRng:
    def random(self):
        return 0.0

    def choice(self, values):
        return values[0]

    def uniform(self, low, high):
        return low


class StubSampler:
    def sample(self, scene, count):
        return ["惊讶", "二", "三", "四", "五"]


class StubClient:
    async def generate(self, **kwargs):
        return json.dumps({
            "oral_text": "真。让我感到惊讶。",
            "selected_seeds": ["惊讶"],
            "speech_phenomena": ["clean"],
        }, ensure_ascii=False)


@pytest.fixture
def isolated_oral_module(monkeypatch):
    """Import Oral generator with a temporary OpenAI stub, then restore import caches."""
    prefixes = ("src", "configs", "pipeline.src")
    relevant = lambda name: any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    before_modules = {name: module for name, module in sys.modules.items() if relevant(name)}
    parent_modules = dict(before_modules)
    if "pipeline" in sys.modules:
        parent_modules["pipeline"] = sys.modules["pipeline"]
    before_attrs = {
        (name, attr): value
        for name, module in parent_modules.items()
        for attr, value in vars(module).items()
    }

    if "openai" not in sys.modules:
        try:
            openai_spec = importlib.util.find_spec("openai")
        except (ImportError, ValueError):
            openai_spec = None
        if openai_spec is None:
            monkeypatch.setitem(
                sys.modules,
                "openai",
                types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object),
            )

    oral = importlib.import_module("pipeline.src.generators.oral_text_generator")
    try:
        yield oral
    finally:
        current_names = [name for name in sys.modules if relevant(name)]
        created_names = [name for name in current_names if name not in before_modules]
        for name in created_names:
            parent, _, attr = name.rpartition(".")
            if parent in parent_modules:
                parent_module = parent_modules[parent]
                if (parent, attr) not in before_attrs:
                    vars(parent_module).pop(attr, None)
                else:
                    prior = before_attrs[(parent, attr)]
                    setattr(parent_module, attr, prior)
            sys.modules.pop(name, None)
        for name, module in before_modules.items():
            sys.modules[name] = module
        if "pipeline" not in parent_modules:
            sys.modules.pop("pipeline", None)


def test_boundary_ratio_controls_chinese_and_mixed_but_not_passthrough_or_english(monkeypatch, isolated_oral_module):
    oral = isolated_oral_module

    monkeypatch.setattr(oral, "PASSTHROUGH_RATIO", 0)
    monkeypatch.setattr(oral, "BOUNDARY_PUNCTUATION_RATIO", 1)
    generator = oral.OralTextGenerator(async_client=StubClient(), rng=FixedRng())
    _, _, additional_zh, phenomena_zh = generator._controls("daily_chat")
    _, _, additional_mix, phenomena_mix = generator._controls("vibe_coding")
    _, _, additional_en, phenomena_en = generator._controls("english_daily")
    assert oral.BOUNDARY_PUNCTUATION_CONTROL in additional_zh
    assert "boundary_punctuation" in phenomena_zh
    assert "boundary_punctuation" in phenomena_mix
    assert "boundary_punctuation" not in phenomena_en
    assert oral.BOUNDARY_PUNCTUATION_CONTROL not in additional_en

    monkeypatch.setattr(oral, "BOUNDARY_PUNCTUATION_RATIO", 0)
    assert "boundary_punctuation" not in generator._controls("daily_chat")[3]

    monkeypatch.setattr(oral, "PASSTHROUGH_RATIO", 1)
    assert generator._controls("daily_chat")[3] == ["clean"]


def test_boundary_control_stacks_with_explanation_and_model_cannot_drop_it(monkeypatch, isolated_oral_module):
    oral = isolated_oral_module

    monkeypatch.setattr(oral, "PASSTHROUGH_RATIO", 0)
    monkeypatch.setattr(oral, "BOUNDARY_PUNCTUATION_RATIO", 1)
    generator = oral.OralTextGenerator(async_client=StubClient(), rng=FixedRng())
    generator.sampler = StubSampler()
    _, _, additional, phenomena = generator._controls("explanation")
    assert "uncommon or ambiguous entity" in additional
    assert oral.BOUNDARY_PUNCTUATION_CONTROL in additional
    result = asyncio.run(generator.generate_one("daily_chat", 1))
    assert result is not None
    assert "boundary_punctuation" in result["speech_phenomena"]
    assert "clean" not in result["speech_phenomena"]


@pytest.mark.parametrize("source", ["clean", "asr"])
@pytest.mark.parametrize("label_location", ["top_level", "meta"])
def test_short_boundary_example_survives_assembly_and_sft_target(source, label_location):
    assemble = _load_assemble()
    raw = {
        "oral_text": "真。让我感到惊讶。",
        "clean_text": "真让我感到惊讶。",
        "id": 7,
    }
    if label_location == "top_level":
        raw["speech_phenomena"] = ["boundary_punctuation"]
    else:
        raw["meta"] = {"speech_phenomena": ["boundary_punctuation"]}
    if source == "asr":
        raw.update({"input": "真。让我感到惊讶。", "output": {"refined_text": "真让我感到惊讶。"}})
    assembled, stats = (
        assemble._assemble_from_asr("daily_chat", [raw])
        if source == "asr"
        else assemble._assemble_from_clean("daily_chat", [raw])
    )
    assert stats["skipped_short"] == 0
    assert assembled[0]["output"]["refined_text"] == "真让我感到惊讶。"
    assert "boundary_punctuation" in assembled[0]["meta"]["speech_phenomena"]
    from pipeline.scripts.export_sft import convert_one

    converted = convert_one(assembled[0])
    assert converted["conversations"][-1]["value"] == "真让我感到惊讶。"


def test_clean_prompt_has_error_and_independent_sentence_counterexamples():
    prompts = json.loads((ROOT / "pipeline/configs/prompts_clean.json").read_text(encoding="utf-8"))
    zh = prompts["system_prompts"]["zh"] + prompts["user_template"]["zh"]
    for phrase in (
        "是。让我有点意外。",
        "是让我有点意外。",
        "这确实；让我有点意外。",
        "这确实让我有点意外。",
        "我到了。你在哪里？",
        "真好。让我看看。",
    ):
        assert phrase in zh


def test_chinese_short_target_is_eligible_without_a_special_boundary_label():
    assemble = _load_assemble()
    raw = {"oral_text": "真。让我感到惊讶。", "clean_text": "真让我感到惊讶。"}
    assembled, stats = assemble._assemble_from_clean("daily_chat", [raw])
    assert len(assembled) == 1
    assert stats["skipped_short"] == 0


def test_short_clean_sentence_counterexample_survives_assembly_and_sft():
    assemble = _load_assemble()
    raw = {
        "oral_text": "真好。让我看看。",
        "clean_text": "真好。让我看看。",
        "speech_phenomena": ["clean"],
    }
    assembled, stats = assemble._assemble_from_clean("daily_chat", [raw])
    assert stats["skipped_short"] == 0
    from pipeline.scripts.export_sft import convert_one

    converted = convert_one(assembled[0])
    assert converted["conversations"][-1]["value"] == "真好。让我看看。"


def test_english_clean_short_target_keeps_english_minimum():
    assemble = _load_assemble()
    raw = {
        "oral_text": "Yes.",
        "clean_text": "Yes.",
        "speech_phenomena": ["clean"],
    }
    assembled, stats = assemble._assemble_from_clean("english_daily", [raw])
    assert assembled == []
    assert stats["skipped_short"] == 1
