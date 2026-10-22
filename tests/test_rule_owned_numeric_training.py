"""Offline regression tests for rule-owned numeric training data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE_LIBRARY = ROOT / "zh-itn" / "zh-itn" / "build" / "libzh_itn_bridge.so"
NATIVE_REQUIRED = pytest.mark.skipif(
    not NATIVE_LIBRARY.is_file(), reason="native zh-itn bridge is not built"
)


def _run_isolated(source: str) -> None:
    """Run imports in a clean interpreter so dependency stubs cannot leak to pytest."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), str(ROOT / "pipeline"), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated check failed:\n{result.stdout}\n{result.stderr}"


@NATIVE_REQUIRED
def test_native_normalization_and_target_restore_use_the_rule_surface():
    _run_isolated(
        """
from src.utils.numeric_policy import normalize_source, prepare_training_pair
source = '嗯明天下午三点开会。'
normalized, meta = normalize_source(source)
assert normalized == '嗯明天下午3点开会。'
assert meta['itn_owner'] == 'rules'
pair = prepare_training_pair(source, source)
assert pair is not None
assert pair[:2] == (normalized, normalized)
"""
    )


@NATIVE_REQUIRED
def test_leading_zero_identifier_and_decimal_version_are_preserved():
    _run_isolated(
        """
from src.utils.numeric_policy import normalize_source, prepare_training_pair
source = '编号0012，版本1.10。'
assert normalize_source(source)[0] == source
pair = prepare_training_pair(source, source)
assert pair is not None and pair[:2] == (source, source)
"""
    )


@NATIVE_REQUIRED
def test_invented_numeric_value_is_rejected():
    _run_isolated(
        """
from src.utils.numeric_policy import prepare_training_pair
assert prepare_training_pair('有3个苹果。', '有4个苹果。') is None
"""
    )


@NATIVE_REQUIRED
def test_explicit_numeric_self_correction_keeps_only_final_value():
    _run_isolated(
        """
from src.utils.numeric_policy import prepare_training_pair
pair = prepare_training_pair('明天3点开会，不对，4点开会。', '明天4点开会。')
assert pair is not None
assert pair[1] == '明天4点开会。'
"""
    )


def test_non_numeric_erroneous_sentence_boundary_remains_eligible():
    _run_isolated(
        """
from src.utils.numeric_policy import prepare_training_pair
pair = prepare_training_pair('真。让我感到惊讶。', '真让我感到惊讶。')
assert pair is not None
assert pair[:2] == ('真。让我感到惊讶。', '真让我感到惊讶。')
"""
    )


def test_sft_system_prompt_is_plain_text_and_does_not_teach_itn():
    _run_isolated(
        """
from src.utils.numeric_policy import TRAINING_SYSTEM_PROMPT
assert '只输出精修后的正文' in TRAINING_SYSTEM_PROMPT
assert '数字格式由外部规则库负责' in TRAINING_SYSTEM_PROMPT
for forbidden in ('请规范化数字', '执行 ITN', '转换数字格式'):
    assert forbidden not in TRAINING_SYSTEM_PROMPT
"""
    )


def test_english_scene_and_english_language_are_rejected_by_sft_export():
    _run_isolated(
        """
import sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from pipeline.scripts.export_sft import convert_one
base = {'input': 'A sentence.', 'output': {'refined_text': 'A sentence.'}}
assert convert_one({**base, 'meta': {'scene': 'english_daily', 'language': 'en'}}) is None
assert convert_one({**base, 'meta': {'scene': 'daily_chat', 'language': 'en'}}) is None
"""
    )


@NATIVE_REQUIRED
def test_sft_export_normalizes_old_itn_target_instead_of_teaching_conversion():
    _run_isolated(
        """
import sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from pipeline.scripts.export_sft import convert_one, SYSTEM_PROMPT
item = {
    'input': '明天下午三点开会。',
    'output': {'refined_text': '明天下午三点开会。'},
    'meta': {'scene': 'daily_chat', 'language': 'zh'},
}
converted = convert_one(item)
assert converted is not None
human, target = converted['conversations'][1:]
assert human['value'] == '明天下午3点开会。'
assert target['value'] == '明天下午3点开会。'
assert '数字格式由外部规则库负责' in SYSTEM_PROMPT
"""
    )


@NATIVE_REQUIRED
def test_clean_generator_sends_rule_normalized_input_and_forces_empty_itn_types():
    _run_isolated(
        """
import asyncio, json, sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
import src.generators.clean_text_generator as clean_module
clean_module._QC_ENABLED = False
CleanTextGenerator = clean_module.CleanTextGenerator
class FakeClient:
    def __init__(self): self.calls = []
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({'clean_text': '嗯明天下午3点开会。', 'itn_types': ['number']}, ensure_ascii=False)
client = FakeClient()
generator = CleanTextGenerator(async_client=client)
result = asyncio.run(generator.generate_one('daily_chat', {
    'id': 1, 'oral_text': '嗯明天下午三点开会。', 'speech_phenomena': ['filler']
}))
assert result is not None
assert '嗯明天下午3点开会。' in client.calls[0]['user_prompt']
assert 'Numeric formatting is handled by external rules' in client.calls[0]['system_prompt']
assert result['clean_text'] == '嗯明天下午3点开会。'
assert result['itn_types'] == []
"""
    )


@NATIVE_REQUIRED
def test_clean_generator_defers_teacher_numeric_drift_to_semantic_qc():
    _run_isolated(
        """
import asyncio, json, sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
import src.generators.clean_text_generator as clean_module
clean_module._QC_ENABLED = False
CleanTextGenerator = clean_module.CleanTextGenerator
class FakeClient:
    async def generate(self, **kwargs):
        return json.dumps({'clean_text': '嗯明天下午4点开会。'}, ensure_ascii=False)
generator = CleanTextGenerator(async_client=FakeClient())
result = asyncio.run(generator.generate_one('daily_chat', {
    'id': 2, 'oral_text': '嗯明天下午三点开会。', 'speech_phenomena': ['filler']
}))
assert result is not None
assert result['clean_text'] == '嗯明天下午4点开会。'
assert result['meta']['generation_plan']['expected_clean'] == '嗯明天下午4点开会。'
"""
    )


@NATIVE_REQUIRED
def test_asr_simulator_rejects_numeric_drift_from_fake_teacher():
    _run_isolated(
        """
import asyncio, json, random, sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from src.generators.asr_simulator import ASRSimulator
class FakeClient:
    def __init__(self): self.calls = 0
    async def generate(self, **kwargs):
        self.calls += 1
        return json.dumps({'input': '明天下午3点开会。', 'target': '明天下午4点开会。'}, ensure_ascii=False)
client = FakeClient()
simulator = ASRSimulator(async_client=client, mode='llm_sim', simulation_ratio=1, rng=random.Random(0))
result = asyncio.run(simulator.simulate_one('daily_chat', {
    'id': 3,
    'oral_text': '明天下午三点开会，然后去吃饭。',
    'clean_text': '明天下午3点开会，然后去吃饭。',
    'speech_phenomena': [],
}))
assert result and result['meta']['is_fragment'] is False
assert result['output']['refined_text'] == '明天下午3点开会，然后去吃饭。'
assert client.calls == 0
"""
    )


@NATIVE_REQUIRED
def test_unsupported_numeric_context_cannot_become_itn_training():
    from pipeline.src.utils.numeric_policy import prepare_training_pair

    # Native rules conservatively leave this unit-free phrase unchanged.
    # Even if the runtime value parser ignores it, the teacher must not turn
    # its Chinese numerals into Arabic conversion labels.
    assert prepare_training_pair("按一到三的范围处理。", "按1到3的范围处理。") is None
    pair = prepare_training_pair("按一到三的范围处理。", "按一到三的范围处理。")
    assert pair is not None and pair[0] == pair[1]


def test_chinese_scene_defaults_are_complete_and_reject_english():
    _run_isolated(
        """
from configs.settings import SCENE_DISTRIBUTION, RAW_DIR, SEEDS_DIR
from configs.scenes import SCENE_REGISTRY, get_enabled_scenes, validate_enabled_scene
enabled = get_enabled_scenes()
assert len(enabled) == 9
assert abs(sum(SCENE_DISTRIBUTION.values()) - 1.0) < 1e-12
assert all(SCENE_REGISTRY[s]['language'] in {'zh', 'zh-en-mix'} for s in enabled)
try:
    validate_enabled_scene('english_daily')
except ValueError:
    pass
else:
    raise AssertionError('English-only scene was still enabled')
import sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from src.generators.oral_text_generator import OralTextGenerator
generator = OralTextGenerator(async_client=object())
assert str(generator.seeds_dir) == SEEDS_DIR
assert str(generator.raw_dir) == RAW_DIR
"""
    )

