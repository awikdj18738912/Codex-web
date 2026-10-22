"""Regression coverage for preserving English entities in Chinese training pairs."""

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

sys.path.insert(0, str(ROOT / "pipeline"))

from src.utils.english_entity_policy import (  # noqa: E402
    english_entities_preserved,
    english_entity_spans,
)
from src.utils.numeric_policy import (  # noqa: E402
    lock_training_target,
    prepare_training_pair,
)


def test_entity_spans_preserve_exact_multiword_and_identifier_forms():
    source = "用FastAPI、RTX 3090、GPT-4o、Visual Studio Code、user_id和Qwen3-0.6B训练"
    assert [value for _, _, value in english_entity_spans(source)] == [
        "FastAPI",
        "RTX 3090",
        "GPT-4o",
        "Visual Studio Code",
        "user_id",
        "Qwen3-0.6B",
    ]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("嗯我想用那个PyTorch来训练", "我想用PyTorch来训练。"),
        ("这个FastAPI接口，接口有点慢", "这个FastAPI接口有点慢。"),
        (
            "用FastAPI、RTX 3090、GPT-4o、Visual Studio Code、user_id",
            "用FastAPI、RTX 3090、GPT-4o、Visual Studio Code、user_id。",
        ),
        ("用Qwen3-0.6B跑一下", "用Qwen3-0.6B跑一下。"),
    ],
)
def test_chinese_cleanup_can_preserve_english_entities_verbatim(source, target):
    assert english_entities_preserved(source, target)
    assert lock_training_target(source, target) == target


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("用PyTorch训练", "用火炬训练。"),
        ("用PyTorch训练", "用Pytorch训练。"),
        ("嗯我想用那个PyTorch来训练", "我想来训练。"),
        ("用FastAPI部署", "用Flask部署。"),
        ("用RTX 3090测试", "用RTX 4090测试。"),
        ("用RTX 3090测试", "用RTX3090测试。"),
        ("用模型训练", "用PyTorch训练。"),
        ("先PyTorch再FastAPI", "先FastAPI再PyTorch。"),
        # A meaningful emphasized English-word repeat is not a whole repeated sentence.
        ("我说no no", "我说no。"),
    ],
)
def test_unjustified_entity_translation_loss_drift_or_reordering_is_rejected(source, target):
    assert not english_entities_preserved(source, target)
    assert lock_training_target(source, target) is None


def test_explicit_self_correction_can_drop_abandoned_entity():
    source = "用TensorFlow，不对，改用PyTorch"
    target = "用PyTorch。"
    assert english_entities_preserved(source, target)
    assert lock_training_target(source, target) == target


def test_exact_repeated_complete_sentence_can_be_deduplicated():
    source = "用FastAPI部署。用FastAPI部署。"
    target = "用FastAPI部署。"
    assert english_entities_preserved(source, target)
    assert lock_training_target(source, target) == target


@NATIVE_REQUIRED
def test_prepare_training_pair_normalizes_numbers_and_preserves_entities():
    source = "嗯明天下午三点用PyTorch训练"
    target = "明天下午3点用PyTorch训练。"
    pair = prepare_training_pair(source, target)
    assert pair is not None
    normalized_source, locked_target, metadata = pair
    assert normalized_source == "嗯明天下午3点用PyTorch训练"
    assert locked_target == target
    assert metadata["itn_owner"] == "rules"
    assert metadata["english_entity_policy"] == "verbatim-v1"


@NATIVE_REQUIRED
def test_prepare_training_pair_rejects_entity_loss_and_numeric_drift():
    assert prepare_training_pair("明天下午三点用PyTorch训练", "明天下午4点训练") is None
    assert prepare_training_pair("明天下午三点用PyTorch训练", "明天下午3点训练") is None


@NATIVE_REQUIRED
def test_sft_export_accepts_preserved_mixed_entities_and_rejects_drift_in_isolated_process():
    code = r'''
import sys, types
sys.modules["openai"] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from pipeline.scripts.export_sft import convert_one

def item(source, target):
    return {"input": source, "output": {"refined_text": target},
            "meta": {"scene": "daily_chat", "language": "zh"}}

accepted = convert_one(item("嗯我想用那个PyTorch来训练", "我想用PyTorch来训练。"))
assert accepted is not None
assert accepted["conversations"][1]["value"] == "嗯我想用那个PyTorch来训练"
assert accepted["conversations"][2]["value"] == "我想用PyTorch来训练。"
assert convert_one(item("用RTX 3090测试", "用RTX 4090测试。")) is None
assert convert_one(item("用FastAPI部署", "用Flask部署。")) is None
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), str(ROOT / "pipeline"), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated SFT check failed:\n{result.stdout}\n{result.stderr}"


def test_clean_generator_defers_teacher_entity_drift_to_semantic_qc():
    code = r'''
import asyncio, json, sys, types
sys.modules["openai"] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
import src.generators.clean_text_generator as clean_module
clean_module._QC_ENABLED = False

class FakeTeacher:
    def __init__(self, target): self.target = target
    async def generate(self, **kwargs):
        return json.dumps({"clean_text": self.target}, ensure_ascii=False)

async def check(target):
    generator = clean_module.CleanTextGenerator(async_client=FakeTeacher(target))
    return await generator.generate_one("daily_chat", {
        "id": 1, "oral_text": "嗯我想用那个PyTorch来训练", "speech_phenomena": ["filler"]
    })

translated = asyncio.run(check("我想用火炬来训练。"))
omitted = asyncio.run(check("我想来训练。"))
assert translated is not None and translated["clean_text"] == "我想用火炬来训练。"
assert omitted is not None and omitted["clean_text"] == "我想来训练。"
assert translated["meta"]["generation_plan"]["expected_clean"] == "我想用火炬来训练。"
accepted = asyncio.run(check("我想用PyTorch来训练。"))
assert accepted is not None and accepted["clean_text"] == "我想用PyTorch来训练。"
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), str(ROOT / "pipeline"), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated Clean check failed:\n{result.stdout}\n{result.stderr}"
