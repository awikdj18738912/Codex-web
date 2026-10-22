"""Regression coverage for strict upstream semantic QC verdict parsing."""
import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = ROOT / "pipeline"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from src.pipeline.semantic_verdict import PARSER_VERSION, parse_semantic_verdict


def test_parser_accepts_only_valid_boolean_verdicts():
    assert PARSER_VERSION == "strict-json-bool-v2"
    assert parse_semantic_verdict('{"pass":true,"reason":"meaning preserved"}').passed is True
    assert parse_semantic_verdict('{"pass":false,"reason":"detail omitted"}').passed is False
    assert parse_semantic_verdict('{"pass":true,"reason":"  good  "}').reason == "good"
    assert parse_semantic_verdict('{"pass":true,"reason":""}').passed is True
    assert parse_semantic_verdict('{"pass":false,"reason":""}').passed is False


@pytest.mark.parametrize("response", [
    '{"reason":"missing pass"}',
    '{"pass":"false","reason":"string"}',
    '{"pass":"true","reason":"string"}',
    '{"pass":0,"reason":"number"}',
    '{"pass":1,"reason":"number"}',
    '{"pass":null,"reason":"null"}',
    '[{"pass":true,"reason":"array"}]',
    'true',
    '{"pass":true,"reason":"cut"',
    'prefix {"pass":true,"reason":"embedded"}',
    '{"pass":true,"pass":false,"reason":"duplicate"}',
    '{"pass":true}',
    '{"pass":true,"reason":3}',
    '{"pass":true,"reason":"ok","extra":NaN}',
    '{"pass":true,"reason":"ok","extra":Infinity}',
    None,
])
def test_parser_rejects_malformed_or_incomplete_verdicts(response):
    verdict = parse_semantic_verdict(response)
    assert verdict.valid is False
    assert verdict.passed is False


def _load_finalize_module():
    path = ROOT / "pipeline/scripts/05_finalize.py"
    spec = importlib.util.spec_from_file_location("test_finalize_semantic_qc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def _run_qc(response=None, error=None):
    finalize = _load_finalize_module()
    llm = FakeLLM(response, error)
    item = {
        "input": "I need the blue folder tomorrow.",
        "output": {"refined_text": "I need the blue folder tomorrow."},
        "meta": {"generation_plan": {"generation_mode": "upstream-teacher-v1"}},
    }
    result = asyncio.run(finalize.qc_one(item, llm, asyncio.Semaphore(1)))
    return result, llm


def test_finalize_rejects_string_true_and_keeps_raw_evidence():
    raw = '{"pass":"true","reason":"looks fine"}'
    (item, passed, _), llm = _run_qc(raw)
    audit = item["meta"]["quality_audit"]
    assert llm.calls == 1
    assert passed is False
    assert audit["consistent"] is False
    assert audit["verdict_valid"] is False
    assert audit["raw_response"] == raw
    assert audit["parser_version"] == "strict-json-bool-v2"


def test_finalize_rejects_llm_call_exception_and_records_invalid_verdict():
    (item, passed, reason), llm = _run_qc(error=RuntimeError("offline"))
    audit = item["meta"]["quality_audit"]
    assert llm.calls == 1
    assert passed is False
    assert reason == "QC call failed"
    assert audit["consistent"] is False
    assert audit["verdict_valid"] is False
    assert audit["raw_response"] is None


def test_finalize_accepts_well_formed_boolean_true():
    raw = '{"pass":true,"reason":"meaning preserved"}'
    (item, passed, reason), llm = _run_qc(raw)
    audit = item["meta"]["quality_audit"]
    assert llm.calls == 2
    assert passed is True
    assert reason == "meaning preserved"
    assert audit["consistent"] is True
    assert audit["verdict_valid"] is True
    assert audit["raw_response"] == raw


def test_source_rejection_stops_before_target_review():
    raw = '{"pass":false,"reason":"editorial stage direction"}'
    (item, passed, reason), llm = _run_qc(raw)
    assert not passed
    assert llm.calls == 1
    assert reason == "source_quality: editorial stage direction"
    assert item["meta"]["quality_audit"]["decision_stage"] == "source"


def test_target_rejection_cannot_be_overridden_by_valid_source():
    finalize = _load_finalize_module()

    class TwoStageJudge:
        def __init__(self):
            self.requests = []

        async def generate(self, **kwargs):
            self.requests.append(kwargs)
            return ('{"pass":true,"reason":""}' if len(self.requests) == 1
                    else '{"pass":false,"reason":"negation omitted"}')

    llm = TwoStageJudge()
    original = "别买红色。"
    item = {"oral_text": original, "meta": {}}
    reviewed, passed, reason = asyncio.run(finalize._qc_upstream_teacher(
        item, "不要买红色。", "买红色。", llm, asyncio.Semaphore(1),
    ))
    assert len(llm.requests) == 2
    assert llm.requests[0]["user_prompt"] == "source: " + original
    assert not passed and reason == "negation omitted"
    evidence = reviewed["meta"]["quality_audit"]
    assert evidence["source_audit"]["consistent"] is True
    assert evidence["decision_stage"] == "meaning"
