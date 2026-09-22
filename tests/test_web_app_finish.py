from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    import system.web_app as web_app
    from system.entity_store import EntityStore
except ModuleNotFoundError:  # The base data-pipeline environment omits FastAPI.
    TestClient = None
    web_app = None


class _SlowRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        time.sleep(0.1)
        return text.replace("原始", "精修"), 100.0


class _FailingRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        raise RuntimeError("simulated model failure")


class _IdentityRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return text, 1.0


class _StreamingRepetitionRefiner(_IdentityRefiner):
    review_calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).review_calls = []

    def review_repetition(self, text: str) -> tuple[str, float]:
        type(self).review_calls.append(text)
        return text.replace("喂，喂", "喂"), 1.0


class _CountingRefiner:
    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls = []

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        type(self).calls.append(text)
        return text.replace("原始", "精修"), 1.0


class _NumericItnRefiner:
    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls = []

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        type(self).calls.append(text)
        return text.replace("二万二千二百元", "22200元"), 1.0


class _IdiomAndNumericRefiner:
    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls = []

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        type(self).calls.append(text)
        return text.replace("三座", "3座"), 1.0


class _DropsPlaceholderOnceRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        if strict_placeholders:
            return text, 2.0
        return text.replace("__ENTITY_000__", "", 1), 1.0


class _CompressesOnceRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return (text, 2.0) if strict_placeholders else ("你怎么拼？", 1.0)


class _SemanticLossOnceRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        if strict_placeholders:
            return text, 2.0
        return text.replace("傻", "的"), 1.0


class _DropsPunctuationRefiner:
    """Simulate a model that returns the words but omits terminal marks."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return text.rstrip("，,、。！？!?；;：:.").rstrip(), 1.0


class _WrongNumberRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return "你好，我有22000元。", 1.0


class _NumericRepairWithLossRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return "你好，我有22200元。", 1.0


def _fake_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "test-session"}
    if endpoint == "/stream/chunk":
        return {"text": "中间原始文本。", "language": "Chinese"}
    if endpoint == "/stream/finish":
        return {"text": "最终原始文本。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _slow_chunk_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/chunk":
        time.sleep(0.03)
    return _fake_stream_request(asr_url, endpoint, session_id, data, params)


def _stable_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "stable-test-session"}
    if endpoint in {"/stream/chunk", "/stream/finish"}:
        return {"text": "第一段原始文本。第二段原始文本。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


class _GrowingStreamRequest:
    """Return a cumulative transcript that grows with every chunk."""

    TRANSCRIPTS = ("第一句原始文本。", "第一句原始文本。第二句原始文本。")

    def __init__(self) -> None:
        self.chunk_count = 0

    def __call__(
        self,
        asr_url: str,
        endpoint: str,
        session_id: str | None = None,
        data: bytes = b"",
        params: dict[str, str] | None = None,
    ) -> dict[str, object]:
        if endpoint == "/stream/start":
            return {"session_id": "growing-test-session"}
        if endpoint == "/stream/chunk":
            index = min(self.chunk_count, len(self.TRANSCRIPTS) - 1)
            self.chunk_count += 1
            return {"text": self.TRANSCRIPTS[index], "language": "Chinese"}
        if endpoint == "/stream/finish":
            return {"text": self.TRANSCRIPTS[-1], "language": "Chinese"}
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        raise AssertionError(f"unexpected endpoint: {endpoint}")


class _VadEventStreamRequest:
    SEGMENTS = ("甲段原始，", "乙段原始。", "丙段原始。")

    def __init__(self) -> None:
        self.chunk_count = 0

    def __call__(
        self,
        asr_url: str,
        endpoint: str,
        session_id: str | None = None,
        data: bytes = b"",
        params: dict[str, str] | None = None,
    ) -> dict[str, object]:
        if endpoint == "/stream/start":
            return {"session_id": "vad-event-session"}
        if endpoint == "/stream/chunk":
            index = min(self.chunk_count, len(self.SEGMENTS) - 1)
            self.chunk_count += 1
            return {
                "text": "".join(self.SEGMENTS[: index + 1]),
                "language": "Chinese",
                "completed_segments": [
                    {
                        "segment_id": index + 1,
                        "text": self.SEGMENTS[index],
                        "vad_boundary": True,
                    }
                ],
            }
        if endpoint == "/stream/finish":
            return {
                "text": "".join(self.SEGMENTS),
                "language": "Chinese",
                "completed_segments": [],
            }
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def _entity_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "entity-test-session"}
    if endpoint in {"/stream/chunk", "/stream/finish"}:
        return {"text": "鬼灵们是这个副本的入口。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


_LONG_COMPLETE_TRANSCRIPT = (
    "怎么可能？你竟结成了元婴？此青火杖乃墨家的不夜之杖。"
    "你这手段怎么比我还像我道？我记住你了。你认识这位道友？"
    "此人就是我与你说过的那个黄风谷姓韩的，放肆！"
    "韩道友已经是元婴修士，又岂会和你一般见识？"
)


def _content_loss_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "content-loss-test-session"}
    if endpoint == "/stream/finish":
        return {"text": _LONG_COMPLETE_TRANSCRIPT, "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _short_high_confidence_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "gate-test-session"}
    if endpoint == "/stream/finish":
        return {"text": "好。", "language": "Chinese", "confidence": 0.99}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _trusted_high_confidence_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "trusted-confidence-test-session"}
    if endpoint == "/stream/finish":
        return {
            "text": "今天天气很好。",
            "language": "Chinese",
            "confidence": 0.99,
            "confidence_calibrated": True,
            "confidence_covers_full_text": True,
            "confidence_scope": "full_text",
            "confidence_source": "qwen_vllm_token_logprobs",
            "confidence_token_count": 8,
        }
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _untrusted_high_confidence_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    payload = _trusted_high_confidence_stream_request(
        asr_url, endpoint, session_id, data, params
    )
    if endpoint == "/stream/finish":
        payload.pop("confidence_calibrated", None)
        payload.pop("confidence_covers_full_text", None)
        payload["confidence_scope"] = "latest_decode_tokens"
    return payload


def _numeric_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "numeric-test-session"}
    if endpoint == "/stream/finish":
        return {"text": "你好，我有二万二千二百元。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _approximate_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "approximate-number-test-session"}
    if endpoint == "/stream/chunk":
        return {
            "text": "我这一千多公里了。一个人。",
            "language": "Chinese",
            "completed_segments": [
                {
                    "segment_id": 1,
                    "text": "我这一千多公里了。一个人。",
                    "vad_boundary": True,
                }
            ],
        }
    if endpoint == "/stream/finish":
        return {"text": "我这一千多公里了。一个人。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _numeric_loss_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "numeric-loss-test-session"}
    if endpoint == "/stream/finish":
        return {
            "text": "你好，我有二万二千二百元。然后我还有三件东西。",
            "language": "Chinese",
        }
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _idiom_and_numeric_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "idiom-numeric-test-session"}
    if endpoint == "/stream/finish":
        return {
            "text": "此人三番五次欲置我于死地，相当于建起了三座三峡。",
            "language": "Chinese",
        }
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


_MULTI_STAGE_CORRECTION_TEXT = (
    "你好，你好，我有一个苹果，不对，我有一个梨，不对，我有一个香蕉。"
)


class _ResolveCorrectionRefiner:
    """A refiner that resolves the correction chain and applies ITN."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return "你好，我有1个香蕉。", 1.0


def _multi_stage_correction_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "multi-correction-test-session"}
    if endpoint == "/stream/finish":
        return {"text": _MULTI_STAGE_CORRECTION_TEXT, "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


@unittest.skipIf(web_app is None, "FastAPI is not installed")
class WebAppFinishTest(unittest.TestCase):
    def test_local_repetition_review_accepts_only_the_candidate_edit(self) -> None:
        from system.refinement_guard import find_repetition_review_candidates

        source = "模拟来电：‘喂，喂，是本人吗？’"
        candidate = next(
            item
            for item in find_repetition_review_candidates(source)
            if item.source == "喂，喂"
        )
        context = candidate.context
        expected = context.replace(candidate.source, candidate.target, 1)
        decide = web_app._repetition_decision_from_refined_context

        self.assertEqual(decide(source, candidate, context)[0]["action"], "keep")
        self.assertEqual(decide(source, candidate, expected)[0]["action"], "remove")
        self.assertEqual(
            decide(source, candidate, expected + "<KEY>[本人]")[0]["action"],
            "remove",
        )
        self.assertEqual(
            decide(source, candidate, expected.split("？")[0] + "？<KEY>[本人]")[0]["action"],
            "remove",
        )
        longer_source = "模拟普通人接到警察电话：“喂，喂，是郭庆子是吧？我是警察。天哪，"
        longer_candidate = next(
            item for item in find_repetition_review_candidates(longer_source)
            if item.source == "喂，喂"
        )
        self.assertEqual(
            decide(
                longer_source,
                longer_candidate,
                "模拟普通人接到警察电话：“喂，是郭庆子是吧？我是警察。<KEY>[郭庆子]",
            )[0]["action"],
            "remove",
        )
        self.assertEqual(
            decide(
                source,
                candidate,
                '{"action":"replace","source":"喂，喂",'
                '"target":"喂","reason":"repetition"}',
            )[0]["action"],
            "remove",
        )
        self.assertIsNone(decide(source, candidate, expected.replace("本人", "他")))
        self.assertIsNone(decide(source, candidate, "喂"))
        self.assertIsNone(
            decide(source, candidate, "上下文：" + expected + "<br>remove 后片段：喂")
        )

    def test_reviewed_repetition_stays_owned_across_asr_tail_revision(self) -> None:
        from system.refinement_guard import find_repetition_review_candidates
        from system.window_refinement import StreamingRefinementDisplay

        source = "模拟来电：‘喂，喂，是本人吗？"
        candidate = next(
            item for item in find_repetition_review_candidates(source)
            if item.source == "喂，喂"
        )
        corrected = source[:candidate.start] + candidate.target + source[candidate.end:]
        spans = [
            {"source_text": "模拟来电：‘喂，", "clean_text": "模拟来电：‘喂，", "state": "committed"},
            {"source_text": "喂，", "clean_text": "喂，", "state": "committed"},
            {"source_text": "是本人吗？", "clean_text": "是本人吗？", "state": "active"},
        ]
        reviewed = web_app._reviewed_source_spans(
            spans, source, corrected,
            [{**candidate.public_dict(), "decision": "remove", "applied": True}],
        )
        self.assertIsNotNone(reviewed)
        display = StreamingRefinementDisplay()
        display.accept(
            {"raw_text": source, "clean_text": corrected,
             "refinement_source_spans": reviewed}, 1,
        )
        revised = display.compose("模拟来电：‘喂，喂，是本人吧？")
        self.assertEqual(revised["display_refined_text"], "模拟来电：‘喂，")
        self.assertEqual(revised["pending_raw_text"], "是本人吧？")

    def test_truncated_review_publishes_repair_before_finish(self) -> None:
        class _TruncatedReviewer(_IdentityRefiner):
            def review_repetition(self, context: str) -> tuple[str, float]:
                return context.replace("喂，喂", "喂").split("天哪")[0] + "<KEY>[郭庆子]", 1.0

        class _RevisedTailASR:
            def __init__(self) -> None:
                self.chunks = 0

            def __call__(self, asr_url, endpoint, session_id=None, data=b"", params=None):
                if endpoint == "/stream/start":
                    return {"session_id": "truncated-review-session"}
                if endpoint == "/stream/chunk":
                    self.chunks += 1
                    ending = "天哪，" if self.chunks == 1 else "天哪！"
                    return {
                        "text": "模拟普通人接到警察电话：“喂，喂，是郭庆子是吧？我是警察。" + ending,
                        "language": "Chinese",
                    }
                if endpoint == "/stream/cancel":
                    return {"cancelled": True}
                raise AssertionError(endpoint)

        with (
            patch.object(web_app, "TransformersRefiner", _TruncatedReviewer),
            patch.object(web_app, "_stream_request", _RevisedTailASR()),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"), "cpu", "http://fake-asr",
                "Chinese", 32, None, refinement_gate_mode="tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=streaming") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    update = self._await_event(websocket, "update")
                    self.assertNotIn("喂，喂", update["display_refined_text"])
                    self.assertTrue(any(
                        review["source"] == "喂，喂" and review["applied"] is True
                        for review in update["repetition_reviews"]
                    ))
                    websocket.send_bytes(b"pcm")
                    revised = self._await_event(websocket, "transcript")
                    self.assertNotIn("喂，喂", revised["display_refined_text"])

    def test_model_omitted_approximation_is_fixed_in_streaming_update(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _IdentityRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(web_app, "_stream_request", _approximate_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"), "cpu", "http://fake-asr",
                "Chinese", 32, None, refinement_gate_mode="tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=streaming") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    update = self._await_event(websocket, "update")
                    self.assertEqual(
                        update["clean_text"], "我这1000多公里了。一个人。"
                    )

    def test_model_omitted_approximation_gets_numeric_fallback(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _IdentityRefiner),
            patch.object(web_app, "_stream_request", _approximate_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertEqual(final["clean_text"], "我这1000多公里了。一个人。")
        self.assertTrue(
            any(
                change["original"] == "一千多"
                and change["replacement"] == "1000多"
                and change["reason"] == "model_omitted_approximate_number"
                for change in final["numeric_fallbacks"]
            )
        )

    def test_numeric_text_reaches_refiner_for_model_driven_itn(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _NumericItnRefiner),
            patch.object(web_app, "_stream_request", _numeric_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertIn("你好，我有二万二千二百元。", _NumericItnRefiner.calls)
        self.assertTrue(
            all("__ENTITY_" not in value for value in _NumericItnRefiner.calls)
        )
        self.assertEqual(final["clean_text"], "你好，我有22200元。")
        self.assertEqual(final["protected_entities"], [])

    def test_wrong_model_number_falls_back_to_original_asr_value(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _WrongNumberRefiner),
            patch.object(web_app, "_stream_request", _numeric_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertEqual(final["clean_text"], "你好，我有22200元。")
        self.assertFalse(final["numeric_normalization_enabled"])
        self.assertEqual(final["numeric_normalizations"], [])
        self.assertTrue(
            any(
                change["original"] == "二万二千二百元"
                and change["replacement"] == "22200元"
                for change in final["numeric_fallbacks"]
            )
        )
        self.assertTrue(
            any(
                "numeric_value_mismatch" in reason
                for reason in final["refiner_reject_reasons"]
            )
        )

    def test_safe_number_is_kept_when_same_window_has_content_loss(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _NumericRepairWithLossRefiner),
            patch.object(web_app, "_stream_request", _numeric_loss_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertEqual(
            final["clean_text"],
            "你好，我有22200元。然后我还有3件东西。",
        )
        self.assertFalse(final["refiner_accepted"])

    def test_model_numeric_itn_keeps_fixed_idiom_verbatim(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _IdiomAndNumericRefiner),
            patch.object(
                web_app,
                "_stream_request",
                _idiom_and_numeric_stream_request,
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertTrue(
            any("__ENTITY_000__" in value for value in _IdiomAndNumericRefiner.calls)
        )
        self.assertEqual(
            final["clean_text"],
            "此人三番五次欲置我于死地，相当于建起了3座三峡。",
        )

    def test_conservative_gate_refines_short_segment(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(
                web_app, "_stream_request", _short_high_confidence_stream_request
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "conservative",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

            self.assertEqual(final["clean_text"], "好。")
            self.assertTrue(final["refiner_executed"])
            self.assertEqual(final["refinement_gate_skipped_segments"], 0)
            self.assertEqual(final["refinement_gate_decisions"][0]["action"], "refine")
            self.assertEqual(_CountingRefiner.calls, ["好。"])
            self.assertEqual(final["session_refiner_call_count"], 1)
            self.assertEqual(
                final["refiner_session_stats"]["gate_skipped_segment_count"], 0
            )
            self.assertTrue(final["refiner_session_stats"]["stats_complete"])

    def test_conservative_gate_uses_only_calibrated_full_text_confidence(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(
                web_app, "_stream_request", _trusted_high_confidence_stream_request
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "conservative",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    transcript = websocket.receive_json()
                    final = websocket.receive_json()

        self.assertEqual(transcript["asr_confidence_metadata"]["scope"], "full_text")
        self.assertFalse(final["refiner_executed"])
        self.assertEqual(final["session_refiner_call_count"], 0)
        self.assertEqual(
            final["refinement_gate_decisions"][0]["reasons"],
            ["high_confidence_clean_segment"],
        )

    def test_untrusted_high_confidence_fails_open_to_refiner(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(
                web_app,
                "_stream_request",
                _untrusted_high_confidence_stream_request,
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "conservative",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    websocket.receive_json()
                    final = websocket.receive_json()

        self.assertTrue(final["refiner_executed"])
        self.assertEqual(final["session_refiner_call_count"], 1)
        self.assertEqual(
            final["refinement_gate_decisions"][0]["reasons"],
            ["confidence_uncalibrated"],
        )

    def test_off_gate_preserves_original_refiner_path(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(
                web_app, "_stream_request", _short_high_confidence_stream_request
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "off",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

            self.assertTrue(final["refiner_executed"])
            self.assertEqual(final["refinement_gate_skipped_segments"], 0)
            self.assertEqual(final["refinement_gate_decisions"][0]["action"], "refine")
            self.assertEqual(len(_CountingRefiner.calls), 1)
            self.assertEqual(final["session_refiner_call_count"], 1)
            self.assertEqual(final["session_refiner_initial_call_count"], 1)
            self.assertEqual(final["session_refiner_retry_call_count"], 0)

    @staticmethod
    def _await_event(websocket, event: str) -> dict:
        for _ in range(60):
            message = websocket.receive_json()
            if message.get("event") == event:
                return message
        raise AssertionError(f"event {event!r} was not received")

    def test_finished_pass_is_published_even_when_asr_advanced(self) -> None:
        """A completed pass must not be thrown away as stale.

        Enqueueing a newer hypothesis used to bump the revision of the pass
        that was already running, so on a fast stream every intermediate
        result was dropped and the refined pane stayed empty until ``finish``.
        """
        with (
            patch.object(web_app, "TransformersRefiner", _SlowRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(web_app, "_stream_request", _GrowingStreamRequest()),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    # The second hypothesis arrives while the first pass runs.
                    websocket.send_bytes(b"pcm")
                    websocket.send_bytes(b"pcm")

                    first_transcript = self._await_event(websocket, "transcript")
                    self.assertEqual(first_transcript["display_refined_text"], "")
                    self.assertEqual(
                        first_transcript["pending_raw_text"], "第一句原始文本。"
                    )
                    second_transcript = self._await_event(websocket, "transcript")
                    self.assertEqual(
                        second_transcript["pending_raw_text"],
                        "第一句原始文本。第二句原始文本。",
                    )
                    first = self._await_event(websocket, "update")
                    self.assertEqual(first["clean_text"], "第一句精修文本。")
                    self.assertEqual(
                        first["display_refined_text"], "第一句精修文本。"
                    )
                    self.assertEqual(first["pending_raw_text"], "第二句原始文本。")
                    self.assertEqual(
                        first["display_text"],
                        "第一句精修文本。第二句原始文本。",
                    )
                    second = self._await_event(websocket, "update")
                    self.assertEqual(
                        second["clean_text"],
                        "第一句精修文本。第二句精修文本。",
                    )
                    self.assertEqual(second["pending_raw_text"], "")
                    self.assertFalse(second["has_pending_refinement"])

    def test_vad_events_preserve_clean_three_segment_gate_skip(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(web_app, "_stream_request", _VadEventStreamRequest()),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    result = None
                    for _ in range(3):
                        websocket.send_bytes(b"pcm")
                        result = self._await_event(websocket, "update")

        self.assertIsNotNone(result)
        self.assertFalse(result["refiner_executed"])
        self.assertEqual(_CountingRefiner.calls, [])
        self.assertEqual(
            result["refinement_gate_decisions"][0]["reasons"],
            ["stable_clean_segment"],
        )

    def test_punctuated_repetition_is_reviewed_during_streaming_vad_window(self) -> None:
        class _PunctuatedRepetitionStreamRequest:
            def __call__(
                self,
                asr_url: str,
                endpoint: str,
                session_id: str | None = None,
                data: bytes = b"",
                params: dict[str, str] | None = None,
            ) -> dict[str, object]:
                if endpoint == "/stream/start":
                    return {"session_id": "streaming-repetition-session"}
                if endpoint == "/stream/chunk":
                    return {
                        "text": "模拟普通人接到警察电话：‘喂，喂，是郭庆子是吧？’",
                        "language": "Chinese",
                        "completed_segments": [
                            {
                                "segment_id": 1,
                                "text": "模拟普通人接到警察电话：‘喂，喂，是郭庆子是吧？’",
                                "vad_boundary": True,
                            }
                        ],
                    }
                if endpoint == "/stream/finish":
                    return {
                        "text": "模拟普通人接到警察电话：‘喂，喂，是郭庆子是吧？’",
                        "language": "Chinese",
                        "completed_segments": [],
                    }
                if endpoint == "/stream/cancel":
                    return {"cancelled": True}
                raise AssertionError(f"unexpected endpoint: {endpoint}")

        with (
            patch.object(web_app, "TransformersRefiner", _StreamingRepetitionRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(
                web_app,
                "_stream_request",
                _PunctuatedRepetitionStreamRequest(),
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    update = self._await_event(websocket, "update")
                    self.assertEqual(
                        update["clean_text"],
                        "模拟普通人接到警察电话：‘喂，是郭庆子是吧？’",
                    )
                    self.assertTrue(
                        any(
                            item["source"] == "喂，喂"
                            and item["applied"] is True
                            and "喂，喂" not in item["model_response"]
                            for item in update["repetition_reviews"]
                        )
                    )

                    websocket.send_json({"event": "finish"})
                    self._await_event(websocket, "transcript")
                    final = self._await_event(websocket, "final")
                    self.assertEqual(
                        final["clean_text"],
                        "模拟普通人接到警察电话：‘喂，是郭庆子是吧？’",
                    )

        self.assertEqual(len(_StreamingRepetitionRefiner.review_calls), 1)
        self.assertNotIn("候选规则", _StreamingRepetitionRefiner.review_calls[0])

    def test_invalid_repetition_review_keeps_original_model_response(self) -> None:
        class _InvalidReviewRefiner(_IdentityRefiner):
            def review_repetition(self, prompt: str) -> tuple[str, float]:
                return "无法确定，请保留原文。", 1.0

        def stream_request(asr_url, endpoint, session_id=None, data=b"", params=None):
            if endpoint == "/stream/start":
                return {"session_id": "review-audit-session"}
            if endpoint in {"/stream/chunk", "/stream/finish"}:
                return {
                    "text": "模拟普通人接到警察电话：‘喂，喂，是郭庆子是吧？’",
                    "language": "Chinese",
                }
            if endpoint == "/stream/cancel":
                return {"cancelled": True}
            raise AssertionError(f"unexpected endpoint: {endpoint}")

        with (
            patch.object(web_app, "TransformersRefiner", _InvalidReviewRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(web_app, "_stream_request", stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"), "cpu", "http://fake-asr",
                "Chinese", 32, None, refinement_gate_mode="tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=streaming") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    update = self._await_event(websocket, "update")
                    self.assertIn("喂，喂", update["clean_text"])
                    review = next(
                        item for item in update["repetition_reviews"]
                        if item["source"] == "喂，喂"
                    )
                    self.assertEqual(review["reason"], "invalid_response")
                    self.assertEqual(review["model_response"], "无法确定，请保留原文。")

    def test_punctuated_repetition_is_reviewed_without_chunk_vad_event(self) -> None:
        """Streaming review must not wait for a backend-specific VAD event."""

        class _NoVadBoundaryStreamRequest:
            # Keep at least the review-context width after the candidate so
            # the second, longer hypothesis takes the cache-hit path.
            text = (
                "模拟普通人接到警察电话：‘喂，喂，是郭庆子是吧？"
                "我是警察。天哪，喂，我是所的警察。"
            )

            def __init__(self) -> None:
                self.chunk_count = 0

            def __call__(
                self,
                asr_url: str,
                endpoint: str,
                session_id: str | None = None,
                data: bytes = b"",
                params: dict[str, str] | None = None,
            ) -> dict[str, object]:
                if endpoint == "/stream/start":
                    return {"session_id": "streaming-repetition-no-vad-session"}
                if endpoint == "/stream/chunk":
                    self.chunk_count += 1
                    suffix = "" if self.chunk_count == 1 else "请尽快配合。"
                    return {"text": self.text + suffix, "language": "Chinese"}
                if endpoint == "/stream/finish":
                    return {
                        "text": self.text + "请尽快配合。",
                        "language": "Chinese",
                    }
                if endpoint == "/stream/cancel":
                    return {"cancelled": True}
                raise AssertionError(f"unexpected endpoint: {endpoint}")

        with (
            patch.object(web_app, "TransformersRefiner", _StreamingRepetitionRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(web_app, "_stream_request", _NoVadBoundaryStreamRequest()),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                None,
                "shadow",
                "tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    update = self._await_event(websocket, "update")
                    self.assertEqual(
                        update["clean_text"],
                        "模拟普通人接到警察电话：‘喂，是郭庆子是吧？"
                        "我是警察。天哪，喂，我是所的警察。",
                    )
                    self.assertTrue(
                        any(
                            item["source"] == "喂，喂"
                            and item["applied"] is True
                            for item in update["repetition_reviews"]
                        )
                    )
                    self.assertNotIn("喂，喂", update["display_refined_text"])

                    # The second pass reuses the earlier local-review
                    # decision.  It must retain the source-span edit instead
                    # of repainting the cached raw ASR span in the browser.
                    websocket.send_bytes(b"pcm")
                    cached_update = self._await_event(websocket, "update")
                    self.assertIn("请尽快配合", cached_update["clean_text"])
                    self.assertNotIn(
                        "喂，喂", cached_update["display_refined_text"]
                    )

        self.assertEqual(len(_StreamingRepetitionRefiner.review_calls), 1)

    def test_intermediate_refinement_interval_limits_repeated_passes(self) -> None:
        """Hypotheses arriving inside the interval collapse into one pass."""
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 1.0),
            patch.object(web_app, "_stream_request", _GrowingStreamRequest()),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    websocket.send_bytes(b"pcm")
                    websocket.send_bytes(b"pcm")

                    self._await_event(websocket, "update")
                    # The first pass runs immediately, the two later hypotheses
                    # are absorbed while the interval is still pending.
                    self.assertEqual(len(_CountingRefiner.calls), 1)

                    websocket.send_json({"event": "finish"})
                    final = self._await_event(websocket, "final")
                    self.assertEqual(
                        final["clean_text"],
                        "第一句精修文本。第二句精修文本。",
                    )

    def test_offline_mode_reuses_completed_window_refinement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=offline"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    message = websocket.receive_json()
                    while message["event"] != "update":
                        message = websocket.receive_json()
                    calls_before_finish = len(_CountingRefiner.calls)
                    self.assertGreater(calls_before_finish, 0)

                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

                    self.assertEqual(final["event"], "final")
                    self.assertEqual(
                        final["clean_text"],
                        "第一段精修文本。第二段精修文本。",
                    )
                    self.assertEqual(len(_CountingRefiner.calls), calls_before_finish)
                    self.assertEqual(
                        final["session_refiner_call_count"], calls_before_finish
                    )
                    self.assertEqual(
                        final["session_refiner_intermediate_call_count"],
                        calls_before_finish,
                    )
                    self.assertEqual(final["session_refiner_final_call_count"], 0)

    def test_online_final_uses_sentence_bounded_window(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=online"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

            self.assertEqual(final["event"], "final")
            self.assertEqual(final["window_size"], 3)
            self.assertEqual(final["window_max_chars"], 80)
            self.assertTrue(all(len(call) <= 80 for call in _CountingRefiner.calls))

    def test_tri_state_uses_one_punctuation_window(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                refinement_gate_mode="tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertEqual(final["window_size"], 3)
        punctuation_counts = [
            sum(call.count(mark) for mark in "，,、。！？!?；;：:.")
            for call in _CountingRefiner.calls
        ]
        self.assertTrue(all(count <= 3 for count in punctuation_counts))
        self.assertTrue(any(count >= 2 for count in punctuation_counts))

    def test_tri_state_waits_for_three_punctuation_chunks_before_model(self) -> None:
        class _ThreeChunkASR:
            def __init__(self) -> None:
                self.chunks = 0

            def __call__(self, asr_url, endpoint, session_id=None, data=b"", params=None):
                if endpoint == "/stream/start":
                    return {"session_id": "fixed-three-session"}
                if endpoint == "/stream/chunk":
                    self.chunks += 1
                    return {
                        "text": "甲，甲，" if self.chunks == 1 else "甲，甲，乙。",
                        "language": "Chinese",
                    }
                if endpoint == "/stream/finish":
                    return {"text": "甲，甲，乙。", "language": "Chinese"}
                if endpoint == "/stream/cancel":
                    return {"cancelled": True}
                raise AssertionError(endpoint)

        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _ThreeChunkASR()),
            patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0.0),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"), "cpu", "http://fake-asr",
                "Chinese", 32, None, refinement_gate_mode="tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=streaming") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    first = self._await_event(websocket, "update")
                    self.assertEqual(first["committed_chunks"], 0)
                    self.assertEqual(_CountingRefiner.calls, [])

                    websocket.send_bytes(b"pcm")
                    second = self._await_event(websocket, "update")
                    self.assertEqual(second["committed_chunks"], 3)
                    self.assertEqual(_CountingRefiner.calls, ["甲，甲，乙。"])

    def test_tri_state_allows_refiner_to_replace_source_punctuation(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _DropsPunctuationRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
                refinement_gate_mode="tri_state",
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=online") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

        self.assertEqual(final["clean_text"], "第一段原始文本。第二段原始文本")

    def test_streaming_window_applies_fuzzy_entity_before_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity_db = Path(directory) / "entities.db"
            EntityStore(entity_db).upsert_entity(
                "鬼灵门",
                entity_type="TERM",
                normalization_policy="normalize",
            )
            with (
                patch.object(web_app, "TransformersRefiner", _IdentityRefiner),
                patch.object(web_app, "_stream_request", _entity_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"),
                    "cpu",
                    "http://fake-asr",
                    "Chinese",
                    32,
                    None,
                    entity_db,
                    "auto",
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?mode=streaming&domain=general"
                    ) as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_bytes(b"pcm")
                        message = websocket.receive_json()
                        while message["event"] != "update":
                            message = websocket.receive_json()
                        self.assertEqual(
                            message["clean_text"], "鬼灵门是这个副本的入口。"
                        )
                        self.assertEqual(
                            message["entity_candidates"][0]["decision"],
                            "AUTO_NORMALIZE",
                        )

                        websocket.send_json({"event": "finish"})
                        self.assertEqual(websocket.receive_json()["event"], "transcript")
                        final = websocket.receive_json()

                        self.assertEqual(final["clean_text"], "鬼灵门是这个副本的入口。")
                        self.assertEqual(
                            final["entity_candidates"][0]["decision"],
                            "AUTO_NORMALIZE",
                        )

    def test_stream_trace_links_model_window_review_and_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "session.jsonl"
            with (
                patch.object(web_app, "TransformersRefiner", _CountingRefiner),
                patch.object(web_app, "_stream_request", _stable_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"), "cpu", "http://fake-asr",
                    "Chinese", 32, output,
                )
                with TestClient(app) as client:
                    with client.websocket_connect("/ws/stream?mode=streaming") as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_bytes(b"pcm")
                        update = self._await_event(websocket, "update")
                        websocket.send_json({"event": "finish"})
                        final = self._await_event(websocket, "final")

            trace_path = output.with_name("session.stream_trace.jsonl")
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(len({event["trace_id"] for event in events}), 1)
            self.assertTrue(any(event["event"] == "asr_hypothesis" for event in events))
            windows = [event for event in events if event["event"] == "window_result"]
            self.assertTrue(windows)
            self.assertTrue(any(
                segment["model_calls"]
                and segment["model_calls"][0]["input"]
                and "output" in segment["model_calls"][0]
                for event in windows for segment in event["segments"]
            ))
            review = next(event for event in events if event["event"] == "stream_review")
            published = next(event for event in events if event["event"] == "pass_published")
            self.assertEqual(review["revision"], update["refinement_revision"])
            self.assertEqual(published["revision"], update["refinement_revision"])
            self.assertEqual(published["output"], update["clean_text"])
            self.assertEqual(
                next(event for event in events if event["event"] == "final_result")["output"],
                final["clean_text"],
            )

    def test_streaming_final_reuses_completed_window_refinement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    message = websocket.receive_json()
                    while message["event"] != "update":
                        message = websocket.receive_json()
                    calls_before_finish = len(_CountingRefiner.calls)
                    self.assertGreater(calls_before_finish, 0)

                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

                    self.assertEqual(final["event"], "final")
                    self.assertEqual(
                        final["clean_text"],
                        "第一段精修文本。第二段精修文本。",
                    )
                    self.assertEqual(len(_CountingRefiner.calls), calls_before_finish)

    def test_severe_content_loss_retries_with_strict_preservation(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CompressesOnceRefiner),
            patch.object(web_app, "_stream_request", _content_loss_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=offline") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(
                        websocket.receive_json()["raw_text"],
                        _LONG_COMPLETE_TRANSCRIPT,
                    )
                    final = websocket.receive_json()
                    self.assertEqual(final["clean_text"], _LONG_COMPLETE_TRANSCRIPT)
                    self.assertTrue(final["refiner_accepted"])
                    # Sentence-bounded finalization may retry each affected
                    # short window independently instead of two large blocks.
                    self.assertGreaterEqual(final["refiner_retry_count"], 2)
                    self.assertEqual(final["placeholder_retry_count"], 0)
                    self.assertIn(
                        "segment_1:severe_content_loss",
                        final["refiner_retry_reasons"],
                    )
                    self.assertGreaterEqual(
                        final["refiner_retry_reasons"].count(
                            "segment_1:severe_content_loss"
                        ),
                        2,
                    )

    def test_multi_stage_self_correction_is_accepted(self) -> None:
        with (
            patch.object(
                web_app, "TransformersRefiner", _ResolveCorrectionRefiner
            ),
            patch.object(
                web_app,
                "_stream_request",
                _multi_stage_correction_stream_request,
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=offline") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(
                        websocket.receive_json()["raw_text"],
                        _MULTI_STAGE_CORRECTION_TEXT,
                    )
                    final = websocket.receive_json()

        self.assertTrue(final["refiner_accepted"], final["refiner_reject_reasons"])
        self.assertEqual(final["refiner_reject_reasons"], [])
        self.assertEqual(final["refiner_masked_outputs"], ["你好，我有1个香蕉。"])
        self.assertEqual(final["clean_text"], "你好，我有1个香蕉。")

    def test_semantic_loss_retries_and_keeps_source_when_needed(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _SemanticLossOnceRefiner),
            patch.object(
                web_app,
                "_stream_request",
                lambda asr_url, endpoint, session_id=None, data=b"", params=None: (
                    {"session_id": "semantic-loss-test-session"}
                    if endpoint == "/stream/start"
                    else (
                        {"text": "你当我傻？", "language": "Chinese"}
                        if endpoint == "/stream/finish"
                        else {"cancelled": True}
                    )
                ),
            ),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=offline") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    websocket.receive_json()
                    final = websocket.receive_json()

        self.assertEqual(final["clean_text"], "你当我傻？")
        self.assertTrue(final["refiner_accepted"])
        self.assertEqual(final["refiner_retry_count"], 1)
        self.assertIn("segment_1:semantic_substitution", final["refiner_retry_reasons"])

    def test_placeholder_failure_retries_once_with_strict_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity_db = Path(directory) / "entities.db"
            EntityStore(entity_db).upsert_entity(
                "鬼灵门",
                entity_type="TERM",
                normalization_policy="normalize",
            )
            with (
                patch.object(
                    web_app, "TransformersRefiner", _DropsPlaceholderOnceRefiner
                ),
                patch.object(web_app, "_stream_request", _entity_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"),
                    "cpu",
                    "http://fake-asr",
                    "Chinese",
                    32,
                    None,
                    entity_db,
                    "auto",
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?mode=offline&domain=general"
                    ) as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_json({"event": "finish"})
                        self.assertEqual(
                            websocket.receive_json()["raw_text"],
                            "鬼灵们是这个副本的入口。",
                        )
                        final = websocket.receive_json()
                        self.assertEqual(final["clean_text"], "鬼灵门是这个副本的入口。")
                        self.assertTrue(final["refiner_accepted"])
                        self.assertEqual(final["placeholder_retry_count"], 1)
                        self.assertEqual(final["refiner_retry_count"], 1)
                        self.assertEqual(len(final["refiner_masked_outputs"]), 2)
                        self.assertEqual(final["session_refiner_call_count"], 2)
                        self.assertEqual(final["session_refiner_initial_call_count"], 1)
                        self.assertEqual(final["session_refiner_retry_call_count"], 1)

    def test_final_auto_mode_applies_and_restores_fuzzy_entity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity_db = Path(directory) / "entities.db"
            EntityStore(entity_db).upsert_entity(
                "鬼灵门",
                entity_type="TERM",
                normalization_policy="normalize",
            )
            with (
                patch.object(web_app, "TransformersRefiner", _IdentityRefiner),
                patch.object(web_app, "_stream_request", _entity_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"),
                    "cpu",
                    "http://fake-asr",
                    "Chinese",
                    32,
                    None,
                    entity_db,
                    "auto",
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?mode=offline&domain=general"
                    ) as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_json({"event": "finish"})
                        self.assertEqual(
                            websocket.receive_json()["raw_text"],
                            "鬼灵们是这个副本的入口。",
                        )
                        final = websocket.receive_json()
                        self.assertEqual(final["clean_text"], "鬼灵门是这个副本的入口。")
                        self.assertEqual(
                            final["entity_candidates"][0]["decision"],
                            "AUTO_NORMALIZE",
                        )
                        self.assertEqual(
                            final["protected_entities"][0]["source"],
                            "fuzzy:general",
                        )

    def test_slow_chunk_reports_progress_before_acknowledgement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _SlowRefiner),
            patch.object(web_app, "_stream_request", _slow_chunk_stream_request),
            patch.object(web_app, "ASR_CHUNK_STATUS_INTERVAL_SECONDS", 0.01),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    progress = websocket.receive_json()
                    self.assertEqual(progress["event"], "status")
                    self.assertEqual(progress["stage"], "asr_chunk")
                    self.assertEqual(progress["chunk_index"], 1)
                    acknowledgement = websocket.receive_json()
                    while acknowledgement["event"] == "status":
                        acknowledgement = websocket.receive_json()
                    self.assertEqual(acknowledgement["event"], "chunk_ack")
                    self.assertEqual(acknowledgement["chunk_index"], 1)

    def test_final_raw_transcript_is_sent_before_slow_refinement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _SlowRefiner),
            patch.object(web_app, "_stream_request", _fake_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )

            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    self.assertEqual(websocket.receive_json()['event'], 'chunk_ack')
                    self.assertEqual(
                        websocket.receive_json()["event"], "transcript"
                    )

                    websocket.send_json({"event": "finish"})
                    pending = websocket.receive_json()
                    intermediate_events = []
                    while pending["event"] != "transcript":
                        intermediate_events.append(pending["event"])
                        pending = websocket.receive_json()
                    self.assertIn("update", intermediate_events)
                    self.assertEqual(pending["event"], "transcript")
                    self.assertEqual(pending["raw_text"], "最终原始文本。")
                    self.assertTrue(pending["refiner_deferred"])

                    final = websocket.receive_json()
                    self.assertEqual(final["event"], "final")
                    self.assertEqual(final["clean_text"], "最终精修文本。")

    def test_refiner_failure_falls_back_to_complete_raw_text(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _FailingRefiner),
            patch.object(web_app, "_stream_request", _fake_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=offline") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    pending = websocket.receive_json()
                    self.assertEqual(pending["raw_text"], "最终原始文本。")
                    final = websocket.receive_json()
                    self.assertEqual(final["event"], "final")
                    self.assertEqual(final["clean_text"], "最终原始文本。")
                    self.assertFalse(final["refiner_accepted"])
                    self.assertIn(
                        "final_refinement_error:RuntimeError",
                        final["refiner_reject_reasons"],
                    )
                    stats = final["refiner_session_stats"]
                    self.assertEqual(stats["call_count"], 1)
                    self.assertEqual(stats["completed_call_count"], 1)
                    self.assertEqual(stats["failed_call_count"], 1)
                    self.assertEqual(stats["inflight_call_count"], 0)
                    self.assertTrue(stats["stats_complete"])


if __name__ == "__main__":
    unittest.main()
