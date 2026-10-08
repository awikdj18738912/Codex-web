from __future__ import annotations

import unittest
import tempfile
import types
from unittest.mock import Mock, patch

from system.qwen_asr_stream_server import (
    _join_transcripts,
    _scope_confidence_payload,
    _should_rotate_segment,
    _vad_finalize_reason,
    main,
)


class QwenStreamRequestValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Exercise the HTTP handlers without loading GPU model weights.
        self.backend = Mock()
        module = types.ModuleType("qwen_asr")
        module.Qwen3ASRModel = types.SimpleNamespace(LLM=Mock(return_value=self.backend))
        apps = []
        with tempfile.TemporaryDirectory() as model_dir:
            with patch.dict("sys.modules", {"qwen_asr": module}), patch(
                "system.qwen_asr_stream_server.Flask.run",
                lambda app, **kwargs: apps.append(app),
            ):
                main(["--model", model_dir, "--vad", "off", "--confidence-logprobs", "0"])
        self.client = apps[0].test_client()

    def test_legacy_rejects_nonfinite_and_invalid_chunk_sizes_before_model_call(self) -> None:
        for value in ("nan", "inf", "-inf", float("nan"), float("inf"),
                      10**400, 0, -1, None, "invalid"):
            with self.subTest(value=value):
                response = self.client.post("/v1/stream/start", json={"chunk_seconds": value})
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)
        self.backend.init_streaming_state.assert_not_called()

    def test_legacy_rejects_nonfinite_integer_configuration(self) -> None:
        for field in ("unfixed_chunk_num", "unfixed_token_num"):
            for value in (float("inf"), float("nan"), -1, None):
                with self.subTest(field=field, value=value):
                    response = self.client.post("/v1/stream/start", json={field: value})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.json)
        self.backend.init_streaming_state.assert_not_called()

    def test_both_protocols_still_accept_finite_chunk_sizes(self) -> None:
        responses = (
            self.client.post("/stream/start?chunk_seconds=0.5"),
            self.client.post("/v1/stream/start", json={"chunk_seconds": 0.5}),
        )
        for response in responses:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["chunk_seconds"], 0.5)
            self.assertTrue(response.json["session_id"])


class QwenStreamSegmentationTest(unittest.TestCase):
    def test_prefers_sentence_boundary_after_soft_limit(self) -> None:
        self.assertFalse(_should_rotate_segment("还在说", 30, 30, 45))
        self.assertTrue(_should_rotate_segment("这句说完了。", 30, 30, 45))

    def test_hard_limit_rotates_without_punctuation(self) -> None:
        self.assertTrue(_should_rotate_segment("还在说", 45, 30, 45))

    def test_vad_endpoint_finishes_a_segment_before_duration_limits(self) -> None:
        self.assertEqual(
            _vad_finalize_reason(
                vad_ended=True,
                text="还在说",
                sample_count=10,
                soft_limit_samples=30,
                hard_limit_samples=45,
            ),
            "vad",
        )

    def test_vad_mode_uses_soft_limit_only_at_sentence_boundary(self) -> None:
        self.assertIsNone(
            _vad_finalize_reason(
                vad_ended=False,
                text="还在说",
                sample_count=30,
                soft_limit_samples=30,
                hard_limit_samples=45,
            )
        )
        self.assertEqual(
            _vad_finalize_reason(
                vad_ended=False,
                text="这句说完了。",
                sample_count=30,
                soft_limit_samples=30,
                hard_limit_samples=45,
            ),
            "soft_limit_sentence_end",
        )

    def test_vad_mode_forces_a_cut_at_the_existing_hard_limit(self) -> None:
        self.assertEqual(
            _vad_finalize_reason(
                vad_ended=False,
                text="连续说话没有句末标点",
                sample_count=45,
                soft_limit_samples=30,
                hard_limit_samples=45,
            ),
            "max_duration",
        )

    def test_transcript_joining_preserves_chinese_and_english_spacing(self) -> None:
        self.assertEqual(_join_transcripts("第一段。", "第二段。"), "第一段。第二段。")
        self.assertEqual(_join_transcripts("hello", "world"), "hello world")


    def test_full_state_score_covers_full_text_without_committed_prefix(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": 0.97,
                "confidence_token_count": 8,
                "confidence_covers_full_state": True,
            },
            has_committed_prefix=False,
        )

        self.assertTrue(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "full_text")

    def test_full_state_score_does_not_cover_prior_committed_text(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": 0.97,
                "confidence_token_count": 8,
                "confidence_covers_full_state": True,
            },
            has_committed_prefix=True,
        )

        self.assertFalse(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "current_state")

    def test_prefix_decode_is_reported_as_generated_suffix_only(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": 0.97,
                "confidence_token_count": 8,
                "confidence_covers_full_state": False,
            },
            has_committed_prefix=False,
        )

        self.assertFalse(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "generated_suffix")

    def test_missing_score_has_unavailable_scope(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": None,
                "confidence_token_count": 0,
                "confidence_covers_full_state": False,
            },
            has_committed_prefix=False,
        )

        self.assertFalse(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "unavailable")


if __name__ == "__main__":
    unittest.main()
