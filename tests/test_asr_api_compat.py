from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from system.web_app import ASRApiStyle, _stream_request


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class ASRApiCompatibilityTest(unittest.TestCase):
    def _capture_request(self, *args, response=None, **kwargs):
        captured: list[object] = []

        def fake_urlopen(request, timeout):
            captured.extend([request, timeout])
            return _Response(response or {"text": "ok"})

        with patch("system.web_app.urllib.request.urlopen", fake_urlopen):
            payload = _stream_request(*args, **kwargs)
        return payload, captured[0], captured[1]

    def test_current_style_preserves_query_parameter_contract(self) -> None:
        payload, request, _ = self._capture_request(
            "http://127.0.0.1:8766",
            "/stream/start",
            params={"language": "Chinese"},
            response={"session_id": "current-session"},
        )

        self.assertEqual(payload["session_id"], "current-session")
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8766/stream/start?language=Chinese",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.data, b"")

    def test_legacy_start_uses_json_body(self) -> None:
        payload, request, _ = self._capture_request(
            "http://127.0.0.1:8766",
            "/stream/start",
            params={"language": "Chinese"},
            api_style="legacy_v1",
            response={"session_id": "legacy-session"},
        )

        self.assertEqual(payload["session_id"], "legacy-session")
        self.assertEqual(
            request.full_url, "http://127.0.0.1:8766/v1/stream/start"
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"language": "Chinese"})
        self.assertEqual(request.get_header("Content-type"), "application/json")

    def test_legacy_chunk_puts_session_in_path(self) -> None:
        _, request, _ = self._capture_request(
            "http://127.0.0.1:8766",
            "/stream/chunk",
            "session/with/slash",
            b"pcm",
            api_style=ASRApiStyle.LEGACY_V1,
        )

        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8766/v1/stream/session%2Fwith%2Fslash/chunk",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.data, b"pcm")

    def test_legacy_finish_and_cancel_use_original_methods(self) -> None:
        _, finish_request, _ = self._capture_request(
            "http://127.0.0.1:8766",
            "/stream/finish",
            "abc123",
            api_style="legacy_v1",
        )
        _, cancel_request, _ = self._capture_request(
            "http://127.0.0.1:8766",
            "/stream/cancel",
            "abc123",
            api_style="legacy_v1",
        )

        self.assertEqual(
            finish_request.full_url,
            "http://127.0.0.1:8766/v1/stream/abc123/finish",
        )
        self.assertEqual(finish_request.get_method(), "POST")
        self.assertEqual(finish_request.data, b"{}")
        self.assertEqual(
            cancel_request.full_url,
            "http://127.0.0.1:8766/v1/stream/abc123",
        )
        self.assertEqual(cancel_request.get_method(), "DELETE")
        self.assertIsNone(cancel_request.data)

    def test_unknown_api_style_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported ASR API style"):
            ASRApiStyle.parse("unknown")


if __name__ == "__main__":
    unittest.main()
