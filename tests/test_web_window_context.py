import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_web_app_finish import TestClient, web_app
from tests.test_window_adjacent_context import FIRST, SECOND


class SourceASR:
    def __init__(self):
        self.count = 0

    def __call__(self, url, endpoint, *args, **kwargs):
        if endpoint == "/stream/start":
            return {"session_id": "window-context"}
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        if endpoint == "/stream/finish":
            return {"text": FIRST + SECOND, "language": "Chinese"}
        self.count += 1
        text = FIRST if self.count == 1 else FIRST + "是人类的发现，"
        return {"text": text, "language": "Chinese", "completed_segments": [
            {"segment_id": 1, "text": FIRST, "vad_boundary": True},
            *([] if self.count == 1 else [
                {"segment_id": 2, "text": "是人类的发现，", "vad_boundary": True}
            ]),
        ]}


class ContextRefiner:
    supports_window_context = True
    calls = []
    response_mode = "normal"

    def __init__(self, *args, **kwargs):
        type(self).calls = []

    def refine(self, text, *, window_context, **kwargs):
        type(self).calls.append((text, window_context))
        if text == FIRST:
            if self.response_mode == "delete_subject":
                return json.dumps(dict(action="replace", source="正义。", target="", reason="repetition"), ensure_ascii=False), 1
            if self.response_mode == "copy_context":
                return text + window_context["right_context"], 1
            return text.replace("就这个", "这个"), 1
        return text, 1


@unittest.skipIf(web_app is None, "FastAPI is not installed")
class WebWindowContextTest(unittest.TestCase):
    @staticmethod
    def receive(ws, kind):
        for _ in range(100):
            payload = ws.receive_json()
            if payload["event"] == "error":
                raise AssertionError(payload)
            if payload["event"] == kind:
                return payload
        raise AssertionError(kind)

    def run_window(self, mode):
        ContextRefiner.response_mode = mode
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "session.jsonl"
            with patch.object(web_app, "TransformersRefiner", ContextRefiner), \
                 patch.object(web_app, "_stream_request", SourceASR()), \
                 patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0):
                app = web_app.create_app(Path("/tmp/model"), "cpu", "http://fake-asr", "Chinese", 256,
                                         output, refinement_gate_mode="off")
                with TestClient(app) as client, client.websocket_connect("/ws/stream?mode=streaming") as ws:
                    self.receive(ws, "ready")
                    ws.send_bytes(b"pcm")
                    pending = self.receive(ws, "update")
                    self.assertEqual(ContextRefiner.calls, [])
                    self.assertEqual(pending["clean_text"], FIRST)
                    ws.send_bytes(b"pcm")
                    update = self.receive(ws, "update")
                    calls_before_finish = list(ContextRefiner.calls)
                    ws.send_json({"event": "finish"})
                    final = self.receive(ws, "final")
                records = [json.loads(line) for line in output.with_suffix(".stream_trace.jsonl").read_text().splitlines()]
        return update, final, calls_before_finish, records

    def test_context_flows_to_model_and_trace_without_duplicate_output(self):
        update, final, calls, records = self.run_window("normal")
        self.assertEqual(calls, [(FIRST, {"left_context": "", "right_context": "是人类的发现，"})])
        self.assertEqual(ContextRefiner.calls[-1], (SECOND, {"left_context": "正义。", "right_context": ""}))
        self.assertEqual(final["clean_text"], (FIRST + SECOND).replace("就这个", "这个"))
        self.assertEqual(final["clean_text"].count("是人类的发现"), 1)
        self.assertEqual(final["window_cache_reused_group_count"], 1)
        called = [segment for record in records for segment in record.get("segments", []) if segment.get("model_calls")]
        self.assertEqual(called[0]["window_context"]["right_context"], "是人类的发现，")
        self.assertEqual(called[0]["model_calls"][0]["window_context"], calls[0][1])

    def test_subject_deletion_patch_falls_back_to_original(self):
        update, final, _, _ = self.run_window("delete_subject")
        self.assertEqual(final["clean_text"], FIRST + SECOND)
        self.assertIn("segment_1:window_boundary_subject_loss", update["refiner_reject_reasons"])

    def test_copied_read_only_context_is_rejected(self):
        update, final, _, _ = self.run_window("copy_context")
        self.assertEqual(final["clean_text"], FIRST + SECOND)
        self.assertIn("segment_1:window_context_copied", update["refiner_reject_reasons"])


if __name__ == "__main__":
    unittest.main()
