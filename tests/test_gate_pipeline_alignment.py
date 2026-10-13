"""Cross-policy source ownership, cache invalidation and retry regressions."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_web_app_finish import TestClient, web_app, _IdentityRefiner


FIRST = ("嗯，", "请查阅附件，", "会议现在开始。")
SECOND = ("资料已经备齐，", "设备正常工作，", "参会人员到场。")
REVISED = ("嗯，", "请查阅文件，", "会议现在开始。")
TAIL = "请开始汇报。"


class _SourceASR:
    def __init__(self):
        self.index = 0

    def __call__(self, url, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/start":
            return {"session_id": "source-alignment"}
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        if endpoint == "/stream/finish":
            return {"text": "".join((*REVISED, *SECOND, TAIL)), "language": "Chinese"}
        if endpoint != "/stream/chunk":
            raise AssertionError(endpoint)
        self.index += 1
        sources = FIRST if self.index < 3 else (*FIRST, *SECOND)
        revision = 0
        if self.index >= 4:
            sources = (*REVISED, *SECOND)
            revision = 1
        return {
            "text": "".join(sources) + (TAIL if self.index >= 3 else ""),
            "language": "Chinese",
            "entity_asr_context_guard_revision": revision,
            "completed_segments": [
                {"segment_id": index + 1, "text": text, "vad_boundary": True}
                for index, text in enumerate(sources)
            ],
        }


@unittest.skipIf(web_app is None, "FastAPI is not installed")
class GatePipelineAlignmentTest(unittest.TestCase):
    @staticmethod
    def receive(ws, event):
        for _ in range(100):
            payload = ws.receive_json()
            if payload["event"] == "error":
                raise AssertionError(payload)
            if payload["event"] == event:
                return payload
        raise AssertionError(f"No {event}")

    def test_all_gate_modes_share_groups_revision_cache_and_final_tail(self):
        from system.learned_window_gate import LearnedWindowGate

        class SkipClassifier:
            def predict_proba(self, values):
                return [[0.01, 0.99]]

        signatures = []
        call_counts = {}
        for native, learned in (("off", "off"), ("conservative", "off"),
                                ("tri_state", "off"), ("tri_state", "shadow"),
                                ("tri_state", "on")):
            with self.subTest(native=native, learned=learned), tempfile.TemporaryDirectory() as tmp:
                gate = LearnedWindowGate("off")
                gate.mode, gate.model, gate.threshold, gate._column = learned, SkipClassifier(), .8, 1
                output = Path(tmp) / "session.jsonl"
                with patch.object(web_app, "TransformersRefiner", _IdentityRefiner), \
                     patch.object(web_app, "LearnedWindowGate", return_value=gate), \
                     patch.object(web_app, "_stream_request", _SourceASR()), \
                     patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0):
                    app = web_app.create_app(Path("/tmp/model"), "cpu", "http://fake-asr", "Chinese", 32,
                                             output, refinement_gate_mode=native,
                                             learned_window_gate_mode=learned)
                    with TestClient(app) as client, client.websocket_connect("/ws/stream?mode=streaming") as ws:
                        ready = self.receive(ws, "ready")
                        self.assertEqual(ready["window_pipeline"], {
                            "version": "fixed_k3_v1", "window_size": 3,
                            "one_punctuation_window": True, "fixed_groups": True,
                            "stream_reprocess_rejected": False,
                            "final_reprocess_rejected": False,
                            "group_numeric_self_corrections": False,
                        })
                        for index in range(5):
                            ws.send_bytes(b"pcm")
                            self.receive(ws, "chunk_ack" if index in (1, 4) else "transcript")
                            if index in (0, 2, 3):
                                update = self.receive(ws, "update")
                            if index == 3:
                                self.assertEqual(update["window_cache_reused_group_count"], 1)
                                self.assertEqual(update["window_cache_reprocessed_group_count"], 1)
                                self.assertEqual(update["window_cache_reprocess_reasons"], {"no_exact_source_group_match": 1})
                        ws.send_json({"event": "finish"})
                        final = self.receive(ws, "final")
                        health = client.get("/health").json()
                self.assertEqual(final["clean_text"], "".join((*REVISED, *SECOND, TAIL)))
                self.assertEqual(final["window_pipeline"], ready["window_pipeline"])
                self.assertEqual(health["window_pipeline"], ready["window_pipeline"])
                self.assertEqual(final["window_cache_reused_group_count"], 2)
                self.assertEqual(final["window_cache_reprocessed_group_count"], 1)
                self.assertEqual(final["refiner_session_stats"]["retry_call_count"], 0)
                self.assertTrue(final["refiner_session_stats"]["stats_complete"])
                records = [json.loads(line) for line in output.with_suffix(".stream_trace.jsonl").read_text().splitlines()]
                windows = [(segment["source"], segment["window_kind"], segment["window_chunk_count"])
                           for row in records if row["event"] == "window_result"
                           for segment in row["segments"] if segment["window_kind"] != "pending_tail"]
                signatures.append(windows)
                pending = [segment for row in records if row["event"] == "window_result"
                           for segment in row["segments"] if segment["window_kind"] == "pending_tail"]
                self.assertTrue(pending)
                self.assertTrue(all(segment["gate"]["action"] == "defer"
                                    and segment["outcome"] == "deferred"
                                    and not segment["model_calls"] for segment in pending))
                saved = json.loads(output.read_text().splitlines()[-1])
                self.assertEqual(saved["output"]["window_pipeline"], ready["window_pipeline"])
                call_counts[(native, learned)] = final["session_refiner_call_count"]
                if native == "off":
                    self.assertEqual(final["session_gate_skipped_segment_count"], 0)
        self.assertTrue(all(signature == signatures[0] for signature in signatures))
        self.assertEqual([source for source, _, _ in signatures[0]],
                         ["".join(FIRST), "".join(SECOND), "".join(REVISED), TAIL])
        self.assertEqual(call_counts[("off", "off")], 4)
        self.assertEqual(call_counts[("tri_state", "off")], 2)
        self.assertEqual(call_counts[("tri_state", "shadow")], 2)
        self.assertEqual(call_counts[("tri_state", "on")], 0)

    def test_rejected_streaming_group_retries_only_with_explicit_finish_option(self):
        class RejectingRefiner(_IdentityRefiner):
            calls = []

            def refine(self, text, **kwargs):
                self.calls.append(text)
                return "好。", 1.0

        class ASR(_SourceASR):
            def __call__(self, url, endpoint, *args, **kwargs):
                if endpoint == "/stream/finish":
                    return {"text": "".join((*FIRST, *SECOND)), "language": "Chinese"}
                result = super().__call__(url, endpoint, *args, **kwargs)
                if endpoint == "/stream/chunk":
                    result["text"] = result["text"].removesuffix(TAIL)
                return result

        for native in ("off", "tri_state"):
            for retry in ("off", "on"):
                with self.subTest(native=native, retry=retry):
                    RejectingRefiner.calls = []
                    with patch.object(web_app, "TransformersRefiner", RejectingRefiner), \
                         patch.object(web_app, "_stream_request", ASR()), \
                         patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0):
                        app = web_app.create_app(Path("/tmp/model"), "cpu", "http://fake-asr", "Chinese", 32,
                                                 None, refinement_gate_mode=native, final_refinement_mode=retry)
                        with TestClient(app) as client, client.websocket_connect("/ws/stream?mode=streaming") as ws:
                            self.receive(ws, "ready")
                            for index in range(3):
                                ws.send_bytes(b"pcm")
                                self.receive(ws, "chunk_ack" if index == 1 else "transcript")
                                if index in (0, 2):
                                    self.receive(ws, "update")
                            before = RejectingRefiner.calls.count("".join(FIRST))
                            self.assertEqual(before, 2)  # initial + preservation retry, once
                            ws.send_json({"event": "finish"})
                            final = self.receive(ws, "final")
                    self.assertEqual(RejectingRefiner.calls.count("".join(FIRST)), before * (2 if retry == "on" else 1))
                    self.assertEqual(final["clean_text"], "".join((*FIRST, *SECOND)))
                    self.assertEqual(final["window_pipeline"]["final_reprocess_rejected"], retry == "on")

    def test_numeric_boundary_guard_depends_on_source_signals_in_every_gate_mode(self):
        from experiments.scripts.collect_gate_pairs import closure_value

        class NumericRefiner(_IdentityRefiner):
            def refine(self, text, **kwargs):
                return "增长了5%", 1.0

        outputs = []
        for native in ("off", "conservative", "tri_state"):
            with self.subTest(native=native), patch.object(web_app, "TransformersRefiner", NumericRefiner):
                app = web_app.create_app(Path("/tmp/model"), "cpu", "http://fake-asr", "Chinese", 32,
                                         None, refinement_gate_mode=native)
                endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/ws/stream")
                refine = closure_value(endpoint, "refine_update")
                result = refine("增长了百分之五，", "Chinese", True,
                                web_app.EntityProtector([]), None, single_window=True)
                self.assertTrue(result["refiner_executed"])
                self.assertEqual(result["clean_text"], "增长了5%，")
                outputs.append(result["clean_text"])
        self.assertEqual(len(set(outputs)), 1)


if __name__ == "__main__":
    unittest.main()
