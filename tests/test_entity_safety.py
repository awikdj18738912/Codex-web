from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys
import numpy as np

from system.entity_store import EntityDefinition, EntityStore
from system.entity_matcher import EntityCandidateMatcher, EntityDecision
from system.entity_pipeline import prepare_entity_segment, finalize_entity_segment
from system.protection import EntityProtector
from system.entity_asr_context import (
    build_entity_asr_context,
    validate_asr_context,
    MAX_ASR_CONTEXT_CHARS,
)


def entity(name, identifier=1, aliases=()):
    return EntityDefinition(
        identifier, name, "TERM", "medical", "normalize", 0, aliases
    )


class EntitySafetyRegressionTest(unittest.TestCase):
    def test_real_medical_and_news_negative_spans_do_not_change(self):
        definitions = (
            entity("手术室"),
            entity("骨髓", 2),
            entity("北京市", 3),
            entity("误工", 4),
        )
        protector = EntityProtector(definitions)
        matcher = EntityCandidateMatcher(
            definitions, selected_domain="medical", mode="auto"
        )
        for source in [
            "手术失败的可能性很低。",
            "这个手术是比较复杂。",
            "保持手术视野。",
            "骨碎补，骨碎了之后。",
            "北京时间到了。",
            "外出务工人员回乡。",
        ]:
            with self.subTest(source=source):
                prepared = prepare_entity_segment(
                    source, protector, matcher, allow_auto=True
                )
                self.assertEqual(prepared.baseline_text, source)
                self.assertFalse(prepared.report.auto_matches)
                self.assertFalse(prepared.hints)

    def test_boundary_conflicting_model_patch_is_rejected(self):
        definitions = (entity("手术室"),)
        p = EntityProtector(definitions)
        m = EntityCandidateMatcher(definitions, selected_domain="medical", mode="auto")
        prepared = prepare_entity_segment("手术失败。", p, m, allow_auto=True)
        result = finalize_entity_segment(
            "手术室败。",
            prepared,
            p,
            entity_patch=dict(
                action="replace",
                source="手术失",
                target="手术室",
                reason="entity_correction",
            ),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.text, "手术失败。")

    def test_english_substrings_are_not_entities_or_partial_placeholders(self):
        p = EntityProtector((entity("CT"), entity("MRI", 2)))
        result = p.protect("actually 做CT扫描，amrita 不是MRI。")
        self.assertEqual([s.original for s in result.spans], ["CT", "MRI"])
        self.assertEqual(
            p.restore(result.masked_text, result).text, result.original_text
        )

    def test_intact_marker_adjacent_to_latin_text_is_not_repaired(self):
        p = EntityProtector((entity("中文"),))
        result = p.protect("a中文b。")
        self.assertEqual(p.restore(result.masked_text, result).text, "a中文b。")

    def test_phonetic_candidate_requires_complete_bounded_confirmation(self):
        p = EntityProtector((entity("鬼灵门"),))
        m = EntityCandidateMatcher(p.entities, selected_domain="medical", mode="auto")
        prepared = prepare_entity_segment("鬼灵们是入口。", p, m, allow_auto=True)
        self.assertEqual(prepared.baseline_text, "鬼灵们是入口。")
        self.assertFalse(
            finalize_entity_segment("鬼灵门是入口。", prepared, p).accepted
        )
        confirmed = finalize_entity_segment(
            "鬼灵门是入口。",
            prepared,
            p,
            entity_patch=dict(
                action="replace",
                source="鬼灵们",
                target="鬼灵门",
                reason="entity_correction",
            ),
        )
        self.assertTrue(confirmed.accepted)
        self.assertEqual(confirmed.normalizations[0]["decision"], "CONTEXT_CONFIRMED")

    def test_trusted_alias_survives_failed_generation(self):
        p = EntityProtector((entity("TAVI手术", aliases=("凯比手术",)),))
        prepared = prepare_entity_segment("完成凯比手术。", p, None, allow_auto=True)
        rejected = finalize_entity_segment("丢掉术式。", prepared, p)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.text, "完成TAVI手术。")
        self.assertEqual(rejected.normalizations[0]["canonical"], "TAVI手术")

    def test_format_normalization_can_be_reverted_without_losing_trusted_alias(self):
        definitions = (entity("AgenticASR"), entity("TAVI手术", 2, ("凯比手术",)))
        p = EntityProtector(definitions, enable_rule_protection=False)
        m = EntityCandidateMatcher(definitions, selected_domain="medical", mode="auto")
        prepared = prepare_entity_segment(
            "Agentic ASR 完成凯比手术。", p, m, allow_auto=True
        )
        self.assertEqual(prepared.baseline_text, "AgenticASR 完成TAVI手术。")
        rejected = finalize_entity_segment("错误输出。", prepared, p)
        self.assertEqual(rejected.text, "Agentic ASR 完成TAVI手术。")
        self.assertEqual(len(rejected.normalizations), 1)


class EntityASRContextTest(unittest.TestCase):
    def test_context_contains_bounded_canonical_names_from_selected_domain(self):
        items = (
            entity("TAVI手术", 1, ("凯比手术",)),
            entity("骨髓", 2),
            EntityDefinition(3, "其它领域术语", "TERM", "legal", "normalize", 100, ()),
        )
        context = build_entity_asr_context(items, "medical")
        self.assertEqual(context.terms, ("TAVI手术",))
        self.assertNotIn("凯比", context.text)
        self.assertLessEqual(len(context.text), 1024)
        self.assertEqual(validate_asr_context(context.text), context.text)

    def test_backend_rejects_invalid_context(self):
        for invalid in [True, [], "x" * (MAX_ASR_CONTEXT_CHARS + 1), "a\x00b"]:
            with self.assertRaises(ValueError):
                validate_asr_context(invalid)

    def test_flask_context_is_scoped_to_session_and_survives_rotation(self):
        import system.qwen_asr_stream_server as server

        starts = []

        class FakeRunner:
            def init_streaming_state(self, **kwargs):
                starts.append(kwargs)
                return SimpleNamespace(text="", language="Chinese")

            def streaming_transcribe(self, pcm, state):
                state.text = "完成术式。"

            def finish_streaming_transcribe(self, state):
                pass

        runner = FakeRunner()
        module = SimpleNamespace(
            Qwen3ASRModel=SimpleNamespace(LLM=lambda **kwargs: runner)
        )
        captured = []

        def capture(app, **kwargs):
            captured.append(app)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"qwen_asr": module}
        ), patch.object(server.Flask, "run", capture):
            server.main(
                [
                    "--model",
                    directory,
                    "--confidence-logprobs",
                    "0",
                    "--vad",
                    "off",
                    "--segment-seconds",
                    "0.01",
                    "--max-segment-seconds",
                    "0.02",
                    "--silence-rms-threshold",
                    "0",
                ]
            )
        client = captured[0].test_client()
        sid = client.post("/stream/start?context=TAVI&language=Chinese").json[
            "session_id"
        ]
        self.assertEqual(starts[-1]["context"], "TAVI")
        client.post(
            "/stream/chunk?session_id=" + sid,
            data=np.ones(1600, dtype="float32").tobytes(),
        )
        self.assertEqual(starts[-1]["context"], "TAVI")
        plain = client.post("/v1/stream/start", json={"language": "Chinese"}).json
        self.assertFalse(plain["context_applied"])
        self.assertEqual(starts[-1]["context"], "")
        self.assertEqual(
            client.post("/v1/stream/start", json={"context": 5}).status_code, 400
        )
        self.assertEqual(
            client.post(
                "/stream/start", query_string={"context": "x" * 2049}
            ).status_code,
            400,
        )

    def test_web_passes_only_selected_glossary_and_reports_backend_ack(self):
        import json
        from fastapi.testclient import TestClient
        import system.web_app as web

        requests = []

        class Identity:
            def __init__(self, *args):
                pass

            def refine(self, text, **kwargs):
                return text, 0.0

        def fake_asr(url, endpoint, session_id=None, data=b"", params=None):
            requests.append((endpoint, params))
            if endpoint == "/stream/start":
                return {
                    "session_id": "ctx",
                    "context_applied": bool(params.get("context")),
                }
            return {"text": "完成TAVI手术。", "language": "Chinese"}

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "entities.db"
            store = EntityStore(database)
            store.upsert_entity("TAVI手术", domain="medical", aliases=("凯比手术",))
            store.upsert_entity("不相关的小说名", domain="general")
            with patch.object(web, "TransformersRefiner", Identity), patch.object(
                web, "_stream_request", fake_asr
            ):
                app = web.create_app(
                    Path(directory),
                    "cpu",
                    "http://fake",
                    "Chinese",
                    32,
                    None,
                    entity_db=database,
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?domain=medical&mode=offline&entity_asr_context=1"
                    ) as socket:
                        ready = socket.receive_json()
                        self.assertTrue(ready["entity_asr_context_applied"])
                        context = requests[0][1]["context"]
                        self.assertIn("TAVI手术", context)
                        self.assertNotIn("凯比手术", context)
                        self.assertNotIn("小说", context)
                        socket.send_json({"event": "finish"})
                        while True:
                            result = socket.receive_json()
                            if result.get("event") == "final":
                                break
                        self.assertTrue(result["entity_asr_context_applied"])
                        self.assertEqual(result["clean_text"], "完成TAVI手术。")
                    with client.websocket_connect(
                        "/ws/stream?domain=medical&mode=offline&entity_asr_context=0"
                    ) as socket:
                        self.assertFalse(
                            socket.receive_json()["entity_asr_context_applied"]
                        )
                        self.assertNotIn("context", requests[-1][1])

    def test_web_rejects_unconfirmed_patch_and_accepts_bounded_candidate_patch(self):
        import json
        from fastapi.testclient import TestClient
        import system.web_app as web

        class Confirming:
            supports_entity_candidates = True

            def __init__(self, *args):
                pass

            def refine(self, text, *, entity_candidates=(), **kwargs):
                assert entity_candidates[0]["observed"] == "鬼灵们"
                return (
                    json.dumps(
                        dict(
                            action="replace",
                            source="鬼灵们",
                            target="鬼灵门",
                            reason="entity_correction",
                        )
                    ),
                    0.0,
                )

        def fake_asr(url, endpoint, session_id=None, data=b"", params=None):
            return (
                {"session_id": "a"}
                if endpoint == "/stream/start"
                else {"text": "鬼灵们是入口。", "language": "Chinese"}
            )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "entities.db"
            EntityStore(database).upsert_entity(
                "鬼灵门", normalization_policy="normalize"
            )
            with patch.object(web, "TransformersRefiner", Confirming), patch.object(
                web, "_stream_request", fake_asr
            ):
                app = web.create_app(
                    Path(directory),
                    "cpu",
                    "http://fake",
                    "Chinese",
                    32,
                    None,
                    entity_db=database,
                    entity_fuzzy_mode="auto",
                )
                with TestClient(app) as client, client.websocket_connect(
                    "/ws/stream?mode=offline"
                ) as socket:
                    socket.receive_json()
                    socket.send_json({"event": "finish"})
                    while True:
                        result = socket.receive_json()
                        if result.get("event") == "final":
                            break
                    self.assertEqual(result["clean_text"], "鬼灵门是入口。")
                    self.assertEqual(
                        result["entity_normalizations"][0]["decision"],
                        "CONTEXT_CONFIRMED",
                    )
