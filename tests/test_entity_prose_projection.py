"""Safety and Web-path checks for bounded entity-only prose confirmation."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from system.entity_matcher import EntityCandidateMatcher, EntityMatchReport
from system.entity_pipeline import finalize_entity_segment, prepare_entity_segment
from system.entity_store import EntityDefinition, EntityStore
from system.protection import EntityProtector


class EntityProseProjectionTest(unittest.TestCase):
    def prepare(self, source, canonical="厉飞宇", **options):
        definition = EntityDefinition(
            1, canonical, options.get("entity_type", "TERM"), "general",
            options.get("policy", "normalize"), 0, (),
        )
        protector = EntityProtector((definition,))
        matcher = EntityCandidateMatcher(
            (definition,), selected_domain="general", mode="auto"
        )
        return prepare_entity_segment(source, protector, matcher, allow_auto=True), protector

    def test_actual_logged_prose_keeps_source_character_and_protected_entity(self):
        definitions = (
            EntityDefinition(1, "厉飞宇", "TERM", "general", "normalize", 0, ()),
            EntityDefinition(2, "SPA", "TERM", "general", "normalize", 0, ()),
        )
        protector = EntityProtector(definitions)
        matcher = EntityCandidateMatcher(definitions, selected_domain="general", mode="auto")
        prepared = prepare_entity_segment(
            "想找你做个SPA，你找错人了。在下厉飞羽，", protector, matcher, allow_auto=True
        )
        result = finalize_entity_segment(
            "想找你做个__ENTITY_000__，你找错人了。下厉飞宇，\n<KEY>[厉飞宇]",
            prepared, protector,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.text, "想找你做个SPA，你找错人了。在下厉飞宇，")
        self.assertEqual(result.normalizations[-1]["confirmation_method"],
                         "bounded_prose_entity_projection_v1")

    def test_rule_also_supports_other_registered_terms(self):
        for observed, canonical in (("月秀区", "越秀区"), ("鬼灵们", "鬼灵门")):
            source = "今天我们一起讨论" + observed + "的相关问题。"
            prepared, protector = self.prepare(source, canonical)
            result = finalize_entity_segment(source.replace(observed, canonical), prepared, protector)
            self.assertTrue(result.accepted)
            self.assertEqual(result.text, source.replace(observed, canonical))

    def test_hint_only_and_unsafe_prose_still_fall_back(self):
        source = "今天我们一起讨论厉飞羽的相关问题。"
        prepared, protector = self.prepare(source)
        for response in (
            source + "<KEY>[厉飞宇]",  # copied annotation, no body correction
            "厉飞宇",  # lost context
            "今天一起讨论厉飞宇的相关问题。",  # two deleted characters
            "明天我们一起讨论厉飞宇的相关问题。",  # other substitution
            "今天我们一起讨论厉飞宇的相关问题，并补充新内容。",
            "今天我们一起讨论厉飞羽的相关问题。厉飞宇。",
            source + "<ENTITY_CANDIDATES>[厉飞宇]",
        ):
            with self.subTest(response=response):
                result = finalize_entity_segment(response, prepared, protector)
                self.assertEqual(result.text, source)
                self.assertFalse(result.normalizations)

    def test_short_ambiguous_and_risky_entities_stay_unconfirmed(self):
        source = "今天我们一起讨论厉飞羽的相关问题。"
        for options in ({"policy": "preserve"}, {"entity_type": "PERSON"}):
            prepared, protector = self.prepare(source, **options)
            result = finalize_entity_segment(source.replace("羽", "宇"), prepared, protector)
            self.assertEqual(result.text, source)
        for short_source in ("在下厉飞羽。", "厉飞羽和厉飞羽都来了。"):
            prepared, protector = self.prepare(short_source)
            result = finalize_entity_segment(short_source.replace("羽", "宇"), prepared, protector)
            self.assertEqual(result.text, short_source)
        prepared, protector = self.prepare(source)
        ambiguous = replace(prepared.report.matches[0], margin=0.01)
        prepared = replace(prepared, report=EntityMatchReport((ambiguous,), 0.0))
        result = finalize_entity_segment(source.replace("羽", "宇"), prepared, protector)
        self.assertEqual(result.text, source)

    def test_damaged_markers_and_invalid_explicit_patch_do_not_use_prose_fallback(self):
        source = "今天我们一起讨论厉飞羽的相关问题。"
        prepared, protector = self.prepare(source)
        response = source.replace("羽", "宇")
        result = finalize_entity_segment(response, prepared, protector, entity_patch={"action": "keep"})
        self.assertEqual(result.text, source)
        result = finalize_entity_segment("__ENTITY_999__" + response, prepared, protector)
        self.assertEqual(result.text, source)

    def test_web_uses_entity_projection_without_retry_or_extra_model_calls(self):
        from fastapi.testclient import TestClient
        import system.web_app as web

        source = "想找你做个SPA，你找错人了。在下厉飞羽，"
        calls = []

        class RecordedRefiner:
            def __init__(self, *args):
                pass

            def refine(self, text, **kwargs):
                calls.append(text)
                return "想找你做个__ENTITY_000__，你找错人了。下厉飞宇，<KEY>[厉飞宇]", 0.0

        def fake_asr(url, endpoint, session_id=None, data=b"", params=None):
            return {"session_id": "projection"} if endpoint == "/stream/start" else {
                "text": source, "language": "Chinese"
            }

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "entities.db"
            store = EntityStore(database)
            store.upsert_entity("厉飞宇", normalization_policy="normalize")
            store.upsert_entity("SPA", normalization_policy="normalize")
            with patch.object(web, "TransformersRefiner", RecordedRefiner), patch.object(
                web, "_stream_request", fake_asr
            ):
                app = web.create_app(Path(directory), "cpu", "http://fake", "Chinese", 64,
                                     None, entity_db=database, entity_fuzzy_mode="auto")
                with TestClient(app) as client, client.websocket_connect(
                    "/ws/stream?mode=offline&entity_audio_repair=0&entity_asr_context=0"
                ) as socket:
                    socket.receive_json()
                    socket.send_json({"event": "finish"})
                    while True:
                        result = socket.receive_json()
                        if result.get("event") == "final":
                            break
                self.assertEqual(result["clean_text"], source.replace("羽", "宇"))
                self.assertEqual(len(calls), 1)
                self.assertEqual(result["entity_normalizations"][-1]["confirmation_method"],
                                 "bounded_prose_entity_projection_v1")


if __name__ == "__main__":
    unittest.main()
