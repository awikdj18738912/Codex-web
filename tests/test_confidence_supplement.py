from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

from system.confidence_supplement import (
    ConfidenceSupplement,
    compose_projections,
    window_evidence,
)
from system.refinement_gate import RefinementGate


def scored_projection(text: str, probability: float = .9):
    size = len(text.encode("utf-8"))
    return {
        "status": "mapped",
        "text": text,
        "spans": [dict(start=0, end=size, original_width=size, token_id=1,
                       token_index=0, logprob=math.log(probability),
                       evidence_id="token", generation_revision=1)],
        "coordinate_space": "raw_ASR_UTF8_bytes",
        "calibrated": False,
    }


class ConfidenceSupplementTest(unittest.TestCase):
    def setUp(self):
        self.gate = RefinementGate("tri_state")

    def native_keep(self, text="今天天气很好。"):
        return self.gate.decide(text, asr_confidence=.99, calibrated=True,
                                covers_segment=True, is_final=True)

    def test_high_score_cannot_cancel_native_signal_or_entity_review(self):
        supplement = ConfidenceSupplement("on")
        cleanup = self.gate.decide("我我我知道了。", asr_confidence=.99,
                                   calibrated=True, covers_segment=True,
                                   is_final=True)
        entity = self.gate.decide("韩立。", asr_confidence=.99,
                                  calibrated=True, covers_segment=True,
                                  entity_hints=("韩立",), is_final=True)
        for native, source in ((cleanup, "我我我知道了。"), (entity, "韩立。")):
            with self.subTest(source=source):
                result, audit = supplement.decide(native, source, scored_projection(source, .999))
                self.assertTrue(result.should_refine)
                self.assertEqual(result.action, "refine")
                self.assertEqual(result.reasons, native.reasons)
                self.assertFalse(audit["applied"])
                self.assertEqual(audit["reason"], "native_action_preserved")

    def test_low_score_adds_review_only_to_native_keep(self):
        native = self.native_keep()
        result, audit = ConfidenceSupplement("on", low_threshold=.8).decide(
            native, "今天天气很好。", scored_projection("今天天气很好。", .2))
        self.assertTrue(result.should_refine)
        self.assertEqual(result.reasons, ("low_confidence_fragment_review",))
        self.assertTrue(audit["applied"])
        self.assertEqual(audit["evidence"]["status"], "complete")

    def test_high_score_clean_segment_is_kept_and_threshold_is_exclusive(self):
        native = self.native_keep()
        supplement = ConfidenceSupplement("on", low_threshold=.8)
        result, audit = supplement.decide(
            native, "今天天气很好。", scored_projection("今天天气很好。", .95))
        self.assertIs(result, native)
        self.assertFalse(result.should_refine)
        self.assertEqual(audit["suggested_action"], "keep")
        self.assertEqual(audit["reason"], "no_extra_risk_detected")

        equal, equal_audit = supplement.decide(
            native, "今天天气很好。", scored_projection("今天天气很好。", .8))
        self.assertIs(equal, native)
        self.assertFalse(equal_audit["applied"])
        below, below_audit = supplement.decide(
            native, "今天天气很好。", scored_projection("今天天气很好。", .8 - 1e-6))
        self.assertTrue(below.should_refine)
        self.assertEqual(below_audit["reason"], "low_confidence_fragment_review")

    def test_off_and_shadow_do_not_change_native_action(self):
        native = self.native_keep()
        projection = scored_projection("今天天气很好。", .1)
        for mode in ("off", "shadow"):
            with self.subTest(mode=mode):
                result, audit = ConfidenceSupplement(mode).decide(
                    native, "今天天气很好。", projection)
                self.assertIs(result, native)
                self.assertFalse(audit["applied"])
                self.assertEqual(audit["native_action"], "keep")
                if mode == "shadow":
                    self.assertEqual(audit["suggested_action"], "refine")

    def test_missing_partial_stale_and_ambiguous_scores_fail_toward_review(self):
        source = "今天天气很好。"
        cases = {
            "missing": None,
            "partial": scored_projection(source),
            "stale": scored_projection("明天天气很好。"),
            "ambiguous": scored_projection(source + source),
        }
        cases["partial"]["spans"] = []
        for label, projection in cases.items():
            with self.subTest(evidence=label):
                result, audit = ConfidenceSupplement("on").decide(
                    self.native_keep(source), source, projection)
                self.assertTrue(result.should_refine)
                self.assertTrue(audit["applied"])
                self.assertEqual(audit["reason"], "confidence_evidence_unknown_review")
                self.assertEqual(audit["evidence"]["status"], "unknown")

    def test_nan_positive_logprob_and_overlapping_spans_are_unknown(self):
        source = "今天天气很好。"
        malformed = {}
        for label, logprob in (("nan", float("nan")), ("positive", .01)):
            projection = scored_projection(source)
            projection["spans"][0]["logprob"] = logprob
            malformed[label] = projection
        overlapping = scored_projection(source)
        overlapping["spans"].append(dict(overlapping["spans"][0]))
        malformed["overlap"] = overlapping

        for label, projection in malformed.items():
            with self.subTest(evidence=label):
                result, audit = ConfidenceSupplement("on").decide(
                    self.native_keep(source), source, projection)
                self.assertTrue(result.should_refine)
                self.assertTrue(audit["applied"])
                self.assertEqual(audit["evidence"]["status"], "unknown")
                self.assertEqual(audit["evidence"]["reason"], "invalid_source_evidence")

    def test_defer_refined_and_empty_text_are_never_upgraded(self):
        supplement = ConfidenceSupplement("on")
        high_risk_projection = scored_projection("今天天气很好。", .01)
        clean = self.native_keep()
        deferred = self.gate.decide("今天天气很好", stable=False, tail_age_ms=100)
        already_refined_keep = self.gate.decide(
            "今天天气很好。", asr_confidence=.99, calibrated=True,
            covers_segment=True, is_final=True, same_as_last_refined=True)
        already_refined = self.gate.decide("我我我知道了。", is_final=True)
        empty = self.gate.decide("", is_final=True)
        for native, source in ((deferred, "今天天气很好。"),
                               (already_refined_keep, "今天天气很好。"),
                               (already_refined, "我我我知道了。"),
                               (empty, "")):
            with self.subTest(action=native.action, source=source):
                result, audit = supplement.decide(native, source, high_risk_projection)
                self.assertIs(result, native)
                self.assertFalse(audit["applied"])
        self.assertEqual(deferred.action, "defer")
        self.assertEqual(already_refined_keep.reasons, ("already_refined_hypothesis",))
        self.assertEqual(already_refined_keep.action, "keep")
        self.assertTrue(already_refined.should_refine)
        self.assertFalse(clean.should_refine)

    def test_compose_uses_utf8_offsets_ascii_spacing_and_vad_local_evidence_ids(self):
        projection = compose_projections([
            {"status": "mapped", "text": "A", "spans": [
                dict(start=0, end=1, evidence_id="token", logprob=-.1)]},
            {"status": "mapped", "text": "B", "spans": [
                dict(start=0, end=1, evidence_id="token", logprob=-.2)]},
            {"status": "mapped", "text": "你", "spans": [
                dict(start=0, end=3, evidence_id="token", logprob=-.3)]},
        ])
        self.assertEqual(projection["text"], "A B你")
        self.assertEqual(projection["coordinate_space"], "raw_ASR_UTF8_bytes")
        self.assertEqual([(s["start"], s["end"]) for s in projection["spans"]],
                         [(0, 1), (2, 3), (3, 6)])
        self.assertEqual([s["evidence_id"] for s in projection["spans"]],
                         ["0:token", "1:token", "2:token"])
        self.assertEqual(projection["spans"][0]["logprob"], -.1)
        self.assertEqual(projection["spans"][1]["logprob"], -.2)

    def test_missing_vad_projection_does_not_borrow_previous_segment_score(self):
        first = "资料齐备。"
        second = "设备就绪。"
        projection = compose_projections([
            {"status": "mapped", "text": first,
             "spans": scored_projection(first)["spans"]},
            {"status": "unsupported", "text": second, "spans": []},
        ])
        self.assertEqual(projection["text"], first + second)
        self.assertEqual(window_evidence(second, projection)["status"], "unknown")
        result, audit = ConfidenceSupplement("on").decide(
            self.native_keep(second), second, projection)
        self.assertTrue(result.should_refine)
        self.assertEqual(audit["reason"], "confidence_evidence_unknown_review")

    def test_create_app_rejects_supplement_on_with_learned_gate_on(self):
        from tests.test_web_app_finish import web_app

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            web_app.create_app(Path("/tmp/model"), "cpu", "http://fake-asr",
                               "Chinese", 32, None,
                               refinement_gate_mode="tri_state",
                               learned_window_gate_mode="on",
                               confidence_supplement_mode="on")


@unittest.skipUnless(__import__("tests.test_web_app_finish", fromlist=["web_app"]).web_app,
                     "FastAPI is not installed")
class ConfidenceSupplementWebTest(unittest.TestCase):
    def test_asr_projection_drives_low_risk_calls_and_preserves_high_risk_and_defer(self):
        from experiments.scripts.collect_gate_pairs import closure_value
        from tests.test_web_app_finish import TestClient, _IdentityRefiner, web_app

        class CountingRefiner(_IdentityRefiner):
            calls = []

            def __init__(self, *args, **kwargs):
                type(self).calls = []

            def refine(self, text, **kwargs):
                type(self).calls.append(text)
                return text, 1.0

        with patch.object(web_app, "TransformersRefiner", CountingRefiner):
            app = web_app.create_app(
                Path("/tmp/model"), "cpu", "http://fake-asr", "Chinese", 32,
                None, refinement_gate_mode="tri_state",
                confidence_supplement_mode="on")
            endpoint = next(route.endpoint for route in app.routes
                            if getattr(route, "path", None) == "/ws/stream")
            refine_update = closure_value(endpoint, "refine_update")
            with TestClient(app):
                def run(source, probability=None, *, pending=False, old_payload=False):
                    asr_payload = {
                        "confidence": .99,
                        "confidence_calibrated": True,
                        "confidence_covers_full_text": True,
                    }
                    if not old_payload:
                        asr_payload["confidence_projection"] = scored_projection(
                            source, probability)
                    metadata = web_app._confidence_metadata(asr_payload)
                    return refine_update(
                        source, "Chinese", True, web_app.EntityProtector([]), .99,
                        single_window=True, confidence_metadata=metadata,
                        pending_only=pending)

                low = "今天天气很好。"
                low_result = run(low, .2)
                self.assertTrue(low_result["refiner_executed"])
                self.assertEqual(CountingRefiner.calls, [low])

                legacy_result = run(low, old_payload=True)
                self.assertTrue(legacy_result["refiner_executed"])
                self.assertEqual(CountingRefiner.calls, [low, low])

                high_signal = "我我我知道了。"
                high_result = run(high_signal, .99)
                self.assertTrue(high_result["refiner_executed"])
                self.assertEqual(CountingRefiner.calls, [low, low, high_signal])

                pending_result = run(low, .01, pending=True)
                self.assertFalse(pending_result["refiner_executed"])
                self.assertEqual(CountingRefiner.calls, [low, low, high_signal])
                self.assertEqual(pending_result["refinement_gate_decisions"][0]["action"],
                                 "defer")


if __name__ == "__main__":
    unittest.main()
