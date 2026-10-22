"""Offline snapshot adapter tests using the real source-owned window cache."""

from collections import Counter

import pytest

from system.offline_text_refinement import OfflineTextRefinementSession


def _fake_web_refine_update(*, edits=None, reject_first=False):
    """Stand in for web_app.refine_update, with a deterministic fake model."""
    edits = dict(edits or {})
    model_inputs = []
    source_attempts = Counter()

    def refine_update(
        raw_text,
        detected_language,
        final,
        protector,
        confidence,
        matcher=None,
        *,
        single_window=False,
        confidence_metadata=None,
        call_stats=None,
        trace_segments=None,
        pending_only=False,
        window_context=None,
    ):
        # The fake model is called only for refinable windows; pending tails
        # exercise the same web callback's deferred, raw-text behavior.
        accepted = True
        clean = raw_text
        if pending_only:
            outcome = "deferred"
        else:
            model_inputs.append(raw_text)
            source_attempts[raw_text] += 1
            clean = raw_text
            for source, target in edits.items():
                clean = clean.replace(source, target)
            accepted = not (reject_first and source_attempts[raw_text] == 1)
            outcome = "accepted" if accepted else "rejected"

        if trace_segments is not None:
            trace_segments.append({
                "source": raw_text,
                "outcome": outcome,
                "output": clean,
                "window_context": window_context,
                "model_calls": ([{"input": raw_text, "output": clean}]
                                if not pending_only else []),
            })
        return {
            "raw_text": raw_text,
            "clean_text": clean,
            "refiner_latency_ms": 1.0 if not pending_only else 0.0,
            "refiner_accepted": accepted,
            "refiner_executed": not pending_only,
            "refiner_reject_reasons": [] if accepted else ["fake_reject"],
            "entity_audit_issues": [],
            "entity_refinement_hints": [],
            "entity_normalizations": [],
            "numeric_normalizations": [],
            "numeric_rule_repairs": [],
            "numeric_fallbacks": [],
            "safe_numeric_repairs": [],
            "safe_repetition_repairs": [],
            "local_deletion_reviews": [],
            "repetition_reviews": [],
            "protected_entities": [],
            "entity_candidates": [],
            "entity_matcher_latency_ms": 0.0,
            "placeholder_retry_count": 0,
            "refiner_retry_count": 0,
            "refiner_retry_reasons": [],
            "refiner_masked_outputs": [clean] if not pending_only else [],
            "refinement_gate_decisions": [],
            "structured_patch_audits": [],
            "boundary_anomalies": [],
            "boundary_reviews": [],
            "refinement_window_context": window_context,
        }

    return refine_update, model_inputs


def _session(*, edits=None, reject_first=False, final_reprocess_rejected=False):
    callback, model_inputs = _fake_web_refine_update(
        edits=edits, reject_first=reject_first
    )
    session = OfflineTextRefinementSession(
        callback,
        (),
        rule_protection=False,
        final_reprocess_rejected=final_reprocess_rejected,
    )
    return session, model_inputs


def test_revised_source_invalidates_old_k3_output_and_display():
    session, model_inputs = _session(edits={"旧词": "旧词精修", "新词": "新词精修"})
    old_snapshot = "甲旧词。乙。丙。丁。"

    first = session.update(old_snapshot, revision=1)
    assert model_inputs == ["甲旧词。乙。丙。"]
    assert "旧词精修" in first["clean_text"]

    revised = session.update("甲新词。乙。丙。丁。", revision=2)

    assert model_inputs[-1] == "甲新词。乙。丙。"
    assert "新词精修" in revised["clean_text"]
    assert "旧词精修" not in revised["clean_text"]
    assert "旧词精修" not in revised["display_text"]
    assert revised["display_text"].startswith("甲新词精修。")


def test_final_flushes_short_tail_and_closes_session():
    session, model_inputs = _session(edits={"乙": "乙精修"})

    final = session.update("甲。乙。", revision=1, final=True)

    assert model_inputs == ["甲。乙。"]
    assert "乙精修" in final["clean_text"]
    assert final["final"] is True
    with pytest.raises(ValueError, match="finished"):
        session.update("甲。乙。丙。", revision=2)


def test_session_rejects_stale_revision():
    session, _ = _session()
    session.update("甲。", revision=2)

    with pytest.raises(ValueError, match="increase monotonically"):
        session.update("乙。", revision=1)


@pytest.mark.parametrize(
    ("final_reprocess_rejected", "expected_first_group_retries"),
    [(True, 1), (False, 0)],
)
def test_final_rejected_group_retry_follows_factory_configuration(
    final_reprocess_rejected, expected_first_group_retries
):
    session, model_inputs = _session(
        reject_first=True,
        final_reprocess_rejected=final_reprocess_rejected,
    )
    snapshot = "甲。乙。丙。丁。"
    first_group = "甲。乙。丙。"

    session.update(snapshot, revision=1)
    assert model_inputs.count(first_group) == 1

    session.update(snapshot, revision=2, final=True)

    assert model_inputs.count(first_group) == 1 + expected_first_group_retries
