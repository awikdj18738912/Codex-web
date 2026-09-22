"""Browser WebSocket UI for a pluggable streaming ASR backend and AgenticASR."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import threading
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .live_qwen_refiner import TransformersRefiner, _append_record
from .entity_store import EntityDefinition, EntityStore
from .deterministic_cleanup import clean_transcript_deterministically
from .entity_matcher import EntityCandidateMatcher, EntityFuzzyMode
from .entity_pipeline import (
    FinalizedEntitySegment,
    finalize_entity_segment,
    has_placeholder_failure,
    has_retryable_integrity_failure,
    prepare_entity_segment,
)
from .protection import EntityProtector, strip_refiner_key_suffix
from .refinement_gate import (
    HypothesisStability,
    HypothesisTracker,
    RefinementGate,
    RefinementGateDecision,
    RefinementGateMode,
)
from .refinement_guard import (
    RepetitionReviewCandidate,
    apply_repetition_review_decisions,
    detect_boundary_anomalies,
    find_repetition_review_candidates,
    join_refined_segments,
    preserve_safe_numeric_edits,
    preserve_safe_repetition_edits,
    split_for_refinement,
)
from .numeric_normalizer import ContextualNumericNormalizer
from .window_refinement import (
    CumulativeWindowRefinement,
    StreamingRefinementDisplay,
)
from .session_memory import SessionEntityMemory
from .refinement_protocol import (
    apply_structured_patch,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
# Refining the complete cumulative hypothesis for every streaming chunk is
# expensive and can monopolize the single GPU-backed refiner when a long file
# is uploaded (especially if the browser has more than one open connection).
# Keep short early hypotheses responsive, then perform one complete refinement
# when the client sends ``finish``.
STREAMING_INTERMEDIATE_MAX_CHARS = 240
# One refinement pass costs about half a second while a sentence window only
# grows a few characters per ASR chunk, so retriggering it immediately mostly
# regenerates the same active window.  Run at most one pass per interval.
# Finished passes are published to the browser, so this interval is also the
# refined-pane update cadence.
STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS = 1.0
REFINER_LOCK_TIMEOUT_SECONDS = 30.0
FINAL_REFINEMENT_TIMEOUT_SECONDS = 30.0
# Long recordings are refined one bounded segment at a time.  Give the final
# pass a length-aware deadline while retaining a hard upper bound for broken
# model calls or disconnected clients.
FINAL_REFINEMENT_MAX_TIMEOUT_SECONDS = 300.0
WEBSOCKET_SEND_TIMEOUT_SECONDS = 10.0
ASR_CONTROL_REQUEST_TIMEOUT_SECONDS = 120.0
# Streaming recognition becomes more expensive as the active Qwen context
# grows.  A long recording can therefore have an occasional slow chunk even
# though the session is healthy.  Keep control requests bounded more tightly,
# but allow chunk processing to finish before the browser's watchdog fires.
ASR_CHUNK_REQUEST_TIMEOUT_SECONDS = 300.0
ASR_CHUNK_STATUS_INTERVAL_SECONDS = 10.0
class ASRApiStyle(str, Enum):
    """Wire protocol used between the Web service and streaming ASR."""

    CURRENT = "current"
    LEGACY_V1 = "legacy_v1"

    @classmethod
    def parse(cls, value: str | "ASRApiStyle") -> "ASRApiStyle":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(style.value for style in cls)
            raise ValueError(
                f"unsupported ASR API style {value!r}; expected one of: {choices}"
            ) from error


class EntityInput(BaseModel):
    canonical_text: str
    entity_type: str = "TERM"
    domain: str = "general"
    normalization_policy: str = "preserve"
    priority: int = 0
    aliases: list[str] = Field(default_factory=list)


class EntityEnabledInput(BaseModel):
    enabled: bool


class _RefinerCallsClosed(RuntimeError):
    """Raised when a timed-out/disconnected session forbids new model calls."""


class _RefinerSessionStats:
    """Thread-safe accounting for every real Refiner invocation in one socket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepting_new_calls = True
        self._call_count = 0
        self._initial_call_count = 0
        self._retry_call_count = 0
        self._intermediate_call_count = 0
        self._final_call_count = 0
        self._completed_call_count = 0
        self._failed_call_count = 0
        self._inflight_call_count = 0
        self._completed_wall_latency_ms = 0.0
        self._gate_skipped_segment_count = 0
        self._busy_segment_count = 0
        self._closed_call_rejection_count = 0

    def begin_call(self, *, final: bool, retry: bool) -> float | None:
        """Reserve one actual call atomically, returning its start timestamp."""

        with self._lock:
            if not self._accepting_new_calls:
                self._closed_call_rejection_count += 1
                return None
            self._call_count += 1
            self._inflight_call_count += 1
            if retry:
                self._retry_call_count += 1
            else:
                self._initial_call_count += 1
            if final:
                self._final_call_count += 1
            else:
                self._intermediate_call_count += 1
        return time.perf_counter()

    def finish_call(self, started_at: float, *, failed: bool) -> None:
        with self._lock:
            self._inflight_call_count -= 1
            self._completed_call_count += 1
            if failed:
                self._failed_call_count += 1
            self._completed_wall_latency_ms += (
                time.perf_counter() - started_at
            ) * 1000

    def record_gate_skip(self) -> None:
        with self._lock:
            self._gate_skipped_segment_count += 1

    def record_busy_segment(self) -> None:
        with self._lock:
            self._busy_segment_count += 1

    def close_new_calls(self) -> None:
        """Prevent a cancelled background thread from starting more calls."""

        with self._lock:
            self._accepting_new_calls = False

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            observed_opportunities = (
                self._initial_call_count
                + self._gate_skipped_segment_count
                + self._busy_segment_count
            )
            return {
                "schema_version": 1,
                "call_count": self._call_count,
                "initial_call_count": self._initial_call_count,
                "retry_call_count": self._retry_call_count,
                "intermediate_call_count": self._intermediate_call_count,
                "final_call_count": self._final_call_count,
                "completed_call_count": self._completed_call_count,
                "failed_call_count": self._failed_call_count,
                "inflight_call_count": self._inflight_call_count,
                "completed_wall_latency_ms": round(
                    self._completed_wall_latency_ms, 3
                ),
                "gate_skipped_segment_count": self._gate_skipped_segment_count,
                "busy_segment_count": self._busy_segment_count,
                "closed_call_rejection_count": self._closed_call_rejection_count,
                "observed_initial_opportunity_count": observed_opportunities,
                "accepting_new_calls": self._accepting_new_calls,
                "stats_complete": (
                    not self._accepting_new_calls
                    and self._inflight_call_count == 0
                ),
            }


def _entity_dict(definition: EntityDefinition) -> dict[str, object]:
    return {
        "entity_id": definition.entity_id,
        "canonical_text": definition.canonical_text,
        "entity_type": definition.entity_type,
        "domain": definition.domain,
        "normalization_policy": definition.normalization_policy,
        "priority": definition.priority,
        "aliases": list(definition.aliases),
        "enabled": definition.enabled,
        "source": definition.source,
        "created_at": definition.created_at,
        "updated_at": definition.updated_at,
    }


def _confidence_metadata(payload: dict[str, object]) -> dict[str, object]:
    """Keep confidence provenance/coverage instead of reducing it to a float."""

    token_count = payload.get("confidence_token_count")
    min_probability = payload.get("confidence_min_probability")
    return {
        "calibrated": payload.get("confidence_calibrated") is True,
        "covers_full_text": payload.get("confidence_covers_full_text") is True,
        "scope": (
            payload.get("confidence_scope")
            if isinstance(payload.get("confidence_scope"), str)
            else None
        ),
        "source": (
            payload.get("confidence_source")
            if isinstance(payload.get("confidence_source"), str)
            else None
        ),
        "token_count": (
            int(token_count)
            if isinstance(token_count, int) and not isinstance(token_count, bool)
            else None
        ),
        "min_probability": (
            float(min_probability)
            if isinstance(min_probability, (int, float))
            and not isinstance(min_probability, bool)
            else None
        ),
    }


def _repetition_review_cache_key(
    candidate: RepetitionReviewCandidate,
) -> tuple[int, int, str, str, str]:
    """Identify a candidate across streaming revisions without using its index."""

    return (
        candidate.start,
        candidate.end,
        candidate.source,
        candidate.target,
        candidate.context,
    )


def _repetition_decision_from_refined_context(
    text: str, candidate: RepetitionReviewCandidate, response: str
) -> list[dict[str, object]] | None:
    """Accept a model edit only when it changes exactly this repeated span."""

    context = candidate.context
    # The same phrase may occur elsewhere in a window.  Align the candidate
    # with its source-owned offset rather than replacing the first occurrence.
    context_start = next(
        (
            start
            for start in range(max(0, candidate.end - len(context)), candidate.start + 1)
            if text[start : start + len(context)] == context
            and start <= candidate.start
            and candidate.end <= start + len(context)
        ),
        None,
    )
    if context_start is None:
        return None
    local_start = candidate.start - context_start
    local_end = candidate.end - context_start
    if context[local_start:local_end] != candidate.source:
        return None
    expected = context[:local_start] + candidate.target + context[local_end:]
    refined, _, issue = apply_structured_patch(
        context, strip_refiner_key_suffix(response)
    )
    if issue is not None:
        return None
    if refined == context:
        action = "keep"
    elif refined == expected:
        action = "remove"
    elif (
        expected.startswith(refined)
        and refined.endswith(tuple("。！？!?；;"))
        and any(
            mark in refined[local_start + len(candidate.target):]
            for mark in "。！？!?；;"
        )
    ):
        # A text Refiner may stop at a complete sentence before the end of
        # the supplied context.  Its entire visible answer must still equal
        # the expected candidate-only edit; never apply the omitted suffix.
        action = "remove"
    else:
        return None
    return [
        {
            "index": candidate.index,
            "action": action,
            "reason": "model_local_text_review",
        }
    ]


def _reviewed_source_spans(
    source_spans: object,
    before: str,
    after: str,
    reviews: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    """Keep accepted local review edits attached to their original ASR spans."""

    if not isinstance(source_spans, (list, tuple)):
        return None
    entries: list[dict[str, object]] = []
    prefix = ""
    for span in source_spans:
        if not isinstance(span, dict):
            return None
        source = span.get("source_text")
        clean = span.get("clean_text")
        if not isinstance(source, str) or not isinstance(clean, str):
            return None
        extended = join_refined_segments((prefix, clean))
        if not extended.endswith(clean):
            return None
        entries.append({
            "source_text": source,
            "clean_text": clean,
            "state": span.get("state", "active"),
            "start": len(extended) - len(clean),
            "end": len(extended),
        })
        prefix = extended
    if prefix != before:
        return None

    applied = sorted(
        (record for record in reviews
         if record.get("decision") == "remove" and record.get("applied") is True),
        key=lambda record: int(record["start"]),
        reverse=True,
    )
    if not applied:
        return None
    for record in applied:
        start, end = record.get("start"), record.get("end")
        source, target = record.get("source"), record.get("target")
        if (
            not isinstance(start, int) or not isinstance(end, int)
            or not isinstance(source, str) or not isinstance(target, str)
            or before[start:end] != source
        ):
            return None
        left = next((i for i, item in enumerate(entries)
                     if item["start"] <= start < item["end"]), None)
        right = next((i for i, item in enumerate(entries)
                      if item["start"] < end <= item["end"]), None)
        if left is None or right is None or left > right:
            return None
        affected = entries[left:right + 1]
        combined = join_refined_segments(item["clean_text"] for item in affected)
        local_start = start - affected[0]["start"]
        local_end = end - affected[0]["start"]
        if combined[local_start:local_end] != source:
            return None
        entries[left:right + 1] = [{
            "source_text": "".join(item["source_text"] for item in affected),
            "clean_text": combined[:local_start] + target + combined[local_end:],
            "state": (
                "active" if any(item["state"] == "active" for item in affected)
                else "committed"
            ),
            "start": affected[0]["start"],
            "end": affected[-1]["end"],
        }]
    if clean_transcript_deterministically(
        join_refined_segments(item["clean_text"] for item in entries)
    ) != after:
        return None
    return [
        {key: item[key] for key in ("source_text", "clean_text", "state")}
        for item in entries
    ]


def _stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
    *,
    api_style: str | ASRApiStyle = ASRApiStyle.CURRENT,
) -> dict[str, object]:
    style = ASRApiStyle.parse(api_style)
    request_data = data
    content_type = "application/octet-stream"
    method = "POST"
    if style is ASRApiStyle.CURRENT:
        query_params = dict(params or {})
        if session_id:
            query_params["session_id"] = session_id
        query = urllib.parse.urlencode(query_params)
        url = f"{asr_url.rstrip('/')}{endpoint}"
        if query:
            url = f"{url}?{query}"
    else:
        if endpoint == "/stream/start":
            url = f"{asr_url.rstrip('/')}/v1/stream/start"
            request_data = json.dumps(params or {}, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"
        else:
            if not session_id:
                raise ValueError(f"{endpoint} requires a streaming ASR session_id")
            encoded_session_id = urllib.parse.quote(session_id, safe="")
            if endpoint == "/stream/chunk":
                url = (
                    f"{asr_url.rstrip('/')}/v1/stream/"
                    f"{encoded_session_id}/chunk"
                )
            elif endpoint == "/stream/finish":
                url = (
                    f"{asr_url.rstrip('/')}/v1/stream/"
                    f"{encoded_session_id}/finish"
                )
                request_data = b"{}"
                content_type = "application/json"
            elif endpoint == "/stream/cancel":
                url = f"{asr_url.rstrip('/')}/v1/stream/{encoded_session_id}"
                request_data = None
                method = "DELETE"
            else:
                raise ValueError(f"unsupported streaming ASR endpoint: {endpoint}")
    request = urllib.request.Request(
        url,
        data=request_data,
        headers={"Content-Type": content_type},
        method=method,
    )
    try:
        timeout = (
            ASR_CHUNK_REQUEST_TIMEOUT_SECONDS
            if endpoint == "/stream/chunk"
            else ASR_CONTROL_REQUEST_TIMEOUT_SECONDS
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"streaming ASR request failed: {error}") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(f"streaming ASR error: {payload}")
    return payload


async def _stream_request_async(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
    *,
    api_style: str | ASRApiStyle = ASRApiStyle.CURRENT,
) -> dict[str, object]:
    """Run the blocking ASR HTTP adapter without stopping the WebSocket loop."""

    style = ASRApiStyle.parse(api_style)
    if style is ASRApiStyle.CURRENT:
        # Preserve the original call shape so existing test doubles and custom
        # adapters that implement the current protocol remain compatible.
        return await asyncio.to_thread(
            _stream_request,
            asr_url,
            endpoint,
            session_id,
            data,
            params,
        )
    return await asyncio.to_thread(
        _stream_request,
        asr_url,
        endpoint,
        session_id,
        data,
        params,
        api_style=style,
    )


def create_app(
    refiner_model: Path,
    refiner_device: str,
    asr_url: str,
    language: str | None,
    max_new_tokens: int,
    output: Path | None,
    entity_db: Path | None = None,
    entity_fuzzy_mode: str = "shadow",
    refinement_gate_mode: str = "off",
    rule_protection: bool = True,
    numeric_normalization: bool = False,
    asr_api_style: str = ASRApiStyle.CURRENT.value,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    print("Loading AgenticASR Refiner...", flush=True)
    refiner = TransformersRefiner(refiner_model, refiner_device, max_new_tokens)
    refiner_lock = threading.Lock()

    def review_final_repetition_text(
        text: str,
        *,
        final: bool,
        call_stats: _RefinerSessionStats | None,
        streaming: bool = False,
        review_cache: dict[
            tuple[int, int, str, str, str], dict[str, object]
        ] | None = None,
    ) -> tuple[str, list[dict[str, object]], float, bool]:
        """Review structural repetitions in a stable stream window or final text."""

        if (not final and not streaming) or not text:
            return text, [], 0.0, False
        candidates = find_repetition_review_candidates(text, max_candidates=128)
        if not candidates:
            return text, [], 0.0, False

        cached_decisions: list[dict[str, object]] = []
        pending_candidates: list[RepetitionReviewCandidate] = []
        for candidate in candidates:
            cached = (
                review_cache.get(_repetition_review_cache_key(candidate))
                if review_cache is not None
                else None
            )
            action = cached.get("action") if isinstance(cached, dict) else None
            if action in {"keep", "remove"}:
                cached_decisions.append(
                    {
                        "index": candidate.index,
                        "action": action,
                        "reason": str(cached.get("reason", "streaming_cache")),
                    }
                )
            else:
                pending_candidates.append(candidate)

        if not pending_candidates:
            if cached_decisions:
                # Cached decisions must retain their applied records.  The
                # streaming display uses those records to attach an edit to
                # its source-owned spans.  Dropping them here left the text
                # clean while the spans reverted to raw ASR on the next pass.
                reviewed_text, applied_records = apply_repetition_review_decisions(
                    text, candidates, cached_decisions
                )
                return reviewed_text, list(applied_records), 0.0, False
            return text, [], 0.0, False

        records: list[dict[str, object]] = []
        if not refiner_lock.acquire(timeout=REFINER_LOCK_TIMEOUT_SECONDS):
            records = [
                {
                    **candidate.public_dict(),
                    "decision": "unresolved",
                    "reason": "review_unavailable",
                }
                for candidate in pending_candidates
            ]
            if cached_decisions:
                text, cached_records = apply_repetition_review_decisions(
                    text, candidates, cached_decisions
                )
                records.extend(cached_records)
            return text, records, 0.0, False

        total_latency_ms = 0.0
        decisions: list[dict[str, object]] = list(cached_decisions)
        model_responses: dict[int, str] = {}
        try:
            reviewer = getattr(refiner, "review_repetition", None)
            if reviewer is None:
                records = [
                    {
                        **candidate.public_dict(),
                        "decision": "unresolved",
                        "reason": "review_unavailable",
                    }
                    for candidate in pending_candidates
                ]
                reviewed_text = text
                if cached_decisions:
                    reviewed_text, cached_records = apply_repetition_review_decisions(
                        text, candidates, cached_decisions
                    )
                    records.extend(cached_records)
                return reviewed_text, records, total_latency_ms, False
            for candidate in pending_candidates:
                started_at = (
                    call_stats.begin_call(final=final, retry=False)
                    if call_stats is not None
                    else time.perf_counter()
                )
                if started_at is None:
                    records.append(
                        {
                            **candidate.public_dict(),
                            "decision": "unresolved",
                            "reason": "review_calls_closed",
                        }
                    )
                    continue
                try:
                    response, latency_ms = reviewer(candidate.context)
                except BaseException:
                    if call_stats is not None:
                        call_stats.finish_call(started_at, failed=True)
                    records.append(
                        {
                            **candidate.public_dict(),
                            "decision": "unresolved",
                            "reason": "review_failed",
                        }
                    )
                    continue
                if call_stats is not None:
                    call_stats.finish_call(started_at, failed=False)
                total_latency_ms += latency_ms
                model_responses[candidate.index] = response
                decision = _repetition_decision_from_refined_context(
                    text, candidate, response
                )
                if decision is None:
                    records.append(
                        {
                            **candidate.public_dict(),
                            "decision": "unresolved",
                            "reason": "invalid_response",
                            "model_response": response,
                        }
                    )
                else:
                    decisions.extend(decision)
            if decisions:
                reviewed_text, applied_records = apply_repetition_review_decisions(
                    text, candidates, decisions
                )
            else:
                reviewed_text, applied_records = text, ()
            applied_by_index = {
                int(record["index"]): record for record in applied_records
            }
            # A cached deletion was already validated on an earlier stream
            # pass.  Still emit its applied record on this pass so the
            # source-span map can carry the edit forward to the browser.
            records.extend(
                applied_by_index[item["index"]]
                for item in cached_decisions
                if int(item["index"]) in applied_by_index
            )
            records.extend(
                {
                    **applied_by_index[candidate.index],
                    "model_response": model_responses[candidate.index],
                }
                for candidate in pending_candidates
                if candidate.index in applied_by_index
            )
            records.sort(key=lambda record: int(record["index"]))

            if review_cache is not None:
                decision_by_index = {
                    int(item["index"]): item for item in decisions
                }
                for candidate in pending_candidates:
                    decision = decision_by_index.get(candidate.index)
                    if decision is None:
                        continue
                    action = str(decision.get("action", "")).lower()
                    applied_record = applied_by_index.get(candidate.index)
                    if action == "keep" or (
                        action == "remove"
                        and isinstance(applied_record, dict)
                        and applied_record.get("applied") is True
                    ):
                        review_cache[_repetition_review_cache_key(candidate)] = {
                            "action": action,
                            "reason": str(
                                decision.get("reason", "streaming_cache")
                            ),
                        }
                if len(review_cache) > 1024:
                    for key in tuple(review_cache)[:256]:
                        review_cache.pop(key, None)
            return reviewed_text, records, total_latency_ms, True
        finally:
            refiner_lock.release()

    entity_store = EntityStore(entity_db) if entity_db is not None else None
    fuzzy_mode = EntityFuzzyMode.parse(entity_fuzzy_mode)
    refinement_gate = RefinementGate(
        refinement_gate_mode,
        # The ASR backend currently exposes an uncalibrated suffix
        # log-probability, not full-segment confidence.  Treating that value
        # as a reason to refine every clean segment caused the Refiner to
        # introduce more errors than it removed.
        refine_on_unverified_confidence=(
            RefinementGateMode.parse(refinement_gate_mode)
            is not RefinementGateMode.TRI_STATE
        ),
    )
    use_punctuation_windows = (
        refinement_gate.mode is RefinementGateMode.TRI_STATE
    )
    numeric_normalizer = ContextualNumericNormalizer()
    selected_asr_api_style = ASRApiStyle.parse(asr_api_style)

    async def request_asr(
        endpoint: str,
        session_id: str | None = None,
        data: bytes = b"",
        params: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return await _stream_request_async(
            asr_url,
            endpoint,
            session_id,
            data,
            params,
            api_style=selected_asr_api_style,
        )

    def managed_entity_store() -> EntityStore:
        if entity_store is None:
            raise HTTPException(
                status_code=409,
                detail="实体数据库未启用；请使用 --entity-db 启动服务。",
            )
        return entity_store

    def refine_update(
        raw_text: str,
        detected_language: str | None,
        final: bool,
        protector: EntityProtector,
        asr_confidence: float | None,
        matcher: EntityCandidateMatcher | None = None,
        *,
        single_window: bool = False,
        confidence_metadata: dict[str, object] | None = None,
        call_stats: _RefinerSessionStats | None = None,
        trace_segments: list[dict[str, object]] | None = None,
        pending_only: bool = False,
    ) -> dict[str, object]:
        confidence_metadata = dict(confidence_metadata or {})
        trace_segment: dict[str, object] | None = None

        def invoke_refiner(
            text: str,
            *,
            hints: tuple[str, ...],
            strict_placeholders: bool = False,
        ) -> tuple[str, float]:
            started_at = (
                call_stats.begin_call(final=final, retry=strict_placeholders)
                if call_stats is not None
                else time.perf_counter()
            )
            if started_at is None:
                raise _RefinerCallsClosed(
                    "refiner calls are closed for this WebSocket session"
                )
            try:
                result = refiner.refine(
                    text,
                    entity_hints=hints,
                    strict_placeholders=strict_placeholders,
                )
            except BaseException as error:
                if trace_segment is not None:
                    trace_segment.setdefault("model_calls", []).append(
                        {"input": text, "strict_placeholders": strict_placeholders,
                         "error": type(error).__name__}
                    )
                if call_stats is not None:
                    call_stats.finish_call(started_at, failed=True)
                raise
            if trace_segment is not None:
                trace_segment.setdefault("model_calls", []).append(
                    {"input": text, "output": result[0],
                     "strict_placeholders": strict_placeholders}
                )
            if call_stats is not None:
                call_stats.finish_call(started_at, failed=False)
            return result

        clean_parts: list[str] = []
        protected_entities: list[dict[str, str]] = []
        entity_hints: list[str] = []
        entity_normalizations: list[dict[str, str]] = []
        numeric_normalizations: list[dict[str, object]] = []
        numeric_fallbacks: list[dict[str, object]] = []
        safe_numeric_repairs: list[dict[str, object]] = []
        safe_repetition_repairs: list[dict[str, object]] = []
        repetition_reviews: list[dict[str, object]] = []
        entity_audit_issues: list[str] = []
        entity_candidates: list[dict[str, object]] = []
        refiner_reject_reasons: list[str] = []
        refiner_masked_outputs: list[str] = []
        structured_patch_audits: list[dict[str, object]] = []
        boundary_anomalies: list[dict[str, object]] = []
        placeholder_retry_count = 0
        refiner_retry_count = 0
        refiner_retry_reasons: list[str] = []
        refinement_gate_decisions: list[dict[str, object]] = []
        refinement_gate_skipped_segments = 0
        refiner_executed = False
        total_latency_ms = 0.0
        matcher_latency_ms = 0.0
        refiner_available = True

        segments = (raw_text,) if single_window and raw_text else split_for_refinement(
            raw_text, one_punctuation_window=use_punctuation_windows
        )
        for segment_index, segment in enumerate(segments):
            prepared = prepare_entity_segment(
                segment,
                protector,
                matcher,
                # Apply high-confidence, low-risk fuzzy normalization in every
                # window.  Streaming hypotheses may still revise their tail,
                # but source-owned cache invalidation will reprocess a revised
                # span; withholding an already verified term until ``finish``
                # made the displayed intermediate text inconsistent with the
                # final result.
                allow_auto=True,
                confidence=asr_confidence,
            )
            protection = prepared.protection
            hints = prepared.hints
            normalized_baseline = (
                numeric_normalizer.normalize(prepared.baseline_text)
                if numeric_normalization
                else None
            )
            baseline_text = (
                normalized_baseline.text
                if normalized_baseline is not None
                else prepared.baseline_text
            )
            boundary_anomalies.extend(
                {
                    "segment_index": segment_index + 1,
                    **anomaly.public_dict(),
                }
                for anomaly in detect_boundary_anomalies(baseline_text)
            )
            masked_text = (
                numeric_normalizer.normalize(protection.masked_text).text
                if numeric_normalization
                else protection.masked_text
            )
            gate_decision = refinement_gate.decide(
                baseline_text,
                asr_confidence=asr_confidence,
                calibrated=(
                    confidence_metadata.get("calibrated") is True
                ),
                covers_segment=(
                    confidence_metadata.get("covers_full_text") is True
                ),
                entity_hints=hints,
                is_final=final,
                numeric_refinement=not numeric_normalization,
            )
            if pending_only and gate_decision.should_refine:
                gate_decision = RefinementGateDecision(
                    mode=gate_decision.mode,
                    should_refine=False,
                    reasons=("incomplete_fixed_group",),
                    visible_chars=gate_decision.visible_chars,
                    asr_confidence=gate_decision.asr_confidence,
                    cleanup_signals=gate_decision.cleanup_signals,
                    calibrated=gate_decision.calibrated,
                    covers_segment=gate_decision.covers_segment,
                    action="defer",
                )
            refinement_gate_decisions.append(
                {"segment_index": segment_index + 1, **gate_decision.public_dict()}
            )
            trace_segment = (
                {"segment_index": segment_index + 1, "source": segment,
                 "baseline": baseline_text, "masked_input": masked_text,
                 "gate": gate_decision.public_dict(), "model_calls": []}
                if trace_segments is not None else None
            )
            if trace_segment is not None:
                trace_segments.append(trace_segment)
            if not gate_decision.should_refine:
                refinement_gate_skipped_segments += 1
                if call_stats is not None:
                    call_stats.record_gate_skip()
                clean_segment = baseline_text
                if not numeric_normalization:
                    approximate_fallback = numeric_normalizer.normalize_approximate(
                        clean_segment
                    )
                    clean_segment = approximate_fallback.text
                    numeric_fallbacks.extend(
                        {
                            **change.public_dict(),
                            "segment_index": segment_index + 1,
                            "reason": "gate_kept_approximate_number",
                        }
                        for change in approximate_fallback.changes
                    )
                changes = list(prepared.normalizations)
                if normalized_baseline is not None:
                    numeric_normalizations.extend(
                        change.public_dict() for change in normalized_baseline.changes
                    )
                clean_parts.append(clean_segment)
                protected_entities.extend(
                    span.public_dict() for span in protection.spans
                )
                entity_hints.extend(hints)
                entity_normalizations.extend(changes)
                entity_candidates.extend(
                    match.public_dict() for match in prepared.report.matches
                )
                matcher_latency_ms += prepared.report.matcher_latency_ms
                entity_audit_issues.extend(
                    protector.audit_unmasked(clean_segment, protection)
                )
                if trace_segment is not None:
                    trace_segment.update({"outcome": "gate_skip", "output": clean_segment})
                continue
            # A stale browser connection must not be able to block the final
            # result forever.  After one timeout, preserve all remaining source
            # segments and report the quality fallback in the response.
            lock_acquired = (
                refiner_available
                and refiner_lock.acquire(timeout=REFINER_LOCK_TIMEOUT_SECONDS)
            )
            if not lock_acquired:
                refiner_available = False
                if call_stats is not None:
                    call_stats.record_busy_segment()
                fallback = numeric_normalizer.normalize(prepared.baseline_text)
                clean_segment = fallback.text
                numeric_fallbacks.extend(
                    {
                        **change.public_dict(),
                        "segment_index": segment_index + 1,
                        "reason": "refiner_unavailable",
                    }
                    for change in fallback.changes
                )
                changes = list(prepared.normalizations)
                if normalized_baseline is not None:
                    numeric_normalizations.extend(
                        change.public_dict() for change in normalized_baseline.changes
                    )
                refiner_reject_reasons.append(
                    f"segment_{segment_index + 1}:refiner_busy"
                )
                if trace_segment is not None:
                    trace_segment.update({"outcome": "refiner_busy", "output": clean_segment})
            else:
                refiner_executed = True
                candidate_for_numeric_salvage: str | None = None
                try:
                    refined_candidate, latency_ms = invoke_refiner(
                        masked_text,
                        hints=hints,
                    )
                    total_latency_ms += latency_ms
                    refiner_masked_outputs.append(refined_candidate)
                    structured_candidate, patch_payload, structured_issue = apply_structured_patch(
                        masked_text, refined_candidate
                    )
                    structured_patch_audits.append(
                        _patch_audit(patch_payload, structured_issue)
                    )
                    if structured_issue:
                        finalized = FinalizedEntitySegment(
                            prepared.baseline_text,
                            False,
                            (structured_issue,),
                        )
                    else:
                        candidate_for_numeric_salvage = structured_candidate
                        finalized = finalize_entity_segment(
                            structured_candidate,
                            prepared,
                            protector,
                        )
                    if has_retryable_integrity_failure(finalized.reject_reasons):
                        initial_reasons = finalized.reject_reasons
                        retry_candidate, retry_latency_ms = invoke_refiner(
                            masked_text,
                            hints=hints,
                            strict_placeholders=True,
                        )
                        refiner_retry_count += 1
                        if has_placeholder_failure(initial_reasons):
                            placeholder_retry_count += 1
                        refiner_retry_reasons.extend(
                            f"segment_{segment_index + 1}:{reason}"
                            for reason in initial_reasons
                        )
                        total_latency_ms += retry_latency_ms
                        refiner_masked_outputs.append(retry_candidate)
                        candidate_for_numeric_salvage = retry_candidate
                        structured_retry, retry_payload, structured_retry_issue = apply_structured_patch(
                            masked_text, retry_candidate
                        )
                        structured_patch_audits.append(
                            _patch_audit(retry_payload, structured_retry_issue)
                        )
                        if structured_retry_issue:
                            finalized = FinalizedEntitySegment(
                                prepared.baseline_text,
                                False,
                                (structured_retry_issue,),
                            )
                        else:
                            finalized = finalize_entity_segment(
                                structured_retry,
                                prepared,
                                protector,
                            )
                finally:
                    refiner_lock.release()
                clean_segment = finalized.text
                salvaged_candidate: str | None = None
                if finalized.reject_reasons and candidate_for_numeric_salvage:
                    restored_candidate = protector.restore(
                        candidate_for_numeric_salvage,
                        protection,
                    )
                    if restored_candidate.accepted:
                        repetition_salvaged = preserve_safe_repetition_edits(
                            prepared.baseline_text,
                            restored_candidate.text,
                        )
                        if repetition_salvaged is not None:
                            safe_repetition_repairs.append(
                                {
                                    "source_text": prepared.baseline_text,
                                    "salvaged_text": repetition_salvaged,
                                    "reason": "safe_adjacent_repetition_after_integrity_fallback",
                                }
                            )
                        numeric_salvaged = preserve_safe_numeric_edits(
                            prepared.baseline_text,
                            restored_candidate.text,
                        )
                        if numeric_salvaged is not None:
                            safe_numeric_repairs.append(
                                {
                                    "source_text": prepared.baseline_text,
                                    "salvaged_text": numeric_salvaged,
                                    "reason": "equivalent_numeric_surface_after_integrity_fallback",
                                }
                            )
                        if repetition_salvaged is not None:
                            combined = (
                                preserve_safe_numeric_edits(
                                    repetition_salvaged,
                                    restored_candidate.text,
                                )
                                or repetition_salvaged
                            )
                            salvaged_candidate = combined
                        elif numeric_salvaged is not None:
                            salvaged_candidate = numeric_salvaged
                if finalized.reject_reasons:
                    # A rejected model result may contain both a wrong value
                    # and a valid conversion elsewhere in the same window.
                    # Use the source-owned text as the authority and run the
                    # deterministic numeric rules over it. If an equivalent
                    # model-only numeric edit was safely salvaged above, use
                    # that as an additional input; no other model edits enter
                    # the fallback result.
                    fallback_source = (
                        salvaged_candidate
                        if salvaged_candidate is not None
                        else prepared.baseline_text
                    )
                    fallback = numeric_normalizer.normalize(fallback_source)
                    clean_segment = fallback.text
                    numeric_fallbacks.extend(
                        {
                            **change.public_dict(),
                            "segment_index": segment_index + 1,
                            "reason": "refiner_rejected",
                            "refiner_reject_reasons": list(
                                finalized.reject_reasons
                            ),
                        }
                        for change in fallback.changes
                    )
                elif not numeric_normalization:
                    approximate_fallback = numeric_normalizer.normalize_approximate(
                        clean_segment
                    )
                    clean_segment = approximate_fallback.text
                    numeric_fallbacks.extend(
                        {
                            **change.public_dict(),
                            "segment_index": segment_index + 1,
                            "reason": "model_omitted_approximate_number",
                        }
                        for change in approximate_fallback.changes
                    )
                changes = list(prepared.normalizations)
                if numeric_normalization:
                    normalized_output = numeric_normalizer.normalize(clean_segment)
                    clean_segment = normalized_output.text
                    segment_changes = (
                        normalized_output.changes
                        if normalized_output.changes
                        else normalized_baseline.changes
                    )
                    numeric_normalizations.extend(
                        change.public_dict() for change in segment_changes
                    )
                if finalized.reject_reasons:
                    refiner_reject_reasons.extend(
                        f"segment_{segment_index + 1}:{reason}"
                        for reason in finalized.reject_reasons
                    )
                if trace_segment is not None:
                    trace_segment.update({
                        "outcome": "rejected" if finalized.reject_reasons else "accepted",
                        "reject_reasons": list(finalized.reject_reasons),
                        "output": clean_segment,
                    })
            clean_parts.append(clean_segment)
            protected_entities.extend(span.public_dict() for span in protection.spans)
            entity_hints.extend(hints)
            entity_normalizations.extend(changes)
            entity_candidates.extend(
                match.public_dict() for match in prepared.report.matches
            )
            matcher_latency_ms += prepared.report.matcher_latency_ms
            entity_audit_issues.extend(protector.audit_unmasked(clean_segment, protection))
        clean_text = clean_transcript_deterministically(
            join_refined_segments(clean_parts)
        )
        return {
            "event": "final" if final else "update",
            "raw_text": raw_text,
            "clean_text": clean_text,
            "asr_language": detected_language,
            "asr_confidence": asr_confidence,
            "asr_confidence_metadata": confidence_metadata,
            "refiner_latency_ms": round(total_latency_ms),
            "refiner_executed": refiner_executed,
            "refiner_accepted": not refiner_reject_reasons,
            "refiner_reject_reasons": refiner_reject_reasons,
            "placeholder_retry_count": placeholder_retry_count,
            "refiner_retry_count": refiner_retry_count,
            "refiner_retry_reasons": refiner_retry_reasons,
            "refiner_masked_outputs": refiner_masked_outputs,
            "structured_patch_audits": structured_patch_audits,
            "boundary_anomalies": boundary_anomalies,
            "boundary_reviews": [],
            "refinement_gate_mode": refinement_gate.mode.value,
            "refinement_gate_config": refinement_gate.config_dict(),
            "refinement_gate_decisions": refinement_gate_decisions,
            "refinement_gate_skipped_segments": refinement_gate_skipped_segments,
            "entity_audit_issues": list(dict.fromkeys(entity_audit_issues)),
            "entity_refinement_hints": list(dict.fromkeys(entity_hints)),
            "entity_normalizations": entity_normalizations,
            "numeric_normalizations": numeric_normalizations,
            "numeric_fallbacks": numeric_fallbacks,
            "safe_numeric_repairs": safe_numeric_repairs,
            "safe_repetition_repairs": safe_repetition_repairs,
            "repetition_reviews": repetition_reviews,
            "boundary_echo_repairs": [],
            "protected_entities": protected_entities,
            "entity_candidates": entity_candidates,
            "entity_matcher_latency_ms": round(matcher_latency_ms, 3),
            "entity_fuzzy_mode": fuzzy_mode.value,
            "entity_matching_config_version": (
                matcher.config.version if matcher is not None else None
            ),
            "rule_protection_enabled": rule_protection,
            "numeric_normalization_enabled": numeric_normalization,
        }

    def _patch_audit(
        payload: dict[str, object] | None, issue: str | None
    ) -> dict[str, object]:
        if payload is None:
            return {
                "format": "legacy_text",
                "accepted": issue is None,
                "issue": issue,
            }
        return {
            "format": "json_patch",
            "action": payload.get("action"),
            "source": payload.get("source", ""),
            "target": payload.get("target", ""),
            "reason": payload.get("reason", ""),
            "accepted": issue is None,
            "issue": issue,
        }

    def fallback_update(
        raw_text: str,
        detected_language: str | None,
        protector: EntityProtector,
        asr_confidence: float | None,
        matcher: EntityCandidateMatcher | None,
        reason: str,
        started_at: float,
        confidence_metadata: dict[str, object] | None = None,
        apply_numeric_fallback: bool = False,
    ) -> dict[str, object]:
        # Preserve the complete ASR text and still apply deterministic entity
        # normalization when the neural Refiner exceeds the final deadline.
        clean_parts: list[str] = []
        protected_entities: list[dict[str, str]] = []
        entity_hints: list[str] = []
        entity_normalizations: list[dict[str, str]] = []
        numeric_normalizations: list[dict[str, object]] = []
        numeric_fallbacks: list[dict[str, object]] = []
        entity_audit_issues: list[str] = []
        entity_candidates: list[dict[str, object]] = []
        matcher_latency_ms = 0.0
        for segment in split_for_refinement(
            raw_text, one_punctuation_window=use_punctuation_windows
        ):
            prepared = prepare_entity_segment(
                segment,
                protector,
                matcher,
                allow_auto=True,
                confidence=asr_confidence,
            )
            protection = prepared.protection
            normalized = (
                numeric_normalizer.normalize(prepared.baseline_text)
                if numeric_normalization or apply_numeric_fallback
                else None
            )
            clean_parts.append(
                normalized.text if normalized is not None else prepared.baseline_text
            )
            if normalized is not None:
                if numeric_normalization:
                    numeric_normalizations.extend(
                        change.public_dict() for change in normalized.changes
                    )
                elif apply_numeric_fallback:
                    numeric_fallbacks.extend(
                        {
                            **change.public_dict(),
                            "reason": "refiner_unavailable",
                        }
                        for change in normalized.changes
                    )
            protected_entities.extend(span.public_dict() for span in protection.spans)
            entity_hints.extend(prepared.hints)
            entity_normalizations.extend(prepared.normalizations)
            entity_candidates.extend(
                match.public_dict() for match in prepared.report.matches
            )
            matcher_latency_ms += prepared.report.matcher_latency_ms
            entity_audit_issues.extend(
                protector.audit_unmasked(prepared.baseline_text, protection)
            )
        return {
            "event": "final",
            "raw_text": raw_text,
            "clean_text": clean_transcript_deterministically(
                join_refined_segments(clean_parts)
            ),
            "asr_language": detected_language,
            "asr_confidence": asr_confidence,
            "asr_confidence_metadata": dict(confidence_metadata or {}),
            "refiner_latency_ms": round((time.perf_counter() - started_at) * 1000),
            "refiner_executed": True,
            "refiner_accepted": False,
            "refiner_reject_reasons": [reason],
            "placeholder_retry_count": 0,
            "refiner_retry_count": 0,
            "refiner_retry_reasons": [],
            "refiner_masked_outputs": [],
            "structured_patch_audits": [],
            "boundary_reviews": [],
            "boundary_anomalies": [
                anomaly
                for segment in split_for_refinement(
                    raw_text, one_punctuation_window=use_punctuation_windows
                )
                for anomaly in (
                    {
                        "segment_text": segment,
                        **item.public_dict(),
                    }
                    for item in detect_boundary_anomalies(segment)
                )
            ],
            "refinement_gate_mode": refinement_gate.mode.value,
            "refinement_gate_config": refinement_gate.config_dict(),
            "refinement_gate_decisions": [],
            "refinement_gate_skipped_segments": 0,
            "entity_audit_issues": list(dict.fromkeys(entity_audit_issues)),
            "entity_refinement_hints": list(dict.fromkeys(entity_hints)),
            "entity_normalizations": entity_normalizations,
            "numeric_normalizations": numeric_normalizations,
            "numeric_fallbacks": numeric_fallbacks,
            "safe_numeric_repairs": [],
            "safe_repetition_repairs": [],
            "repetition_reviews": [],
            "boundary_echo_repairs": [],
            "protected_entities": protected_entities,
            "entity_candidates": entity_candidates,
            "entity_matcher_latency_ms": round(matcher_latency_ms, 3),
            "entity_fuzzy_mode": fuzzy_mode.value,
            "entity_matching_config_version": (
                matcher.config.version if matcher is not None else None
            ),
            "rule_protection_enabled": rule_protection,
            "numeric_normalization_enabled": numeric_normalization,
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            WEB_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "entity_db": entity_store is not None,
            "refinement_gate_mode": refinement_gate.mode.value,
            "asr_api_style": selected_asr_api_style.value,
            "numeric_normalization_enabled": numeric_normalization,
        }

    @app.get("/api/entities")
    def list_entities(
        query: str = "",
        entity_type: str | None = None,
        include_disabled: bool = True,
    ) -> dict[str, object]:
        definitions = managed_entity_store().list_entities(include_disabled=include_disabled)
        keyword = query.strip().casefold()
        type_filter = entity_type.strip().upper() if entity_type and entity_type.strip() else None
        filtered = [
            definition
            for definition in definitions
            if (
                not keyword
                or keyword in definition.canonical_text.casefold()
                or any(keyword in alias.casefold() for alias in definition.aliases)
            )
            and (type_filter is None or definition.entity_type == type_filter)
        ]
        return {"entities": [_entity_dict(definition) for definition in filtered]}

    @app.post("/api/entities", status_code=201)
    def create_entity(payload: EntityInput) -> dict[str, object]:
        store = managed_entity_store()
        try:
            store.get_entity(payload.canonical_text.strip(), payload.domain.strip())
        except KeyError:
            pass
        else:
            raise HTTPException(
                status_code=409,
                detail="该标准名称与领域组合已存在，请使用编辑操作。",
            )
        try:
            definition = store.upsert_entity(
                payload.canonical_text,
                entity_type=payload.entity_type,
                domain=payload.domain,
                normalization_policy=payload.normalization_policy,
                priority=payload.priority,
                aliases=payload.aliases,
                source="ui",
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _entity_dict(definition)

    @app.put("/api/entities/{entity_id}")
    def update_entity(entity_id: int, payload: EntityInput) -> dict[str, object]:
        try:
            definition = managed_entity_store().update_entity(
                entity_id,
                payload.canonical_text,
                entity_type=payload.entity_type,
                domain=payload.domain,
                normalization_policy=payload.normalization_policy,
                priority=payload.priority,
                aliases=payload.aliases,
                source="ui",
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="实体不存在或已被删除。") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _entity_dict(definition)

    @app.patch("/api/entities/{entity_id}/enabled")
    def set_entity_enabled(entity_id: int, payload: EntityEnabledInput) -> dict[str, object]:
        store = managed_entity_store()
        if not store.set_enabled_by_id(entity_id, enabled=payload.enabled):
            raise HTTPException(status_code=404, detail="实体不存在或已被删除。")
        return _entity_dict(store.get_entity_by_id(entity_id))

    @app.delete("/api/entities/{entity_id}")
    def delete_entity(entity_id: int) -> dict[str, bool]:
        if not managed_entity_store().delete_entity(entity_id):
            raise HTTPException(status_code=404, detail="实体不存在或已被删除。")
        return {"deleted": True}

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()
        connection_open = True

        async def send_json(payload: dict[str, object]) -> bool:
            nonlocal connection_open
            if not connection_open:
                return False
            async with send_lock:
                if not connection_open:
                    return False
                try:
                    await asyncio.wait_for(
                        websocket.send_json(payload),
                        timeout=WEBSOCKET_SEND_TIMEOUT_SECONDS,
                    )
                except (RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                    connection_open = False
                    return False
            return True

        requested_language = websocket.query_params.get("language") or language
        requested_mode = (websocket.query_params.get("mode") or "online").lower()
        requested_domain = (websocket.query_params.get("domain") or "general").strip() or "general"
        if requested_mode not in {"online", "offline", "streaming"}:
            await send_json(
                {"event": "error", "detail": "mode must be online, offline, or streaming"}
            )
            await websocket.close()
            return
        session_id: str | None = None
        last_raw_text = ""
        finished = False
        received_chunk_count = 0
        asr_activity: dict[str, object] = {}
        session_memory = SessionEntityMemory()
        refiner_session_stats = _RefinerSessionStats()
        hypothesis_tracker = HypothesisTracker()
        trace_id = uuid.uuid4().hex
        trace_path = (
            output.with_name(f"{output.stem}.stream_trace.jsonl")
            if output is not None else None
        )
        trace_lock = threading.Lock()
        trace_context: dict[str, object] = {"phase": "stream", "revision": None}

        def record_trace(event: str, **details: object) -> None:
            if trace_path is None:
                return
            try:
                with trace_lock:
                    _append_record(trace_path, {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "trace_id": trace_id,
                        "event": event,
                        **details,
                    })
            except (OSError, TypeError, ValueError):
                # Diagnostics must never change a transcription decision.
                pass

        def attach_refiner_session_stats(
            result: dict[str, object], *, close: bool = False
        ) -> dict[str, object]:
            if close:
                refiner_session_stats.close_new_calls()
            snapshot = refiner_session_stats.snapshot()
            result["refiner_session_stats"] = snapshot
            # Flat aliases keep JSONL analysis and the browser console simple.
            result["session_refiner_call_count"] = snapshot["call_count"]
            result["session_refiner_initial_call_count"] = snapshot[
                "initial_call_count"
            ]
            result["session_refiner_retry_call_count"] = snapshot[
                "retry_call_count"
            ]
            result["session_refiner_intermediate_call_count"] = snapshot[
                "intermediate_call_count"
            ]
            result["session_refiner_final_call_count"] = snapshot[
                "final_call_count"
            ]
            result["session_gate_skipped_segment_count"] = snapshot[
                "gate_skipped_segment_count"
            ]
            return result

        def session_refine_update(*args, **kwargs) -> dict[str, object]:
            kwargs["call_stats"] = refiner_session_stats
            segments: list[dict[str, object]] = []
            phase = trace_context["phase"]
            revision = trace_context["revision"]
            if trace_path is not None:
                kwargs["trace_segments"] = segments
            try:
                result = refine_update(*args, **kwargs)
            except Exception as error:
                record_trace(
                    "window_error", phase=phase, revision=revision,
                    input=args[0] if args else None, segments=segments,
                    error=type(error).__name__,
                )
                raise
            record_trace(
                "window_result", phase=phase, revision=revision,
                input=args[0] if args else None, segments=segments,
                output=result.get("clean_text"),
                reject_reasons=result.get("refiner_reject_reasons"),
            )
            return result

        def session_prepare_pending(*args, **kwargs) -> dict[str, object]:
            kwargs["pending_only"] = True
            return session_refine_update(*args, **kwargs)

        definitions = (
            entity_store.list_entities(domain=requested_domain)
            if entity_store is not None
            else ()
        )
        protector = EntityProtector(
            definitions,
            session_memory=session_memory,
            enable_rule_protection=rule_protection,
            # Numeric values are normalized deterministically before the
            # Refiner and verified again afterwards.  Do not expose them as
            # fragile __ENTITY_NNN__ placeholders.
            protect_numeric_spans=False,
        )
        matcher = (
            EntityCandidateMatcher(
                definitions,
                selected_domain=requested_domain,
                mode=fuzzy_mode,
            )
            if definitions and fuzzy_mode is not EntityFuzzyMode.OFF
            else None
        )
        pending_refinement: tuple[
            str,
            str | None,
            float | None,
            dict[str, object],
            str,
            int,
            int,
            tuple[str, ...] | None,
        ] | None = None
        streaming_refiner_task: asyncio.Task[None] | None = None
        streaming_finish_requested = False
        latest_refinement_revision = 0
        last_refinement_started_at: float | None = None
        vad_source_segments: list[str] = []
        seen_vad_segment_ids: set[int] = set()
        # In tri_state, wait for three punctuation chunks and refine them as
        # one source-owned group. Legacy modes retain rolling K=3 behavior.
        window_refinement = CumulativeWindowRefinement(
            session_refine_update,
            window_size=3,
            one_punctuation_window=use_punctuation_windows,
            fixed_groups=use_punctuation_windows,
            prepare_pending=session_prepare_pending if use_punctuation_windows else None,
        )
        refinement_display = StreamingRefinementDisplay()
        streaming_repetition_cache: dict[
            tuple[int, int, str, str, str], dict[str, object]
        ] = {}
        streaming_repetition_reviews: list[dict[str, object]] = []

        def consume_completed_segments(payload: dict[str, object]) -> int:
            """Consume request-scoped ASR segment events exactly once."""

            events = payload.get("completed_segments")
            if not isinstance(events, list):
                return 0
            added = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                segment_id = event.get("segment_id")
                text_value = event.get("text")
                if (
                    not isinstance(segment_id, int)
                    or isinstance(segment_id, bool)
                    or segment_id < 1
                    or segment_id in seen_vad_segment_ids
                    or not isinstance(text_value, str)
                    or not text_value.strip()
                ):
                    continue
                seen_vad_segment_ids.add(segment_id)
                vad_source_segments.append(text_value.strip())
                added += 1
            return added

        def transcript_event(
            raw_text: str,
            detected_language: str | None,
            asr_confidence: float | None,
            confidence_metadata: dict[str, object],
            *,
            deferred: bool,
            gate_decision: RefinementGateDecision | None = None,
            stability: HypothesisStability | None = None,
        ) -> dict[str, object]:
            """Publish raw ASR immediately while retaining valid refinements."""

            payload: dict[str, object] = {
                "event": "transcript",
                "raw_text": raw_text,
                "asr_language": detected_language,
                "asr_confidence": asr_confidence,
                "asr_confidence_metadata": confidence_metadata,
                "refiner_deferred": deferred,
            }
            if gate_decision is not None:
                public_gate = gate_decision.public_dict()
                payload["refinement_gate_action"] = public_gate["action"]
                payload["refinement_gate_state"] = public_gate["state"]
                payload["refinement_gate_reasons"] = public_gate["reasons"]
            if stability is not None:
                payload["refinement_hypothesis"] = {
                    "stable": stability.stable,
                    "unchanged_updates": stability.unchanged_updates,
                    "revision_ratio": round(stability.revision_ratio, 4),
                    "tail_age_ms": round(stability.tail_age_ms, 1),
                    "sentence_complete": stability.sentence_complete,
                }
            payload.update(refinement_display.compose(raw_text))
            return payload

        def intermediate_gate(
            raw_text: str,
            asr_confidence: float | None,
            confidence_metadata: dict[str, object],
        ) -> tuple[HypothesisStability, RefinementGateDecision | None]:
            stability = hypothesis_tracker.observe(raw_text)
            if refinement_gate.mode is not RefinementGateMode.TRI_STATE:
                return stability, None
            decision = refinement_gate.decide(
                raw_text,
                asr_confidence=asr_confidence,
                calibrated=confidence_metadata.get("calibrated") is True,
                covers_segment=confidence_metadata.get("covers_full_text") is True,
                is_final=False,
                stable=stability.stable,
                tail_age_ms=stability.tail_age_ms,
                revision_ratio=stability.revision_ratio,
                unchanged_updates=stability.unchanged_updates,
            )
            return stability, decision

        def finalize_streaming_windows(
            raw_text: str,
            detected_language: str | None,
            final: bool,
            protector: EntityProtector,
            asr_confidence: float | None,
            confidence_metadata: dict[str, object],
            matcher_value: EntityCandidateMatcher | None,
            source_segments: tuple[str, ...] | None = None,
        ) -> dict[str, object]:
            """Reuse committed streaming refinements and finalize only the tail.

            Intermediate and final windows both allow high-confidence, low-risk
            fuzzy entity normalization.  The source-owned window cache still
            invalidates any span revised by ASR, while the final entity-only
            fallback remains as an idempotent consistency pass.
            """

            # Finalize through the same sentence-bounded window path even if
            # the client finished before an intermediate refinement completed.
            # This keeps online, offline, and streaming modes from sending the
            # entire final hypothesis to the Refiner in one request.
            if source_segments:
                result = window_refinement.update_segments(
                    source_segments,
                    detected_language,
                    final,
                    protector,
                    asr_confidence,
                    matcher_value,
                    confidence_metadata=confidence_metadata,
                    raw_text=raw_text,
                )
            else:
                result = window_refinement.update(
                    raw_text,
                    detected_language,
                    final,
                    protector,
                    asr_confidence,
                    matcher_value,
                    confidence_metadata=confidence_metadata,
                )
            result["asr_confidence"] = asr_confidence
            result["asr_confidence_metadata"] = dict(confidence_metadata)
            entity_result = fallback_update(
                str(result["clean_text"]),
                detected_language,
                protector,
                asr_confidence,
                matcher_value,
                "streaming_final_entity_normalization",
                time.perf_counter(),
                confidence_metadata,
            )
            result["clean_text"] = entity_result["clean_text"]
            result["entity_audit_issues"] = list(
                dict.fromkeys(
                    [
                        *result.get("entity_audit_issues", []),
                        *entity_result["entity_audit_issues"],
                    ]
                )
            )
            result["entity_refinement_hints"] = list(
                dict.fromkeys(
                    [
                        *result.get("entity_refinement_hints", []),
                        *entity_result["entity_refinement_hints"],
                    ]
                )
            )
            result["entity_normalizations"] = [
                *result.get("entity_normalizations", []),
                *entity_result["entity_normalizations"],
            ]
            result["numeric_normalizations"] = [
                *result.get("numeric_normalizations", []),
                *entity_result["numeric_normalizations"],
            ]
            result["numeric_fallbacks"] = [
                *result.get("numeric_fallbacks", []),
                *entity_result["numeric_fallbacks"],
            ]
            result["protected_entities"] = [
                *result.get("protected_entities", []),
                *entity_result["protected_entities"],
            ]
            result["entity_candidates"] = [
                *entity_result["entity_candidates"],
                *result.get("entity_candidates", []),
            ]
            result["entity_matcher_latency_ms"] = round(
                float(result.get("entity_matcher_latency_ms", 0.0))
                + float(entity_result["entity_matcher_latency_ms"]),
                3,
            )
            reviewed_text, review_records, review_latency_ms, review_executed = (
                review_final_repetition_text(
                    str(result["clean_text"]),
                    final=final,
                    call_stats=refiner_session_stats,
                    review_cache=streaming_repetition_cache,
                )
            )
            result["clean_text"] = clean_transcript_deterministically(
                reviewed_text
            )
            result["repetition_reviews"] = [
                *streaming_repetition_reviews,
                *result.get("repetition_reviews", []),
                *review_records,
            ]
            result["refiner_latency_ms"] = round(
                float(result.get("refiner_latency_ms", 0.0)) + review_latency_ms
            )
            result["refiner_executed"] = bool(
                result.get("refiner_executed", False) or review_executed
            )
            return result

        async def process_audio_chunk(audio: bytes) -> tuple[dict[str, object], int]:
            """Forward one chunk while keeping a slow, healthy session visible."""

            nonlocal received_chunk_count
            received_chunk_count += 1
            chunk_index = received_chunk_count
            started_at = time.perf_counter()
            task = asyncio.create_task(
                request_asr("/stream/chunk", session_id, audio)
            )
            while True:
                try:
                    payload = await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=ASR_CHUNK_STATUS_INTERVAL_SECONDS,
                    )
                    return payload, chunk_index
                except asyncio.TimeoutError:
                    if not await send_json(
                        {
                            "event": "status",
                            "stage": "asr_chunk",
                            "chunk_index": chunk_index,
                            "elapsed_seconds": round(
                                time.perf_counter() - started_at
                            ),
                        }
                    ):
                        raise RuntimeError(
                            "client disconnected while ASR chunk was processing"
                        )

        async def run_streaming_refiner() -> None:
            nonlocal pending_refinement, streaming_finish_requested
            nonlocal latest_refinement_revision, last_refinement_started_at
            while pending_refinement is not None:
                # Refine only a bounded tail of the cumulative ASR hypothesis;
                # committed results stay cached for finalization.
                if streaming_finish_requested:
                    pending_refinement = None
                    return
                if last_refinement_started_at is not None:
                    remaining = (
                        STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS
                        - (time.perf_counter() - last_refinement_started_at)
                    )
                    if remaining > 0:
                        # Newer hypotheses keep replacing the pending request
                        # while we wait, so the next pass always refines the
                        # freshest cumulative transcript.
                        await asyncio.sleep(remaining)
                        if streaming_finish_requested or pending_refinement is None:
                            pending_refinement = None
                            return
                (
                    tail_value,
                    language_value,
                    confidence_value,
                    confidence_metadata_value,
                    full_raw_value,
                    tail_start,
                    revision,
                    source_segments_value,
                ) = pending_refinement
                pending_refinement = None
                # Claim the revision only now that this pass really starts.
                # One pass runs at a time, so the publish guard below can no
                # longer see a newer revision and discard a finished result.
                latest_refinement_revision = revision
                last_refinement_started_at = time.perf_counter()
                trace_context.update(phase="stream", revision=revision)
                record_trace(
                    "pass_started", revision=revision,
                    raw_text=full_raw_value, tail=tail_value,
                    source_segments=source_segments_value,
                )
                try:
                    if source_segments_value:
                        owned_raw_value = join_refined_segments(
                            source_segments_value
                        )
                        result = await asyncio.to_thread(
                            window_refinement.update_segments,
                            source_segments_value,
                            language_value,
                            not use_punctuation_windows,
                            protector,
                            confidence_value,
                            matcher,
                            confidence_metadata=confidence_metadata_value,
                            raw_text=owned_raw_value,
                        )
                    else:
                        result = await asyncio.to_thread(
                            window_refinement.update,
                            full_raw_value,
                            language_value,
                            False,
                            protector,
                            confidence_value,
                            matcher,
                            confidence_metadata=confidence_metadata_value,
                        )
                    # Repetition review belongs to the streaming result, not
                    # only to a VAD-finalized source segment.  Some ASR
                    # backends expose VAD boundaries only on ``finish``; if
                    # this remains conditional on ``source_segments_value``,
                    # punctuation-separated repetitions stay visible until
                    # the final pass even though the active window is ready.
                    before_review = str(result["clean_text"])
                    if use_punctuation_windows and result["committed_chunks"] == 0:
                        # The current ASR hypothesis has not produced a full
                        # K=3 group.  Keep its source-owned text pending until
                        # the third punctuation chunk is available.
                        reviewed_text = before_review
                        review_records = []
                        review_latency_ms = 0.0
                        review_executed = False
                    else:
                        (
                            reviewed_text,
                            review_records,
                            review_latency_ms,
                            review_executed,
                        ) = await asyncio.to_thread(
                            review_final_repetition_text,
                            before_review,
                            final=False,
                            streaming=True,
                            call_stats=refiner_session_stats,
                            review_cache=streaming_repetition_cache,
                        )
                    result["clean_text"] = clean_transcript_deterministically(
                        reviewed_text
                    )
                    reviewed_spans = _reviewed_source_spans(
                        result.get("refinement_source_spans"),
                        before_review,
                        str(result["clean_text"]),
                        review_records,
                    )
                    if reviewed_spans is not None:
                        result["refinement_source_spans"] = reviewed_spans
                    record_trace(
                        "stream_review", revision=revision,
                        input=before_review, model_review_output=reviewed_text,
                        output=result["clean_text"], reviews=review_records,
                        source_spans_updated=reviewed_spans is not None,
                    )
                    result["repetition_reviews"] = [
                        *result.get("repetition_reviews", []),
                        *review_records,
                    ]
                    result["refiner_latency_ms"] = round(
                        float(result.get("refiner_latency_ms", 0.0))
                        + review_latency_ms
                    )
                    result["refiner_executed"] = bool(
                        result.get("refiner_executed", False)
                        or review_executed
                    )
                    streaming_repetition_reviews.extend(review_records)
                    # A finish request may be waiting for this pass.  Publish
                    # it unless the pass was explicitly invalidated after a
                    # timeout or replaced by a newer hypothesis.
                    if streaming_finish_requested or revision != latest_refinement_revision:
                        record_trace(
                            "pass_not_published", revision=revision,
                            reason=("finish_requested" if streaming_finish_requested
                                    else "superseded"),
                            output=result.get("clean_text"),
                        )
                        continue
                    result["refinement_revision"] = revision
                    result["event"] = "update"
                    refinement_display.accept(result, revision)
                    result["raw_text"] = full_raw_value
                    # ASR may have advanced while this model call was running.
                    # Publish the real result for history/metrics, but render it
                    # together with the newest source-owned pending tail.
                    result.update(
                        refinement_display.compose(last_raw_text or full_raw_value)
                    )
                    attach_refiner_session_stats(result)
                    sent = await send_json(result)
                    record_trace(
                        "pass_published" if sent else "pass_send_failed",
                        revision=revision, output=result.get("clean_text"),
                        display_refined_text=result.get("display_refined_text"),
                        pending_raw_text=result.get("pending_raw_text"),
                        source_spans=result.get("refinement_source_spans"),
                    )
                    if not sent:
                        return
                except Exception as error:
                    record_trace(
                        "pass_error", revision=revision,
                        error=type(error).__name__, detail=str(error),
                    )
                    if (
                        not streaming_finish_requested
                        and revision == latest_refinement_revision
                    ):
                        await send_json({"event": "error", "detail": str(error)})
                if streaming_finish_requested:
                    pending_refinement = None
                    return

        def queue_streaming_refinement(
            tail_value: str,
            language_value: str | None,
            confidence_value: float | None,
            confidence_metadata_value: dict[str, object],
            full_raw_value: str,
            tail_start: int,
            source_segments_value: tuple[str, ...] | None = None,
        ) -> None:
            nonlocal pending_refinement, streaming_refiner_task
            # Enqueueing must not invalidate the pass that is already
            # running: the revision is claimed in run_streaming_refiner.
            revision = latest_refinement_revision + 1
            record_trace(
                "pass_queued", revision=revision,
                replaced_pending=pending_refinement is not None,
                raw_text=full_raw_value, tail=tail_value,
                source_segments=source_segments_value,
            )
            pending_refinement = (
                tail_value,
                language_value,
                confidence_value,
                confidence_metadata_value,
                full_raw_value,
                tail_start,
                revision,
                source_segments_value,
            )
            if streaming_refiner_task is None or streaming_refiner_task.done():
                streaming_refiner_task = asyncio.create_task(run_streaming_refiner())
        async def refine_final(
            raw_text: str,
            detected_language: str | None,
            protector: EntityProtector,
            asr_confidence: float | None,
            confidence_metadata: dict[str, object],
            source_segments: tuple[str, ...] | None = None,
        ) -> dict[str, object]:
            started_at = time.perf_counter()
            # All browser modes use the same bounded sentence-window finalizer;
            # the mode only controls when intermediate updates are published.
            finalizer = finalize_streaming_windows
            task = asyncio.create_task(
                asyncio.to_thread(
                    finalizer,
                    raw_text,
                    detected_language,
                    True,
                    protector,
                    asr_confidence,
                    confidence_metadata,
                    matcher,
                    source_segments,
                )
            )
            loop = asyncio.get_running_loop()
            segment_count = max(
                1,
                len(
                    split_for_refinement(
                        raw_text,
                        one_punctuation_window=use_punctuation_windows,
                    )
                ),
            )
            deadline_seconds = min(
                FINAL_REFINEMENT_MAX_TIMEOUT_SECONDS,
                max(
                    FINAL_REFINEMENT_TIMEOUT_SECONDS,
                    15.0 + segment_count * 8.0,
                ),
            )
            deadline = loop.time() + deadline_seconds
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    task.cancel()
                    raise asyncio.TimeoutError
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task), timeout=min(8.0, remaining)
                    )
                except asyncio.TimeoutError:
                    # Keep reverse proxies and the browser informed while a
                    # long transcript is being refined.
                    await send_json(
                        attach_refiner_session_stats(
                            {
                                "event": "status",
                                "stage": "final_refinement",
                                "elapsed_seconds": round(
                                    time.perf_counter() - started_at
                                ),
                            }
                        )
                    )

        try:
            start = await request_asr(
                "/stream/start",
                params={"language": requested_language} if requested_language else None,
            )
            session_id = str(start["session_id"])
            await send_json({"event": "ready"})
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    connection_open = False
                    break
                text = message.get("text")
                if text:
                    command = json.loads(text)
                    if command.get("event") != "finish":
                        continue
                    payload = await request_asr("/stream/finish", session_id)
                    if requested_mode in {"online", "offline", "streaming"}:
                        # Let the in-flight streaming pass publish before the
                        # final pass starts.  Previously the finish command
                        # invalidated it immediately, so a repetition found
                        # by the streaming reviewer was discarded and only
                        # the final Refiner could change the visible text.
                        if (
                            streaming_refiner_task is not None
                            and not streaming_refiner_task.done()
                        ):
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(streaming_refiner_task),
                                    timeout=REFINER_LOCK_TIMEOUT_SECONDS,
                                )
                            except asyncio.TimeoutError:
                                # The final pass remains the safe fallback for
                                # a slow streaming generation.  Invalidate its
                                # revision so a late result cannot overwrite
                                # the final text.
                                latest_refinement_revision += 1
                                streaming_finish_requested = True
                            except Exception:
                                streaming_finish_requested = True
                        if not streaming_finish_requested:
                            streaming_finish_requested = True
                        pending_refinement = None
                    consume_completed_segments(payload)
                    asr_activity = {
                        key: payload.get(key)
                        for key in (
                            "vad",
                            "vad_speech_detected",
                            "vad_segment_ended",
                            "vad_skipped_chunks",
                            "silence_skipped_chunks",
                            "asr_segment_count",
                            "segmentation_mode",
                        )
                        if key in payload
                    }
                    final_raw = str(payload.get("text", "")).strip()
                    final_source_segments = tuple(vad_source_segments) or None
                    if (
                        final_source_segments is not None
                        and join_refined_segments(final_source_segments) != final_raw
                    ):
                        # A mixed/older backend may omit an event. Never let an
                        # incomplete event stream drop text from final output.
                        final_source_segments = None
                    last_raw_text = final_raw
                    detected_language = payload.get("language")
                    asr_confidence = _optional_confidence(payload.get("confidence"))
                    confidence_metadata = _confidence_metadata(payload)
                    if final_raw:
                        # Publish the complete ASR result before starting the
                        # potentially slow neural pass. This keeps the raw
                        # transcript available even while refinement is busy.
                        if not await send_json(
                            transcript_event(
                                final_raw,
                                (
                                    detected_language
                                    if isinstance(detected_language, str)
                                    else None
                                ),
                                asr_confidence,
                                confidence_metadata,
                                deferred=True,
                            )
                        ):
                            finished = True
                            break
                        refinement_started = time.perf_counter()
                        trace_context.update(phase="final", revision=latest_refinement_revision)
                        record_trace(
                            "final_started", revision=latest_refinement_revision,
                            raw_text=final_raw,
                            source_segments=final_source_segments,
                        )
                        try:
                            result = await refine_final(
                                final_raw,
                                detected_language if isinstance(detected_language, str) else None,
                                protector,
                                asr_confidence,
                                confidence_metadata,
                                final_source_segments,
                            )
                        except asyncio.TimeoutError:
                            refiner_session_stats.close_new_calls()
                            result = fallback_update(
                                final_raw,
                                detected_language if isinstance(detected_language, str) else None,
                                protector,
                                asr_confidence,
                                matcher,
                                "final_refinement_timeout",
                                refinement_started,
                                confidence_metadata,
                                apply_numeric_fallback=True,
                            )
                        except Exception as error:
                            refiner_session_stats.close_new_calls()
                            # A model/runtime failure must not erase a
                            # complete ASR result that was already published.
                            result = fallback_update(
                                final_raw,
                                detected_language if isinstance(detected_language, str) else None,
                                protector,
                                asr_confidence,
                                matcher,
                                f"final_refinement_error:{type(error).__name__}",
                                refinement_started,
                                confidence_metadata,
                                apply_numeric_fallback=True,
                            )
                        result["refinement_revision"] = latest_refinement_revision
                        attach_refiner_session_stats(result, close=True)
                        record_trace(
                            "final_result", revision=latest_refinement_revision,
                            raw_text=final_raw,
                            output=result.get("clean_text"),
                            reviews=result.get("repetition_reviews"),
                            reject_reasons=result.get("refiner_reject_reasons"),
                        )
                        if output is not None:
                            _append_record(
                                output,
                                {
                                    "captured_at": datetime.now(timezone.utc).isoformat(),
                                    "mode": requested_mode,
                                    "entity_domain": requested_domain,
                                    "asr_language": result["asr_language"],
                                    "asr_confidence": result["asr_confidence"],
                                    "asr_confidence_metadata": result[
                                        "asr_confidence_metadata"
                                    ],
                                    "asr_activity": asr_activity,
                                    "output": {
                                        "raw_text": result["raw_text"],
                                        "clean_text": result["clean_text"],
                                        "refinement_revision": result.get(
                                            "refinement_revision"
                                        ),
                                        "llm_latency_ms": result["refiner_latency_ms"],
                                        "refiner_executed": result["refiner_executed"],
                                        "refiner_accepted": result["refiner_accepted"],
                                        "refiner_reject_reasons": result[
                                            "refiner_reject_reasons"
                                        ],
                                        "placeholder_retry_count": result[
                                            "placeholder_retry_count"
                                        ],
                                        "refiner_retry_count": result[
                                            "refiner_retry_count"
                                        ],
                                        "refiner_retry_reasons": result[
                                            "refiner_retry_reasons"
                                        ],
                                        "refiner_masked_outputs": result[
                                            "refiner_masked_outputs"
                                        ],
                                        "structured_patch_audits": result[
                                            "structured_patch_audits"
                                        ],
                                        "boundary_anomalies": result[
                                            "boundary_anomalies"
                                        ],
                                        "boundary_reviews": result.get(
                                            "boundary_reviews", []
                                        ),
                                        "refinement_gate_mode": result[
                                            "refinement_gate_mode"
                                        ],
                                        "refinement_gate_config": result[
                                            "refinement_gate_config"
                                        ],
                                        "refinement_gate_decisions": result[
                                            "refinement_gate_decisions"
                                        ],
                                        "refinement_gate_skipped_segments": result[
                                            "refinement_gate_skipped_segments"
                                        ],
                                        "entity_audit_issues": result[
                                            "entity_audit_issues"
                                        ],
                                        "entity_refinement_hints": result[
                                            "entity_refinement_hints"
                                        ],
                                        "entity_normalizations": result[
                                            "entity_normalizations"
                                        ],
                                        "numeric_normalizations": result[
                                            "numeric_normalizations"
                                        ],
                                        "numeric_fallbacks": result[
                                            "numeric_fallbacks"
                                        ],
                                        "safe_numeric_repairs": result[
                                            "safe_numeric_repairs"
                                        ],
                                        "safe_repetition_repairs": result.get(
                                            "safe_repetition_repairs", []
                                        ),
                                        "repetition_reviews": result.get(
                                            "repetition_reviews", []
                                        ),
                                        "boundary_echo_repairs": result[
                                            "boundary_echo_repairs"
                                        ],
                                        "protected_entities": result[
                                            "protected_entities"
                                        ],
                                        "entity_candidates": result[
                                            "entity_candidates"
                                        ],
                                        "entity_matcher_latency_ms": result[
                                            "entity_matcher_latency_ms"
                                        ],
                                        "entity_fuzzy_mode": result[
                                            "entity_fuzzy_mode"
                                        ],
                                        "entity_matching_config_version": result[
                                            "entity_matching_config_version"
                                        ],
                                        "rule_protection_enabled": result[
                                            "rule_protection_enabled"
                                        ],
                                        "numeric_normalization_enabled": result[
                                            "numeric_normalization_enabled"
                                        ],
                                        "refiner_session_stats": result[
                                            "refiner_session_stats"
                                        ],
                                    },
                                },
                            )
                        await send_json(result)
                    else:
                        empty_result: dict[str, object] = {
                            "event": "final",
                            "asr_confidence": asr_confidence,
                            "asr_confidence_metadata": confidence_metadata,
                        }
                        attach_refiner_session_stats(empty_result, close=True)
                        await send_json(empty_result)
                    finished = True
                    break
                audio = message.get("bytes")
                if not audio:
                    continue
                if requested_mode == "offline":
                    payload, chunk_index = await process_audio_chunk(audio)
                    completed_now = consume_completed_segments(payload)
                    if not await send_json(
                        {"event": "chunk_ack", "chunk_index": chunk_index}
                    ):
                        break
                    raw_text = str(payload.get("text", "")).strip()
                    if raw_text and (raw_text != last_raw_text or completed_now):
                        last_raw_text = raw_text
                        detected_language = payload.get("language")
                        asr_confidence = _optional_confidence(
                            payload.get("confidence")
                        )
                        confidence_metadata = _confidence_metadata(payload)
                        language_value = (
                            detected_language
                            if isinstance(detected_language, str)
                            else None
                        )
                        stability, gate_decision = intermediate_gate(
                            raw_text, asr_confidence, confidence_metadata
                        )
                        is_deferred = (
                            gate_decision is not None
                            and gate_decision.public_dict()["action"] == "defer"
                        )
                        has_repetition_candidate = bool(
                            find_repetition_review_candidates(raw_text, max_candidates=1)
                        )
                        record_trace(
                            "asr_hypothesis", mode=requested_mode,
                            chunk_index=chunk_index, raw_text=raw_text,
                            completed_segments=completed_now,
                            gate=gate_decision.public_dict() if gate_decision else None,
                            repetition_candidate=has_repetition_candidate,
                            deferred=is_deferred,
                        )
                        if not await send_json(
                            transcript_event(
                                raw_text,
                                language_value,
                                asr_confidence,
                                confidence_metadata,
                                deferred=is_deferred,
                                gate_decision=gate_decision,
                                stability=stability,
                            )
                        ):
                            break
                        if completed_now:
                            tail_start = max(
                                0, len(raw_text) - STREAMING_INTERMEDIATE_MAX_CHARS
                            )
                            queue_streaming_refinement(
                                raw_text[tail_start:],
                                language_value,
                                asr_confidence,
                                confidence_metadata,
                                raw_text,
                                tail_start,
                                tuple(vad_source_segments),
                            )
                            continue
                        if (is_deferred or (
                            gate_decision is not None
                            and gate_decision.public_dict()["action"] == "keep"
                        )) and not has_repetition_candidate:
                            continue
                        tail_start = max(
                            0, len(raw_text) - STREAMING_INTERMEDIATE_MAX_CHARS
                        )
                        queue_streaming_refinement(
                            raw_text[tail_start:],
                            language_value,
                            asr_confidence,
                            confidence_metadata,
                            raw_text,
                            tail_start,
                        )
                    continue
                payload, chunk_index = await process_audio_chunk(audio)
                completed_now = consume_completed_segments(payload)
                if requested_mode == "streaming":
                    await send_json(
                        {"event": "chunk_ack", "chunk_index": chunk_index}
                    )
                raw_text = str(payload.get("text", "")).strip()
                if raw_text and (raw_text != last_raw_text or completed_now):
                    last_raw_text = raw_text
                    detected_language = payload.get("language")
                    asr_confidence = _optional_confidence(payload.get("confidence"))
                    confidence_metadata = _confidence_metadata(payload)
                    language_value = (
                        detected_language if isinstance(detected_language, str) else None
                    )
                    if requested_mode in {"online", "streaming"}:
                        stability, gate_decision = intermediate_gate(
                            raw_text, asr_confidence, confidence_metadata
                        )
                        is_deferred = (
                            gate_decision is not None
                            and gate_decision.public_dict()["action"] == "defer"
                        )
                        has_repetition_candidate = bool(
                            find_repetition_review_candidates(raw_text, max_candidates=1)
                        )
                        record_trace(
                            "asr_hypothesis", mode=requested_mode,
                            chunk_index=chunk_index, raw_text=raw_text,
                            completed_segments=completed_now,
                            gate=gate_decision.public_dict() if gate_decision else None,
                            repetition_candidate=has_repetition_candidate,
                            deferred=is_deferred,
                        )
                        if not await send_json(
                            transcript_event(
                                raw_text,
                                language_value,
                                asr_confidence,
                                confidence_metadata,
                                deferred=is_deferred,
                                gate_decision=gate_decision,
                                stability=stability,
                            )
                        ):
                            break
                        if completed_now:
                            tail_start = max(
                                0, len(raw_text) - STREAMING_INTERMEDIATE_MAX_CHARS
                            )
                            queue_streaming_refinement(
                                raw_text[tail_start:],
                                language_value,
                                asr_confidence,
                                confidence_metadata,
                                raw_text,
                                tail_start,
                                tuple(vad_source_segments),
                            )
                            continue
                        if (is_deferred or (
                            gate_decision is not None
                            and gate_decision.public_dict()["action"] == "keep"
                        )) and not has_repetition_candidate:
                            continue
                        tail_start = max(
                            0, len(raw_text) - STREAMING_INTERMEDIATE_MAX_CHARS
                        )
                        queue_streaming_refinement(
                            raw_text[tail_start:],
                            language_value,
                            asr_confidence,
                            confidence_metadata,
                            raw_text,
                            tail_start,
                        )
                    else:
                        if not await send_json(
                            attach_refiner_session_stats(
                                await asyncio.to_thread(
                                    session_refine_update,
                                    raw_text,
                                    language_value,
                                    False,
                                    protector,
                                    asr_confidence,
                                    matcher,
                                    confidence_metadata=confidence_metadata,
                                )
                            )
                        ):
                            break
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            await send_json({"event": "error", "detail": str(error)})
        except WebSocketDisconnect:
            pass
        except Exception as error:
            await send_json({"event": "error", "detail": f"后台处理失败：{error}"})
        finally:
            connection_open = False
            refiner_session_stats.close_new_calls()
            if streaming_refiner_task is not None and not streaming_refiner_task.done():
                streaming_refiner_task.cancel()
            if session_id is not None and not finished:
                try:
                    await request_asr("/stream/cancel", session_id)
                except (RuntimeError, TimeoutError):
                    pass

    return app


def _optional_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    return confidence if 0 <= confidence <= 1 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local AgenticASR streaming browser UI")
    parser.add_argument("--refiner-model", type=Path, required=True)
    parser.add_argument("--refiner-device", default="cuda:1")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--asr-api-style",
        choices=[style.value for style in ASRApiStyle],
        default=ASRApiStyle.CURRENT.value,
        help="current uses /stream endpoints; legacy_v1 uses the original /v1/stream/{session_id} contract",
    )
    parser.add_argument("--language", default="Chinese", help="use auto for automatic detection")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--entity-db",
        type=Path,
        help="optional SQLite database containing verified protected entities",
    )
    parser.add_argument(
        "--entity-fuzzy-mode",
        choices=[mode.value for mode in EntityFuzzyMode],
        default="shadow",
        help="off, shadow, hint, or final-only automatic entity normalization",
    )
    parser.add_argument(
        "--refinement-gate-mode",
        choices=[mode.value for mode in RefinementGateMode],
        default="off",
        help="off preserves the baseline; conservative uses two-state gating; tri_state emits KEEP/DEFER/REFINE",
    )
    parser.add_argument(
        "--disable-rule-protection",
        action="store_true",
        help="disable automatic URL/email/date/time/number/identifier/acronym masking",
    )
    parser.add_argument(
        "--disable-numeric-normalization",
        action="store_true",
        help="keep model-driven numeric normalization enabled (legacy alias)",
    )
    parser.add_argument(
        "--enable-numeric-normalization",
        action="store_true",
        help="enable legacy deterministic context-bound Chinese number normalization",
    )
    args = parser.parse_args(argv)
    if not args.refiner_model.exists():
        parser.error(f"Refiner model not found: {args.refiner_model}")
    if args.max_new_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("--max-new-tokens must be positive and --port must be valid")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(
        args.refiner_model.resolve(),
        args.refiner_device,
        args.asr_url,
        None if args.language.lower() == "auto" else args.language,
        args.max_new_tokens,
        args.output.resolve() if args.output else None,
        args.entity_db.resolve() if args.entity_db else None,
        args.entity_fuzzy_mode,
        args.refinement_gate_mode,
        not args.disable_rule_protection,
        args.enable_numeric_normalization and not args.disable_numeric_normalization,
        args.asr_api_style,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
