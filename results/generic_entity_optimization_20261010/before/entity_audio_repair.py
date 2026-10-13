"""Type-independent, opt-in glossary correction verified against the same audio.

The optional session wrapper never copies unrelated replay wording. It uses
only a frozen glossary and the initial ASR hypothesis, without reference text.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import lru_cache
import json
import logging
import re
import tempfile
import time
import unicodedata

from .entity_asr_context import EntityASRContext
from .entity_asr_echo_guard import EntityASREchoGuard
from .entity_retrieval import EntityRetriever
from .entity_store import EntityDefinition
from .refinement_guard import join_refined_segments

MAX_REPLAY_TERMS = 8
REPLAY_PREFIX = "参考术语仅用于辅助拼写，不代表音频内容；未在音频中听到的词不要补充。术语列表："
_LOG = logging.getLogger(__name__)


def normalized(text):
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in value if unicodedata.category(c)[0] not in {"P", "Z", "C"})


class GenericEntitySelector:
    """The same lexical/sound retrieval settings for every entity type."""
    def __init__(self, definitions):
        self.definitions = tuple(d for d in definitions if d.enabled and d.normalization_policy == "normalize")
        surfaces, self.owner = [], {}
        for definition in self.definitions:
            for surface in (definition.canonical_text, *definition.aliases):
                if re.fullmatch(r"[\u3400-\u9fff]{2,24}", surface) is None:
                    continue
                identifier = len(surfaces) + 1
                # Type is deliberately not a retrieval feature. Registered
                # aliases supply cross-script pronunciations, never guesses.
                surfaces.append(EntityDefinition(identifier, surface, "TERM", "general", "normalize", 0, ()))
                self.owner[identifier] = definition
        self.retriever = EntityRetriever(surfaces, minimum_score=0.52, max_candidates=5)

    def select(self, text):
        blocked = []
        for definition in self.definitions:
            for surface in (definition.canonical_text, *definition.aliases):
                blocked.extend((m.start(), m.end()) for m in re.finditer(re.escape(surface), text))
        matches = self.retriever.retrieve(text, blocked_spans=blocked)
        dedup = {}
        for match in matches:
            if match.boundary_issue not in (None, "short_chinese_span_requires_alias"):
                continue
            definition = self.owner[match.entity_id]
            key = match.start, match.end, definition.entity_id
            candidate = dict(match.public_dict(), canonical=definition.canonical_text,
                             entity_id=definition.entity_id, evidence_surface=match.canonical)
            candidate.pop("entity_type")
            if key not in dedup or candidate["retrieval_score"] > dedup[key]["retrieval_score"]:
                dedup[key] = candidate
        ordered = sorted(dedup.values(), key=lambda c: (-c["retrieval_score"], -len(c["observed"]), c["start"], c["canonical"]))
        terms = list(dict.fromkeys(c["canonical"] for c in ordered))[:MAX_REPLAY_TERMS]
        selected = [c for c in ordered if c["canonical"] in terms]
        return selected, terms


def verify_entity_audio_changes(original, replay, candidates):
    """Allow adjacent registered corrections in the surrounding anchors.

    Each side still consumes the same eight source characters. Its only
    alternatives are original characters or retrieved glossary spellings;
    arbitrary changes to surrounding wording cannot provide support.
    """
    source, target = normalized(original), normalized(replay)
    spans = [(len(normalized(original[:c["start"]])),
              len(normalized(original[:c["end"]])), normalized(c["canonical"]), c)
             for c in candidates]

    def side_matches(a, b, replay_side, query_start, query_end, reverse=False):
        value = source[a:b]
        if reverse:
            value = value[::-1]
        edges = {}
        for start, end, canonical, _ in spans:
            if not a <= start < end <= b or start < query_end and query_start < end:
                continue
            x,y,word = (b-end,b-start,canonical[::-1]) if reverse else (start-a,end-a,canonical)
            edges.setdefault(x, []).append((y,word))

        @lru_cache(maxsize=None)
        def match(i,j):
            if i == len(value):
                return True
            if j >= len(replay_side):
                return False
            if value[i] == replay_side[j] and match(i+1,j+1):
                return True
            return any(replay_side.startswith(word,j) and match(end,j+len(word))
                       for end,word in edges.get(i,()))
        return match(0,0)

    confirmed = []
    for start,end,canonical,candidate in spans:
        if not source[:start] and not source[end:]:
            continue
        for occurrence in re.finditer(re.escape(canonical),target):
            if (side_matches(max(0,start-8),start,target[:occurrence.start()][::-1],start,end,True)
                    and side_matches(end,min(len(source),end+8),target[occurrence.end():],start,end)):
                confirmed.append(candidate)
                break
    accepted = []
    for candidate in confirmed:
        if any(other["canonical"] != candidate["canonical"] and
               candidate["start"] < other["end"] and other["start"] < candidate["end"]
               for other in confirmed):
            continue
        if any(candidate["start"] < old["end"] and old["start"] < candidate["end"] for old in accepted):
            continue
        accepted.append(candidate)
    result = original
    for candidate in sorted(accepted, key=lambda c:c["start"], reverse=True):
        result = result[:candidate["start"]] + candidate["canonical"] + result[candidate["end"]:]
    return result,accepted


def revise_entity_audio_payload(payload, segments, corrected, changes):
    """Publish a bounded entity revision without reusing old token scores."""
    original = str(payload.get("text", ""))
    revised = deepcopy(payload)
    events = deepcopy(list(segments))
    complete = join_refined_segments(str(s.get("text", "")) for s in events) == original.strip()
    applied, cursor = 0, 0
    if complete:
        for event in events:
            text = str(event.get("text", "")).strip()
            start = original.find(text, cursor)
            if start < 0:
                complete = False
                break
            end = start + len(text)
            local = [c for c in changes if start <= c["start"] < c["end"] <= end]
            for change in sorted(local, key=lambda c: c["start"], reverse=True):
                a, b = change["start"] - start, change["end"] - start
                if text[a:b] != change["observed"]:
                    raise ValueError("Entity change does not refer to its source span")
                text = text[:a] + change["canonical"] + text[b:]
                applied += 1
            event["text"] = text
            if local:
                event["confidence"] = None
                event["confidence_metadata"] = dict(
                    confidence=None, confidence_calibrated=False,
                    confidence_covers_full_state=False, confidence_covers_full_text=False,
                    confidence_scope="unscored_entity_revision", confidence_source="entity_audio_repair",
                )
            cursor = end
    # Existing Web finalization already falls back to the authoritative full
    # text when an older backend lacks a complete segment event stream.
    revised["completed_segments"] = events if complete and applied == len(changes) else []
    revised.update(
        text=corrected, entity_audio_repair_revision=1,
        confidence=None, confidence_calibrated=False,
        confidence_covers_full_state=False, confidence_covers_full_text=False,
        confidence_scope="unscored_entity_revision", confidence_source="entity_audio_repair",
        confidence_mean_logprob=None, confidence_min_probability=None, confidence_token_count=0,
    )
    return revised


class EntityAudioRepair:
    """Session-owned audio replay; failures preserve the completed first pass."""

    def __init__(self, request, definitions, *, on_status=None,
                 replay_chunk_bytes=64000, max_audio_bytes=460800000,
                 max_source_chars=6000, verification_timeout=None):
        self._request = request
        self.definitions = tuple(d for d in definitions if d.enabled and d.normalization_policy == "normalize"
                                 and any(re.fullmatch(r"[\u3400-\u9fff]{2,24}", s)
                                         for s in (d.canonical_text, *d.aliases)))
        self._on_status = on_status
        self.replay_chunk_bytes = replay_chunk_bytes
        self.max_audio_bytes = max_audio_bytes
        self.max_source_chars = max_source_chars
        self.verification_timeout = verification_timeout
        self._audio = None
        self._audio_size = 0
        self._segments = {}
        self._params = {}
        self._sid = None
        self._verification_task = None
        self.enabled = bool(self.definitions)
        self.phase = "pending" if self.enabled else "skipped"
        self.reason = None if self.enabled else "no_normalizable_entities"
        self.changes = []
        self.terms = []
        self.candidate_count = 0
        self.checks = 0
        self.replay_context_applied = False
        self._started = None
        self.elapsed_seconds = 0.0
        self.revision = 0

    def snapshot(self):
        elapsed = time.monotonic() - self._started if self._started and self.phase in {"retrieving", "verifying"} else self.elapsed_seconds
        return dict(enabled=self.enabled, phase=self.phase, reason=self.reason,
                    entity_count=len(self.definitions), candidate_count=self.candidate_count,
                    prompt_terms=list(self.terms), checks=self.checks,
                    replay_context_applied=self.replay_context_applied,
                    applied=bool(self.changes), change_count=len(self.changes),
                    changes=list(self.changes), revision=self.revision,
                    elapsed_seconds=round(elapsed, 3))

    def close(self):
        task = self._verification_task
        if task is not None and not task.done() and not task.cancelling():
            task.cancel()
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self._segments.clear()

    def _annotate(self, payload):
        return dict(payload, entity_audio_repair=self.snapshot(), entity_audio_repair_revision=self.revision)

    def _record_segments(self, payload):
        if not self.enabled:
            return
        for event in payload.get("completed_segments", ()):
            if isinstance(event, dict) and isinstance(event.get("segment_id"), int) and isinstance(event.get("text"), str):
                self._segments[event["segment_id"]] = deepcopy(event)

    async def _status(self):
        if self._on_status is not None:
            if await self._on_status(self.snapshot()) is False:
                raise asyncio.CancelledError

    async def request(self, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/start":
            self._params = dict(params or {})
            self._params.pop("context", None)
            result = await self._request(endpoint, session_id, data, self._params)
            self._sid = result["session_id"]
            if result.get("context_applied") is True:
                self.enabled = False
                self.phase, self.reason = "skipped", "initial_context_already_applied"
            if self.enabled:
                self._audio = tempfile.SpooledTemporaryFile(max_size=1048576, mode="w+b")
                self.phase = "capturing"
            return self._annotate(result)
        if endpoint == "/stream/cancel":
            try:
                return await self._request(endpoint, self._sid or session_id, data, params)
            finally:
                self.close()
        if endpoint == "/stream/chunk":
            if self._audio is not None:
                if self._audio_size + len(data) > self.max_audio_bytes:
                    self.enabled = False
                    self.phase, self.reason = "skipped", "audio_cache_limit"
                    self.close()
                else:
                    self._audio.seek(0, 2)
                    self._audio.write(data)
                    self._audio_size += len(data)
            result = await self._request(endpoint, self._sid or session_id, data, params)
            self._record_segments(result)
            return self._annotate(result)
        if endpoint != "/stream/finish":
            return await self._request(endpoint, self._sid or session_id, data, params)
        try:
            original = await self._request(endpoint, self._sid or session_id, data, params)
            self._record_segments(original)
            text = str(original.get("text", ""))
            if not self.enabled or self._audio is None or not self._audio_size or not text.strip():
                if self.enabled:
                    self.phase, self.reason = "complete", "no_audio_or_text"
                return self._annotate(original)
            if len(text) > self.max_source_chars:
                self.phase, self.reason = "skipped", "source_text_limit"
                return self._annotate(original)
            self._started = time.monotonic()
            self.phase = "retrieving"
            try:
                await self._status()
                self._verification_task = asyncio.create_task(self._repair(original))
                budget = self.verification_timeout
                if budget is None:
                    budget = min(300.0, max(60.0, 30.0 + self._audio_size / 64000 * 0.3))
                deadline = time.monotonic() + budget
                while not self._verification_task.done():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._verification_task.cancel()
                        await asyncio.gather(self._verification_task, return_exceptions=True)
                        raise TimeoutError("Entity audio verification timed out")
                    await asyncio.wait({self._verification_task}, timeout=min(5.0, remaining))
                    if not self._verification_task.done():
                        await self._status()
                result = self._verification_task.result()
                self.elapsed_seconds = time.monotonic() - self._started
                return self._annotate(result)
            except asyncio.CancelledError:
                self.phase, self.reason = "cancelled", "session_cancelled"
                if self._verification_task is not None:
                    self._verification_task.cancel()
                    await asyncio.gather(self._verification_task, return_exceptions=True)
                raise
            except Exception as error:
                self.phase, self.reason = "fallback", type(error).__name__
                self.elapsed_seconds = time.monotonic() - self._started
                _LOG.warning("Entity audio verification failed; keeping first pass (%s)", type(error).__name__)
                return self._annotate(original)
        finally:
            self.close()

    async def _repair(self, original):
        text = str(original.get("text", ""))

        def select():
            return GenericEntitySelector(self.definitions).select(text)

        candidates, self.terms = await asyncio.to_thread(select)
        self.candidate_count = len(candidates)
        if not self.terms:
            self.phase = "complete"
            self.reason = "no_candidates"
            return original
        context = EntityASRContext(REPLAY_PREFIX + json.dumps(self.terms, ensure_ascii=False), tuple(self.terms))
        guard = EntityASREchoGuard(self._request, context, replay_chunk_bytes=self.replay_chunk_bytes,
                                   max_audio_bytes=self.max_audio_bytes)
        self.phase = "verifying"
        ready = None
        try:
            params = dict(self._params, context=context.text)
            ready = await guard.request("/stream/start", params=params)
            if ready.get("context_applied") is not True:
                self.phase, self.reason = "skipped", "backend_context_unsupported"
                return original
            self.replay_context_applied = True
            self.checks += 1
            self._audio.seek(0)
            while True:
                pcm = self._audio.read(self.replay_chunk_bytes)
                if not pcm:
                    break
                await guard.request("/stream/chunk", ready["session_id"], pcm)
            replay = await guard.request("/stream/finish", ready["session_id"])
            if guard.fallback_active:
                self.phase, self.reason = "fallback", "replay_prompt_echo"
                return original
            corrected, changes = verify_entity_audio_changes(text, str(replay.get("text", "")), candidates)
            if not changes:
                self.phase, self.reason = "complete", "no_confirmed_changes"
                return original
            # Build the full revision before committing any session state.
            result = revise_entity_audio_payload(original, self._segments.values(), corrected, changes)
            types = {d.entity_id: d.entity_type for d in self.definitions}
            self.changes = [dict(c, entity_type=types.get(c["entity_id"], ""), match_type="AUDIO_REPLAY_CONFIRMED") for c in changes]
            self.revision = 1
            self.phase, self.reason = "complete", None
            return result
        finally:
            if ready is not None:
                try:
                    await guard.request("/stream/cancel", ready["session_id"])
                except Exception:
                    pass  # cleanup must not replace a successful or failed verification
            guard.close()
