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
REPLAY_DECODE_SECONDS = 10.0
MIN_REPLAY_SCORE = 0.63
SHORT_EXPANSION_RULE = "spelling_or_complete_edge_sound_v1"
LOCAL_REPLAY_RULE = "segment_windows_v1"
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


def _short_expansion_supported(candidate):
    observed = candidate["observed"]
    surface = candidate.get("evidence_surface", candidate["canonical"])
    if (len(observed) > 2 or len(surface) <= len(observed)
            or set(observed) & set(surface)):
        return True
    # Exact complete syllables at one edge can be an ASR-truncated name.
    # Internal syllable insertion without any spelling evidence is too weak
    # for a short fragment, even when a hinted replay chooses that word.
    from .entity_matcher import _pinyin_tokens
    source_sound = _pinyin_tokens(observed)
    target_sound = _pinyin_tokens(surface)
    width = len(source_sound)
    return bool(width and (target_sound[:width] == source_sound
                           or target_sound[-width:] == source_sound))


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
    # Preserve declared short aliases and complete prefix/suffix sounds.
    # This gate only removes confirmations, never changes prompt terms.
    accepted = [c for c in accepted if _short_expansion_supported(c)]
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


class _LegacyEntityAudioRepair:
    """Session-owned audio replay; failures preserve the completed first pass."""

    def __init__(self, request, definitions, *, on_status=None,
                 replay_chunk_bytes=64000, max_audio_bytes=460800000,
                 max_source_chars=6000, verification_timeout=None,
                 minimum_replay_score=MIN_REPLAY_SCORE, local_replay=True,
                 streaming_replay=True):
        self._request = request
        from .entity_mixed_spelling import supported_surface
        self.definitions = tuple(d for d in definitions if d.enabled and d.normalization_policy == "normalize"
                                 and any(supported_surface(s)
                                         for s in (d.canonical_text, *d.aliases)))
        self._on_status = on_status
        self.replay_chunk_bytes = replay_chunk_bytes
        self.max_audio_bytes = max_audio_bytes
        self.max_source_chars = max_source_chars
        self.verification_timeout = verification_timeout
        self.minimum_replay_score = minimum_replay_score
        self._audio = None
        self._audio_size = 0
        self._segments = {}
        self._segment_audio = {}
        self._boundary_floor = 0
        self._timeline_valid = True
        self.local_replay = local_replay
        self.replay_mode = "pending"
        self.local_replay_reason = None
        self.replay_audio_bytes = 0
        self.replay_windows = []
        self._params = {}
        self._sid = None
        self._verification_task = None
        self.enabled = bool(self.definitions)
        self.phase = "pending" if self.enabled else "skipped"
        self.reason = None if self.enabled else "no_normalizable_entities"
        self.changes = []
        self.terms = []
        self.candidate_count = 0
        self.max_candidate_score = 0.0
        self.checks = 0
        self.replay_context_applied = False
        self._started = None
        self.elapsed_seconds = 0.0
        self.revision = 0
        self._streaming = None
        if streaming_replay and self.enabled:
            from .entity_streaming_replay import StreamingEntityReplay
            self._streaming = StreamingEntityReplay(self)
        self.streaming_cache_used = False

    def snapshot(self):
        elapsed = time.monotonic() - self._started if self._started and self.phase in {"retrieving", "verifying"} else self.elapsed_seconds
        return dict(enabled=self.enabled, phase=self.phase, reason=self.reason,
                    entity_count=len(self.definitions), candidate_count=self.candidate_count,
                    max_candidate_score=self.max_candidate_score,
                    minimum_replay_score=self.minimum_replay_score,
                    short_expansion_rule=SHORT_EXPANSION_RULE,
                    prompt_terms=list(self.terms), checks=self.checks,
                    replay_context_applied=self.replay_context_applied,
                    applied=bool(self.changes), change_count=len(self.changes),
                    changes=list(self.changes), revision=self.revision,
                    elapsed_seconds=round(elapsed, 3), replay_mode=self.replay_mode,
                    local_replay_reason=self.local_replay_reason,
                    local_replay_enabled=self.local_replay, local_replay_rule=LOCAL_REPLAY_RULE,
                    source_audio_seconds=round(self._audio_size / 64000, 3),
                    replay_audio_seconds=round(self.replay_audio_bytes / 64000, 3),
                    replay_windows=list(self.replay_windows),
                    replay_decode_seconds=REPLAY_DECODE_SECONDS,
                    streaming_cache_used=self.streaming_cache_used,
                    streaming_replay=self._streaming.snapshot() if self._streaming else None)

    def close(self):
        if self._streaming is not None:
            self._streaming.close()
        task = self._verification_task
        if task is not None and not task.done() and not task.cancelling():
            task.cancel()
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self._segments.clear()
        self._segment_audio.clear()

    def _annotate(self, payload):
        return dict(payload, entity_audio_repair=self.snapshot(), entity_audio_repair_revision=self.revision)

    def _record_segments(self, payload, audio_before=None):
        if not self.enabled:
            return
        before = self._audio_size if audio_before is None else audio_before
        new = []
        for event in payload.get("completed_segments", ()):
            if not isinstance(event, dict) or not isinstance(event.get("segment_id"), int) or not isinstance(event.get("text"), str):
                continue
            sid = event["segment_id"]
            if sid in self._segments:
                if self._segments[sid]["text"] != event["text"]:
                    self._timeline_valid = False
            else:
                if sid != len(self._segments) + 1:
                    self._timeline_valid = False
                self._segment_audio[sid] = (self._boundary_floor, self._audio_size)
                new.append(sid)
            self._segments[sid] = deepcopy(event)
        if new:
            self._boundary_floor = before

    async def _status(self):
        if self._on_status is not None:
            if await self._on_status(self.snapshot()) is False:
                raise asyncio.CancelledError

    async def request(self, endpoint, session_id=None, data=b"", params=None):
        primary_request = self._streaming.primary_request if self._streaming else self._request
        if endpoint == "/stream/start":
            self._params = dict(params or {})
            self._params.pop("context", None)
            result = await primary_request(endpoint, session_id, data, self._params)
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
                return await primary_request(endpoint, self._sid or session_id, data, params)
            finally:
                self.close()
        if endpoint == "/stream/chunk":
            audio_before = self._audio_size
            if self._audio is not None:
                if self._audio_size + len(data) > self.max_audio_bytes:
                    self.enabled = False
                    self.phase, self.reason = "skipped", "audio_cache_limit"
                    self.close()
                else:
                    self._audio.seek(0, 2)
                    self._audio.write(data)
                    self._audio_size += len(data)
                    if self._streaming is not None:
                        self._streaming.capture(data)
            result = await primary_request(endpoint, self._sid or session_id, data, params)
            self._record_segments(result, audio_before)
            if self._streaming is not None:
                self._streaming.consider()
            return self._annotate(result)
        if endpoint != "/stream/finish":
            return await self._request(endpoint, self._sid or session_id, data, params)
        try:
            original = await primary_request(endpoint, self._sid or session_id, data, params)
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

        candidates, terms = await asyncio.to_thread(select)
        self.candidate_count = len(candidates)
        self.max_candidate_score = max((c["retrieval_score"] for c in candidates), default=0.0)
        if not terms:
            self.phase = "complete"
            self.reason = "no_candidates"
            return original
        # Broad retrieval includes ordinary phrases that weakly resemble the
        # glossary. Skip their whole replay, without altering the term list or
        # verification rules of sessions that pass. Scores are retrieval
        # evidence, not calibrated correctness probabilities.
        if self.max_candidate_score < self.minimum_replay_score:
            self.phase, self.reason = "complete", "weak_candidates"
            return original
        self.terms = terms
        if self._streaming is not None:
            cached = await self._streaming.finish()
            if cached is not None and all(term in self._streaming.terms for term in terms):
                value, changes = verify_entity_audio_changes(text, cached[0], candidates)
                resolved = True
                for candidate in candidates:
                    if any(candidate["start"] < c["end"] and c["start"] < candidate["end"] for c in changes):
                        continue
                    keep = dict(candidate, canonical=candidate["observed"], evidence_surface=candidate["observed"])
                    if keep not in verify_entity_audio_changes(text, cached[0], [keep, *changes])[1]:
                        resolved = False
                        break
                # Echo fallback is reusable only as confirmation of unchanged
                # text. It cannot supply a canonical entity correction.
                if resolved and (not cached[1] or not changes):
                    self.streaming_cache_used = True
                    self.replay_mode = "streaming"
                    self.checks += 1
                    self.replay_context_applied = True
                    self.replay_audio_bytes = self._streaming.processed_bytes
                    return self._publish_changes(original, changes)
        context = EntityASRContext(REPLAY_PREFIX + json.dumps(self.terms, ensure_ascii=False), tuple(self.terms))
        windows = self._localized_windows(text, candidates)
        if windows:
            self.replay_mode = "localized"
            proposals = []
            resolved = True
            for start, end, local in windows:
                replay = await self._replay_range(start, end, context)
                if replay is None:
                    return original
                if replay[1]:
                    resolved = False
                    break
                _, changes = verify_entity_audio_changes(text, replay[0], local)
                proposals.extend(changes)
                for candidate in local:
                    if any(candidate["start"] < c["end"] and c["start"] < candidate["end"] for c in changes):
                        continue
                    keep = dict(candidate, canonical=candidate["observed"], evidence_surface=candidate["observed"])
                    _, verified = verify_entity_audio_changes(text, replay[0], [keep, *changes])
                    if keep not in verified:
                        resolved = False
                        break
                if not resolved:
                    break
            if resolved:
                unique = {(c["start"], c["end"], c["canonical"]): c for c in proposals}
                return self._publish_changes(original, list(unique.values()))
            self.replay_mode = "full_after_local"
            self.local_replay_reason = "local_anchors_unresolved"
        else:
            self.replay_mode = "full"
        replay = await self._replay_range(0, self._audio_size, context)
        if replay is None:
            return original
        if replay[1]:
            self.phase, self.reason = "fallback", "replay_prompt_echo"
            return original
        _, changes = verify_entity_audio_changes(text, replay[0], candidates)
        return self._publish_changes(original, changes)

    def _publish_changes(self, original, changes):
        if not changes:
            self.phase, self.reason = "complete", "no_confirmed_changes"
            return original
        corrected = str(original.get("text", ""))
        for change in sorted(changes, key=lambda c: c["start"], reverse=True):
            if corrected[change["start"]:change["end"]] != change["observed"]:
                raise ValueError("Entity source span changed during localized verification")
            corrected = corrected[:change["start"]] + change["canonical"] + corrected[change["end"]:]
        result = revise_entity_audio_payload(original, self._segments.values(), corrected, changes)
        types = {d.entity_id: d.entity_type for d in self.definitions}
        self.changes = [dict(c, entity_type=types.get(c["entity_id"], ""), match_type="AUDIO_REPLAY_CONFIRMED") for c in changes]
        self.revision = 1
        self.phase, self.reason = "complete", None
        return result

    def _localized_windows(self, text, candidates):
        if not self.local_replay or self._audio_size < 45 * 64000:
            self.local_replay_reason = "short_audio_or_disabled"
            return None
        events = list(self._segments.values())
        if not self._timeline_valid or join_refined_segments(e["text"] for e in events) != text.strip():
            self.local_replay_reason = "unreliable_segment_text"
            return None
        spans = []
        cursor = 0
        for event in events:
            value = event["text"].strip()
            start = text.find(value, cursor) if value else -1
            bounds = self._segment_audio.get(event["segment_id"])
            if start < 0 or bounds is None:
                self.local_replay_reason = "unmapped_segment"
                return None
            spans.append((start, start + len(value), *bounds))
            cursor = start + len(value)
        required = []
        for candidate in candidates:
            left, right = candidate["start"], candidate["end"]
            count = 0
            while left and count < 8:
                left -= 1
                count += len(normalized(text[left]))
            count = 0
            while right < len(text) and count < 8:
                count += len(normalized(text[right]))
                right += 1
            relevant = [s for s in spans if s[0] < right and left < s[1]]
            if not relevant:
                self.local_replay_reason = "unmapped_candidate"
                return None
            start = max(0, min(s[2] for s in relevant) - 2 * 64000)
            end = min(self._audio_size, max(s[3] for s in relevant) + 2 * 64000)
            required.append([start, end, candidate])
        windows = [[start, end, [candidate]] for start, end, candidate in required
                   if _short_expansion_supported(candidate)]
        merged = []
        for start, end, local in sorted(windows, key=lambda w:w[0]):
            if merged and start <= merged[-1][1] + 64000:
                merged[-1][1] = max(end, merged[-1][1])
                merged[-1][2].extend(local)
            else:
                merged.append([start, end, list(local)])
        if not merged or len(merged) > 2 or sum(end-start for start,end,_ in merged) > self._audio_size * 0.6:
            self.local_replay_reason = "insufficient_audio_saving"
            return None
        for window in merged:
            window[2] = [c for start, end, c in required
                         if window[0] <= start and end <= window[1]]
        self.local_replay_reason = None
        return merged

    async def _replay_range(self, start, end, context):
        guard = EntityASREchoGuard(self._request, context, replay_chunk_bytes=self.replay_chunk_bytes,
                                   max_audio_bytes=self.max_audio_bytes)
        self.phase = "verifying"
        ready = None
        try:
            # Only final replay evidence is consumed. A larger decode step
            # avoids rebuilding intermediate hypotheses we never publish;
            # the first-pass session retains its requested chunk interval.
            ready = await guard.request("/stream/start", params=dict(
                self._params, context=context.text, chunk_seconds=REPLAY_DECODE_SECONDS))
            if ready.get("context_applied") is not True:
                self.phase, self.reason = "skipped", "backend_context_unsupported"
                return None
            self.replay_context_applied = True
            self.checks += 1
            self.replay_windows.append(dict(start_seconds=round(start / 64000, 3), end_seconds=round(end / 64000, 3)))
            self._audio.seek(start)
            remaining = end - start
            while remaining:
                pcm = self._audio.read(min(self.replay_chunk_bytes, remaining))
                if not pcm:
                    raise ValueError("Incomplete entity audio cache")
                remaining -= len(pcm)
                self.replay_audio_bytes += len(pcm)
                await guard.request("/stream/chunk", ready["session_id"], pcm)
            replay = await guard.request("/stream/finish", ready["session_id"])
            return str(replay.get("text", "")), guard.fallback_active
        finally:
            if ready is not None:
                try:
                    await guard.request("/stream/cancel", ready["session_id"])
                except Exception:
                    pass
            guard.close()


from .entity_streaming_live import LiveEntityMixin


class EntityAudioRepair(LiveEntityMixin, _LegacyEntityAudioRepair):
    """Incremental confirmation with a bounded final check for short audio.

    The legacy state/helpers remain available for archived regression tests;
    this public controller does not use their whole-recording repair path.
    """
    def __init__(self, request, definitions, **kwargs):
        definitions = tuple(definitions)
        self._live_spelling_definitions = definitions
        ctc_session = kwargs.pop("ctc_session", None)
        ctc_mode = kwargs.pop("ctc_mode", "off")
        names = ("candidate_score", "candidate_margin", "window_seconds",
                 "cache_seconds", "finish_wait", "maximum_queue", "replay_ratio",
                 "maximum_replay_seconds", "request_timeout", "enable_live",
                 "short_finish_max_seconds", "short_finish_wait")
        options = {name: kwargs.pop(name) for name in names if name in kwargs}
        options.setdefault("candidate_score", kwargs.get("minimum_replay_score", MIN_REPLAY_SCORE))
        if kwargs.get("verification_timeout") is not None:
            options.setdefault("request_timeout", min(3., kwargs["verification_timeout"]))
        options.setdefault("enable_live", kwargs.pop("streaming_replay", True))
        kwargs["streaming_replay"] = False
        super().__init__(request, definitions, **kwargs)
        self._init_live(request, ctc_session=ctc_session, ctc_mode=ctc_mode, **options)
        if not options["enable_live"] and self.enabled:
            self.enabled = False
            self.phase, self.reason = "skipped", "streaming_entity_review_disabled"

    async def request(self, endpoint, session_id=None, data=b"", params=None):
        return await self._request_live(endpoint, session_id, data, params)

    def snapshot(self):
        result = super().snapshot()
        if hasattr(self, "_live_stats"):
            result["incremental"] = self._live_snapshot()
            result["elapsed_seconds"] = round(self._live_stats["review_wall_seconds"], 3)
        return result

    def close(self):
        self._close_live()
        super().close()
