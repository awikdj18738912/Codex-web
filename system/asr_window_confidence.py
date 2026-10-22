"""Source-aligned diagnostic ASR scores using the existing K=3 partition.

No score is calibrated or permitted to change production refinement routing.
The byte-level representation handles tokens containing partial UTF-8 chars.
"""
from __future__ import annotations

from collections import defaultdict
import json
from math import exp, isfinite
import unicodedata

from .qwen_confidence import QwenVLLMConfidenceDecoder, update_state_confidence
from .refinement_guard import join_refined_segments
from .window_refinement import CumulativeWindowRefinement


def chosen_token_scores(completion):
    """Preserve missing positions, rather than compacting away missing scores."""
    result = []
    values = getattr(completion, "logprobs", None) or []
    for index, token in enumerate(completion.token_ids):
        row = values[index] if index < len(values) else None
        value = row.get(token, row.get(str(token))) if isinstance(row, dict) else row
        value = getattr(value, "logprob", value)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or value > 0):
            value = None
        result.append(float(value) if value is not None else None)
    return result


def byte_alphabet_inverse():
    # GPT-2/Qwen ByteLevel vocabulary encodes all 256 bytes as printable chars.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    code_points = list(byte_values)
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + len(code_points) - 188)
    return {chr(code): value for value, code in zip(byte_values, code_points)}


class TokenBytes:
    def __init__(self, tokenizer):
        decoder = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
        if decoder.get("type") != "ByteLevel":
            raise ValueError("This diagnostic mapper requires a ByteLevel tokenizer")
        self.tokenizer = tokenizer
        self.inverse = byte_alphabet_inverse()
        self.special = set(tokenizer.all_special_ids)
        self.added = tokenizer.added_tokens_decoder

    def decode(self, token_ids, scores):
        spans, chunks, cursor = [], [], 0
        for index, (token, score) in enumerate(zip(token_ids, scores)):
            if token in self.special:
                continue  # vLLM default skips EOS/control tokens in completion.text.
            if token in self.added:
                payload = str(self.added[token]).encode("utf-8")
            else:
                encoded = self.tokenizer.convert_ids_to_tokens(token)
                payload = bytes(self.inverse[char] for char in encoded)
            chunks.append(payload)
            spans.append(dict(start=cursor, end=cursor + len(payload), original_width=len(payload),
                              token_id=token, token_index=index, logprob=score))
            cursor += len(payload)
        return b"".join(chunks), spans


class TokenScoreLedger:
    """Keep scored prefix bytes and replace rolled-back suffix evidence."""
    def __init__(self):
        self.raw = ""
        self.spans = []
        self.revision = 0
        self.issues = []

    def update(self, prefix, completion_text, generated_bytes, generated_spans):
        self.revision += 1
        cut = len(prefix.encode("utf-8"))
        issues, retained = [], []
        if prefix and not self.raw.startswith(prefix):
            issues.append("unscored_or_changed_prefix")
        else:
            for span in self.spans:
                if span["start"] >= cut:
                    continue
                item = dict(span)
                if item["end"] > cut:
                    # Retokenization can cut through an old generated token.
                    # Its full-token probability does not score the new token.
                    item.update(end=cut, logprob=None, truncated_by_rollback=True)
                    issues.append("rollback_cut_through_scored_token")
                retained.append(item)
        if generated_bytes != completion_text.encode("utf-8"):
            issues.append("generated_bytes_differ_from_completion_text")
            generated_spans = []
        for span in generated_spans:
            item = dict(span)
            item.update(start=span["start"] + cut, end=span["end"] + cut,
                        evidence_id=f"{self.revision}:{span['token_index']}", generation_revision=self.revision)
            retained.append(item)
        self.raw = prefix + completion_text
        self.spans = retained
        self.issues = issues

    def text_projection(self, text, forced_language):
        # SDK parsing may collapse excessive repetition. Do not guess offsets
        # for transformed text, even if a repeated suffix happens to match.
        raw = self.raw
        if '\ufffd' in text:
            return dict(status="unsupported", reason="replacement_character_in_transcription", text=text,
                        spans=[], calibrated=False, production_skip_allowed=False)
        payload = raw if forced_language or "<asr_text>" not in raw else raw.split("<asr_text>", 1)[1]
        if payload.strip() != text:
            return dict(status="unsupported", reason="parser_changed_transcription_text", text=text,
                        spans=[], calibrated=False, production_skip_allowed=False)
        end = len(raw.rstrip())
        start = end - len(text)
        offset = len(raw[:start].encode("utf-8"))
        length = len(text.encode("utf-8"))
        spans = []
        for span in self.spans:
            a, b = max(span["start"], offset), min(span["end"], offset + length)
            if a < b:
                spans.append({**span, "start": a - offset, "end": b - offset})
        return dict(status="mapped", text=text, spans=spans, revision=self.revision,
                    issues=self.issues, coordinate_space="raw_ASR_UTF8_bytes", calibrated=False,
                    score_source="chosen_generation_token_logprob_with_retained_prefix_history",
                    production_skip_allowed=False)


class QwenWindowConfidenceDecoder(QwenVLLMConfidenceDecoder):
    """Opt-in isolated decoder preserving the default streaming transcript."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_bytes = TokenBytes(self.asr.processor.tokenizer)

    def _decode(self, state, prefix):
        request = dict(prompt=state.prompt_raw + prefix, multi_modal_data={"audio": [state.audio_accum]})
        outputs = self.asr.model.generate([request], sampling_params=self.sampling_params, use_tqdm=False)
        completion = outputs[0].outputs[0]
        scores = chosen_token_scores(completion)
        generated, spans = self.token_bytes.decode(completion.token_ids, scores)
        ledger = getattr(state, "_window_score_ledger", None)
        if ledger is None:
            ledger = TokenScoreLedger()
            state._window_score_ledger = ledger
        ledger.update(prefix, completion.text, generated, spans)
        state._raw_decoded = prefix + completion.text
        language, text = self._parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language, state.text = language, text
        update_state_confidence(state, completion, temperature=self.temperature,
                                calibrated=self.calibrated, covers_full_state=not prefix)
        state._window_confidence_projection = ledger.text_projection(text, state.force_language)
        state._window_generation_trace = dict(revision=ledger.revision, prefix=prefix,
            completion_text=completion.text, token_ids=list(completion.token_ids), chosen_logprobs=scores,
            raw_decoded=state._raw_decoded, parsed_text=text, ledger_issues=ledger.issues,
            finish_reason=getattr(completion, "finish_reason", None))


def lexical_byte_ranges(text):
    cursor, result = 0, []
    for index, char in enumerate(text):
        end = cursor + len(char.encode("utf-8"))
        if not char.isspace() and unicodedata.category(char)[0] in {"L", "N", "M"}:
            result.append((index, cursor, end))
        cursor = end
    return result


def score_source_range(projection, start, end, low_probability=.8):
    ranges = [item for item in lexical_byte_ranges(projection["text"]) if start <= item[0] < end]
    weights, selected, covered = defaultdict(float), {}, 0
    for _, a, b in ranges:
        overlap = [(span, max(0, min(b, span["end"]) - max(a, span["start"])))
                   for span in projection["spans"] if span["logprob"] is not None]
        if sum(width for _, width in overlap) != b - a:
            continue  # A partial UTF-8 character does not count as covered.
        covered += 1
        for span, width in overlap:
            if width:
                identity = span["evidence_id"]
                weights[identity] += width / span["original_width"]
                selected[identity] = span
    total_weight = sum(weights.values())
    log_score = (sum(selected[key]["logprob"] * weight for key, weight in weights.items()) / total_weight
                 if total_weight else None)
    covered_score = exp(log_score) if log_score is not None else None
    probabilities = sorted(exp(item["logprob"]) for item in selected.values())
    full = bool(ranges) and covered == len(ranges) and projection["status"] == "mapped"
    return dict(score=covered_score if full else None, covered_text_score=covered_score,
        coverage=covered / len(ranges) if ranges else None,
        lexical_characters=len(ranges), covered_lexical_characters=covered,
        scored_tokens=len(selected), token_weight=total_weight,
        min_token_probability=min(probabilities) if probabilities else None,
        low_probability_token_ratio=sum(p < low_probability for p in probabilities) / len(probabilities) if probabilities else None,
        low_probability_reference_threshold=low_probability,
        evidence_revisions=sorted({item["generation_revision"] for item in selected.values()}),
        calibrated=False, production_skip_allowed=False,
        aggregation="UTF8_overlap_weighted_geometric_mean_of_generation_token_probabilities; pure punctuation excluded",
        scope="raw_ASR_spelling_score_not_cleanup_need_or_probability_whole_window_correct")


def locate_chunks(text, chunks):
    """Map exact source chars, allowing only removed boundary whitespace."""
    cursor, result = 0, []
    for chunk in chunks:
        positions = []
        for char in chunk:
            while cursor < len(text) and text[cursor] != char and text[cursor].isspace():
                cursor += 1
            if cursor == len(text) or text[cursor] != char:
                raise ValueError("Project partition cannot be aligned to original ASR characters")
            positions.append(cursor)
            cursor += 1
        if positions:
            result.append((positions[0], positions[-1] + 1))
    return result


def project_window_scores(projection, *, final, source_chunks=None,
                          group_numeric_self_corrections=True, window_size=3, max_chars=80,
                          adjacent_context=True):
    text = projection["text"]
    splitter = CumulativeWindowRefinement(None, window_size=window_size, window_max_chars=max_chars,
        one_punctuation_window=True, fixed_groups=True,
        group_numeric_self_corrections=group_numeric_self_corrections,
        adjacent_context=adjacent_context)
    chunks, pending = splitter._fixed_chunks(text, source_chunks=source_chunks, final=final,
                                             preserve_cached_partition=False)
    offsets = locate_chunks(text, chunks)
    fragments = [dict(index=index, text=chunk, start=a, end=b,
                      confidence=score_source_range(projection, a, b))
                 for index, (chunk, (a, b)) in enumerate(zip(chunks, offsets))]
    complete = splitter._complete_fixed_count(chunks, final=final)
    windows = []
    for index in range(0, complete, window_size):
        group = fragments[index:min(index + window_size, complete)]
        a, b = group[0]["start"], group[-1]["end"]
        result = score_source_range(projection, a, b)
        fragment_scores = [item["confidence"]["score"] for item in group]
        result["min_fragment_score"] = min(fragment_scores) if all(score is not None for score in fragment_scores) else None
        windows.append(dict(index=len(windows), text=join_refined_segments(item["text"] for item in group),
            raw_source_text=text[a:b], start=a, end=b, fragment_indices=[item["index"] for item in group],
            fragment_count=len(group), final_tail=len(group) < window_size, confidence=result))
    return dict(text=text, fragments=fragments, windows=windows, pending_text=pending,
        pending_fragment_indices=list(range(complete, len(chunks))),
        partition=dict(window_size=window_size, one_punctuation_window=True, fixed_groups=True,
            window_max_chars=max_chars, group_numeric_self_corrections=group_numeric_self_corrections,
            adjacent_context=adjacent_context,
            source_chunks_supplied=source_chunks is not None,
            implementation="CumulativeWindowRefinement._fixed_chunks"),
        calibrated=False, production_skip_allowed=False)
