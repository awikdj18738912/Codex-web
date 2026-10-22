"""Experimental, incremental Viterbi CTC word spotter (NumPy, CPU).

This is a small prototype, not a reproduction of the complete streaming CTC-WS
paper (https://arxiv.org/abs/2605.18222). The context-trie/token-passing design
is informed by NVIDIA NeMo Speech's Apache-2.0 implementations:
https://github.com/NVIDIA-NeMo/Speech/tree/main/nemo/collections/asr/parts/context_biasing
No NeMo runtime dependency or source code is copied. Scores use the best CTC
alignment, not the summed CTC posterior and not text/fuzzy matching.

Frames are absolute, end_frame is inclusive, and the default frame shift is
40ms. Only previously unseen contiguous frames may be accepted. State carries
across calls; acoustic history is never replayed or retained. finish() returns
consolidated detections. accept() returns new/improved evidence in this chunk;
consumers should use finish() for final span deduplication. Bias is awarded once
per trie edge, never on a token self-loop or blank. The unbiased acoustic gap is
against the sum of the framewise maximum log probability *on the same span*;
it is not a competing transcript/CTC beam likelihood ratio.
"""
from __future__ import annotations

import copy
import itertools
import math
import time
from typing import Any

import numpy as np


class StreamingWordSpotter:
    """One best blank/nonblank alignment per trie node, retained across chunks.

    Tokenization enumerates at most 32 deterministic bare/word-boundary variants
    per canonical word. This character tokenizer deliberately does not invent
    aliases or consult transcripts. UTF-8 byte fallback is used only when a
    character has neither a bare nor a boundary-prefixed vocabulary token.
    """

    def __init__(self, tokens: list[str], words: list[str], context_weight: float = 2.,
                 beam: float = 12., minimum_gap: float = -3., *,
                 blank_id: int | None = None, frame_seconds: float = .04,
                 max_tokenizations: int = 32, max_hypotheses_per_chunk: int = 1000,
                 max_seconds_per_character: float = 1., byte_fallback: bool = True):
        if not tokens or len(tokens) != len(set(tokens)):
            raise ValueError('tokens must be nonempty and unique')
        if beam <= 0 or frame_seconds <= 0 or max_seconds_per_character <= 0:
            raise ValueError('beam and timing limits must be positive')
        if max_tokenizations < 1 or max_hypotheses_per_chunk < 1:
            raise ValueError('tokenization and output limits must be positive')
        if not all(math.isfinite(x) for x in [context_weight, beam, minimum_gap,
                                             frame_seconds, max_seconds_per_character]):
            raise ValueError('configuration must be finite')
        self.tokens = list(tokens)
        self.token_ids = {token: i for i, token in enumerate(tokens)}
        if blank_id is None:
            blank_id = next((self.token_ids[t] for t in ('<blk>', '<blank>', '[blank]', '<eps>')
                             if t in self.token_ids), 0)
        if not 0 <= blank_id < len(tokens):
            raise ValueError('blank_id outside vocabulary')
        self.blank_id = blank_id
        self.context_weight = float(context_weight)
        self.beam = float(beam)
        self.minimum_gap = float(minimum_gap)
        self.frame_seconds = float(frame_seconds)
        self.max_tokenizations = min(int(max_tokenizations), 32)
        self.max_hypotheses_per_chunk = int(max_hypotheses_per_chunk)
        self.max_seconds_per_character = max_seconds_per_character
        self.byte_fallback = byte_fallback
        parents, labels, depths = [0], [blank_id], [0]
        children: list[dict[int, int]] = [{}]
        terminals: dict[int, list[str]] = {}
        max_lengths = [0]
        unique_words = sorted(set(word.strip() for word in words if word.strip()))
        coverage: dict[str, Any] = dict(requested_words=len(unique_words), covered_words=0,
            skipped_short_words=[], unsupported_words={}, byte_fallback_words=[],
            truncated_tokenization_words=[], tokenizations=0, variants_per_word={})
        for word in unique_words:
            if len(word.replace(' ', '')) < 2:
                coverage['skipped_short_words'].append(word)
                continue
            variants, missing, used_bytes, truncated = self._tokenize(word)
            if missing:
                coverage['unsupported_words'][word] = missing
                continue
            coverage['covered_words'] += 1
            coverage['variants_per_word'][word] = len(variants)
            coverage['tokenizations'] += len(variants)
            if used_bytes:
                coverage['byte_fallback_words'].append(word)
            if truncated:
                coverage['truncated_tokenization_words'].append(word)
            for variant in variants:
                node = 0
                for token in variant:
                    if token not in children[node]:
                        child = len(parents)
                        children[node][token] = child
                        parents.append(node)
                        labels.append(token)
                        depths.append(depths[node] + 1)
                        children.append({})
                        max_lengths.append(0)
                    node = children[node][token]
                    max_lengths[node] = max(max_lengths[node], len(word))
                terminals.setdefault(node, []).append(word)
        self.parents = np.asarray(parents, dtype=np.int32)
        self.labels = np.asarray(labels, dtype=np.int32)
        self.depths = np.asarray(depths, dtype=np.int32)
        self._repeats = self.labels == self.labels[self.parents]
        self._bonus = self.depths * self.context_weight
        self._max_age = np.ceil(np.maximum(1, np.asarray(max_lengths)) *
                                max_seconds_per_character / frame_seconds).astype(np.int64)
        self._terminal_nodes = np.asarray(sorted(terminals), dtype=np.int32)
        self._terminals = terminals
        for graph_array in (self.parents, self.labels, self.depths, self._repeats,
                            self._bonus, self._max_age, self._terminal_nodes):
            graph_array.flags.writeable = False
        size = len(parents)
        self._blank = np.full(size, -np.inf)
        self._nonblank = np.full(size, -np.inf)
        self._blank_gap = np.full(size, -np.inf)
        self._nonblank_gap = np.full(size, -np.inf)
        self._blank_start = np.full(size, -1, dtype=np.int64)
        self._nonblank_start = np.full(size, -1, dtype=np.int64)
        self._next_frame: int | None = None
        self._detections: dict[tuple[str, int], dict] = {}
        self._finished = False
        self.stats: dict[str, Any] = dict(prototype='incremental_viterbi_ctc_ws',
            frames_processed=0, chunks_processed=0, graph_states=size,
            graph_ctc_states=2 * size, coverage=coverage, compute_seconds=0.,
            terminal_candidates=0, output_truncated=0, frame_seconds=frame_seconds,
            context_weight=context_weight, beam=beam, minimum_gap=minimum_gap)

    def create_stream(self) -> StreamingWordSpotter:
        """Create an empty stream sharing this instance's compiled graph.

        Works on either a fresh template or an already used/finished stream.
        Graph arrays are read-only. Vocabulary, terminal metadata and the large
        coverage dictionary are shared and must be treated as read-only by
        callers; all acoustic arrays, detections and counter stats are private.
        No word tokenization or graph construction is repeated.
        """
        stream = copy.copy(self)
        size = len(self.parents)
        stream._blank = np.full(size, -np.inf)
        stream._nonblank = np.full(size, -np.inf)
        stream._blank_gap = np.full(size, -np.inf)
        stream._nonblank_gap = np.full(size, -np.inf)
        stream._blank_start = np.full(size, -1, dtype=np.int64)
        stream._nonblank_start = np.full(size, -1, dtype=np.int64)
        stream._next_frame = None
        stream._detections = {}
        stream._finished = False
        coverage = self.stats['coverage']
        stream.stats = copy.deepcopy(self.stats, {id(coverage): coverage})
        for counter in ('frames_processed', 'chunks_processed', 'terminal_candidates',
                        'output_truncated'):
            stream.stats[counter] = 0
        stream.stats['compute_seconds'] = 0.
        return stream

    def clone_reset(self) -> StreamingWordSpotter:
        """Alias for create_stream(); the source stream is left unchanged."""
        return self.create_stream()

    def _tokenize(self, word):
        choices, missing, used_bytes = [], [], False
        for char in word:
            options = [(self.token_ids[token],) for token in (char, '▁' + char)
                       if token in self.token_ids and self.token_ids[token] != self.blank_id]
            if not options and self.byte_fallback:
                byte_ids = [self.token_ids.get(f'<0x{byte:02X}>') for byte in char.encode('utf-8')]
                if all(token is not None and token != self.blank_id for token in byte_ids):
                    options = [tuple(byte_ids)]
                    used_bytes = True
            if not options:
                missing.append(char)
            choices.append(options)
        if missing:
            return [], sorted(set(missing)), used_bytes, False
        count = math.prod(len(c) for c in choices)
        variants = []
        seen = set()
        # Prioritize all-bare, first-boundary, all-boundary before mixed forms.
        preferred = [tuple(c[0] for c in choices),
                     tuple(c[-1] if i == 0 else c[0] for i, c in enumerate(choices)),
                     tuple(c[-1] for c in choices)]
        for pieces in itertools.chain(preferred, itertools.product(*choices)):
            flattened = tuple(itertools.chain.from_iterable(pieces))
            if flattened not in seen:
                seen.add(flattened)
                variants.append(flattened)
                if len(variants) >= self.max_tokenizations:
                    break
        return variants, [], used_bytes, count > len(variants)

    @staticmethod
    def _rank(hyp):
        return (hyp['acoustic_gap_per_token'], hyp['raw_score'],
                -(hyp['end_frame'] - hyp['start_frame']), -hyp['start_frame'])

    def accept(self, log_probs, frame_offset: int, available_audio_seconds: float) -> list[dict]:
        if self._finished:
            raise RuntimeError('accept after finish')
        values = np.asarray(log_probs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.tokens):
            raise ValueError('log_probs must have shape [frames, vocabulary]')
        if not np.isfinite(values.max(axis=1)).all() or np.isnan(values).any():
            raise ValueError('each frame needs a finite log probability; NaN is invalid')
        if np.any(values > 1e-5):
            raise ValueError('expected log probabilities <= 0')
        if int(frame_offset) != frame_offset or frame_offset < 0:
            raise ValueError('frame_offset must be a nonnegative integer')
        if not math.isfinite(available_audio_seconds) or available_audio_seconds < 0:
            raise ValueError('available_audio_seconds must be finite and nonnegative')
        if self._next_frame is not None and frame_offset != self._next_frame:
            raise ValueError(f'expected contiguous frame_offset {self._next_frame}, got {frame_offset}')
        if not len(values):
            return []
        began = time.perf_counter()
        changed = {}
        p = self.parents
        for local_frame, row in enumerate(values):
            frame = frame_offset + local_frame
            frame_max = float(row.max())
            choose_nb = self._nonblank >= self._blank
            best = np.where(choose_nb, self._nonblank, self._blank)
            best_gap = np.where(choose_nb, self._nonblank_gap, self._blank_gap)
            best_start = np.where(choose_nb, self._nonblank_start, self._blank_start)
            # Root is a fresh, zero-cost entry at EVERY frame.
            best[0], best_gap[0], best_start[0] = 0., 0., frame
            blank = best + row[self.blank_id]
            blank_gap = best_gap + row[self.blank_id] - frame_max
            blank_start = best_start.copy()
            source = np.where(self._repeats, self._blank[p], best[p])
            source_gap = np.where(self._repeats, self._blank_gap[p], best_gap[p])
            source_start = np.where(self._repeats, self._blank_start[p], best_start[p])
            emission = row[self.labels]
            advance = source + emission
            loop = self._nonblank + emission
            # Re-emitting the same label loops at the same node; no edge bonus.
            use_advance = advance >= loop
            nonblank = np.where(use_advance, advance, loop)
            nonblank_gap = np.where(use_advance, source_gap, self._nonblank_gap) + emission - frame_max
            nonblank_start = np.where(use_advance, source_start, self._nonblank_start)
            best_biased = max(0., float(np.max(np.maximum(blank, nonblank) + self._bonus)))
            for score, gap, start in ((blank, blank_gap, blank_start),
                                      (nonblank, nonblank_gap, nonblank_start)):
                dead = ((score + self._bonus < best_biased - self.beam) |
                        (frame - start + 1 > self._max_age))
                score[dead], gap[dead], start[dead] = -np.inf, -np.inf, -1
                score[0], gap[0], start[0] = -np.inf, -np.inf, -1
            self._blank, self._blank_gap, self._blank_start = blank, blank_gap, blank_start
            self._nonblank, self._nonblank_gap, self._nonblank_start = nonblank, nonblank_gap, nonblank_start
            nodes = self._terminal_nodes
            valid = np.isfinite(nonblank[nodes]) & (nonblank_gap[nodes] / self.depths[nodes] >= self.minimum_gap)
            for node in nodes[valid]:
                start = int(nonblank_start[node])
                for word in self._terminals[int(node)]:
                    if frame - start + 1 > math.ceil(len(word) * self.max_seconds_per_character / self.frame_seconds):
                        continue
                    raw = float(nonblank[node])
                    hyp = dict(word=word, start_frame=start, end_frame=int(frame),
                        score=raw + float(self._bonus[node]), raw_score=raw,
                        acoustic_gap_per_token=min(0., float(nonblank_gap[node]) / int(self.depths[node])),
                        token_count=int(self.depths[node]), word_length=len(word),
                        available_audio_seconds=float(available_audio_seconds),
                        start_seconds=start * self.frame_seconds,
                        end_seconds=(int(frame) + 1) * self.frame_seconds)
                    self.stats['terminal_candidates'] += 1
                    key = (word, start)
                    old = self._detections.get(key)
                    if old is None or self._rank(hyp) > self._rank(old):
                        self._detections[key] = hyp
                        changed[key] = hyp
        self._next_frame = frame_offset + len(values)
        self.stats['frames_processed'] += len(values)
        self.stats['chunks_processed'] += 1
        self.stats['compute_seconds'] += time.perf_counter() - began
        ranked = sorted(changed.values(), key=lambda h: (-h['acoustic_gap_per_token'],
                        -h['raw_score'], h['start_frame'], h['word']))
        self.stats['output_truncated'] += max(0, len(ranked) - self.max_hypotheses_per_chunk)
        return [dict(h) for h in ranked[:self.max_hypotheses_per_chunk]]

    def finish(self) -> list[dict]:
        """Consolidate overlapping detections per word; keep disjoint repeats.

        This is greedy score-ranked interval suppression, not a global word
        alignment. Different words may overlap; downstream selection needs ASR
        evidence and thresholds calibrated on development data.
        """
        self._finished = True
        selected: list[dict] = []
        per_word: dict[str, list[dict]] = {}
        for hyp in sorted(self._detections.values(), key=self._rank, reverse=True):
            siblings = per_word.setdefault(hyp['word'], [])
            if any(hyp['start_frame'] <= h['end_frame'] and h['start_frame'] <= hyp['end_frame']
                   for h in siblings):
                continue
            siblings.append(hyp)
            selected.append(dict(hyp))
        return sorted(selected, key=lambda h: (h['start_frame'], h['end_frame'], h['word']))
