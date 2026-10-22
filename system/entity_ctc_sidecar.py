"""Optional per-session incremental CTC evidence for entity audio repair.

The acoustic model and immutable context graphs are shared. Every recording
owns its encoder caches, word-spotter state and one ordered CPU worker. The
inference code is the implementation verified by the isolated comparison; it
is imported lazily so installations with this feature disabled need no ORT.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time


class EntityCtcRuntime:
    max_active_sessions = 8

    def __init__(self, model_dir: Path, *, num_threads: int = 2):
        from experiments.scripts.entity_ctc_frontend import StreamingCtcModel

        directory = Path(model_dir).resolve()
        self.model = StreamingCtcModel(directory / "model.int8.onnx",
                                       directory / "tokens.txt", num_threads)
        self._templates = OrderedDict()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.max_active_sessions)
        self.model_dir = directory

    def create_session(self, definitions, *, minimum_gap: float = -2.,
                       minimum_margin: float = .1):
        from experiments.scripts.entity_ctc_word_spotter import StreamingWordSpotter

        definitions = tuple(d for d in definitions if d.enabled and d.normalization_policy == "normalize")
        words = tuple(sorted({d.canonical_text for d in definitions if len(d.canonical_text) >= 2}))
        if not words:
            return None
        if not self._slots.acquire(blocking=False):
            return None
        try:
            with self._lock:
                template = self._templates.get(words)
                if template is None:
                    template = StreamingWordSpotter(
                        self.model.tokens, list(words), context_weight=2., beam=12.,
                        minimum_gap=-3., blank_id=self.model.blank_id,
                        frame_seconds=self.model.frame_shift_seconds)
                    self._templates[words] = template
                    if len(self._templates)>8:
                        self._templates.popitem(last=False)
                else:
                    self._templates.move_to_end(words)
            return EntityCtcSession(self.model, template.create_stream(), definitions,
                                    minimum_gap, minimum_margin,
                                    on_close=self._slots.release)
        except Exception:
            self._slots.release()
            raise


class EntityCtcSession:
    max_pending_audio_seconds = 8.

    def __init__(self, model, spotter, definitions, minimum_gap, minimum_margin,
                 on_close=None):
        self.model = model
        self.stream = model.create_stream()
        self.spotter = spotter
        self.definitions = definitions
        self.minimum_gap = minimum_gap
        self.minimum_margin = minimum_margin
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="entity-ctc")
        self._futures = []
        # Retain roughly the most recent 80 seconds of chunk evidence. Full
        # encoder caches remain stateful; old published entity patches live in
        # the audio-repair controller, not in this event history.
        self._events = deque(maxlen=256)
        self._total_events = 0
        self._final_hypotheses = None
        self._audio_bytes = 0
        self._processed_audio_bytes = 0
        self._closed = False
        self._finished = False
        self._error = None
        self._on_close = on_close
        self.compute_seconds = 0.

    @property
    def audio_seconds(self):
        return self._audio_bytes / 64000

    @property
    def event_count(self):
        return self._total_events

    @property
    def error(self):
        return self._error

    def submit_audio(self, pcm: bytes):
        """Queue only newly arrived 16-kHz mono float32 PCM, in order."""
        if self._closed or self._finished:
            return
        if self._error is not None:
            raise RuntimeError("CTC worker failed") from self._error
        if len(pcm) % 4:
            raise ValueError("CTC audio needs complete float32 samples")
        if not pcm:
            return
        pending_bytes = self._audio_bytes + len(pcm) - self._processed_audio_bytes
        if pending_bytes > int(self.max_pending_audio_seconds * 64000):
            raise RuntimeError("CTC audio backlog exceeded its per-session limit")
        self._audio_bytes += len(pcm)
        future = self._worker.submit(self._consume, bytes(pcm))
        future.add_done_callback(self._remember_failure)
        if len(self._futures)>2:
            self._futures=self._futures[-1:]
        self._futures.append(future)

    def _remember_failure(self, future):
        if not future.cancelled():
            error=future.exception()
            if error is not None:self._error=error

    def _consume(self, pcm):
        import numpy as np

        began = time.perf_counter()
        try:
            audio = np.frombuffer(pcm, dtype="<f4")
            if not np.isfinite(audio).all():
                raise ValueError("CTC audio contains a non-finite sample")
            for block in self.stream.accept_audio(audio):
                hypotheses = self.spotter.accept(block["log_probs"], block["frame_offset"],
                                                  block["available_audio_seconds"])
                self._events.append(dict(
                    audio_seconds=block["available_audio_seconds"],
                    after_input_finished=False, ctc_text=block["text"],
                    ctc_tokens=list(block["token_ids"]),
                    ctc_frame_times=list(self.stream.token_frames[:len(block["token_ids"])]),
                    hypotheses=hypotheses))
                self._total_events += 1
        finally:
            self._processed_audio_bytes += len(pcm)
            self.compute_seconds += time.perf_counter() - began

    def _finish(self):
        began = time.perf_counter()
        for block in self.stream.finish():
            hypotheses = self.spotter.accept(block["log_probs"], block["frame_offset"],
                                              block["available_audio_seconds"])
            self._events.append(dict(
                audio_seconds=block["available_audio_seconds"],
                after_input_finished=True, ctc_text=block["text"],
                ctc_tokens=list(block["token_ids"]),
                ctc_frame_times=list(self.stream.token_frames[:len(block["token_ids"])]),
                hypotheses=hypotheses))
            self._total_events += 1
        self._final_hypotheses = self.spotter.finish()
        self.compute_seconds += time.perf_counter() - began

    async def finish(self, *, timeout: float = 1.):
        if self._closed or self._finished:
            return
        if self._error is not None:
            raise RuntimeError("CTC worker failed") from self._error
        self._finished = True
        future = self._worker.submit(self._finish)
        self._futures.append(future)
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
        except Exception as error:
            self._error = error
            raise

    def propose(self, raw_text: str, *, final: bool = False):
        """Return localized CTC-supported candidates against untouched Qwen text."""
        from experiments.scripts.evaluate_entity_ctc_fusion import fuse

        if self._closed or self._error or not raw_text:
            return []
        # A worker may be appending the next immutable event. A list snapshot
        # is safe under the GIL and never includes unsubmitted future audio.
        events = list(self._events)
        if final and self._final_hypotheses is None:
            return []
        last = events[-1] if events else {}
        record = dict(audio_seconds=self.audio_seconds, events=events,
                      ctc_tokens=last.get("ctc_tokens", []),
                      ctc_token_frames=last.get("ctc_frame_times", []),
                      hypotheses=self._final_hypotheses or [])
        _, proposals = fuse(raw_text, record, self.definitions, self.model.tokens,
                            self.minimum_gap, self.minimum_margin,
                            available=self.audio_seconds, final=final)
        return proposals

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._worker.shutdown(wait=False, cancel_futures=True)
        if self._on_close is not None:
            self._on_close()
            self._on_close = None
