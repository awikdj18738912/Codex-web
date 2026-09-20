"""Local vLLM streaming service for Qwen3-ASR."""

from __future__ import annotations

import argparse
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from .audio_activity import is_silence, pcm_peak, pcm_rms
from .backends import SAMPLE_RATE, VAD_WINDOW, build_vad
from .qwen_confidence import QwenVLLMConfidenceDecoder, state_confidence_payload
from .stream_vad import StreamingVADGate


@dataclass
class Session:
    state: object
    last_seen: float
    vad_gate: StreamingVADGate | None = None
    language: str | None = None
    detected_language: str = ""
    committed_text: str = ""
    pending_pcm: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32), repr=False
    )
    completed_segments: int = 0
    segment_samples: int = 0
    last_confidence: dict[str, object] | None = None
    silence_skipped_chunks: int = 0
    vad_skipped_chunks: int = 0
    vad_last_speech: bool = False
    vad_segment_ended: bool = False
    cancelled: bool = False
    chunk_size_seconds: float | None = None
    unfixed_chunk_num: int | None = None
    unfixed_token_num: int | None = None


_SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’》〉』】）)]*$")


def _scope_confidence_payload(
    payload: dict[str, object],
    *,
    has_committed_prefix: bool,
) -> dict[str, object]:
    """Describe which portion of the returned transcript owns the score."""

    scoped = dict(payload)
    token_count = scoped.get("confidence_token_count")
    available = (
        scoped.get("confidence") is not None
        and isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count > 0
    )
    covers_full_state = (
        available and scoped.get("confidence_covers_full_state") is True
    )
    covers_full_text = covers_full_state and not has_committed_prefix
    if not available:
        scope = "unavailable"
    elif covers_full_text:
        scope = "full_text"
    elif covers_full_state:
        scope = "current_state"
    else:
        scope = "generated_suffix"
    scoped["confidence_covers_full_text"] = covers_full_text
    scoped["confidence_scope"] = scope
    return scoped


def _join_transcripts(prefix: str, suffix: str) -> str:
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    separator = " " if prefix[-1].isascii() and suffix[0].isascii() else ""
    return f"{prefix}{separator}{suffix}"


def _should_rotate_segment(
    text: str,
    sample_count: int,
    soft_limit_samples: int,
    hard_limit_samples: int,
) -> bool:
    if sample_count >= hard_limit_samples:
        return True
    return sample_count >= soft_limit_samples and bool(
        _SENTENCE_END_RE.search(text.strip())
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local Qwen3-ASR streaming")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM context limit; bounded streaming does not require the model's 65K maximum",
    )
    parser.add_argument("--chunk-size-seconds", type=float, default=1.0)
    parser.add_argument(
        "--segmentation-mode",
        choices=("continuous", "vad_finalize"),
        default="vad_finalize",
        help="continuous keeps one ASR state; vad_finalize starts a new state at each VAD endpoint",
    )
    parser.add_argument(
        "--asr-send-chunk-seconds",
        type=float,
        default=0.5,
        help="speech audio aggregation size before sending it to Qwen",
    )
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument(
        "--silence-rms-threshold",
        type=float,
        default=0.002,
        help="RMS fallback for chunks below this level; set 0 to disable",
    )
    parser.add_argument(
        "--vad",
        choices=("off", "silero", "energy", "firered"),
        default="silero",
        help="speech activity detector used before Qwen decoding",
    )
    parser.add_argument(
        "--vad-model",
        type=Path,
        default=Path("models/silero_vad.onnx"),
        help="Silero VAD ONNX model path",
    )
    parser.add_argument(
        "--firered-dir",
        type=Path,
        default=Path("models/firered_vad"),
        help="FireRedVAD model directory when --vad firered is selected",
    )
    parser.add_argument(
        "--vad-provider",
        default="cpu",
        help="sherpa-onnx provider for Silero VAD (normally cpu)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="speech probability threshold for Silero/FireRed VAD",
    )
    parser.add_argument(
        "--vad-min-silence",
        type=float,
        default=0.7,
        help="seconds of silence required to end a speech region",
    )
    parser.add_argument(
        "--vad-min-speech",
        type=float,
        default=0.25,
        help="seconds of speech required to start a speech region",
    )
    parser.add_argument(
        "--vad-energy-threshold",
        type=float,
        default=0.02,
        help="RMS threshold when --vad energy is selected",
    )
    parser.add_argument(
        "--vad-preroll",
        type=float,
        default=0.7,
        help="seconds of audio retained before VAD speech start",
    )
    parser.add_argument(
        "--vad-tail-pad",
        type=float,
        default=1.0,
        help="seconds of silence appended before finishing a VAD segment",
    )
    parser.add_argument(
        "--confidence-logprobs",
        type=int,
        default=0,
        help="capture this many vLLM token alternatives; 0 keeps the original fast path",
    )
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=1.0,
        help="temperature used to convert mean token logprob into a score",
    )
    parser.add_argument(
        "--confidence-calibrated",
        action="store_true",
        help="mark the score as calibrated after fitting temperature on a dev set",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="prefer starting a fresh bounded ASR state after this duration",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=45.0,
        help="always start a fresh ASR state after this duration",
    )
    args = parser.parse_args(argv)
    if not args.model.exists():
        parser.error(f"Qwen3-ASR model not found: {args.model}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len < 1024:
        parser.error("--max-model-len must be at least 1024")
    if args.chunk_size_seconds <= 0:
        parser.error("--chunk-size-seconds must be positive")
    if args.asr_send_chunk_seconds <= 0:
        parser.error("--asr-send-chunk-seconds must be positive")
    if args.segment_seconds <= 0:
        parser.error("--segment-seconds must be positive")
    if args.max_segment_seconds < args.segment_seconds:
        parser.error("--max-segment-seconds must be at least --segment-seconds")
    if args.confidence_logprobs < 0:
        parser.error("--confidence-logprobs must be non-negative")
    if args.confidence_temperature <= 0:
        parser.error("--confidence-temperature must be positive")
    if args.confidence_calibrated and args.confidence_logprobs == 0:
        parser.error("--confidence-calibrated requires --confidence-logprobs")
    if args.silence_rms_threshold < 0:
        parser.error("--silence-rms-threshold must be non-negative")
    if not 0 <= args.vad_threshold <= 1:
        parser.error("--vad-threshold must be between 0 and 1")
    if args.vad_min_silence <= 0:
        parser.error("--vad-min-silence must be positive")
    if args.vad_min_speech <= 0:
        parser.error("--vad-min-speech must be positive")
    if args.vad_energy_threshold < 0:
        parser.error("--vad-energy-threshold must be non-negative")
    if args.vad_preroll < 0:
        parser.error("--vad-preroll must be non-negative")
    if args.vad_tail_pad < 0:
        parser.error("--vad-tail-pad must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise RuntimeError("Streaming service requires `qwen-asr[vllm]`") from error

    print("Loading Qwen3-ASR vLLM streaming backend...", flush=True)
    asr = Qwen3ASRModel.LLM(
        model=str(args.model.resolve()),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=32,
    )
    decoder = (
        QwenVLLMConfidenceDecoder(
            asr,
            top_logprobs=args.confidence_logprobs,
            temperature=args.confidence_temperature,
            calibrated=args.confidence_calibrated,
        )
        if args.confidence_logprobs
        else None
    )
    asr_runner = decoder or asr
    app = Flask(__name__)
    sessions: dict[str, Session] = {}
    model_lock = threading.Lock()
    session_lock = threading.Lock()
    soft_limit_samples = round(args.segment_seconds * SAMPLE_RATE)
    hard_limit_samples = round(args.max_segment_seconds * SAMPLE_RATE)
    asr_send_chunk_samples = max(
        VAD_WINDOW, round(args.asr_send_chunk_seconds * SAMPLE_RATE)
    )
    vad_tail_pad_samples = round(args.vad_tail_pad * SAMPLE_RATE)

    if args.vad == "silero" and not args.vad_model.exists():
        raise RuntimeError(
            f"Silero VAD model not found: {args.vad_model}. "
            "Run `bash system/download_vad.sh models` or use --vad off."
        )

    def new_vad_gate() -> StreamingVADGate | None:
        if args.vad == "off":
            return None
        detector = build_vad(
            args.vad,
            str(args.vad_model.resolve()),
            str(args.firered_dir.resolve()),
            args.vad_threshold,
            args.vad_min_silence,
            args.vad_min_speech,
            args.vad_energy_threshold,
            args.vad_provider,
        )
        return StreamingVADGate(
            detector,
            sample_rate=SAMPLE_RATE,
            preroll_seconds=args.vad_preroll,
            window_samples=VAD_WINDOW,
        )

    def new_state(
        language: str | None,
        *,
        chunk_size_seconds: float | None = None,
        unfixed_chunk_num: int | None = None,
        unfixed_token_num: int | None = None,
    ):
        return asr_runner.init_streaming_state(
            language=language,
            unfixed_chunk_num=(
                args.unfixed_chunk_num
                if unfixed_chunk_num is None
                else unfixed_chunk_num
            ),
            unfixed_token_num=(
                args.unfixed_token_num
                if unfixed_token_num is None
                else unfixed_token_num
            ),
            chunk_size_sec=(
                args.chunk_size_seconds
                if chunk_size_seconds is None
                else chunk_size_seconds
            ),
        )

    def feed_speech(session: Session, samples: np.ndarray) -> None:
        """Buffer speech before sending it to Qwen in stable-sized batches."""

        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        session.pending_pcm = np.concatenate((session.pending_pcm, chunk))
        session.segment_samples += chunk.size
        while session.pending_pcm.size >= asr_send_chunk_samples:
            asr_runner.streaming_transcribe(
                session.pending_pcm[:asr_send_chunk_samples], session.state
            )
            session.pending_pcm = session.pending_pcm[asr_send_chunk_samples:]

    def has_active_audio(session: Session) -> bool:
        return bool(
            session.segment_samples
            or session.pending_pcm.size
            or (getattr(session.state, "text", "") or "").strip()
        )

    def finalize_segment(
        session: Session, *, reset_state: bool = True
    ) -> dict[str, object] | None:
        """Finish one speech region and optionally prepare the next state."""

        if session.pending_pcm.size:
            asr_runner.streaming_transcribe(session.pending_pcm, session.state)
            session.pending_pcm = np.zeros(0, dtype=np.float32)
        if vad_tail_pad_samples:
            asr_runner.streaming_transcribe(
                np.zeros(vad_tail_pad_samples, dtype=np.float32), session.state
            )
        asr_runner.finish_streaming_transcribe(session.state)
        segment_text = str(getattr(session.state, "text", "") or "").strip()
        detected_language = getattr(session.state, "language", "") or ""
        if detected_language:
            session.detected_language = detected_language
        segment_confidence = _scope_confidence_payload(
            state_confidence_payload(session.state),
            has_committed_prefix=bool(session.committed_text.strip()),
        )
        session.last_confidence = segment_confidence
        completed_event: dict[str, object] | None = None
        if segment_text:
            session.committed_text = _join_transcripts(
                session.committed_text, segment_text
            )
            session.completed_segments += 1
            completed_event = {
                "segment_id": session.completed_segments,
                "text": segment_text,
                "vad_boundary": True,
                "confidence": segment_confidence.get("confidence"),
                "confidence_metadata": segment_confidence,
            }
        session.segment_samples = 0
        session.vad_segment_ended = True
        if reset_state:
            session.state = new_state(
                session.language,
                chunk_size_seconds=session.chunk_size_seconds,
                unfixed_chunk_num=session.unfixed_chunk_num,
                unfixed_token_num=session.unfixed_token_num,
            )
        return completed_event

    def start_session(
        language: str | None,
        *,
        chunk_size_seconds: float | None = None,
        unfixed_chunk_num: int | None = None,
        unfixed_token_num: int | None = None,
    ):
        with model_lock:
            state = new_state(
                language,
                chunk_size_seconds=chunk_size_seconds,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
            )
            vad_gate = new_vad_gate()
        session_id = uuid.uuid4().hex
        with session_lock:
            sessions[session_id] = Session(
                state=state,
                last_seen=time.monotonic(),
                vad_gate=vad_gate,
                language=language,
                chunk_size_seconds=chunk_size_seconds,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
            )
        return jsonify(session_id=session_id)

    def full_text(session: Session) -> str:
        return _join_transcripts(
            session.committed_text,
            getattr(session.state, "text", "") or "",
        )

    def get_session(session_id: str) -> Session | None:
        with session_lock:
            session = sessions.get(session_id)
            if session is not None:
                session.last_seen = time.monotonic()
            return session

    def confidence_payload(session: Session) -> dict[str, object]:
        current = state_confidence_payload(session.state)
        if current.get("confidence") is not None:
            return _scope_confidence_payload(
                current,
                has_committed_prefix=bool(session.committed_text.strip()),
            )
        # Once a new state has produced text without usable logprobs, an older
        # segment score no longer describes the returned transcript.
        if (getattr(session.state, "text", "") or "").strip():
            return _scope_confidence_payload(
                current,
                has_committed_prefix=bool(session.committed_text.strip()),
            )
        return session.last_confidence or _scope_confidence_payload(
            current,
            has_committed_prefix=bool(session.committed_text.strip()),
        )

    def response_payload(
        session: Session,
        *,
        audio_rms: float = 0.0,
        audio_peak: float = 0.0,
        silence_skipped: bool = False,
        completed_segment_events: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        response: dict[str, object] = {
            "language": session.detected_language,
            "text": full_text(session),
            "audio_rms": audio_rms,
            "audio_peak": audio_peak,
            "silence_skipped": silence_skipped,
            "silence_skipped_chunks": session.silence_skipped_chunks,
            "vad": args.vad,
            "vad_speech_detected": session.vad_last_speech,
            "vad_segment_ended": session.vad_segment_ended,
            "vad_skipped_chunks": session.vad_skipped_chunks,
            "asr_segment_count": session.completed_segments,
            "segmentation_mode": args.segmentation_mode,
            # Events are scoped to this request.  Consumers can process each
            # VAD-finalized source segment exactly once instead of trying to
            # recover boundaries from the cumulative transcript.
            "completed_segments": completed_segment_events or [],
            "active_segment": str(
                getattr(session.state, "text", "") or ""
            ).strip(),
        }
        response.update(confidence_payload(session))
        return response

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "mode": "streaming",
            "api_styles": ["current", "legacy_v1"],
            "confidence_logprobs": args.confidence_logprobs,
            "confidence_calibrated": args.confidence_calibrated,
            "silence_rms_threshold": args.silence_rms_threshold,
            "vad": args.vad,
            "vad_model": str(args.vad_model),
            "vad_threshold": args.vad_threshold,
            "vad_min_silence": args.vad_min_silence,
            "vad_min_speech": args.vad_min_speech,
            "vad_preroll": args.vad_preroll,
            "vad_tail_pad": args.vad_tail_pad,
            "asr_send_chunk_seconds": args.asr_send_chunk_seconds,
            "segmentation_mode": args.segmentation_mode,
        }

    @app.post("/stream/start")
    def start():
        language = request.args.get("language") or None
        return start_session(language)

    @app.post("/v1/stream/start")
    def legacy_start():
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return jsonify(error="JSON request body must be an object"), 400
        language_value = payload.get("language")
        language = language_value if isinstance(language_value, str) else None
        try:
            chunk_size_seconds = float(
                payload.get("chunk_seconds", args.chunk_size_seconds)
            )
            unfixed_chunk_num = int(
                payload.get("unfixed_chunk_num", args.unfixed_chunk_num)
            )
            unfixed_token_num = int(
                payload.get("unfixed_token_num", args.unfixed_token_num)
            )
        except (TypeError, ValueError):
            return jsonify(error="invalid legacy streaming configuration"), 400
        if (
            chunk_size_seconds <= 0
            or unfixed_chunk_num < 0
            or unfixed_token_num < 0
        ):
            return jsonify(error="invalid legacy streaming configuration"), 400
        return start_session(
            language,
            chunk_size_seconds=chunk_size_seconds,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
        )

    def process_chunk(session_id: str):
        session = get_session(session_id)
        if session is None:
            return jsonify(error="invalid session_id"), 400
        payload = request.get_data(cache=False)
        if len(payload) % 4:
            return jsonify(error="PCM payload must contain float32 samples"), 400
        pcm16k = np.frombuffer(payload, dtype=np.float32)
        audio_rms = pcm_rms(pcm16k)
        audio_peak = pcm_peak(pcm16k)
        forwarded_pcm: np.ndarray | None = pcm16k
        vad_skipped = False
        completed_segment_events: list[dict[str, object]] = []
        session.vad_segment_ended = False
        with model_lock:
            # The browser may disconnect while this request waits behind a
            # long GPU call. Do not spend more GPU time on a session that was
            # cancelled in the meantime.
            if session.cancelled:
                return jsonify(error="session cancelled"), 409
            if (
                args.segmentation_mode == "vad_finalize"
                and session.vad_gate is not None
            ):
                # Transport chunks are large for HTTP efficiency, but VAD
                # transitions are observed at the original 512-sample
                # (~32 ms) granularity.
                any_forwarded = False
                for offset in range(0, pcm16k.size, VAD_WINDOW):
                    vad_window = pcm16k[offset : offset + VAD_WINDOW]
                    decision = session.vad_gate.accept(vad_window)
                    session.vad_last_speech = decision.speech_detected
                    if decision.forward is not None and decision.forward.size:
                        any_forwarded = True
                        feed_speech(session, decision.forward)
                    if decision.ended:
                        event = finalize_segment(session)
                        if event is not None:
                            completed_segment_events.append(event)
                forwarded_pcm = pcm16k if any_forwarded else None
                vad_skipped = not any_forwarded
                if not any_forwarded:
                    session.silence_skipped_chunks += 1
                    session.vad_skipped_chunks += 1
            elif session.vad_gate is not None:
                decision = session.vad_gate.accept(pcm16k)
                session.vad_last_speech = decision.speech_detected
                forwarded_pcm = decision.forward
                vad_skipped = forwarded_pcm is None
            elif args.silence_rms_threshold > 0 and is_silence(
                pcm16k, rms_threshold=args.silence_rms_threshold
            ):
                # RMS remains an explicit fallback when VAD is disabled.
                forwarded_pcm = None
                vad_skipped = True

            if (
                args.segmentation_mode == "vad_finalize"
                and session.vad_gate is not None
            ):
                # Speech was fed window-by-window above.  VAD endpoints
                # already finish and rotate the state.
                pass
            elif forwarded_pcm is None or forwarded_pcm.size == 0:
                session.silence_skipped_chunks += 1
                if session.vad_gate is not None:
                    session.vad_skipped_chunks += 1
            else:
                asr_runner.streaming_transcribe(forwarded_pcm, session.state)
                session.segment_samples += forwarded_pcm.size
            segment_text = getattr(session.state, "text", "") or ""
            detected_language = getattr(session.state, "language", "") or ""
            if detected_language:
                session.detected_language = detected_language
            if (
                (
                    args.segmentation_mode != "vad_finalize"
                    or session.vad_gate is None
                )
                and forwarded_pcm is not None
                and _should_rotate_segment(
                    segment_text,
                    session.segment_samples,
                    soft_limit_samples,
                    hard_limit_samples,
                )
            ):
                event = finalize_segment(session)
                if event is not None:
                    completed_segment_events.append(event)
        return jsonify(
            response_payload(
                session,
                audio_rms=audio_rms,
                audio_peak=audio_peak,
                silence_skipped=vad_skipped,
                completed_segment_events=completed_segment_events,
            )
        )

    @app.post("/stream/chunk")
    def chunk():
        return process_chunk(request.args.get("session_id", ""))

    @app.post("/v1/stream/<session_id>/chunk")
    def legacy_chunk(session_id: str):
        return process_chunk(session_id)

    def finish_session(session_id: str):
        with session_lock:
            session = sessions.pop(session_id, None)
        if session is None:
            return jsonify(error="invalid session_id"), 400
        completed_segment_events: list[dict[str, object]] = []
        with model_lock:
            if has_active_audio(session):
                event = finalize_segment(session)
                if event is not None:
                    completed_segment_events.append(event)
        return jsonify(
            response_payload(
                session,
                completed_segment_events=completed_segment_events,
            )
        )

    @app.post("/stream/finish")
    def finish():
        return finish_session(request.args.get("session_id", ""))

    @app.post("/v1/stream/<session_id>/finish")
    def legacy_finish(session_id: str):
        return finish_session(session_id)

    def cancel_session(session_id: str):
        with session_lock:
            session = sessions.pop(session_id, None)
            if session is not None:
                session.cancelled = True
        if session is None:
            return jsonify(error="invalid session_id"), 400
        return jsonify(cancelled=True)

    @app.post("/stream/cancel")
    def cancel():
        return cancel_session(request.args.get("session_id", ""))

    @app.delete("/v1/stream/<session_id>")
    def legacy_cancel(session_id: str):
        return cancel_session(session_id)

    print(f"Streaming service listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(1) from error
