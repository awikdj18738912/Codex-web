"""Microphone AgenticASR client using a local Qwen3-ASR service.

Run ``system.qwen_asr_server`` in the qwen3-asr environment first. This client
runs in the agentic-asr environment and sends each VAD-delimited utterance to
that localhost service before applying the local Transformers Refiner.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .backends import EnergyVad, SAMPLE_RATE, VAD_WINDOW, normalize_cjk
from .deterministic_cleanup import clean_transcript_deterministically
from .entity_store import EntityStore
from .entity_matcher import EntityCandidateMatcher, EntityFuzzyMode
from .entity_pipeline import (
    FinalizedEntitySegment,
    finalize_entity_segment,
    has_placeholder_failure,
    has_retryable_integrity_failure,
    prepare_entity_segment,
)
from .protection import EntityProtector
from .refinement_gate import RefinementGate, RefinementGateMode
from .refinement_guard import preserve_safe_repetition_edits
from .numeric_normalizer import ContextualNumericNormalizer
from .refinement_protocol import (
    STRICT_PLACEHOLDER_PROMPT,
    STRUCTURED_REFINER_SYSTEM_PROMPT,
    apply_structured_patch,
)
from .session_memory import SessionEntityMemory

SYSTEM_PROMPT = STRUCTURED_REFINER_SYSTEM_PROMPT

class TransformersRefiner:
    """Local Transformers backend matching the offline Refiner prompt."""

    # Prevent a malformed or unusually long request from keeping the WebSocket
    # open forever. ``generate(max_time=...)`` returns the best partial output
    # available; the caller quality-checks it and falls back to source text
    # when it is incomplete.
    GENERATION_MAX_TIME_SECONDS = 12.0

    def __init__(self, model_path: Path, device_map: str, max_new_tokens: int) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Refiner requires torch and transformers in the active environment"
            ) from error
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype="auto", device_map=device_map
        ).eval()
        self._max_new_tokens = max_new_tokens
        template = self._tokenizer.get_chat_template()
        self._template_kwargs = (
            {"enable_thinking": False} if "enable_thinking" in template else {}
        )

    def refine(
        self,
        raw_text: str,
        *,
        entity_hints: Iterable[str] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        hints = tuple(dict.fromkeys(item.strip() for item in entity_hints if item.strip()))
        user_content = raw_text
        if hints:
            user_content = f"{raw_text}\n<KEY>[{'、'.join(hints[:16])}]"
        messages = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}{STRICT_PLACEHOLDER_PROMPT}"
                    if strict_placeholders
                    else SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": user_content},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self._template_kwargs,
        )
        device = self._model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                max_time=self.GENERATION_MAX_TIME_SECONDS,
                do_sample=False
            )
        input_width = inputs["input_ids"].shape[1]
        text = self._tokenizer.decode(
            generated[0, input_width:], skip_special_tokens=True
        ).strip()
        return text, (time.perf_counter() - started) * 1000

    def review_repetition(self, text: str) -> tuple[str, float]:
        """Refine only the candidate's local transcript context."""

        return self.refine(text)


def _sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Microphone input requires sounddevice and the system PortAudio library. "
            "On Ubuntu: `sudo apt install libportaudio2`"
        ) from error
    return sd


def _wav_payload(samples: list[np.ndarray]) -> tuple[bytes, float]:
    audio = np.concatenate(samples).astype("float32", copy=False)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue(), len(audio) / SAMPLE_RATE


def _transcribe(
    asr_url: str, audio: bytes, language: str | None, timeout: float
) -> tuple[str, str | None, float | None]:
    query = urllib.parse.urlencode({"language": language}) if language else ""
    separator = "&" if "?" in asr_url else "?"
    url = f"{asr_url}{separator}{query}" if query else asr_url
    request = urllib.request.Request(
        url, data=audio, headers={"Content-Type": "audio/wav"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Qwen3-ASR service request failed: {error}") from error
    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Qwen3-ASR returned no text: {payload}")
    detected_language = payload.get("language")
    confidence_value = payload.get("confidence")
    confidence = (
        float(confidence_value)
        if isinstance(confidence_value, (int, float))
        and not isinstance(confidence_value, bool)
        and 0 <= float(confidence_value) <= 1
        else None
    )
    return (
        text,
        detected_language if isinstance(detected_language, str) else None,
        confidence,
    )


def _append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _patch_audit(
    payload: dict[str, object] | None, issue: str | None
) -> dict[str, object]:
    """Make structured-patch decisions visible in the JSONL audit record."""

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microphone AgenticASR via a local Qwen3-ASR service"
    )
    parser.add_argument("--refiner-model", type=Path)
    parser.add_argument("--refiner-device", default="cuda:1")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8765/transcribe")
    parser.add_argument("--asr-timeout", type=float, default=120.0)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--language", default="Chinese", help="use auto for automatic detection")
    parser.add_argument("--vad-threshold", type=float, default=0.02)
    parser.add_argument("--vad-min-silence", type=float, default=0.7)
    parser.add_argument("--vad-min-speech", type=float, default=0.25)
    parser.add_argument("--preroll", type=float, default=0.5)
    parser.add_argument("--refiner-max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, help="append utterance records as JSONL")
    parser.add_argument(
        "--entity-db",
        type=Path,
        help="optional SQLite database containing verified protected entities",
    )
    parser.add_argument(
        "--entity-domain",
        default="general",
        help="load this entity domain plus general",
    )
    parser.add_argument(
        "--entity-fuzzy-mode",
        choices=[mode.value for mode in EntityFuzzyMode],
        default="shadow",
        help="off, shadow, hint, or automatic normalization for complete utterances",
    )
    parser.add_argument(
        "--refinement-gate-mode",
        choices=[mode.value for mode in RefinementGateMode],
        default="off",
        help="off preserves the baseline; conservative enables pre-Refiner routing",
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
    if args.preroll < 0 or args.asr_timeout <= 0:
        parser.error("--preroll must be non-negative and --asr-timeout must be positive")
    if args.vad_min_silence <= 0 or args.vad_min_speech <= 0:
        parser.error("VAD durations must be positive")
    if args.refiner_max_new_tokens < 1:
        parser.error("--refiner-max-new-tokens must be at least 1")
    if not args.entity_domain.strip():
        parser.error("--entity-domain must not be empty")
    if not args.list_devices and args.refiner_model is None:
        parser.error("--refiner-model is required for transcription")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sd = _sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return 0

    refiner_model_path = args.refiner_model.resolve()
    if not refiner_model_path.exists():
        raise FileNotFoundError(f"Refiner model not found: {refiner_model_path}")
    print("Loading AgenticASR Refiner...", flush=True)
    refiner = TransformersRefiner(
        refiner_model_path, args.refiner_device, args.refiner_max_new_tokens
    )
    memory = SessionEntityMemory()
    if args.entity_db is not None:
        entity_store = EntityStore(args.entity_db.resolve())
        entity_definitions = entity_store.list_entities(domain=args.entity_domain.strip())
        print(
            f"Loaded {len(entity_definitions)} protected entities from "
            f"{args.entity_db.resolve()}",
            flush=True,
        )
    else:
        entity_definitions = ()
    protector = EntityProtector(
        entity_definitions,
        session_memory=memory,
        enable_rule_protection=not args.disable_rule_protection,
        # Numeric normalization and value validation make placeholder masking
        # unnecessary here; leave placeholders for terms and fixed idioms.
        protect_numeric_spans=False,
    )
    fuzzy_mode = EntityFuzzyMode.parse(args.entity_fuzzy_mode)
    matcher = (
        EntityCandidateMatcher(
            entity_definitions,
            selected_domain=args.entity_domain.strip(),
            mode=fuzzy_mode,
        )
        if entity_definitions and fuzzy_mode is not EntityFuzzyMode.OFF
        else None
    )
    refinement_gate = RefinementGate(
        args.refinement_gate_mode,
        refine_on_unverified_confidence=(
            RefinementGateMode.parse(args.refinement_gate_mode)
            is not RefinementGateMode.TRI_STATE
        ),
    )
    numeric_normalizer = ContextualNumericNormalizer()
    print(f"Using Qwen3-ASR service at {args.asr_url}")
    print("Listening. Press Ctrl+C to stop.", flush=True)

    vad = EnergyVad(args.vad_threshold, args.vad_min_silence, args.vad_min_speech)
    windows: queue.Queue[np.ndarray] = queue.Queue()
    preroll = deque(maxlen=max(1, round(args.preroll * SAMPLE_RATE / VAD_WINDOW)))
    active = False
    utterance: list[np.ndarray] = []
    language = None if args.language.lower() == "auto" else args.language

    def transcribe_and_refine(samples: list[np.ndarray]) -> None:
        payload, duration = _wav_payload(samples)
        raw_text, detected_language, asr_confidence = _transcribe(
            args.asr_url, payload, language, args.asr_timeout
        )
        raw_text = normalize_cjk(raw_text).strip()
        if not raw_text:
            return
        prepared = prepare_entity_segment(
            raw_text,
            protector,
            matcher,
            allow_auto=True,
            confidence=asr_confidence,
        )
        protection = prepared.protection
        normalized_baseline = (
            numeric_normalizer.normalize(prepared.baseline_text)
            if args.enable_numeric_normalization and not args.disable_numeric_normalization
            else None
        )
        baseline_text = (
            normalized_baseline.text
            if normalized_baseline is not None
            else prepared.baseline_text
        )
        masked_text = (
            numeric_normalizer.normalize(protection.masked_text).text
            if args.enable_numeric_normalization and not args.disable_numeric_normalization
            else protection.masked_text
        )
        gate_decision = refinement_gate.decide(
            baseline_text,
            asr_confidence=asr_confidence,
            entity_hints=prepared.hints,
            is_final=True,
            numeric_refinement=(
                not args.enable_numeric_normalization
                or args.disable_numeric_normalization
            ),
        )
        refiner_executed = gate_decision.should_refine
        latency_ms = 0.0
        refiner_masked_outputs: list[str] = []
        structured_patch_audits: list[dict[str, object]] = []
        safe_repetition_repairs: list[dict[str, object]] = []
        candidate_for_salvage: str | None = None
        placeholder_retry_count = 0
        refiner_retry_count = 0
        refiner_retry_reasons: tuple[str, ...] = ()
        if gate_decision.should_refine:
            refined_text, latency_ms = refiner.refine(
                masked_text, entity_hints=prepared.hints
            )
            refiner_masked_outputs.append(refined_text)
            structured_candidate, patch_payload, structured_issue = apply_structured_patch(
                masked_text, refined_text
            )
            structured_patch_audits.append(
                _patch_audit(patch_payload, structured_issue)
            )
            finalized = (
                FinalizedEntitySegment(
                    prepared.baseline_text, False, (structured_issue,)
                )
                if structured_issue
                else finalize_entity_segment(
                    structured_candidate, prepared, protector
                )
            )
            if not structured_issue:
                candidate_for_salvage = structured_candidate
            if has_retryable_integrity_failure(finalized.reject_reasons):
                refiner_retry_reasons = finalized.reject_reasons
                placeholder_failed = has_placeholder_failure(
                    finalized.reject_reasons
                )
                refined_text, retry_latency_ms = refiner.refine(
                    masked_text,
                    entity_hints=prepared.hints,
                    strict_placeholders=True,
                )
                latency_ms += retry_latency_ms
                refiner_retry_count = 1
                placeholder_retry_count = int(placeholder_failed)
                refiner_masked_outputs.append(refined_text)
                structured_retry, retry_payload, structured_retry_issue = apply_structured_patch(
                    masked_text, refined_text
                )
                structured_patch_audits.append(
                    _patch_audit(retry_payload, structured_retry_issue)
                )
                finalized = (
                    FinalizedEntitySegment(
                        prepared.baseline_text, False, (structured_retry_issue,)
                    )
                    if structured_retry_issue
                    else finalize_entity_segment(
                        structured_retry, prepared, protector
                    )
                )
                if not structured_retry_issue:
                    candidate_for_salvage = structured_retry
            refined_text = finalized.text
            if finalized.reject_reasons and candidate_for_salvage:
                restored = protector.restore(candidate_for_salvage, protection)
                if restored.accepted:
                    salvaged = preserve_safe_repetition_edits(
                        prepared.baseline_text,
                        restored.text,
                    )
                    if salvaged is not None:
                        refined_text = salvaged
                        safe_repetition_repairs.append(
                            {
                                "source_text": prepared.baseline_text,
                                "salvaged_text": salvaged,
                                "reason": "safe_adjacent_repetition_after_integrity_fallback",
                            }
                        )
        else:
            refined_text = baseline_text
            finalized_accepted = True
            finalized_reasons: tuple[str, ...] = ()
        numeric_changes = ()
        if args.enable_numeric_normalization and not args.disable_numeric_normalization:
            normalized_output = numeric_normalizer.normalize(refined_text)
            refined_text = normalized_output.text
            numeric_changes = (
                normalized_output.changes
                if normalized_output.changes
                else normalized_baseline.changes
            )
        refined_text = clean_transcript_deterministically(refined_text)
        if not gate_decision.should_refine:
            print(
                "[refinement gate] skipped: " + ", ".join(gate_decision.reasons),
                flush=True,
            )
        else:
            finalized_accepted = finalized.accepted
            finalized_reasons = finalized.reject_reasons
        entity_normalizations = prepared.normalizations
        entity_audit_issues = protector.audit_unmasked(refined_text, protection)
        print(f"[raw] {raw_text}")
        print(f"[refined {latency_ms:.0f}ms] {refined_text}", flush=True)
        if refiner_retry_count:
            print(
                "[refinement guard] retried: " + ", ".join(refiner_retry_reasons),
                flush=True,
            )
        if entity_audit_issues:
            print(f"[entity audit] {', '.join(entity_audit_issues)}", flush=True)
        for match in prepared.report.auto_matches:
            print(
                f"[entity auto {match.final_score:.3f}] "
                f"{match.observed} -> {match.canonical}",
                flush=True,
            )
        if args.output:
            _append_record(
                args.output.resolve(),
                {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "audio_seconds": round(duration, 3),
                    "asr_language": detected_language,
                    "asr_confidence": asr_confidence,
                    "output": {
                        "raw_text": raw_text,
                        "clean_text": refined_text,
                        "llm_latency_ms": latency_ms,
                        "refiner_executed": refiner_executed,
                        "refiner_accepted": finalized_accepted,
                        "refiner_reject_reasons": list(finalized_reasons),
                        "placeholder_retry_count": placeholder_retry_count,
                        "refiner_retry_count": refiner_retry_count,
                        "refiner_retry_reasons": list(refiner_retry_reasons),
                        "refiner_masked_outputs": refiner_masked_outputs,
                        "structured_patch_audits": structured_patch_audits,
                        "refinement_gate_mode": refinement_gate.mode.value,
                        "refinement_gate_config": refinement_gate.config_dict(),
                        "refinement_gate_decisions": [gate_decision.public_dict()],
                        "refinement_gate_skipped_segments": int(
                            not gate_decision.should_refine
                        ),
                        "entity_audit_issues": list(entity_audit_issues),
                        "entity_normalizations": list(entity_normalizations),
                        "numeric_normalizations": [
                            change.public_dict() for change in numeric_changes
                        ],
                        "safe_repetition_repairs": safe_repetition_repairs,
                        "protected_entities": [
                            span.public_dict() for span in protection.spans
                        ],
                        "entity_candidates": [
                            match.public_dict() for match in prepared.report.matches
                        ],
                        "entity_matcher_latency_ms": round(
                            prepared.report.matcher_latency_ms, 3
                        ),
                        "entity_fuzzy_mode": fuzzy_mode.value,
                        "entity_matching_config_version": (
                            matcher.config.version if matcher is not None else None
                        ),
                        "rule_protection_enabled": not args.disable_rule_protection,
                        "numeric_normalization_enabled": (
                            args.enable_numeric_normalization
                            and not args.disable_numeric_normalization
                        ),
                    },
                },
            )

    def callback(indata, frames, timing, status) -> None:  # noqa: ANN001, ARG001
        if status:
            print(status, file=sys.stderr)
        windows.put(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_WINDOW,
            device=args.device_index,
            callback=callback,
        ):
            while True:
                window = windows.get()
                vad.accept_waveform(window)
                speech = vad.is_speech_detected()
                if speech and not active:
                    active = True
                    utterance = list(preroll)
                if active:
                    utterance.append(window)
                preroll.append(window)
                if active and not speech:
                    transcribe_and_refine(utterance)
                    active = False
                    utterance = []
                while not vad.empty():
                    vad.pop()
    except KeyboardInterrupt:
        print()
        if active and utterance:
            transcribe_and_refine(utterance)
        print("Stopped.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
