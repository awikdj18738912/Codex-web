"""Model-based, vocabulary-free comparison of repeated-span alternatives.

This is a fallback for a text Refiner that cannot make a candidate-only edit.
The masked language model is loaded from the local cache; an unavailable model
never authorizes a deletion.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


MODEL_ID = "hfl/chinese-roberta-wwm-ext"
REMOVE_MARGIN = 0.7
KEEP_MARGIN = -0.7
MAX_SENTENCE_CHARS = 96


@dataclass(frozen=True)
class PlausibilityResult:
    decision: str
    margin: float | None
    reason: str


class RepetitionPlausibility:
    """Compare mean pseudo-log-likelihood of a sentence and one local edit."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._load_error: str | None = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, local_files_only=True
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                MODEL_ID, local_files_only=True
            ).to(self.device).eval()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self._load_error = type(error).__name__
            return False
        return True

    def _mean_log_probability(self, sentence: str) -> float:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None
        encoded = tokenizer(sentence, return_tensors="pt", truncation=True,
                            max_length=MAX_SENTENCE_CHARS + 2)
        tokens = encoded["input_ids"][0].to(self.device)
        positions = list(range(1, len(tokens) - 1))
        if not positions:
            return float("-inf")
        total = 0.0
        with torch.inference_mode():
            for offset in range(0, len(positions), 24):
                batch_positions = positions[offset:offset + 24]
                masked = tokens.repeat(len(batch_positions), 1)
                for row, position in enumerate(batch_positions):
                    masked[row, position] = tokenizer.mask_token_id
                attention = torch.ones_like(masked)
                logits = model(input_ids=masked, attention_mask=attention).logits
                rows = torch.arange(len(batch_positions), device=self.device)
                chosen = logits[rows, batch_positions, tokens[batch_positions]]
                normalizer = torch.logsumexp(logits[rows, batch_positions], dim=-1)
                total += (chosen - normalizer).sum().item()
        return total / len(positions)

    def decide(self, source: str, start: int, end: int, target: str) -> PlausibilityResult:
        if (start < 0 or end > len(source) or start >= end or not target
                or len(source) > MAX_SENTENCE_CHARS):
            return PlausibilityResult("unresolved", None, "plausibility_input_limit")
        edited = source[:start] + target + source[end:]
        with self._lock:
            if not self._ensure_model():
                return PlausibilityResult("unresolved", None, "plausibility_unavailable")
            try:
                margin = self._mean_log_probability(edited) - self._mean_log_probability(source)
            except (RuntimeError, ValueError):
                return PlausibilityResult("unresolved", None, "plausibility_failed")
        if margin >= REMOVE_MARGIN:
            return PlausibilityResult("remove", round(margin, 4), "model_plausibility_remove")
        if margin <= KEEP_MARGIN:
            return PlausibilityResult("keep", round(margin, 4), "model_plausibility_keep")
        return PlausibilityResult("unresolved", round(margin, 4), "model_plausibility_uncertain")
