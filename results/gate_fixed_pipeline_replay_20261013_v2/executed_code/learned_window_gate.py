"""Optional CPU window filter; failures preserve the native scheduler's action."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
import time

from .weak_refinement_gate import input_text

GATE_MODES = ('off', 'shadow', 'on')
DEFAULT_MODEL_DIR = (Path(__file__).resolve().parents[1] / 'results' /
                     'gate_terminal_gain_round4_20261012' / 'linear_training')


class LearnedWindowGate:
    def __init__(self, mode: str = 'off', model_dir: Path = DEFAULT_MODEL_DIR):
        if mode not in GATE_MODES:
            raise ValueError('learned window gate mode must be off, shadow or on')
        self.mode, self.model_dir = mode, Path(model_dir).resolve()
        self.model = None
        self.threshold = self.model_sha256 = self.load_error = None
        self._lock = threading.Lock()
        self._scores = self._would_skip = self._vetoes = self._errors = 0
        if mode != 'off':
            try:
                selection = json.loads((self.model_dir / 'selection.json').read_text())
                # Keep the experimentally selected model and threshold together.
                if selection['primary'] != 'terminal_quality':
                    raise ValueError('Expected the frozen terminal_quality selection')
                metadata = selection['variants']['terminal_quality']
                threshold = float(metadata['threshold'])
                if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise ValueError('Invalid skip threshold')
                data = (self.model_dir / 'terminal_quality/model.pkl').read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != metadata['model_sha256']:
                    raise ValueError('Model does not match selection SHA256')
                # Only load the locally trained artifact identified by selection.
                model = pickle.loads(data)
                if set(model.classes_) != {'refine_required', 'skip_allowed'}:
                    raise ValueError('Unsupported model classes')
                self._column = list(model.classes_).index('skip_allowed')
                self.model, self.threshold, self.model_sha256 = model, threshold, digest
            except Exception as error:
                self.load_error = f'{type(error).__name__}: {error}'[:240]

    def decide(self, value: dict, native_action: str) -> dict:
        result = dict(mode=self.mode, action=native_action, suggested_action=native_action,
                      applied=False, calibrated=False, score=None, threshold=self.threshold)
        if self.mode == 'off':
            return result | dict(reason='disabled')
        if native_action != 'refine' or value.get('task_kind') != 'window_refine':
            return result | dict(reason='native_action_preserved')
        if self.model is None:
            return result | dict(reason='model_unavailable_native_fallback', error=self.load_error)
        started = time.perf_counter()
        with self._lock:
            try:
                description, target, context = input_text(value)
                probabilities = self.model.predict_proba([description + context + '\n' + target])[0]
                if (len(probabilities) != 2 or
                    any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities) or
                    abs(float(sum(probabilities)) - 1) > 1e-4):
                    raise ValueError('Invalid classifier probabilities')
                score = float(probabilities[self._column])
                skip = score >= self.threshold
                self._scores += 1
                self._would_skip += int(skip)
                applied = skip and self.mode == 'on'
                self._vetoes += int(applied)
                result.update(score=score, suggested_action='keep' if skip else native_action,
                              action='keep' if applied else native_action, applied=applied,
                              reason='shadow_observation' if self.mode == 'shadow' else
                                     'learned_window_veto' if applied else 'native_refine_retained')
            except Exception as error:
                self._errors += 1
                result.update(reason='score_error_native_fallback',
                              error=f'{type(error).__name__}: {error}'[:240])
        result['gate_ms'] = (time.perf_counter() - started) * 1000
        return result

    def status(self) -> dict:
        with self._lock:
            return dict(mode=self.mode, ready=self.model is not None, model_dir=str(self.model_dir),
                        model_sha256=self.model_sha256, threshold=self.threshold,
                        load_error=self.load_error, scored_windows=self._scores,
                        would_skip_windows=self._would_skip, vetoed_windows=self._vetoes,
                        scoring_errors=self._errors, scope='window_refine_only', calibrated=False)
