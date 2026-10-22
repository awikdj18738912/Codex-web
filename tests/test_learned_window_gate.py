from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from system.learned_window_gate import LearnedWindowGate


class _Classifier:
    classes_ = ['refine_required', 'skip_allowed']

    def __init__(self, probabilities=(0.05, 0.95), error=False):
        self.probabilities, self.error = probabilities, error

    def predict_proba(self, values):
        if self.error:
            raise RuntimeError('simulated scoring failure')
        return [self.probabilities]


def write_model(root, classifier=None):
    (root / 'terminal_quality').mkdir()
    data = pickle.dumps(classifier or _Classifier())
    (root / 'terminal_quality/model.pkl').write_bytes(data)
    (root / 'selection.json').write_text(json.dumps(dict(
        primary='terminal_quality', variants=dict(terminal_quality=dict(
            threshold=0.8, model_sha256=hashlib.sha256(data).hexdigest())))))


WINDOW = dict(task_kind='window_refine', domain='general', raw_text='请确认确认编号。',
              baseline_text='请确认确认编号。', available_context='')


class LearnedWindowGateTests(unittest.TestCase):
    def test_off_does_not_load_or_score(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('Unexpected loading')):
            gate = LearnedWindowGate('off', Path('/missing'))
        self.assertEqual(gate.decide(WINDOW, 'refine')['action'], 'refine')
        self.assertEqual(gate.status()['scored_windows'], 0)

    def test_shadow_records_skip_without_changing_native_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_model(root)
            gate = LearnedWindowGate('shadow', root)
            result = gate.decide(WINDOW, 'refine')
            self.assertEqual(result['suggested_action'], 'keep')
            self.assertEqual(result['action'], 'refine')
            self.assertFalse(result['applied'])
            self.assertEqual(gate.status()['would_skip_windows'], 1)
            self.assertEqual(gate.status()['vetoed_windows'], 0)

    def test_on_only_vetoes_high_skip_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_model(root)
            gate = LearnedWindowGate('on', root)
            self.assertTrue(gate.decide(WINDOW, 'refine')['applied'])
            gate.model = _Classifier((0.4, 0.6))
            self.assertEqual(gate.decide(WINDOW, 'refine')['action'], 'refine')
            self.assertEqual(gate.status()['vetoed_windows'], 1)

    def test_keep_defer_and_local_review_never_scored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_model(root)
            gate = LearnedWindowGate('on', root)
            gate.model = _Classifier(error=True)
            for action in ('keep', 'defer'):
                self.assertEqual(gate.decide(WINDOW, action)['action'], action)
            self.assertEqual(gate.decide(dict(task_kind='local_edit_review'), 'refine')['action'], 'refine')
            self.assertEqual(gate.status()['scoring_errors'], 0)

    def test_missing_or_hash_mismatched_model_preserves_refine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = LearnedWindowGate('on', root)
            self.assertEqual(missing.decide(WINDOW, 'refine')['action'], 'refine')
            self.assertFalse(missing.status()['ready'])
            write_model(root)
            (root / 'terminal_quality/model.pkl').write_bytes(b'wrong checkpoint')
            gate = LearnedWindowGate('on', root)
            self.assertEqual(gate.decide(WINDOW, 'refine')['action'], 'refine')
            self.assertIn('SHA256', gate.status()['load_error'])

    def test_invalid_score_or_scoring_error_preserves_refine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_model(root)
            gate = LearnedWindowGate('on', root)
            for classifier in (_Classifier((float('nan'), 0.9)), _Classifier((0.8, 0.9)),
                               _Classifier(error=True)):
                gate.model = classifier
                result = gate.decide(WINDOW, 'refine')
                self.assertEqual(result['action'], 'refine')
                self.assertFalse(result['applied'])
                self.assertEqual(result['reason'], 'score_error_native_fallback')
            self.assertEqual(gate.status()['scoring_errors'], 3)

    def test_cli_default_and_explicit_mode(self):
        from system.web_app import parse_args
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(parse_args(['--refiner-model', temp]).learned_window_gate_mode, 'off')
            self.assertEqual(parse_args(['--refiner-model', temp, '--learned-window-gate-mode', 'shadow']).learned_window_gate_mode, 'shadow')

    def test_start_script_does_not_reuse_service_with_different_mode(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/start_web.sh'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            curl = root / 'curl'
            curl.write_text('#!/bin/bash\nprintf \'{"window_edit_policy":"compat","refinement_gate_mode":"tri_state","final_refinement_mode":"off","zh_itn_enabled":false,"window_pipeline":{"version":"fixed_k3_context_v2","window_size":3,"one_punctuation_window":true,"fixed_groups":true,"stream_reprocess_rejected":false,"final_reprocess_rejected":false,"group_numeric_self_corrections":false,"adjacent_context":true,"context_lookahead_chunks":1,"context_protocol":"v1-read-only-adjacent-context"}}\\n\'\n')
            curl.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'],
                       WINDOW_EDIT_POLICY='compat', LEARNED_WINDOW_GATE_MODE='shadow',
                       REFINEMENT_GATE_MODE='tri_state', FINAL_REFINEMENT_MODE='off', ZH_ITN_ENABLED='0')
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('uses learned gate off; requested shadow', result.stdout + result.stderr)
            env['LEARNED_WINDOW_GATE_MODE'] = 'off'
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn('reusing it', result.stdout)
            for key, requested, message in (
                ('REFINEMENT_GATE_MODE', 'off', 'refinement_gate_mode'),
                ('FINAL_REFINEMENT_MODE', 'on', 'final_refinement_mode'),
            ):
                with self.subTest(key=key):
                    changed = dict(env, **{key: requested})
                    result = subprocess.run(['bash', str(script)], env=changed, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('configuration mismatch: ' + message, result.stderr)
            result = subprocess.run(['bash', str(script)], env=dict(env, ZH_ITN_ENABLED='1'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('pipeline or zh-itn configuration mismatch', result.stderr)
            curl.write_text('#!/bin/bash\nprintf \'{"window_edit_policy":"compat","refinement_gate_mode":"tri_state","final_refinement_mode":"off"}\\n\'\n')
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('older window pipeline', result.stderr)


if __name__ == '__main__':
    unittest.main()
