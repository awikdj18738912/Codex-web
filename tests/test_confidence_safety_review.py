from copy import deepcopy
from contextlib import redirect_stdout
from argparse import Namespace
import io
import json
from pathlib import Path
import tempfile
import unittest

from system.edit_detection_data import review_basis
from experiments.scripts.benchmark_fixed_gate_pipeline import sha
from experiments.scripts.prepare_confidence_safety_review import packet_basis, select_candidates, write_rows
from experiments.scripts.import_confidence_safety_reviews import run, summarize_verified, validate_review


class ConfidenceSafetyReviewTest(unittest.TestCase):
    def fixture(self, directory):
        (directory / 'audio').mkdir()
        for name in ('source.wav', 'original.mp3', 'audio/case.wav'):
            (directory / name).write_bytes(b'audio_identity_fixture')
        row = dict(id='window', raw_asr='嗯，请确认。', baseline_text='嗯，请确认。',
            recording_group='consumed_recording', split='regression',
            source_audio_path=str(directory / 'source.wav'), source_audio_sha256=sha(directory / 'source.wav'),
            packaged_audio='audio/case.wav', audio_path=str(directory / 'audio/case.wav'),
            audio_sha256=sha(directory / 'audio/case.wav'),
            audio_provenance=dict(original_path=str(directory / 'original.mp3'), original_sha256=sha(directory / 'original.mp3')),
            all_refine_result=dict(output='请确认。'), supplement_result=dict(output='请确认。'),
            supplement_reviewed=True, desired_text=None, edits=None, human_audio_gold=False)
        row['review_basis_sha256'] = review_basis(row)
        row['packet_basis_sha256'] = packet_basis(row)
        review = dict(id='window', review_basis_sha256=row['review_basis_sha256'],
            packet_basis_sha256=row['packet_basis_sha256'], reviewer='test_fixture_only',
            review_kind='human_audio_cleaned', audio_checked=True, decision='edit', desired_text='请确认。',
            edit_tasks=['disfluency'], single_task_for_all_edits=True, asr_fidelity='correct',
            refiner_effect='beneficial', note='test fixture only')
        return row, review

    def test_selection_keeps_all_changes_and_supplement_reviews(self):
        rows = []
        for stratum in ('accepted_changed', 'accepted_unchanged', 'rejected'):
            for i in range(16):
                rows.append(dict(id=f'{stratum}-{i}', selection_stratum=stratum, change_kind='lexical_change',
                    case=i, window_index=0, dependency_group=str(i % 4), score_band='high' if i % 2 else 'other',
                    category='existing_test_recording', supplement_reviewed=i == 15))
        selected = select_candidates(rows)
        self.assertEqual(len(selected), 40)
        self.assertEqual(sum(r['selection_stratum'] == 'accepted_changed' for r in selected), 16)
        self.assertTrue(all(r in selected for r in rows if r['supplement_reviewed']))
        self.assertEqual(selected, select_candidates(rows))
        with self.assertRaises(ValueError):
            select_candidates(rows, controls=17)

    def test_model_or_context_change_invalidates_packet(self):
        row = dict(baseline_text='原文', all_refine_result={'output': '候选'}, left_context='前文')
        before = packet_basis(row)
        for field, value in [('all_refine_result', {'output': '别的候选'}), ('left_context', '别的前文')]:
            with self.subTest(field=field):
                self.assertNotEqual(before, packet_basis({**row, field: value}))

    def test_audio_review_is_diagnostic_and_unknown_is_not_keep(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); row, review = self.fixture(directory)
            known = validate_review(row, review, directory)
            self.assertTrue(known['necessary_refinement'])
            self.assertTrue(known['human_audio_gold'])
            self.assertFalse(known['training_eligible'])
            self.assertFalse(known['production_eligible'])
            unknown = validate_review(row, {**review, 'decision': 'uncertain', 'desired_text': None,
                'edit_tasks': [], 'refiner_effect': 'unassessable', 'note': '无法确定'}, directory)
            self.assertIsNone(unknown['necessary_refinement'])
            self.assertFalse(unknown['human_audio_gold'])
            stats = summarize_verified([known, unknown])
            self.assertEqual(stats['necessary_refinement_coverage_on_reviewed_native_skips'], 1.)
            self.assertEqual(stats['uncertain_records'], 1)

    def test_unheard_stale_and_tampered_audio_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); row, review = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'human audio'):
                validate_review(row, {**review, 'audio_checked': False}, directory)
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate_review(row, {**review, 'packet_basis_sha256': 'old'}, directory)
            (directory / row['packaged_audio']).write_bytes(b'changed_audio')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                validate_review(row, review, directory)

    def test_keep_and_effect_cannot_contradict_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); row, review = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'KEEP assessment'):
                validate_review(row, {**review, 'decision': 'keep', 'desired_text': row['baseline_text'],
                    'edit_tasks': []}, directory)
            unchanged = deepcopy(row); unchanged['all_refine_result']['output'] = unchanged['baseline_text']
            unchanged['packet_basis_sha256'] = packet_basis(unchanged)
            with self.assertRaisesRegex(ValueError, 'Unchanged'):
                validate_review(unchanged, {**review, 'packet_basis_sha256': unchanged['packet_basis_sha256']}, directory)

    def test_numeric_value_edits_cannot_be_refiner_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); row, review = self.fixture(directory)
            row['raw_asr'] = row['baseline_text'] = '预算12元，请确认。'
            row['review_basis_sha256'] = review_basis(row); row['packet_basis_sha256'] = packet_basis(row)
            review.update(review_basis_sha256=row['review_basis_sha256'], packet_basis_sha256=row['packet_basis_sha256'],
                          desired_text='预算13元，请确认。', edit_tasks=['local_typo'])
            with self.assertRaisesRegex(ValueError, 'numeric'):
                validate_review(row, review, directory)

    def test_no_labels_produce_no_recall_or_correctness_claim(self):
        stats = summarize_verified([])
        self.assertIsNone(stats['necessary_refinement_coverage_on_reviewed_native_skips'])
        self.assertEqual(stats['necessary_refinement_windows'], 0)
        self.assertFalse(stats['production_eligible'])

    def test_conflicting_duplicate_submissions_cannot_create_a_gold_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); row, review = self.fixture(directory)
            write_rows(directory / 'review_queue.jsonl', [row])
            (directory / 'manifest.json').write_text(json.dumps({'output_sha256': {
                'review_queue.jsonl': sha(directory / 'review_queue.jsonl')}}))
            second = {**review, 'decision': 'keep', 'desired_text': row['baseline_text'],
                      'edit_tasks': [], 'refiner_effect': 'harmful'}
            write_rows(directory / 'submitted.jsonl', [review, second])
            output = directory / 'imported'
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                run(Namespace(queue=directory, reviews=directory / 'submitted.jsonl', output=output))
            result = json.loads((output / 'summary.json').read_text())
            self.assertEqual(result['rejected_reviews'], 2)
            self.assertEqual(result['reviewed_records'], 0)
            self.assertEqual((output / 'verified_diagnostics.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
