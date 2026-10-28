from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
from refiner_scoped_runner import ScopedRunner


class Ledger:
    def __init__(self):
        self.accepted = Counter(A01=90000)
        self.sequence = 0

    def stats(self):
        result = {c: {'accepted': self.accepted[c], 'total': self.accepted[c]}
                  for c in ('A01', 'B01', 'B02')}
        result['total'] = {'accepted': sum(self.accepted.values())}
        return result

    def next_sequence(self):
        self.sequence += 1
        return self.sequence


class ScopedRunnerTest(unittest.TestCase):
    def runner(self):
        runner = ScopedRunner.__new__(ScopedRunner)
        runner.args = SimpleNamespace(workers=4, batch_size=1, stop_after=2)
        runner.categories = ['B01', 'B02']
        runner.ledger = Ledger()
        runner.initial_scope_count = 0
        runner.scope_target = 20000
        runner.stop_event = threading.Event()
        runner.audit_deferred = False
        runner.failure_streak = 0
        runner.last_export = 90000
        runner.states = []
        runner.status = lambda state='running', *a, **kw: runner.states.append(state)
        runner.retry_deferred = lambda: True
        runner.export_review = lambda **kw: None
        runner.submitted = []
        runner.committed = []
        def batch(category, sequence, n):
            runner.submitted.append(category)
            return {'rows': [{'id': str(sequence), 'category': category, 'rejection_reason': None}]}
        def commit(record):
            runner.ledger.accepted[record['category']] += 1
            runner.committed.append(record['id'])
            return 'accepted'
        runner.batch = batch
        runner.commit_record = commit
        return runner

    def test_historical_other_classes_do_not_count_toward_gap_phase(self):
        runner = self.runner()
        runner.run()
        self.assertGreaterEqual(len(runner.committed), 2)
        self.assertEqual(set(runner.submitted), {'B01', 'B02'})
        self.assertEqual(runner.ledger.accepted['A01'], 90000)
        self.assertEqual(runner.states[-1], 'coverage_gaps_stage_finished')

    def test_stop_drains_and_commits_submitted_work(self):
        runner = self.runner()
        original = runner.batch
        def stopping_batch(*args):
            runner.stop_event.set()
            return original(*args)
        runner.batch = stopping_batch
        runner.run()
        self.assertGreater(len(runner.committed), 0)
        self.assertEqual(len(runner.submitted), len(runner.committed))
        self.assertEqual(runner.states[-1], 'stopped_after_draining')


if __name__ == '__main__':
    unittest.main()
