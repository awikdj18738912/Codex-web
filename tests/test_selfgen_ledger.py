import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/scripts/selfgen_ledger.py'
SPEC = importlib.util.spec_from_file_location('selfgen_ledger', PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'ledger.sqlite'
        self.ledger = MOD.Ledger(self.path, {'A': 2, 'B': 1}, 'cfg')

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, ident, source=None, target='t', category='A', verdict='passed'):
        source = source or ident
        return self.ledger.record_candidate(ident, category, 'g', source, target,
                                            {'id': ident}, verdict, 'bad')

    def test_resume_idempotency_and_config_guard(self):
        self.assertEqual(self.record('x'), 'accepted')
        self.assertEqual(self.record('x'), 'accepted')
        self.assertEqual(self.ledger.next_sequence(), 1)
        again = MOD.Ledger(self.path, {'A': 2, 'B': 1}, 'cfg')
        self.assertEqual(again.next_sequence(), 2)
        self.assertEqual(len(list(again.iter_accepted())), 1)
        with self.assertRaisesRegex(ValueError, 'ledger_config_mismatch'):
            MOD.Ledger(self.path, {'A': 3, 'B': 1}, 'cfg')

    def test_quota_and_rejections(self):
        self.assertEqual(self.record('r', verdict='failed'), 'rejected')
        self.assertEqual(self.record('1'), 'accepted')
        self.assertEqual(self.record('2'), 'accepted')
        self.assertEqual(self.record('3'), 'quota_full')
        self.assertEqual(self.record('b', category='B'), 'accepted')
        self.assertEqual(self.ledger.stats()['total']['accepted'], 3)
        self.assertEqual(len(list(self.ledger.iter_accepted('A'))), 2)

    def test_concurrent_quota_never_overfills(self):
        ledger = MOD.Ledger(Path(self.tmp.name) / 'concurrent.sqlite', {'A': 7}, 'concurrent')
        def add(i):
            return ledger.record_candidate(str(i), 'A', 'g', f'source{i}', 't', {}, 'passed', None)
        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(add, range(50)))
        self.assertEqual(statuses.count('accepted'), 7)
        self.assertEqual(ledger.stats()['A']['accepted'], 7)
        self.assertEqual(statuses.count('quota_full'), 43)

    def test_normalized_duplicate_and_conflict_quarantine(self):
        self.assertEqual(self.record('one', '  Ａ  b。 ', 'target'), 'accepted')
        self.assertEqual(self.record('two', 'A b。', 'target'), 'duplicate')
        self.assertEqual(self.record('three', 'A b。', 'other'), 'conflict')
        self.assertEqual(list(self.ledger.iter_accepted()), [])
        self.assertEqual(self.ledger.stats()['A']['conflict'], 2)
        # Punctuation remains significant during normalization.
        self.assertEqual(self.record('punct', 'A b!', 'target'), 'accepted')

    def test_export_rows_stay_within_each_category_quota(self):
        for i in range(8):
            category = 'A' if i < 5 else 'B'
            self.record(f'export-{i}', category=category)
        exported = list(self.ledger.iter_accepted())
        by_category = {category: [row for row in exported if row['category'] == category]
                       for category in self.ledger.quotas}
        self.assertLessEqual(len(by_category['A']), self.ledger.quotas['A'])
        self.assertLessEqual(len(by_category['B']), self.ledger.quotas['B'])
        self.assertEqual(len(exported), self.ledger.stats()['total']['accepted'])

    def test_pending_audit_survives_resume_and_clears_atomically_on_terminal_record(self):
        payload = {'id': 'pending-1', 'category': 'A', 'input': '来源句子内容足够长。',
                   'reference': '目标句子内容足够长。'}
        self.ledger.defer_audit('pending-1', payload)
        stats = self.ledger.stats()
        self.assertEqual(stats['A']['pending_audit'], 1)
        self.assertEqual(stats['total']['pending_audit'], 1)
        self.assertEqual(stats['total']['total'], 0)
        self.assertEqual(stats['A']['accepted'], 0)
        self.assertEqual(list(self.ledger.iter_accepted()), [])

        resumed = MOD.Ledger(self.path, {'A': 2, 'B': 1}, 'cfg')
        pending = list(resumed.iter_pending_audits())
        self.assertEqual(pending, [payload])
        self.assertEqual(list(resumed.iter_pending_audits('B')), [])
        status = resumed.record_candidate('pending-1', 'A', 'g', payload['input'],
                                          payload['reference'], payload, 'passed', None)
        self.assertEqual(status, 'accepted')
        self.assertEqual(list(resumed.iter_pending_audits()), [])
        self.assertEqual(resumed.stats()['A']['pending_audit'], 0)
        self.assertEqual(resumed.stats()['A']['accepted'], 1)

    def test_terminal_record_for_existing_id_clears_pending_entry(self):
        self.assertEqual(self.record('same'), 'accepted')
        self.ledger.defer_audit('same', {'id': 'same', 'category': 'A'})
        self.assertEqual(self.record('same'), 'accepted')
        self.assertEqual(list(self.ledger.iter_pending_audits()), [])


if __name__ == '__main__':
    unittest.main()
