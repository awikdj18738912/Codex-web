import unittest
from collections import defaultdict

from experiments.scripts import expand_gate_templates as expansion
from experiments.scripts import synthetic_gate_data as data


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = data.make_seeds(1000, 20261009)
        cls.rows = expansion.generate(cls.old)

    def test_reproducible_unique_scenarios_with_no_old_template_overlap(self):
        self.assertEqual(self.rows, expansion.generate(self.old))
        self.assertEqual(len(self.rows), 108)
        self.assertEqual(len({r['template_group'] for r in self.rows}), 60)
        old = {(expansion.delexical(r['spoken_text']), expansion.delexical(r['intended_text'])) for r in self.old}
        new = {(expansion.delexical(r['spoken_text']), expansion.delexical(r['intended_text'])) for r in self.rows}
        self.assertFalse(old & new)
        self.assertEqual(len({r['seed_id'] for r in self.rows}), len(self.rows))

    def test_counterparts_and_slot_values_share_one_partition(self):
        groups = defaultdict(set)
        for row in expansion.generate(self.old, slots_per_template=2):
            groups[row['template_group']].add(row['synthetic_split'])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual({r['synthetic_split'] for r in self.rows}, {'train','dev','test'})
        self.assertEqual(sum(r['synthetic_split']=='train' for r in self.rows), 64)

    def test_risk_scenarios_preserve_script_and_do_not_assign_gate_labels(self):
        for row in self.rows:
            if row['category'] not in {'clean','filler','stutter'}:
                self.assertEqual(row['spoken_text'], row['intended_text'])
            self.assertNotIn('label', row)
            self.assertFalse(row['audio_script_verified'])
            self.assertEqual(row['seed_sha256'],data.digest({k:v for k,v in row.items() if k!='seed_sha256'}))

    def test_invalid_expansion_size_fails(self):
        with self.assertRaises(ValueError): expansion.generate(self.old, slots_per_template=0)


if __name__ == '__main__':
    unittest.main()
