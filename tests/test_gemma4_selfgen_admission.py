"""Protect pilot admission from fabricated corrections and broken construction."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/scripts/generate_gemma4_selfgen_pilot.py'
SPEC = importlib.util.spec_from_file_location('selfgen_pilot', PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class AdmissionTest(unittest.TestCase):
    def test_rejects_invented_content(self):
        with self.assertRaisesRegex(ValueError, 'unsupported_content'):
            MOD.prepare({'input': '明天送资料。', 'reference': '明天下午送资料。'}, 'B01', {})

    def test_d02_changes_only_one_boundary(self):
        item = MOD.prepare({'reference': '请把准备好的材料放在前台。', 'break_after': '请把准备好的'}, 'D02', {})
        self.assertEqual(item['input'], '请把准备好的。材料放在前台。')
        self.assertEqual(MOD.content(item['input']), MOD.content(item['reference']))
        with self.assertRaisesRegex(ValueError, 'invalid_break_boundary'):
            MOD.prepare({'reference': '请把材料放在前台。', 'break_after': '把材料'}, 'D02', {})

    def test_placeholder_mapping_and_order(self):
        row = {'input': '给松澜科技呃发文件，再联系澄川团队。',
               'reference': '给松澜科技发文件，再联系澄川团队。', 'entity_names': ['澄川团队', '松澜科技']}
        item = MOD.prepare(row, 'E06', {'entity_count': 2})
        self.assertEqual(item['reference'], '给__ENTITY_000__发文件，再联系__ENTITY_001__。')
        self.assertEqual(item['construction']['mapping']['__ENTITY_000__'], '松澜科技')
        row['reference'] = '给松澜科技发文件，再联系松澜科技。'
        with self.assertRaises(ValueError):
            MOD.prepare(row, 'E06', {'entity_count': 2})

    def test_rejects_placeholder_fabricated_by_teacher(self):
        with self.assertRaisesRegex(ValueError, 'unexpected_markup'):
            MOD.prepare({'input': '__ENTITY_000__呃。', 'reference': '__ENTITY_000__。'}, 'E06', {})

    def test_rejects_string_booleans_and_changed_audit_target(self):
        verdict = {k: True for k in ('source_valid', 'category_valid', 'meaning_preserved', 'minimal_edit', 'cleanup_complete')}
        verdict.update(issues=[], expected_target='请先确认。')
        self.assertTrue(MOD.audit_pass(verdict, '请先确认。'))
        verdict['source_valid'] = 'true'
        self.assertFalse(MOD.audit_pass(verdict, '请先确认。'))
        verdict['source_valid'] = True
        self.assertFalse(MOD.audit_pass(verdict, '请先审批。'))

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_json_key'):
            MOD.strict_json('{"input":"a","input":"b"}')

    def test_spelling_removal_cannot_delete_other_instructions(self):
        row = {'input': '收件人沈禾，沈阳的沈，禾苗的禾，明天寄。', 'reference': '收件人沈禾，明天寄。',
               'entity_names': ['沈禾'], 'removable_span': '沈阳的沈，禾苗的禾',
               'spelling_cues': [{'character': '沈', 'word': '沈阳'}, {'character': '禾', 'word': '禾苗'}]}
        MOD.prepare(row, 'B03', {})
        row['reference'] = '收件人沈禾。'
        with self.assertRaisesRegex(ValueError, 'edits_outside_spelling_span'):
            MOD.prepare(row, 'B03', {})

    def test_negation_protection_rejects_deletion_beyond_filler(self):
        row = {'input': '除非审核通过，呃，否则不能发布。', 'reference': '除非审核通过，否则不能发布。'}
        MOD.prepare(row, 'E03', {'mode': 'clean'})
        row['reference'] = '除非审核通过，否则能发布。'
        with self.assertRaisesRegex(ValueError, 'e03_edits_beyond_one_filler'):
            MOD.prepare(row, 'E03', {'mode': 'clean'})

    def test_positive_stance_is_protected_content_too(self):
        MOD.prepare({'input': '我个人认为这项安排值得支持。',
                     'reference': '我个人认为这项安排值得支持。'}, 'E03', {'mode': 'keep'})
        MOD.prepare({'input': '那个，我还是觉得先做验证更合适。',
                     'reference': '我还是觉得先做验证更合适。'}, 'E03', {'mode': 'clean'})

    def test_rollback_evidence_checked(self):
        row = {'input': '送青禾路，不对，送松泉路，还是青禾路。', 'reference': '送青禾路。',
               'candidate_sequence': ['青禾路', '松泉路', '青禾路'], 'final_candidate': '青禾路'}
        MOD.prepare(row, 'B02', {'subtype': 'rollback'})
        row['reference'] = '送松泉路。'
        with self.assertRaisesRegex(ValueError, 'abandoned_candidate'):
            MOD.prepare(row, 'B02', {'subtype': 'rollback'})


if __name__ == '__main__':
    unittest.main()
