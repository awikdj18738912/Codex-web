import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
import generate_refiner_100k as mod


class GenerationTest(unittest.TestCase):
    def test_quotas_cover_exactly_current_17_classes(self):
        self.assertEqual(sum(mod.QUOTAS.values()), 100000)
        self.assertEqual(len(mod.QUOTAS), 17)
        self.assertFalse(set(mod.QUOTAS) & {'E01', 'E02', 'E04', 'F01', 'F02'})

    def test_missing_word_cannot_add_fact_or_arbitrary_rewrite(self):
        row = {'input': '他把书放桌子上了。', 'reference': '他把书放在桌子上了。', 'missing_span': '在'}
        mod.construct_validate(row, 'C03', {})
        row.update(reference='他把书放在桌子上，晚上再读。')
        with self.assertRaisesRegex(ValueError, 'not_one_small'):
            mod.construct_validate(row, 'C03', {})

    def test_reorder_must_preserve_characters_and_context(self):
        row = {'input': '我把已经文件发给你了。', 'reference': '我已经把文件发给你了。',
               'edit_source': '把已经', 'edit_target': '已经把'}
        mod.construct_validate(row, 'C04', {})
        row['reference'] = '我已经把文件发给他了。'
        with self.assertRaises(ValueError):
            mod.construct_validate(row, 'C04', {})

    def test_sentence_boundary_not_just_arbitrary_subject_pause(self):
        mod.construct_validate({'left': '请把文件交给', 'right': '前台的工作人员。'}, 'D02', {'boundary_word': '交给'})
        with self.assertRaisesRegex(ValueError, 'weak_or_unplanned'):
            mod.construct_validate({'left': '这份报告的数据', 'right': '有一点问题。'}, 'D02', {'boundary_word': '交给'})

    def test_controlled_spelling_preserves_all_context(self):
        row = {'reference': '请把资料寄给沈禾，下午再确认。'}
        plan = {'entity_name': '沈禾', 'spelling_cues': [{'character': '沈', 'word': '沈阳'},
                {'character': '禾', 'word': '禾苗'}], 'cue_position': 'utterance_end'}
        out = mod.construct_validate(row, 'B03', plan)
        self.assertEqual(out['reference'], row['reference'])
        self.assertIn('沈阳的沈，禾苗的禾', out['input'])

    def test_near_index_keeps_exact_source_for_conflict_handling(self):
        n = mod.NearIndex()
        text = '请把资料交给前台工作人员，明天下午领取。'
        n.add('x', text)
        self.assertEqual(n.match(text), 'x')
        n.drop_source(text)
        self.assertIsNone(n.match(text))

    def test_redundancy_cannot_delete_multiple_facts(self):
        mod.construct_validate({'input': '大约需要半个小时左右才能办完。',
                                'reference': '大约需要半个小时才能办完。'}, 'C02', {})
        with self.assertRaises(ValueError):
            mod.construct_validate({'input': '大约需要半个小时左右才能办完。',
                                   'reference': '需要半个小时才能办完。'}, 'C02', {})

    def test_clean_reference_construction_never_reverses_correction_direction(self):
        out = mod.construct_validate({'reference': '我已经把文件发给前台了。',
              'correct_fragment': '已经把', 'wrong_fragment': '把已经'}, 'C04', {})
        self.assertEqual(out['reference'], '我已经把文件发给前台了。')
        self.assertEqual(out['input'], '我把已经文件发给前台了。')
        out = mod.construct_validate({'reference': '这能帮助我们养成及时备份的习惯。',
              'missing_span': '习惯'}, 'C03', {'missing_component': '习惯'})
        self.assertEqual(out['input'], '这能帮助我们养成及时备份的。')
        self.assertIn('习惯', out['reference'])

    def test_controlled_stutter_changes_only_starting_character(self):
        out = mod.construct_validate({'reference': '我想把这份材料交给前台。'}, 'A02', {'stutter_char': '我'})
        self.assertEqual(out['input'], '我我想把这份材料交给前台。')

    def test_reference_boundary_anchor_constructs_one_abnormal_period(self):
        out = mod.construct_validate({'reference': '请把整理好的材料交给前台同事。'}, 'D02', {'boundary_word': '交给'})
        self.assertEqual(out['input'], '请把整理好的材料交给。前台同事。')


if __name__ == '__main__':
    unittest.main()
