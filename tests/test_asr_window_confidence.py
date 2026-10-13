import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from system.asr_window_confidence import (
    TokenBytes, TokenScoreLedger, byte_alphabet_inverse, chosen_token_scores,
    project_window_scores, score_source_range,
)


def scored_projection(text, probability=.8):
    ledger = TokenScoreLedger()
    payload = text.encode()
    ledger.update("", text, payload, [dict(start=0,end=len(payload),original_width=len(payload),
                  token_id=1,token_index=0,logprob=math.log(probability))])
    return ledger.text_projection(text, "Chinese")


class WindowConfidenceTest(unittest.TestCase):
    def test_byte_alphabet_covers_all_bytes(self):
        inverse = byte_alphabet_inverse()
        self.assertEqual(len(inverse), 256)
        self.assertEqual(set(inverse.values()), set(range(256)))
        self.assertEqual(inverse['Ġ'], 32)
        self.assertEqual(inverse['Ċ'], 10)

    def test_missing_scores_keep_their_token_positions(self):
        completion = SimpleNamespace(token_ids=[1, 2, 3], logprobs=[{1: -.1}, {99: -.2}, {3: -.3}])
        self.assertEqual(chosen_token_scores(completion), [-.1, None, -.3])
        self.assertEqual(chosen_token_scores(SimpleNamespace(token_ids=[1, 2], logprobs=[{1: -.1}])), [-.1, None])

    def test_rolled_back_suffix_is_replaced_not_accumulated(self):
        ledger = TokenScoreLedger()
        ledger.update('', '甲乙', '甲乙'.encode(), [
            dict(start=0,end=3,original_width=3,token_id=1,token_index=0,logprob=math.log(.9)),
            dict(start=3,end=6,original_width=3,token_id=2,token_index=1,logprob=math.log(.1))])
        ledger.update('甲', '丙', '丙'.encode(), [dict(start=0,end=3,original_width=3,token_id=3,token_index=0,logprob=math.log(.8))])
        result = score_source_range(ledger.text_projection('甲丙', 'Chinese'),0,2)
        self.assertAlmostEqual(result['score'], math.sqrt(.9 * .8))
        self.assertEqual(result['scored_tokens'], 2)
        self.assertEqual(result['evidence_revisions'], [1,2])
        self.assertFalse(any(span['token_id']==2 for span in ledger.spans))

    def test_prefix_cut_through_old_token_is_unknown(self):
        ledger = TokenScoreLedger()
        ledger.update('', '甲乙', '甲乙'.encode(), [dict(start=0,end=6,original_width=6,token_id=1,token_index=0,logprob=-.1)])
        ledger.update('甲', '丙', '丙'.encode(), [dict(start=0,end=3,original_width=3,token_id=3,token_index=0,logprob=-.2)])
        result = score_source_range(ledger.text_projection('甲丙','Chinese'),0,2)
        self.assertEqual(result['coverage'], .5)
        self.assertIsNone(result['score'])
        self.assertIn('rollback_cut_through_scored_token',ledger.issues)

    def test_header_does_not_contribute_to_text_score(self):
        ledger = TokenScoreLedger()
        header = 'language Chinese<asr_text>'
        size = len(header.encode())
        ledger.update('',header+'你好', (header+'你好').encode(), [
            dict(start=0,end=size,original_width=size,token_id=1,token_index=0,logprob=math.log(.99)),
            dict(start=size,end=size+6,original_width=6,token_id=2,token_index=1,logprob=math.log(.3))])
        result=score_source_range(ledger.text_projection('你好',None),0,2)
        self.assertAlmostEqual(result['score'],.3)
        self.assertEqual(result['scored_tokens'],1)

    def test_parser_repetition_repair_is_not_guessed(self):
        ledger=TokenScoreLedger()
        raw='嗯'*30
        ledger.update('',raw,raw.encode(),[])
        self.assertEqual(ledger.text_projection('嗯','Chinese')['status'],'unsupported')

    def test_missing_byte_piece_does_not_cover_a_character(self):
        result=scored_projection('龘')
        result['spans'][0]['end']=2
        score=score_source_range(result,0,1)
        self.assertEqual(score['coverage'],0.)
        self.assertIsNone(score['score'])

    def test_current_punctuation_decimal_and_enumeration_rules(self):
        text='材料已备齐，会议将在今天开始；请核对金额：金额是3.14元。苹果、梨都备齐！'
        result=project_window_scores(scored_projection(text),final=True)
        self.assertEqual([item['text'] for item in result['fragments']],
            ['材料已备齐，','会议将在今天开始；','请核对金额：','金额是3.14元。','苹果、梨都备齐！'])
        self.assertEqual([w['fragment_count'] for w in result['windows']],[3,2])
        self.assertTrue(result['windows'][-1]['final_tail'])
        for window in result['windows']:
            self.assertAlmostEqual(window['confidence']['score'],.8)
            self.assertEqual(window['confidence']['coverage'],1.)
            self.assertFalse(window['confidence']['production_skip_allowed'])

    def test_streaming_short_group_remains_pending(self):
        text='材料已备齐，会议马上开始。'
        result=project_window_scores(scored_projection(text),final=False)
        self.assertEqual(result['windows'],[])
        self.assertEqual(result['pending_fragment_indices'],[0,1])

    def test_numeric_self_correction_uses_project_merge(self):
        text='预算是5元，不对，应该是6元。材料已经备齐，会议马上开始。'
        result=project_window_scores(scored_projection(text),final=True)
        self.assertIn('5元',result['fragments'][0]['text'])
        self.assertIn('6元',result['fragments'][0]['text'])
        self.assertEqual(result['fragments'][0]['confidence']['coverage'],1.)

    def test_changed_completion_bytes_cannot_create_a_high_score(self):
        ledger=TokenScoreLedger()
        ledger.update('', '乙', '甲'.encode(), [dict(start=0,end=3,original_width=3,token_id=1,token_index=0,logprob=-.001)])
        result=score_source_range(ledger.text_projection('乙','Chinese'),0,1)
        self.assertIsNone(result['score'])
        self.assertEqual(result['coverage'],0.)


class RealQwenTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path('/home/aim0/data/models/ASR/Qwen3-ASR-1.7B')
        if not (path/'tokenizer_config.json').exists():
            raise unittest.SkipTest('Local Qwen tokenizer not available')
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,fix_mistral_regex=True)
        cls.mapper=TokenBytes(cls.tokenizer)

    def test_real_tokenizer_bytes_reconstruct_unicode_and_decimal(self):
        text='龘𠮷，金额3.14元。苹果、梨都备齐！'
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        payload,spans=self.mapper.decode(ids,[-.1]*len(ids))
        self.assertEqual(payload,text.encode())
        self.assertEqual(sum(item['end']-item['start'] for item in spans),len(payload))
        self.assertTrue(any('�' in self.tokenizer.decode([token]) for token in ids))

    def test_end_token_is_not_transcript_evidence(self):
        ids=self.tokenizer.encode('你好',add_special_tokens=False)+[self.tokenizer.eos_token_id]
        payload,spans=self.mapper.decode(ids,[-.1]*len(ids))
        self.assertEqual(payload,'你好'.encode())
        self.assertEqual(len(spans),len(ids)-1)


if __name__=='__main__':
    unittest.main()
