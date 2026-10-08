"""Check exact suffix scoring and reuse of identical source probabilities."""

import math
import unittest

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'Torch/Transformers are optional outside the Refiner runtime')
class RefinerPairScoringTests(unittest.TestCase):
    def setUp(self):
        from system.live_qwen_refiner import TransformersRefiner
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, pad_token_id=0, attention_dropout=0,
            )).eval()

        class CountingModel(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.calls = 0
            def forward(self, **kwargs):
                self.calls += 1
                return self.wrapped(**kwargs)
            def get_input_embeddings(self):
                return self.wrapped.get_input_embeddings()

        class Tokenizer:
            pad_token_id = 0
            def __call__(self, text, **kwargs):
                return {'input_ids': torch.tensor([[ord(ch) % 28 + 1 for ch in text]], dtype=torch.long)}
            def apply_chat_template(self, *args, **kwargs):
                return {'input_ids': torch.tensor([[2, 4, 6, 8]])}

        self.refiner = TransformersRefiner.__new__(TransformersRefiner)
        self.refiner._torch = torch
        self.refiner._model = CountingModel(model)
        self.refiner._tokenizer = Tokenizer()
        self.refiner._template_kwargs = {}
        from collections import OrderedDict
        self.refiner._repetition_source_scores = OrderedDict()

    def check_pairs(self, cropped):
        self.refiner._scoring_logits_to_keep = cropped
        for prompt in (torch.tensor([2, 4, 6]), torch.tensor([2, 4, 6, 8, 10])):
            for pair in [('abc', 'abcdef'), ('x', 'x'), ('', 'abcdef'), ('longer phrase', 'z')]:
                with self.subTest(cropped=cropped, prompt_length=len(prompt), pair=pair):
                    expected = []
                    for completion in pair:
                        ids = self.refiner._tokenizer(completion)['input_ids'][0]
                        joined = torch.cat((prompt, ids)).unsqueeze(0)
                        with torch.inference_mode():
                            logits = self.refiner._model(input_ids=joined).logits[0, len(prompt)-1:-1, :].float()
                            expected.append(torch.log_softmax(logits, -1).gather(1, ids.unsqueeze(1)).mean().item())
                    self.refiner._model.calls = 0
                    actual = self.refiner._completion_log_probabilities(prompt, pair)
                    self.assertEqual(self.refiner._model.calls, len(pair))
                    for observed, reference in zip(actual, expected):
                        if math.isnan(reference):
                            self.assertTrue(math.isnan(observed))
                        else:
                            self.assertAlmostEqual(observed, reference, places=6)

    def test_uncropped_completion_alignment(self):
        self.check_pairs(cropped=False)

    def test_suffix_logits_preserve_probabilities(self):
        self.check_pairs(cropped=True)

    def test_source_probability_reused_only_with_identical_prompt_and_sentence(self):
        self.refiner._scoring_logits_to_keep = True
        context = '他们他们又又说话。'
        self.refiner.review_repetition_candidate_with_status(context, '他们他们', '他们', 0)
        self.assertEqual(self.refiner._model.calls, 2)
        cached = self.refiner.review_repetition_candidate_with_status(context, '又又', '又', 4)
        self.assertEqual(self.refiner._model.calls, 3)
        self.refiner._repetition_source_scores.clear()
        fresh = self.refiner.review_repetition_candidate_with_status(context, '又又', '又', 4)
        self.assertEqual(cached[0], fresh[0])
        self.assertEqual(cached[3], fresh[3])
        before = self.refiner._model.calls
        self.refiner.review_repetition_candidate_with_status('前文。'+context, '又又', '又', 7)
        self.assertEqual(self.refiner._model.calls-before, 2)

    def test_source_score_cache_is_bounded(self):
        self.refiner._completion_log_probabilities = lambda _prompt, values: [-1.0 for _ in values]
        for i in range(140):
            self.refiner.review_repetition_candidate_with_status(f'前文{i}。他们他们说话。', '他们他们', '他们', len(f'前文{i}。'))
        self.assertEqual(len(self.refiner._repetition_source_scores), 128)


if __name__ == '__main__':
    unittest.main()
