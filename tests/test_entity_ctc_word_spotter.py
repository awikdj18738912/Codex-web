"""Synthetic posterior tests; no model, audio corpus, or test gold is read."""
import numpy as np
import pytest

from experiments.scripts.entity_ctc_word_spotter import StreamingWordSpotter


def posterior(tokens, path, high=.995):
    result = np.full((len(path), len(tokens)), (1. - high) / (len(tokens) - 1))
    for frame, token in enumerate(path):
        result[frame, tokens.index(token)] = high
    return np.log(result)


def acoustic_evidence(hyps):
    return [{k: v for k, v in h.items() if k != 'available_audio_seconds'} for h in hyps]


def test_word_crosses_chunk_and_frames_not_replayed():
    tokens = ['<blk>', '厉', '飞', '宇', '他']
    spotter = StreamingWordSpotter(tokens, ['厉飞宇'])
    x = posterior(tokens, ['<blk>', '厉', '<blk>', '飞', '<blk>', '宇', '<blk>'])
    assert spotter.accept(x[:4], 0, .16) == []
    assert spotter.stats['frames_processed'] == 4
    seen = spotter.accept(x[4:], 4, .28)
    assert any(h['word'] == '厉飞宇' for h in seen)
    hyp, = spotter.finish()
    assert hyp['start_frame'] == 1 and hyp['end_frame'] == 5
    assert hyp['available_audio_seconds'] == .28
    assert hyp['token_count'] == 3
    assert hyp['acoustic_gap_per_token'] == pytest.approx(0)
    assert hyp['score'] == pytest.approx(hyp['raw_score'] + 6)
    assert spotter.stats['frames_processed'] == 7


def test_repeated_characters_need_blank_transition():
    tokens = ['<blk>', '哈', '他']
    no_blank = StreamingWordSpotter(tokens, ['哈哈'], minimum_gap=-.5)
    no_blank.accept(posterior(tokens, ['哈'] * 6), 0, .24)
    assert no_blank.finish() == []
    with_blank = StreamingWordSpotter(tokens, ['哈哈'], minimum_gap=-.5)
    with_blank.accept(posterior(tokens, ['哈', '<blk>', '哈']), 0, .12)
    hyp, = with_blank.finish()
    assert (hyp['start_frame'], hyp['end_frame']) == (0, 2)


def test_token_self_loop_does_not_earn_context_bonus():
    tokens = ['<blk>', '甲', '乙', '他']
    spotter = StreamingWordSpotter(tokens, ['甲乙'], context_weight=20, beam=100,
                                   minimum_gap=-.5)
    spotter.accept(posterior(tokens, ['甲', '乙', '乙', '乙']), 0, .16)
    for hyp in spotter.finish():
        assert hyp['score'] - hyp['raw_score'] == pytest.approx(40)


def test_blank_only_never_becomes_hotword_even_with_high_bias():
    tokens = ['<blk>', '甲', '乙', '▁甲', '▁乙']
    spotter = StreamingWordSpotter(tokens, ['甲乙'], context_weight=100, beam=1000)
    assert spotter.accept(posterior(tokens, ['<blk>'] * 30), 0, 1.2) == []
    assert spotter.finish() == []


def test_rare_character_byte_fallback_and_unsupported_coverage():
    rare = '龘'
    byte_tokens = [f'<0x{x:02X}>' for x in rare.encode('utf-8')]
    tokens = ['<blk>', '甲', '乙', *byte_tokens]
    spotter = StreamingWordSpotter(tokens, ['甲龘', '甲𠮷', '甲'])
    coverage = spotter.stats['coverage']
    assert coverage['covered_words'] == 1
    assert coverage['byte_fallback_words'] == ['甲龘']
    assert coverage['unsupported_words'] == {'甲𠮷': ['𠮷']}
    assert coverage['skipped_short_words'] == ['甲']
    spotter.accept(posterior(tokens, ['甲', *byte_tokens]), 0, .16)
    hyp, = spotter.finish()
    assert hyp['word'] == '甲龘' and hyp['token_count'] == 4


def test_tokenization_variants_are_bounded_and_include_mixed_boundary():
    word = '甲乙丙丁戊己庚辛'
    tokens = ['<blk>', *word, *['▁' + char for char in word]]
    spotter = StreamingWordSpotter(tokens, [word])
    assert spotter.stats['coverage']['variants_per_word'][word] == 32
    assert spotter.stats['coverage']['truncated_tokenization_words'] == [word]
    small = StreamingWordSpotter(['<blk>', '甲', '乙', '▁甲', '▁乙'], ['甲乙'])
    small.accept(posterior(small.tokens, ['甲', '▁乙']), 0, .08)
    assert small.finish()[0]['word'] == '甲乙'


def test_chunking_does_not_change_acoustic_results():
    tokens = ['<blk>', '甲', '乙', '丙', '哈']
    x = posterior(tokens, ['<blk>', '甲', '甲', '<blk>', '乙', '<blk>',
                            '丙', '<blk>', '哈', '<blk>', '哈', '<blk>',
                            '甲', '乙', '丙'])
    words = ['甲乙', '甲乙丙', '哈哈']
    one = StreamingWordSpotter(tokens, words)
    one.accept(x, 0, len(x) * .04)
    many = StreamingWordSpotter(tokens, words)
    for start in range(0, len(x), 2):
        end = min(start + 2, len(x))
        many.accept(x[start:end], start, end * .04)
    assert acoustic_evidence(one.finish()) == acoustic_evidence(many.finish())
    assert one.stats['frames_processed'] == many.stats['frames_processed'] == len(x)


def test_invalid_offsets_are_rejected_without_processing():
    tokens = ['<blk>', '甲', '乙']
    s = StreamingWordSpotter(tokens, ['甲乙'])
    x = posterior(tokens, ['甲'])
    s.accept(x, 10, 1)
    with pytest.raises(ValueError, match='contiguous'):
        s.accept(x, 10, 1)
    with pytest.raises(ValueError, match='contiguous'):
        s.accept(x, 12, 1)
    assert s.stats['frames_processed'] == 1
    s.finish()
    with pytest.raises(RuntimeError, match='finish'):
        s.accept(x, 11, 1)


def test_output_cap_is_counted_without_changing_final_internal_evidence():
    tokens = ['<blk>', '甲', '乙', '丙']
    spotter = StreamingWordSpotter(tokens, ['甲乙', '甲乙丙'], max_hypotheses_per_chunk=1)
    result = spotter.accept(posterior(tokens, ['甲', '乙', '丙']), 0, .12)
    assert len(result) == 1
    assert spotter.stats['output_truncated'] >= 1
    assert {h['word'] for h in spotter.finish()} == {'甲乙', '甲乙丙'}


def test_shared_graph_streams_have_independent_state_and_reset_counters(monkeypatch):
    tokens = ['<blk>', '甲', '乙', '丙']
    template = StreamingWordSpotter(tokens, ['甲乙'])
    template.accept(posterior(tokens, ['甲', '乙']), 90, 4.)
    assert template.finish()
    def no_retokenization(*args):
        raise AssertionError('create_stream must reuse the compiled graph')
    monkeypatch.setattr(StreamingWordSpotter, '_tokenize', no_retokenization)
    first = template.create_stream()
    second = template.clone_reset()
    for stream in (first, second):
        assert stream.parents is template.parents
        assert stream._terminal_nodes is template._terminal_nodes
        assert stream._terminals is template._terminals
        assert not stream.parents.flags.writeable
        assert stream.stats is not template.stats
        assert stream.stats['coverage'] is template.stats['coverage']
        assert stream.stats['frames_processed'] == 0
        assert stream.stats['chunks_processed'] == 0
        assert stream.stats['terminal_candidates'] == 0
        assert stream.stats['output_truncated'] == 0
        assert stream.stats['compute_seconds'] == 0
        assert stream._detections is not template._detections
        for name in ('_blank', '_nonblank', '_blank_gap', '_nonblank_gap',
                     '_blank_start', '_nonblank_start'):
            assert not np.shares_memory(getattr(stream, name), getattr(template, name))
    assert not np.shares_memory(first._nonblank, second._nonblank)
    first.accept(posterior(tokens, ['甲']), 0, .04)
    second.accept(posterior(tokens, ['乙']), 0, .04)
    assert second.finish() == []
    first.accept(posterior(tokens, ['乙']), 1, .08)
    hyp, = first.finish()
    assert (hyp['start_frame'], hyp['end_frame']) == (0, 1)
    assert template.stats['frames_processed'] == 2
    assert first.stats['frames_processed'] == 2
    assert second.stats['frames_processed'] == 1
    assert template._next_frame == 92
