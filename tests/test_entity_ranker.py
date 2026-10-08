"""Safety checks for the isolated experimental entity verifier."""
import random

import pytest

from system.entity_store import EntityDefinition
from system.entity_retrieval import EntityRetriever,sound_similarity
from system.entity_ranker import apply_verified_choices,candidate_input


def definition(word='化疗',entity_id=1,**kwargs):
    return EntityDefinition(entity_id,word,'TERM','medical',kwargs.pop('normalization_policy','normalize'),0,(),**kwargs)


def candidate(text='我花了后要剃头发'):
    return next(c.public_dict() for c in EntityRetriever([definition()]).alternatives(text,1,3) if c.canonical=='化疗')


def test_retrieval_is_not_authority_to_edit_short_valid_words():
    text='我花了后要剃头发';c=candidate(text)
    assert c['boundary_issue']=='short_chinese_span_requires_alias'
    assert text[c['start']:c['end']]=='花了'
    output,changes=apply_verified_choices(text,[c],[.9],threshold=.99)
    assert output==text and not changes


def test_disabled_and_preserve_entities_cannot_be_candidates():
    for d in [definition(enabled=False),definition(normalization_policy='preserve')]:
        assert EntityRetriever([d]).alternatives('我花了后要剃头发',1,3)==()


def test_registered_source_and_blocked_rule_spans_are_not_candidates():
    text='我花了后要剃头发';retriever=EntityRetriever([definition()])
    assert retriever.alternatives('我化疗',1,3)==()
    assert all(not(c.start<3 and 1<c.end) for c in retriever.retrieve(text,blocked_spans=[(1,3)]))


def test_uncertain_competing_entities_keep_source():
    text='我花了后要剃头发';first=candidate(text)
    second=dict(first,entity_id=2,canonical='花疗')
    output,changes=apply_verified_choices(text,[first,second],[.999,.99],threshold=.99)
    assert output==text and not changes


def test_confident_edit_copies_only_span_and_preserves_numbers():
    text='我花了后要剃头发，费用3200元';c=candidate(text)
    output,changes=apply_verified_choices(text,[c],[.999],threshold=.99)
    assert output=='我化疗后要剃头发，费用3200元'
    assert len(changes)==1


def test_stale_offsets_keep_source_and_model_input_rejects_them():
    text='我花了后要剃头发';c=dict(candidate(text),start=2)
    output,changes=apply_verified_choices(text,[c],[.999],threshold=.99)
    assert output==text and not changes
    with pytest.raises(ValueError):candidate_input(text,c)


@pytest.mark.parametrize('score',[float('nan'),float('inf'),-.1,1.1])
def test_invalid_confidence_fails_closed(score):
    text='我花了后要剃头发'
    with pytest.raises(ValueError):apply_verified_choices(text,[candidate(text)],[score],threshold=.99)


def test_candidate_does_not_cross_punctuation_or_ascii_tokens():
    candidates=EntityRetriever([definition('手术室')]).retrieve('手术，是a_CT_ually，2026年10月6日')
    assert all(all('\u3400'<=c<='\u9fff' for c in item.observed) for item in candidates)


def test_model_input_contains_context_and_candidate_without_reference():
    text='我花了后要剃头发';original,proposed,features=candidate_input(text,candidate(text))
    assert '[unused1]花了[unused2]' in original
    assert '[unused1]化疗[unused2]' in proposed
    assert '剃头发' in original and len(features)==8


def test_sound_pruning_never_discards_a_match_above_cutoff():
    rng=random.Random(20261006);sounds=['shen','ceng','hua','liao','la','chang','cang','jie','qie','chu','liu','ai','lin']
    for _ in range(500):
        left=tuple(rng.choices(sounds,k=rng.randrange(2,8)))
        right=tuple(rng.choices(sounds,k=rng.randrange(2,8)))
        exact=sound_similarity(left,right)
        pruned=sound_similarity(left,right,.55)
        if exact>=.55:assert pruned==exact
        else:assert pruned<.55
