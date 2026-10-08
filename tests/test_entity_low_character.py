"""Complete sound evidence is checked before mixed glyph-score filtering."""
from dataclasses import replace

from system.entity_store import EntityDefinition, EntitySpellingContext
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector


def definition(canonical='截肢手术', entity_id=1):
    p=EntitySpellingContext((('患者',1.0),('医生',1.0),('接受',1.0)),1.0,
        'declared procedure list',canonical,'TERM','medical')
    return EntityDefinition(entity_id,canonical,'TERM','medical','normalize',0,(),spelling_context=p)


def prepare(text, definitions=None, mode='auto', allow_auto=True):
    definitions=definitions or (definition(),)
    m=EntityCandidateMatcher(definitions,selected_domain='medical',mode=mode)
    return prepare_entity_segment(text,EntityProtector(definitions,protect_numeric_spans=False),m,allow_auto=allow_auto)


def test_user_example_is_recalled_without_a_misspelling_alias():
    p=prepare('患者需要接受介质手书，原剂量20毫克不变。')
    assert p.baseline_text=='患者需要接受截肢手术，原剂量20毫克不变。'
    assert p.report.auto_matches[0].final_score<0.8
    assert p.normalizations[0]['match_type']=='CONTEXT_PINYIN'


def test_plain_prebuilt_procedure_does_not_require_error_aliases_or_a_profile():
    ds=(replace(definition(),spelling_context=None),)
    for source,expected in [('介质手书','截肢手术'),
                            ('患者需要接受介质手书','患者需要接受截肢手术')]:
        assert prepare(source,ds).baseline_text==expected
    assert prepare('医生拿到了介质手书。',ds).baseline_text=='医生拿到了介质手书。'
    assert prepare('患者需要接受截指手书。',ds).baseline_text=='患者需要接受截指手书。'


def test_context_alone_does_not_turn_handwriting_into_surgery():
    for text in ['医生拿到了介质手书。','医生接受了介质手书作为礼物。',
                 '患者谈到了介质手书。','医生介绍了介质手书的作者。']:
        assert prepare(text).baseline_text==text


def test_medical_dictionary_does_not_rename_a_doctor_as_a_disease():
    for text in ['患者请医生高雪雅查看报告。','患者高雪雅说昨天很难受。']:
        assert prepare(text,(definition('高血压'),)).baseline_text==text
    assert prepare('患者需要接受截指手书。').baseline_text=='患者需要接受截指手书。'


def test_ambiguity_policies_modes_and_boundaries_remain_hard_guards():
    text='患者需要接受介质手书。'
    for ds in [(definition(),definition('截指手术',2)),
               (replace(definition(),normalization_policy='preserve'),),
               (replace(definition(),enabled=False),)]:
        assert prepare(text,ds).baseline_text==text
    for mode,allow in [('hint',True),('shadow',True),('off',True),('auto',False)]:
        assert prepare(text,mode=mode,allow_auto=allow).baseline_text==text
    for text in ['患者需要接受介质，手书。','患者需要接受@介质手书_2026。']:
        assert prepare(text).baseline_text==text
