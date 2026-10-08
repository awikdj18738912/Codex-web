"""Only full-sound registered names in explicit disease definitions qualify."""
from dataclasses import replace
import pytest
from system.entity_store import EntityDefinition
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment,fallback_entity_segment
from system.protection import EntityProtector


def definition():
    return EntityDefinition(1,'腹股沟疝','TERM','medical','normalize',0,())


def prepare(text,ds=None,mode='auto',allow_auto=True):
    ds=(definition(),) if ds is None else ds
    return prepare_entity_segment(text,EntityProtector(ds,protect_numeric_spans=False),
        EntityCandidateMatcher(ds,selected_domain='medical',mode=mode),allow_auto=allow_auto)


@pytest.mark.parametrize('tail',['是一种相当常见的疾病。','是一类常见病症。','是一种疾病。'])
def test_registered_standard_name_alone_can_verify_a_disease_definition(tail):
    p=prepare('复古钩扇'+tail)
    assert p.baseline_text=='腹股沟疝'+tail
    assert p.report.auto_matches[0].final_score<0.8
    assert p.normalizations[0]['match_type']=='CONTEXT_PINYIN'
    assert 'context:medical_disease_definition_role' in p.report.auto_matches[0].reasons
    assert fallback_entity_segment(p,('forced_failure',)).text==p.baseline_text


@pytest.mark.parametrize('text',[
    '复古钩扇是一种相当常见的装饰。','医生在讨论复古钩扇作品。',
    '傅古勾珊是一位医生。','医生复古钩扇是一种疾病。',
    '患者复古钩扇是一种疾病。','小说介绍复古钩扇是一种常见疾病。',
    '《复古钩扇》是一种疾病的比喻。','复古钩扇是一种疾病的比喻。',
    '复古钩扇不是一种疾病。','复骨钩扇是一种疾病。',
    '复二沟扇是一种疾病。','布谷钩扇是一种疾病。',
    '@复古钩扇是一种疾病。','/复古钩扇是一种疾病。',
    'a复古钩扇是一种疾病。','复古钩，扇是一种疾病。',
])
def test_ordinary_words_people_other_sounds_facts_and_identifiers_stay(text):
    assert prepare(text).baseline_text==text


def test_policies_modes_and_namespace_ambiguity_remain_hard_guards():
    text='复古钩扇是一种疾病。';d=definition()
    for item in [replace(d,enabled=False),replace(d,normalization_policy='preserve'),replace(d,entity_type='PERSON')]:
        assert prepare(text,(item,)).baseline_text==text
    for mode,allow in [('off',True),('shadow',True),('hint',True),('auto',False)]:
        assert prepare(text,mode=mode,allow_auto=allow).baseline_text==text
    for rival in [EntityDefinition(2,'富谷沟珊','PERSON','medical','preserve',0,()),
                  EntityDefinition(2,'另一名称','PERSON','medical','preserve',0,('傅古勾珊',))]:
        assert prepare(text,(d,rival)).baseline_text==text


def test_definition_does_not_complete_a_missing_or_short_term():
    assert prepare('复古钩是一种疾病。').baseline_text=='复古钩是一种疾病。'
    short=EntityDefinition(2,'中耳炎','TERM','medical','normalize',0,())
    assert prepare('钟尔言是一种疾病。',(short,)).baseline_text=='钟尔言是一种疾病。'
