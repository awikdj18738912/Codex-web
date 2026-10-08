"""Registered equivalent aliases can supply full-sound medical evidence."""
from dataclasses import replace
import pytest

from system.entity_store import EntityDefinition
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment, fallback_entity_segment
from system.protection import EntityProtector


def definitions():
    return (EntityDefinition(1,'ECMO','TERM','medical','normalize',0,('艾克膜',)),
            EntityDefinition(2,'ECMO手术','TERM','medical','normalize',0,('艾克膜手术',)))


def prepare(text, ds=None, mode='auto', allow_auto=True):
    ds=definitions() if ds is None else ds
    return prepare_entity_segment(text,EntityProtector(ds,protect_numeric_spans=False),
        EntityCandidateMatcher(ds,selected_domain='medical',mode=mode),allow_auto=allow_auto)


@pytest.mark.parametrize('source,expected',[
    ('这台埃克莫支持生命。','这台ECMO支持生命。'),
    ('第二台埃克莫不但没有耗材。','第二台ECMO不但没有耗材。'),
    ('患者接受埃克莫手术。','患者接受ECMO手术。'),
    ('启动了埃克莫。','启动了ECMO。'),
])
def test_complete_alias_sound_in_a_medical_object_slot(source,expected):
    p=prepare(source)
    assert p.baseline_text==expected
    assert p.normalizations[0]['match_type']=='CONTEXT_ALIAS_PINYIN'
    assert fallback_entity_segment(p,('forced_refiner_failure',)).text==expected


@pytest.mark.parametrize('text',[
    '医生埃克莫建议继续治疗。','请埃克莫医生过来。',
    '这台埃克莫医生推荐的仪器很好。','患者埃克莫说很难受。',
    '接受埃克莫作为礼物。','进行埃克莫的签名核对。',
    '做埃克莫的作者访谈。','使用《埃克莫》这本小说。',
    '使用@埃克莫_2026。','使用/埃克莫/file。',
    '使用埃克莫这个词的写法。','谈到了埃克莫。',
    '这台艾克曼支持生命。','这台埃克蒙支持生命。',
    '这台埃克，莫支持生命。','使用埃克莫作为人名。',
])
def test_no_renaming_people_titles_identifiers_or_other_sounds(text):
    assert prepare(text).baseline_text==text


def test_only_declared_alias_and_auto_normalization_are_authorized():
    text='这台埃克莫支持生命。'
    for mode,allow in [('off',True),('shadow',True),('hint',True),('auto',False)]:
        assert prepare(text,mode=mode,allow_auto=allow).baseline_text==text
    base=definitions()[0]
    for d in [replace(base,aliases=()),replace(base,enabled=False),
              replace(base,normalization_policy='preserve'),replace(base,entity_type='PERSON')]:
        assert prepare(text,(d,)).baseline_text==text


def test_namespace_checks_canonicals_and_aliases_even_preserved_entries():
    text='这台埃克莫支持生命。';base=definitions()[0]
    for competitor in [EntityDefinition(3,'爱克模','TERM','medical','preserve',0,()),
                       EntityDefinition(3,'另一设备','TERM','medical','preserve',0,('爱克模',))]:
        assert prepare(text,(base,competitor)).baseline_text==text


def test_long_alias_preserves_prefix_normalization_and_suffix_once():
    p=prepare('患者做埃克莫手术，剂量20毫克。')
    assert p.baseline_text=='患者做ECMO手术，剂量20毫克。'
    assert len(p.normalizations)==1
    assert p.normalizations[0]['canonical']=='ECMO手术'


def test_short_exact_alias_still_protects_its_span():
    ds=(*definitions(),EntityDefinition(3,'埃克莫','PERSON','medical','preserve',0,()))
    text='这台埃克莫支持生命。'
    assert prepare(text,ds).baseline_text==text
