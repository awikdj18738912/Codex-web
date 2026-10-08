"""Opt-in near sounds of registered medical acronyms and verified repeats."""
from dataclasses import replace
import pytest
from system.entity_store import EntityDefinition,EntitySpellingContext
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment,fallback_entity_segment
from system.protection import EntityProtector


def definitions(enabled=True):
    result=[]
    for i,c,a in [(1,'ECMO','艾克膜'),(2,'ECMO手术','艾克膜手术')]:
        p=EntitySpellingContext((('机器',1.0),('手术',1.0)),1.0,'Reviewed medical alias',c,'TERM','medical',allow_near_spelling=enabled)
        result.append(EntityDefinition(i,c,'TERM','medical','normalize',0,(a,),spelling_context=p))
    return tuple(result)


def prepare(text,ds=None,mode='auto',allow_auto=True):
    ds=definitions() if ds is None else ds
    return prepare_entity_segment(text,EntityProtector(ds,protect_numeric_spans=False),
        EntityCandidateMatcher(ds,selected_domain='medical',mode=mode),allow_auto=allow_auto)


@pytest.mark.parametrize('text,expected',[
    ('第二台埃克姆的到来。','第二台ECMO的到来。'),
    ('埃克姆机器连接了起来。','ECMO机器连接了起来。'),
    ('他做了数十例艾克姆手术。','他做了数十例ECMO手术。'),
    ('这台爱克牧的耗材终于配齐了。','这台ECMO的耗材终于配齐了。'),
    ('启动了埃克姆。','启动了ECMO。'),
])
def test_one_segment_alias_error_in_a_declared_object_slot(text,expected):
    p=prepare(text)
    assert p.baseline_text==expected
    assert p.normalizations[0]['match_type']=='CONTEXT_ALIAS_NEAR_PINYIN'
    assert fallback_entity_segment(p,('forced_failure',)).text==expected


@pytest.mark.parametrize('text',[
    '医生埃克姆正在查房。','患者请埃克姆医生看报告。',
    '这台埃克姆医生推荐的机器很贵。','患者埃克姆说头痛。',
    '使用埃克姆作为人名。','这台艾克曼运行正常。','这台埃克蒙运行正常。',
    '使用@埃克姆_01。','连接/埃克姆/file。','使用《埃克姆》这本小说。',
    '连接爱科目教学设备。','医生在机器旁讨论埃克姆的拼写。',
    '谈到了埃克姆。','患者接受埃骨姆手术。',
])
def test_people_other_words_other_sounds_and_body_characters_stay(text):
    assert prepare(text).baseline_text==text


def test_near_permission_binding_policy_modes_and_namespace_are_required():
    text='这台埃克姆运行正常。';d=definitions()[0]
    for item in [definitions(False)[0],replace(d,spelling_context=None),replace(d,enabled=False),
                 replace(d,normalization_policy='preserve'),replace(d,entity_type='PERSON'),
                 replace(d,spelling_context=replace(d.spelling_context,canonical_text='别的设备'))]:
        assert prepare(text,(item,)).baseline_text==text
    for mode,allow in [('off',True),('shadow',True),('hint',True),('auto',False)]:
        assert prepare(text,mode=mode,allow_auto=allow).baseline_text==text
    for other in [EntityDefinition(3,'爱克母','TERM','medical','preserve',0,()),
                  EntityDefinition(3,'另一设备','TERM','medical','preserve',0,('爱克木',)),
                  EntityDefinition(3,'另一设备','TERM','medical','preserve',0,('爱克墨',))]:
        assert prepare(text,(d,other)).baseline_text==text


def test_weak_usage_does_not_reuse_an_earlier_alias_as_identity_proof():
    p=prepare('这台埃克莫已经就位，今天用埃克莫支持生命。',definitions(False))
    assert p.baseline_text=='这台ECMO已经就位，今天用埃克莫支持生命。'
    assert len(p.normalizations)==1


@pytest.mark.parametrize('text',[
    'ECMO用于生命支持，医生用埃克莫作为人名。',
    'ECMO用于生命支持。今天用埃克莫。',
    'xECMOy支持生命，今天用埃克莫。',
    '《ECMO》是一本书，今天用埃克莫。',
    '今天用埃克莫支持生命，稍后有ECMO。',
])
def test_a_topic_mention_does_not_authorize_weak_usage_slots(text):
    assert prepare(text,definitions(False)).baseline_text==text
