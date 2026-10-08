"""Registered anatomical stems in complete, explicit procedure objects."""
from dataclasses import replace
import pytest
from system.entity_store import EntityDefinition
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment,fallback_entity_segment
from system.protection import EntityProtector


def definition():
    return EntityDefinition(1,'胰十二指肠','TERM','medical','normalize',0,())


def prepare(text,ds=None,mode='auto',allow_auto=True):
    ds=(definition(),) if ds is None else ds
    return prepare_entity_segment(text,EntityProtector(ds,protect_numeric_spans=False),
        EntityCandidateMatcher(ds,selected_domain='medical',mode=mode),allow_auto=allow_auto)


@pytest.mark.parametrize('source,expected',[
    ('以十二指肠切除术。','胰十二指肠切除术。'),
    ('麻醉后，医生马上为他展开预计长达七个小时的手术。以十二指肠切除术。',
     '麻醉后，医生马上为他展开预计长达七个小时的手术。胰十二指肠切除术。'),
    ('患者接受以十二指肠切除术。','患者接受胰十二指肠切除术。'),
])
def test_user_segment_and_full_window_normalize_only_the_existing_stem(source,expected):
    p=prepare(source)
    assert p.baseline_text==expected
    assert p.normalizations[0]['observed']=='以十二指肠'
    assert p.normalizations[0]['canonical']=='胰十二指肠'
    assert p.normalizations[0]['match_type']=='CONTEXT_PINYIN'
    assert 'context:medical_procedure_stem_role' in p.report.auto_matches[0].reasons
    assert fallback_entity_segment(p,('forced_failure',)).text==expected


@pytest.mark.parametrize('text',[
    '医生以十二指肠切除术为例解释流程。',
    '以十二指肠切除术为例解释流程。',
    '患者接受以十二指肠切除术为主的治疗。',
    '以十二指肠切除术，作为第一步。',
    '以十二指肠切除术。作为第一步。',
    '你以哪一项手术作为依据？以十二指肠切除术。',
    '以十二指肠为研究对象。','以十二指肠为基础。',
    '医生介绍以十二指肠切除术的作者。',
    '作者接受以十二指肠切除术作为小说素材。',
    '患者接受十三指肠切除术。','患者接受肺十二指肠切除术。',
    '患者接受以十二指肠切断术。','患者接受以十二指肠手术。',
    '患者接受@以十二指肠切除术。','a以十二指肠切除术。',
    '患者接受以十二，指肠切除术。','《以十二指肠切除术》。',
])
def test_prepositions_other_facts_missing_actions_and_nonclinical_uses_stay(text):
    assert prepare(text).baseline_text==text


def test_policies_modes_known_source_and_spelling_ambiguity_abstain():
    text='以十二指肠切除术。';d=definition()
    for item in [replace(d,enabled=False),replace(d,normalization_policy='preserve'),replace(d,entity_type='PERSON')]:
        assert prepare(text,(item,)).baseline_text==text
    for mode,allow in [('off',True),('hint',True),('shadow',True),('auto',False)]:
        assert prepare(text,mode=mode,allow_auto=allow).baseline_text==text
    for rival in [EntityDefinition(2,'怡十二指肠','TERM','medical','preserve',0,()),
                  EntityDefinition(2,'另一个名称','TERM','medical','preserve',0,('怡十二指肠',)),
                  EntityDefinition(2,'怡十二指肠切除术','TERM','medical','preserve',0,()),
                  EntityDefinition(2,'另一完整术式','TERM','medical','preserve',0,('怡十二指肠切除术',)),
                  EntityDefinition(2,'以十二指肠','TERM','medical','preserve',0,())]:
        assert prepare(text,(d,rival)).baseline_text==text


def test_rule_applies_to_registered_stems_and_does_not_invent_a_library_entry():
    ds=(EntityDefinition(2,'甲状旁腺','TERM','medical','normalize',0,()),)
    assert prepare('甲壮旁现切除术。',ds).baseline_text=='甲状旁腺切除术。'
    assert prepare('以十二指肠切除术。',ds).baseline_text=='以十二指肠切除术。'
