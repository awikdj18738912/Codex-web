"""Medical symbols must keep complete letters, numbers and lexical roles."""
from dataclasses import replace
import pytest

from system.entity_store import EntityDefinition,EntitySpellingContext
from system.protection import EntityProtector
from system.entity_pipeline import prepare_entity_segment,fallback_entity_segment
from system.entity_matcher import EntityCandidateMatcher


def entity(name,identifier=1,**kwargs):
    return EntityDefinition(identifier,name,'TERM','medical','normalize',0,(),**kwargs)


def normalized(text,definitions=None):
    protector=EntityProtector(definitions or (entity('HER2'),entity('PD-1',2),entity('PD-L1',3)))
    result=protector.protect(text)
    return protector.restore(result.masked_text,result).text


@pytest.mark.parametrize('source,target',[
    ('表达Her two蛋白。','表达HER2蛋白。'),
    ('HER two基因阳性。','HER2基因阳性。'),
    ('阻止Her two和配体结合。','阻止HER2和配体结合。'),
    ('靶向H E R二的抗体。','靶向HER2的抗体。'),
    ('P D一、P D L一抗体。','PD-1、PD-L1抗体。'),
    ('PD一P D L一抗体。','PD-1PD-L1抗体。'),
    ('靶向PD one抗体，每次2毫克。','靶向PD-1抗体，每次2毫克。'),
    ('表达CD二十的细胞。','表达CD20的细胞。'),
    ('CD twenty抗体。','CD20抗体。'),
    ('CD二零抗体。','CD20抗体。'),
])
def test_same_complete_symbol_and_number(source,target):
    assert normalized(source,(entity('HER2'),entity('PD-1',2),entity('PD-L1',3),entity('CD20',4)))==target


@pytest.mark.parametrize('source',[
    'I saw her two days ago.',
    'HER two是编号。',
    '表达xHER二蛋白。',
    '表达HER二十蛋白。',
    '表达HER3蛋白。',
    '表达HERtwo蛋白。',
    'PD一二抗体。',
    'PD一a抗体。',
    'PD，L一抗体。',
    '医生HER two介绍基因检测。',
    '叫HER two的医生介绍抗体。',
    '抗体样本在HER two年采集。',
    '基因报告中HER two这个写法对吗？',
    '基因数据在x/HER two.txt里面。',
    '抗体样本地址是https://example.org/HER2。',
    '抗体邮件寄到HER2@example.org。',
    '喝兔这个蛋白。',
    'HER阴性，HER2阳性。',
])
def test_no_partial_guess_ordinary_words_or_changed_facts(source):
    assert normalized(source)==source


def test_policy_domain_type_and_ambiguity_are_respected():
    source='表达Her two蛋白。';definition=entity('HER2')
    for d in [replace(definition,normalization_policy='preserve'),replace(definition,enabled=False),
              replace(definition,domain='general'),replace(definition,entity_type='PERSON')]:
        assert normalized(source,(d,))==source
    source='PD一抗体。'
    assert normalized(source,(entity('PD-1'),entity('PD1',2)))==source
    assert normalized('HER one蛋白。',(entity('HER01'),))=='HER one蛋白。'


def test_registered_subname_and_alias_are_not_overwritten():
    source='表达HER two蛋白。'
    assert normalized(source,(entity('HER2'),replace(entity('HER',2),normalization_policy='preserve')))==source
    alias=replace(entity('另一种蛋白',2),aliases=('Her two',),normalization_policy='preserve')
    assert normalized(source,(entity('HER2'),alias))==source


def test_verified_symbol_survives_model_failure_without_changing_dose():
    defs=(entity('PD-1'),);p=EntityProtector(defs)
    source='PD一抗体，每次2毫克。'
    prepared=prepare_entity_segment(source,p,EntityCandidateMatcher(defs,selected_domain='medical',mode='auto'),allow_auto=True)
    assert prepared.baseline_text=='PD-1抗体，每次2毫克。'
    assert prepared.hints==()
    result=fallback_entity_segment(prepared,('model_failed',))
    assert result.text==prepared.baseline_text
    assert result.normalizations[0]['match_type']=='EXACT_ACRONYM_PRESENTATION'


def drug(name='贝伐珠单抗',identifier=1,opt_in=True):
    context=EntitySpellingContext((('治疗',1.0),('药物',1.0),('抑制',1.0),('医生',1.0)),1.0,
        'Independent prebuilt medical standard names',name,'TERM','medical',allow_internal_omission=opt_in)
    return entity(name,identifier,spelling_context=context)


def prepare_drug(source,definitions):
    return prepare_entity_segment(source,EntityProtector(definitions,protect_numeric_spans=False),
        EntityCandidateMatcher(definitions,selected_domain='medical',mode='auto'),allow_auto=True)


def test_rare_monoclonal_standard_name_does_not_require_jieba_entry():
    source='一个呢是贝伐单抗，这个是抑制血管生成的。'
    prepared=prepare_drug(source,(drug(),))
    assert prepared.baseline_text==source.replace('贝伐单抗','贝伐珠单抗')
    assert prepared.normalizations[0]['match_type']=='CONTEXT_OMISSION'


@pytest.mark.parametrize('source',[
    '医生叫贝伐单抗，介绍这种治疗。',
    '贝伐单抗医生介绍这种药物。',
    '医生说贝伐单抗。',
    '贝伐单抗这种写法用于药物名称吗？',
    '这种药物是贝伐珠单抗，剂量20毫克。',
    '医生使用非小细胞肺癌这种诊断。',
])
def test_drug_omission_does_not_force_person_names_or_already_correct_text(source):
    assert prepare_drug(source,(drug(),)).baseline_text==source


def test_rare_drug_completion_needs_permission_and_a_unique_standard():
    source='使用贝伐单抗治疗。'
    assert prepare_drug(source,(drug(opt_in=False),)).baseline_text==source
    assert prepare_drug(source,(drug(),drug('贝伐甲单抗',2))).baseline_text==source
    # Unknown general disease spellings must not inherit the rare-drug exception.
    assert prepare_drug('医生治疗未知长症。',(drug('未知甲长症'),)).baseline_text=='医生治疗未知长症。'
