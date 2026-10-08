"""Contextual omission correction must abstain on competing facts and words."""
from dataclasses import replace
import json
import sqlite3
import subprocess
import sys

import pytest

from system.entity_store import EntityDefinition, EntitySpellingContext, EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment, finalize_entity_segment, fallback_entity_segment
from system.protection import EntityProtector


def term(canonical='骨质疏松症', entity_id=1):
    context = EntitySpellingContext((('医生', 1.0), ('患者', 1.0)), 1.0,
        'declared medical glossary', canonical, 'TERM', 'medical', allow_internal_omission=True)
    return EntityDefinition(entity_id, canonical, 'TERM', 'medical', 'normalize', 0, (),
                            spelling_context=context)


def prepare(text, definitions=None, mode='auto', allow_auto=True):
    definitions = definitions or (term(),)
    protector = EntityProtector(definitions, protect_numeric_spans=False)
    matcher = EntityCandidateMatcher(definitions, selected_domain='medical', mode=mode)
    return prepare_entity_segment(text, protector, matcher, allow_auto=allow_auto), protector


def test_internal_omission_changes_only_the_registered_term():
    text = '医生说患者有骨疏松症，原用量20毫克，每天2次。'
    prepared, _ = prepare(text)
    assert prepared.baseline_text == text.replace('骨疏松症', '骨质疏松症')
    assert prepared.normalizations[0]['match_type'] == 'CONTEXT_OMISSION'


def test_unrelated_model_edits_cannot_erase_verified_spelling():
    prepared, protector = prepare('医生说患者有骨疏松症。')
    result = finalize_entity_segment('把原文全部删除。', prepared, protector)
    assert not result.accepted
    assert result.text == '医生说患者有骨质疏松症。'
    assert fallback_entity_segment(prepared, ('model_failed',)).text == result.text


def test_context_and_normalization_authorization_are_required():
    text = '医生说患者有骨疏松症。'
    for definition in [replace(term(), spelling_context=None),
                       replace(term(), spelling_context=replace(term().spelling_context, allow_internal_omission=False)),
                       replace(term(), spelling_context=replace(term().spelling_context, allow_internal_omission='true')),
                       replace(term(), enabled=False),
                       replace(term(), normalization_policy='preserve'),
                       replace(term(), canonical_text='骨量疏松症')]:
        assert prepare(text, (definition,))[0].baseline_text == text
    for mode, allow in [('off', True), ('shadow', True), ('hint', True), ('auto', False)]:
        assert prepare(text, mode=mode, allow_auto=allow)[0].baseline_text == text
    for text in ['可能是骨疏松症。', '医生来了。可能是骨疏松症。']:
        assert prepare(text)[0].baseline_text == text


def test_competing_registered_expansions_abstain():
    text = '医生说患者有骨疏松症。'
    definitions = (term(), term('骨量疏松症', 2))
    assert prepare(text, definitions)[0].baseline_text == text


def test_anatomical_and_clinical_qualifiers_are_not_invented():
    cases = [
        ('医生诊断甲状腺功能亢进症。', '甲状旁腺功能亢进症'),
        ('患者接受经肾镜取石术。', '经皮肾镜取石术'),
        ('医生诊断小细胞肺癌。', '非小细胞肺癌'),
        ('医生诊断骨关节炎。', '骨性关节炎'),
    ]
    for text, canonical in cases:
        assert prepare(text, (term(canonical),))[0].baseline_text == text


def test_other_illness_and_ordinary_grammar_are_preserved():
    for text in ['患者有骨质增生，医生建议检查。', '患者曾经系统学习了急救。',
                 '医生说骨质疏松正在加重。']:
        assert prepare(text)[0].baseline_text == text


def test_identifiers_punctuation_and_spelling_discussion_are_preserved():
    for text in ['医生查看@骨疏松症_2026。', '医生查看x/骨疏松症.txt。',
                 '医生说患者有骨、疏松症。', '患者问骨疏松症这个写法对吗？']:
        assert prepare(text)[0].baseline_text == text


def test_complete_registered_source_is_never_overwritten():
    source = term('骨疏松症', 2)
    source = replace(source, normalization_policy='preserve', spelling_context=None)
    text = '医生说患者有骨疏松症。'
    assert prepare(text, (term(), source))[0].baseline_text == text


def test_opt_in_roundtrip_disable_and_old_library_read(tmp_path):
    path = tmp_path / 'entities.db'
    store = EntityStore(path)
    d = store.upsert_entity('骨质疏松症', entity_type='TERM', domain='medical', normalization_policy='normalize')
    store.set_spelling_context(d.entity_id, keywords={'患者': 1}, minimum_score=1, source='verified list')
    before = path.read_bytes()
    assert not EntityStore(path).get_entity_by_id(d.entity_id).spelling_context.allow_internal_omission
    assert path.read_bytes() == before
    enabled = store.set_spelling_context(d.entity_id, keywords={'患者': 1}, minimum_score=1,
                                         source='verified list', allow_internal_omission=True)
    assert enabled.spelling_context.allow_internal_omission
    assert EntityStore(path).get_entity_by_id(d.entity_id).spelling_context.allow_internal_omission
    disabled = store.set_spelling_context(d.entity_id, keywords={'患者': 1}, minimum_score=1, source='verified list')
    assert not disabled.spelling_context.allow_internal_omission
    for invalid in ['true', 1, None]:
        with pytest.raises(ValueError, match='must be a boolean'):
            store.set_spelling_context(d.entity_id, keywords={'患者': 1}, source='list', allow_internal_omission=invalid)


def test_import_migrates_atomically_and_exports_permission(tmp_path):
    db = tmp_path / 'entities.db'
    store = EntityStore(db)
    store.upsert_entity('既有实体', domain='medical')
    record = dict(canonical_text='骨质疏松症', entity_type='TERM', domain='medical', source='verified list',
        spelling_context=dict(keywords={'患者': 1}, minimum_score=1, source='verified list', allow_internal_omission=True))
    path = tmp_path / 'terms.json'
    path.write_text(json.dumps([record], ensure_ascii=False))
    command = [sys.executable, '-m', 'system.entity_context_cli', '--db', str(db)]
    subprocess.run([*command, 'import', '--input', str(path)], check=True, capture_output=True)
    assert store.get_entity('骨质疏松症', 'medical').spelling_context.allow_internal_omission
    exported = tmp_path / 'export.json'
    subprocess.run([*command, 'export', '--output', str(exported)], check=True, capture_output=True)
    assert next(r for r in json.loads(exported.read_text()) if r['canonical_text']=='骨质疏松症')['spelling_context']['allow_internal_omission']
    # A malformed later entry cannot commit the first staged entry or migrate
    # the destination's old schema as a partial import side effect.
    old_db = tmp_path / 'old.db'
    old_store = EntityStore(old_db)
    original = old_store.upsert_entity('保留实体')
    invalid = dict(record, canonical_text='另一个术语', spelling_context=dict(record['spelling_context'], allow_internal_omission='true'))
    path.write_text(json.dumps([record, invalid], ensure_ascii=False))
    result = subprocess.run([sys.executable, '-m', 'system.entity_context_cli', '--db', str(old_db), 'import', '--input', str(path)], capture_output=True)
    assert result.returncode != 0
    assert old_store.list_entities() == (original,)
    with sqlite3.connect(old_db) as connection:
        assert 'allow_internal_omission' not in {r[1] for r in connection.execute('PRAGMA table_info(entity_spelling_contexts)')}
