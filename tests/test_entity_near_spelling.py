"""Near-spelling opt-in must preserve diagnoses, routes, atoms, and doses."""
from dataclasses import replace
import json
import sqlite3
import subprocess
import sys

import pytest

from system.entity_store import EntityDefinition,EntitySpellingContext,EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment,finalize_entity_segment
from system.protection import EntityProtector


def term(canonical='二氧化碳潴留',entity_id=1):
    p=EntitySpellingContext((('医生',1.),('呼吸机',1.),('潮气量',1.)),1.,'declared medical terms',
        canonical,'TERM','medical',allow_near_spelling=True)
    return EntityDefinition(entity_id,canonical,'TERM','medical','normalize',0,(),spelling_context=p)


def prepare(text,ds=None,mode='auto',allow=True):
    ds=ds or (term(),);p=EntityProtector(ds,protect_numeric_spans=False)
    m=EntityCandidateMatcher(ds,selected_domain='medical',mode=mode)
    return prepare_entity_segment(text,p,m,allow_auto=allow),p


def test_one_character_near_sound_keeps_rest_of_sentence_and_quantities():
    s='医生说二氧化碳储留，潮气量20毫升不变。'
    r,p=prepare(s)
    assert r.baseline_text==s.replace('储留','潴留')
    assert r.normalizations[0]['match_type']=='CONTEXT_NEAR_PINYIN'
    failed=finalize_entity_segment('丢弃全文。',r,p)
    assert not failed.accepted and failed.text==r.baseline_text


def test_permission_context_modes_and_identity_are_required():
    s='医生说二氧化碳储留。'
    for d in [replace(term(),spelling_context=None),
              replace(term(),spelling_context=replace(term().spelling_context,allow_near_spelling=False)),
              replace(term(),normalization_policy='preserve'),
              replace(term(),enabled=False)]:
        assert prepare(s,(d,))[0].baseline_text==s
    for mode,allow in [('off',True),('hint',True),('shadow',True),('auto',False)]:
        assert prepare(s,mode=mode,allow=allow)[0].baseline_text==s
    assert prepare('这叫二氧化碳储留。')[0].baseline_text=='这叫二氧化碳储留。'
    assert prepare('医生来了。这叫二氧化碳储留。')[0].baseline_text=='医生来了。这叫二氧化碳储留。'


def test_anatomical_route_chemical_and_drug_contrasts_are_not_rewritten():
    for s,c in [('医生诊断二氧化氮潴留。','二氧化碳潴留'),
                ('医生安排经鼻胆道引流术。','经皮胆道引流术'),
                ('医生诊断慢性肝囊炎。','慢性胆囊炎'),
                ('医生给患者阿洛西林钠。','阿莫西林钠')]:
        assert prepare(s,(term(c),))[0].baseline_text==s


def test_registered_source_word_boundaries_and_spelling_discussion_are_preserved():
    s='医生说二氧化碳储留。'
    source=replace(term('二氧化碳储留',2),normalization_policy='preserve',spelling_context=None)
    assert prepare(s,(term(),source))[0].baseline_text==s
    for s in ['医生讨论二氧化碳储留这个词的写法。',
              '医生说二氧化碳，储留。','医生查看x/二氧化碳储留.txt。',
              '医生说二氧化碳储留20。']:
        assert prepare(s)[0].baseline_text==s


def test_independent_options_survive_store_and_atomic_cli_import(tmp_path):
    db=tmp_path/'terms.db';store=EntityStore(db)
    d=store.upsert_entity('二氧化碳潴留',domain='medical',normalization_policy='normalize')
    store.set_spelling_context(d.entity_id,keywords={'医生':1},minimum_score=1,source='list',allow_near_spelling=True)
    loaded=EntityStore(db).get_entity_by_id(d.entity_id).spelling_context
    assert loaded.allow_near_spelling and not loaded.allow_internal_omission
    for bad in ['true',1,None]:
        with pytest.raises(ValueError):
            store.set_spelling_context(d.entity_id,keywords={'医生':1},source='list',allow_near_spelling=bad)
    data=[dict(canonical_text='骨质疏松症',domain='medical',source='list',spelling_context=dict(
        keywords={'医生':1},minimum_score=1,source='list',allow_internal_omission=True,allow_near_spelling=True))]
    path=tmp_path/'import.json';path.write_text(json.dumps(data,ensure_ascii=False))
    subprocess.run([sys.executable,'-m','system.entity_context_cli','--db',str(db),'import','--input',str(path)],check=True,capture_output=True)
    loaded=store.get_entity('骨质疏松症','medical').spelling_context
    assert loaded.allow_internal_omission and loaded.allow_near_spelling
    # near was the first lazily added column; internal was added later. Named
    # copies must preserve values despite a different declaration order.
    with sqlite3.connect(db) as c:
        assert [r[1] for r in c.execute('PRAGMA table_info(entity_spelling_contexts)')][-2:]==['allow_near_spelling','allow_internal_omission']
