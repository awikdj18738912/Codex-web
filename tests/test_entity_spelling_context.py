"""Behavioral safety and persistence tests for optional standard-name spelling."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import json
import subprocess
import sys

from system.entity_store import EntityStore,EntityDefinition,EntitySpellingContext
from system.entity_matcher import EntityCandidateMatcher
from system.protection import EntityProtector
from system.entity_pipeline import prepare_entity_segment,finalize_entity_segment,fallback_entity_segment


def name(canonical='刘艾林',entity_id=1):
    context=EntitySpellingContext((('记者',3.0),),2.0,'confirmed participant list',canonical,'PERSON','general')
    return EntityDefinition(entity_id,canonical,'PERSON','general','normalize',0,(),spelling_context=context)


def prepare(text,definitions=None,mode='auto',allow_auto=True):
    definitions=definitions or (name(),)
    p=EntityProtector(definitions);m=EntityCandidateMatcher(definitions,selected_domain='general',mode=mode)
    return prepare_entity_segment(text,p,m,allow_auto=allow_auto),p


class EntitySpellingContextTest(unittest.TestCase):
    def test_complete_registered_name_changes_only_its_spelling(self):
        p,_=prepare('本报记者刘爱林在10点发布消息。')
        self.assertEqual(p.baseline_text,'本报记者刘艾林在10点发布消息。')
        self.assertEqual(p.normalizations[0]['match_type'],'CONTEXT_PINYIN')

    def test_default_entry_requires_existing_confirmation(self):
        p,_=prepare('本报记者刘爱林。',(replace(name(),spelling_context=None),))
        self.assertEqual(p.baseline_text,'本报记者刘爱林。')

    def test_unmatched_context_abstains(self):
        p,_=prepare('我认识刘爱林。')
        self.assertEqual(p.baseline_text,'我认识刘爱林。')

    def test_context_in_different_sentence_does_not_authorize_name(self):
        p,_=prepare('记者来到现场。我认识刘爱林。')
        self.assertEqual(p.baseline_text,'记者来到现场。我认识刘爱林。')

    def test_registered_homophone_ambiguity_abstains(self):
        p,_=prepare('本报记者刘爱琳。',(name(),name('刘爱林',2)))
        self.assertEqual(p.baseline_text,'本报记者刘爱琳。')

    def test_different_surname_character_is_retained(self):
        p,_=prepare('本报记者留爱林。')
        self.assertEqual(p.baseline_text,'本报记者留爱林。')

    def test_different_pronunciation_is_retained(self):
        p,_=prepare('本报记者刘爱明。')
        self.assertEqual(p.baseline_text,'本报记者刘爱明。')

    def test_preserve_policy_is_never_overridden(self):
        p,_=prepare('本报记者刘爱林。',(replace(name(),normalization_policy='preserve'),))
        self.assertEqual(p.baseline_text,'本报记者刘爱林。')

    def test_modes_and_partial_permission_are_respected(self):
        for mode,allow in [('off',True),('hint',True),('shadow',True),('auto',False)]:
            with self.subTest(mode=mode,allow=allow):
                p,_=prepare('本报记者刘爱林。',mode=mode,allow_auto=allow)
                self.assertEqual(p.baseline_text,'本报记者刘爱林。')

    def test_identifier_and_book_title_are_retained(self):
        for text in ['记者账号@刘爱林_2026','记者文件x/刘爱林.txt','记者阅读《刘爱林传》。']:
            with self.subTest(text=text):
                p,_=prepare(text)
                self.assertEqual(p.baseline_text,text)

    def test_numeric_phrase_is_not_a_name(self):
        d=name('三石仪')
        p,_=prepare('记者公布的预算是三十亿。',(d,))
        self.assertEqual(p.baseline_text,'记者公布的预算是三十亿。')

    def test_invalid_refinement_preserves_verified_context_correction(self):
        p,protector=prepare('本报记者刘爱林。')
        final=finalize_entity_segment('全部删除。',p,protector)
        self.assertFalse(final.accepted)
        self.assertEqual(final.text,'本报记者刘艾林。')
        self.assertEqual(final.normalizations[0]['match_type'],'CONTEXT_PINYIN')
        fallback=fallback_entity_segment(p,('model_failure',))
        self.assertEqual(fallback.text,final.text)

    def test_old_unsafe_short_word_examples_remain_raw(self):
        cases=[('手术失败的概率很低。','手术室'),('骨碎了之后能够补回去。','骨髓'),('他们外出务工。','误工')]
        for text,canonical in cases:
            d=name(canonical)
            ctx=replace(d.spelling_context,entity_type='TERM',keywords=(('之后',3.0),('概率',3.0),('外出',3.0)))
            d=replace(d,entity_type='TERM',spelling_context=ctx)
            p,_=prepare(text,(d,))
            self.assertEqual(p.baseline_text,text)


class EntitySpellingContextStoreTest(unittest.TestCase):
    def test_roundtrip_rename_and_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            store=EntityStore(Path(folder)/'entities.db')
            d=store.upsert_entity('刘艾林',entity_type='PERSON',normalization_policy='normalize')
            store.set_spelling_context(d.entity_id,keywords={'记者':3},source='confirmed list')
            reopened=EntityStore(store.path).get_entity_by_id(d.entity_id)
            self.assertEqual(reopened.spelling_context.keywords,(('记者',3.0),))
            store.update_entity(d.entity_id,'刘艾明',entity_type='PERSON',normalization_policy='normalize')
            self.assertIsNone(store.get_entity_by_id(d.entity_id).spelling_context)
            store.set_spelling_context(d.entity_id,keywords={'记者':3},source='updated list')
            self.assertTrue(store.delete_entity(d.entity_id))
            with store._connect() as c:
                self.assertEqual(c.execute('SELECT COUNT(*) FROM entity_spelling_contexts').fetchone()[0],0)

    def test_invalid_context_never_enables_a_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            store=EntityStore(Path(folder)/'entities.db');d=store.upsert_entity('刘艾林')
            for keywords,minimum in [({},2),({'刘艾林':3},2),({'记者':float('nan')},2),({'记者':3},0)]:
                with self.subTest(keywords=keywords):
                    with self.assertRaises(ValueError):
                        store.set_spelling_context(d.entity_id,keywords=keywords,minimum_score=minimum,source='list')
            self.assertIsNone(store.get_entity_by_id(d.entity_id).spelling_context)

    def test_context_can_be_disabled_without_removing_the_entity(self):
        with tempfile.TemporaryDirectory() as folder:
            store=EntityStore(Path(folder)/'entities.db');d=store.upsert_entity('刘艾林')
            store.set_spelling_context(d.entity_id,keywords={'记者':3},source='list')
            self.assertTrue(store.clear_spelling_context(d.entity_id))
            self.assertIsNone(store.get_entity_by_id(d.entity_id).spelling_context)

    def test_import_preserves_omitted_aliases_and_disabled_state(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);db=folder/'entities.db';store=EntityStore(db)
            d=store.upsert_entity('刘艾林',entity_type='PERSON',aliases=['刘爱林'],normalization_policy='normalize')
            store.set_enabled_by_id(d.entity_id,enabled=False)
            records=[dict(canonical_text='刘艾林',source='confirmed list',spelling_context=dict(keywords={'记者':3},source='list'))]
            path=folder/'import.json';path.write_text(json.dumps(records,ensure_ascii=False))
            subprocess.run([sys.executable,'-m','system.entity_context_cli','--db',str(db),'import','--input',str(path)],check=True,capture_output=True)
            loaded=store.get_entity_by_id(d.entity_id)
            self.assertEqual(loaded.aliases,('刘爱林',))
            self.assertFalse(loaded.enabled)
            self.assertIsNotNone(loaded.spelling_context)

    def test_invalid_import_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);db=folder/'entities.db';store=EntityStore(db)
            original=store.upsert_entity('旧实体',normalization_policy='normalize')
            records=[dict(canonical_text='新实体',source='list'),dict(canonical_text='坏实体',source='list',spelling_context=dict(keywords={},source='bad'))]
            path=folder/'import.json';path.write_text(json.dumps(records,ensure_ascii=False))
            result=subprocess.run([sys.executable,'-m','system.entity_context_cli','--db',str(db),'import','--input',str(path)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(store.list_entities(),(original,))


if __name__=='__main__':unittest.main()
