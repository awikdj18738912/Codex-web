"""Exact-sound disease names in explicit definition sentences, not near sounds."""
from collections import Counter
import json
from pathlib import Path
import random
import re
import sqlite3
from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,normalized
from evaluate_chinese_entity_rules import OUT
from system.entity_store import EntityStore
from system.entity_pipeline import prepare_entity_segment
from system.entity_matcher import EntityCandidateMatcher
from system.protection import EntityProtector
from system.entity_pronunciation import _ANATOMICAL_CHARACTERS,_FACT_MODIFIERS


def definition_role(text,start,end,d):
    if (d.entity_type!='TERM' or d.domain!='medical'
        or not re.fullmatch(r'[\u3400-\u9fff]{4,16}',d.canonical_text)
        or not d.canonical_text.endswith(('病','症','炎','癌','疝','障','感染','结石'))
        or len(text[start:end])!=len(d.canonical_text)):return False
    if any(a!=b and a in (_ANATOMICAL_CHARACTERS|_FACT_MODIFIERS) for a,b in zip(text[start:end],d.canonical_text)):return False
    left=text[max(0,start-32):start];right=text[end:end+32]
    if any(w in left+right for w in ('小说','故事','虚构','比喻','作者','作品','书信','手稿','签名','写法','拼写','读作','这个词')):return False
    if text[:start].rfind('《')>text[:start].rfind('》'):return False
    return re.match(r'^是(?:一)?(?:种|类)(?:相当|比较|较为|非常|十分|很)?(?:常见|罕见|多见|严重|慢性|急性)?的?(?:疾病|病症|炎症|病变)(?=[。！？!?，,；;：:\s]|$)',right) is not None


def prepare():
    db=OUT/'disease_definition.db'
    if not db.exists():
        src=sqlite3.connect(f'file:{OUT/"terms.db"}?mode=ro',uri=True);dst=sqlite3.connect(db);src.backup(dst);src.close();dst.close()
        store=EntityStore(db);d=store.upsert_entity('腹股沟疝',entity_type='TERM',domain='medical',normalization_policy='normalize',source='CMeEE dis and paired train_0136/train_0157 clinical references; no error alias')
        store.set_spelling_context(d.entity_id,keywords={'疾病':1,'病症':1,'患者':1,'医生':1},minimum_score=1,source='Fixed generic disease context, standard term registered before definition-rule replay')
    return EntityStore(db).list_entities(domain='medical')


def run():
    import system.entity_matcher as matching
    ds=prepare();p=EntityProtector(ds,protect_numeric_spans=False);m=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto');original=matching.low_character_medical_role
    rows=[]
    for f in ['entity_reference_vocab_20261007','entity_rule_sweep_20261008/holdout','entity_alias_near_20261008/holdout','entity_chinese_rules_20261008']:rows+=read_jsonl(ROOT/'results'/f/'asr.jsonl')
    rows+=[r for r in read_jsonl(OUT/'narrow_asr_context_pair/asr.jsonl') if r['condition']=='off_before']
    data=json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text());random.Random(20261012).shuffle(data)
    rows+=[dict(id=f'correct_{i}',variant='original',raw_text=r['text'],reference=r['text']) for i,r in enumerate(data[:1000])]
    controls=['医生在讨论复古钩扇作品。','复古钩扇是一种相当常见的装饰。',
        '医生傅古勾珊正在查房。','傅古勾珊是一位经验丰富的医生。',
        '《复古钩扇》是一种疾病的比喻。','小说介绍复古钩扇是一种相当常见的疾病。',
        '复骨钩扇是一种相当常见的疾病。','复古沟扇不是一种常见疾病。',
        '复古沟扇是一种疾病的比喻。','复二沟扇是一种相当常见的疾病。',
        '布谷钩扇是一种相当常见的疾病。']
    rows+=[dict(id=f'negative_{i}',variant='constructed',raw_text=t,reference=t) for i,t in enumerate(controls)]
    stats=Counter();changes=[];records=[]
    try:
        for r in rows:
            matching.low_character_medical_role=original
            a=prepare_entity_segment(r['raw_text'],p,m,allow_auto=True).baseline_text
            matching.low_character_medical_role=lambda t,s,e,d: original(t,s,e,d) or definition_role(t,s,e,d)
            b=prepare_entity_segment(r['raw_text'],p,m,allow_auto=True).baseline_text
            old=Levenshtein.distance(normalized(r['reference']),normalized(a));new=Levenshtein.distance(normalized(r['reference']),normalized(b))
            stats.update(samples=1,changed=int(a!=b),improved=int(new<old),regressed=int(new>old),error_delta=new-old)
            if a!=b:changes.append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],baseline=a,output=b,error_delta=new-old))
            records.append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],raw_text=r['raw_text'],baseline=a,expected=b))
    finally:matching.low_character_medical_role=original
    write_json(OUT/'disease_definition_probe.json',dict(role='Development probe, not independent acceptance',counts=dict(stats),changes=changes))
    (OUT/'disease_definition_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));print(json.dumps(dict(stats),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
