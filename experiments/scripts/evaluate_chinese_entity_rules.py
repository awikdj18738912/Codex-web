"""Fixed Chinese procedure-role experiments with paired public medical audio."""
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import random
import re
from types import SimpleNamespace
from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,decode,normalized
from evaluate_reference_entity_vocab import extract
from evaluate_medical_entities import lexicon_occurrences
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector
from system.entity_pronunciation import procedure_spelling_evidence,medical_person_slot
from system.entity_boundaries import candidate_boundary_issue

OUT=ROOT/'results/entity_chinese_rules_20261008'
VARIANTS=['current','independent_sentence','procedure_heading','quantified_action']
REVIEWED=('白内障手术','腹股沟疝修补手术','前列腺癌','慢性肾脏病','胰十二指肠切除术','甲状旁腺激素')


def extra_procedure_evidence(text,start,end,definition,namespace,variant):
    source=text[start:end]
    # Delegate complete pronunciation, namespace and anatomical-source checks
    # to the existing verifier using its already-supported standalone role.
    if not procedure_spelling_evidence(source,0,len(source),definition,namespace):return ()
    if medical_person_slot(text,start,end) or candidate_boundary_issue(text,start,end):return ()
    left=text[max(0,start-64):start];right=text[end:end+24]
    if any(w in left+right for w in ('礼物','书信','手稿','作者','签名','笔迹','小说','故事','写法','拼写','写成','读作','字形','这个词','错别字','人名')):return ()
    if text[:start].rfind('《')>text[:start].rfind('》'):return ()
    if variant=='quantified_action':
        clause=re.split(r'[，,；;。！？!?\n\r]',left)[-1]
        if re.search(r'(接受|进行|实施|安排|完成|施行|开展|展开|做)(了|过)?(预计|一共|共)?(长达)?([0-9一二三四五六七八九十百数几]+)(例|次|项|小时|个小时)(的)?$',clause):return ('medical_quantified_procedure_role',)
        return ()
    clause_left=re.split(r'[。！？!?\n\r]',left)[-1]
    clause_right=re.split(r'[。！？!?\n\r]',right)[0]
    if clause_left.strip(' \t，,；;：:') or clause_right.strip(' \t，,；;：:'):return ()
    if variant=='independent_sentence':return ('independent_procedure_sentence',)
    previous=re.split(r'[。！？!?\n\r]',left)
    if len(previous)<2:return ()
    previous=previous[-2].strip()
    if re.search(r'(接受|进行|实施|安排|完成|施行|开展|展开|做)[^。！？!?\n\r]{0,28}手术$',previous):return ('medical_procedure_heading',)
    return ()


def controls():
    return [
        '医生把礼物放到了桌上。介质手书。',
        '医生在小说里展开一场手术。介质手书。',
        '作者安排了预计七小时的手术。介质手书。',
        '医生介绍这种手术的拼写。宜十二指肠切除术。',
        '患者接受了手术。截指手书。',
        '患者做了三次介质手书，作为礼物交给作者。',
        '医生安排了三次《介质手书》的阅读。',
        '医生接受介质手书作为手稿。',
        '医生让孩子假装螃蟹练习动作。',
        '患者请高雪雅医生查看报告。',
        '患者准备了一幅作品。介质手书。',
        '主持人介绍了这篇文献。介质手书。',
    ]


def prepare():
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    if (OUT/'selection.json').exists():raise FileExistsError('Frozen selection exists')
    excluded=set()
    folders=['entity_medical_scenes_20261006','entity_medical_scenes_20261006/holdout','entity_reference_vocab_20261007','entity_rule_sweep_20261008/holdout','entity_alias_near_20261008/holdout']
    for f in folders:excluded.update(r['source_row'] for r in json.loads((ROOT/'results'/f/'selection.json').read_text())['samples'])
    labels=defaultdict(set)
    for r in json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text()):
        for e in r['entities']:labels[e['entity']].add(e['type'])
    rows=pq.read_table(ROOT/'results/medical_entity_20261003/source/chinese_train.parquet').to_pylist()
    pools={'procedure':[],'disease':[],'control':[]};selection=[];names=defaultdict(list)
    for i,r in enumerate(rows):
        if i in excluded or not 3<=r['duration']<=20:continue
        if not 2<=sum('\u3400'<=c<='\u9fff' for c in r['text'])/r['duration']<=8:continue
        if re.search(r'[A-Za-z]',r['text']):continue
        entities=extract(r['text'],labels)
        for phrase in REVIEWED:
            for m in re.finditer(re.escape(phrase),r['text']):entities.append(dict(text=phrase,start=m.start(),end=m.end(),annotation_types=['reviewed_reference_phrase']))
        chosen=[]
        for e in sorted(entities,key=lambda e:(-(e['end']-e['start']),e['start'])):
            if not any(e['start']<q['end'] and q['start']<e['end'] for q in chosen):chosen.append(e)
        entities=chosen
        if any(e['text'].endswith(('手术','切除术')) or ('pro' in e['annotation_types'] and re.fullmatch(r'[\u3400-\u9fff]{3,16}',e['text'])) for e in entities):group='procedure'
        elif any((set(e['annotation_types']) & {'dis','sym'} and 'bod' not in e['annotation_types']) or e['text'].endswith(('癌','病','症','炎','瘤','障')) for e in entities):group='disease'
        elif not entities:group='control'
        else:continue
        pools[group].append((i,r,entities))
    for group,n in [('procedure',16),('disease',20),('control',8)]:
        pool=pools[group];random.Random('20261010:'+group).shuffle(pool)
        for i,r,entities in pool[:n]:
            rid=f'MultiMed_Chinese_train_{i:04d}';path=OUT/'audio'/(rid+'.wav');path.parent.mkdir(exist_ok=True)
            wave,rate=sf.read(io.BytesIO(r['audio']['bytes']),dtype='float32')
            if wave.ndim>1:wave=wave.mean(axis=1)
            if rate!=16000:
                f=gcd(rate,16000);wave=resample_poly(wave,16000//f,rate//f).astype('float32')
            sf.write(path,wave,16000,subtype='FLOAT')
            selection.append(dict(id=rid,source_row=i,audio_path=str(path),reference=r['text'],entities=entities,group=group,original_audio_sha256=hashlib.sha256(r['audio']['bytes']).hexdigest()))
            for e in entities:names[e['text']].append(dict(id=rid,reference=r['text'],start=e['start'],end=e['end']))
    # Keep the prior validated configuration as well as new reference names.
    src=ROOT/'results/entity_alias_near_20261008/official_alias_terms.db'
    import sqlite3
    s=sqlite3.connect(f'file:{src}?mode=ro',uri=True);d=sqlite3.connect(OUT/'terms.db');s.backup(d);s.close();d.close()
    store=EntityStore(OUT/'terms.db');known={d.canonical_text for d in store.list_entities(domain='medical')}
    for name in sorted(names):
        if name not in known:
            d=store.upsert_entity(name,domain='medical',entity_type='TERM',normalization_policy='normalize',source='Paired medical reference, frozen before ASR')
            if re.fullmatch(r'[\u3400-\u9fff]{3,16}',name):
                words={w:1.0 for w in ['患者','病人','医生','医院','手术','治疗','检查','诊断','症状','疾病','接受','进行','患有','患上','需要','开刀'] if w not in name}
                store.set_spelling_context(d.entity_id,keywords=words,minimum_score=1,source='Fixed generic medical context, not a per-record answer')
    # Two constructed standard procedures support negative controls only.
    write_json(OUT/'selection.json',dict(seed=20261010,noise_snr_db=5,samples=selection,pool_counts={k:len(v) for k,v in pools.items()},previous_rows_excluded=sorted(excluded),source_url='https://huggingface.co/datasets/leduckhai/MultiMed',role='Known reference-derived vocabulary; not unknown-entity generalization'))
    write_json(OUT/'reference_entities.json',[dict(canonical_text=k,evidence=v) for k,v in names.items()])
    write_json(OUT/'protocol.json',dict(variants=VARIANTS,rule_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_sha256=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),terms_sha256=hashlib.sha256((OUT/'terms.db').read_bytes()).hexdigest(),reference_at_inference=False))
    print('FROZEN',len(selection),Counter(r['group'] for r in selection),'library',len(store.list_entities(domain='medical')),flush=True)


def evaluate(label):
    import system.entity_matcher as matching
    ds=EntityStore(OUT/'terms.db').list_entities(domain='medical')
    if label=='development':
        rows=[]
        for folder in ['entity_reference_vocab_20261007','entity_rule_sweep_20261008/holdout','entity_alias_near_20261008/holdout']:rows+=read_jsonl(ROOT/'results'/folder/'asr.jsonl')
        data=json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text());random.Random(20261010).shuffle(data)
        rows+=[dict(id=f'correct_{i}',variant='original',raw_text=r['text'],reference=r['text']) for i,r in enumerate(data[:1000])]
        rows+=[dict(id=f'negative_{i}',variant='constructed',raw_text=t,reference=t) for i,t in enumerate(controls())]
        # Separate constructed standard name, no guessed error alias.
        from system.entity_store import EntityDefinition
        ds=(*ds,EntityDefinition(10000,'截肢手术','TERM','medical','normalize',0,()))
    else:rows=read_jsonl(OUT/'asr.jsonl')
    p=EntityProtector(ds,protect_numeric_spans=False);m=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto');original=matching.procedure_spelling_evidence
    counts={v:Counter() for v in VARIANTS};changes={v:[] for v in VARIANTS};records=[];names={d.canonical_text for d in ds}
    try:
        for r in rows:
            matching.procedure_spelling_evidence=original
            baseline=prepare_entity_segment(r['raw_text'],p,m,allow_auto=True).baseline_text;old=Levenshtein.distance(normalized(r['reference']),normalized(baseline));outputs={}
            for v in VARIANTS:
                if v=='current':matching.procedure_spelling_evidence=original
                else:
                    def evidence(text,start,end,d,namespace,rule=v):
                        return original(text,start,end,d,namespace) or extra_procedure_evidence(text,start,end,d,namespace,rule)
                    matching.procedure_spelling_evidence=evidence
                result=prepare_entity_segment(r['raw_text'],p,m,allow_auto=True);new=Levenshtein.distance(normalized(r['reference']),normalized(result.baseline_text));stats=counts[v]
                stats.update(samples=1,changed=int(result.baseline_text!=baseline),improved=int(new<old),regressed=int(new>old),error_delta=new-old)
                if r['variant'] in {'clean','noise'}:
                    expected=Counter(e['text'] for e in lexicon_occurrences(r['reference'],names));have=Counter(e['text'] for e in lexicon_occurrences(result.baseline_text,names))
                    stats['expected_mentions']+=sum(expected.values());stats['recovered_mentions']+=sum(min(n,have[w]) for w,n in expected.items())
                if result.baseline_text!=baseline:changes[v].append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],baseline=baseline,output=result.baseline_text,normalizations=result.normalizations,error_delta=new-old))
                outputs[v]=result.baseline_text
            records.append(dict(id=r['id'],variant=r['variant'],raw_text=r['raw_text'],reference=r['reference'],outputs=outputs))
    finally:matching.procedure_spelling_evidence=original
    write_json(OUT/f'{label}.json',dict(counts={v:dict(s) for v,s in counts.items()},changes=changes))
    (OUT/f'{label}_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    print(json.dumps({v:dict(s) for v,s in counts.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','asr','development','fresh']);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='asr':decode(SimpleNamespace(output=OUT,asr_url='http://127.0.0.1:8766'))
    else:evaluate(a.stage)
