"""Offline scoring only. Never imported by the inference collector/worker."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np
import opencc
from rapidfuzz.distance import Levenshtein

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(OUT / 'source_snapshot'))
from system.entity_audio_repair import normalized
from system.zh_itn_bridge import ZhITNNormalizer

ARMS = ('context', 'audio')
cc = opencc.OpenCC('t2s')
itn = ZhITNNormalizer(ROOT / 'zh-itn/zh-itn/build/libzh_itn_bridge.so')


@lru_cache(maxsize=30000)
def norm(text):
    return normalized(cc.convert(itn.normalize(text).text))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return {k: float(v) for k, v in zip(('mean', 'median', 'p90'),
            (np.mean(values), np.median(values), np.percentile(values, 90)))}


def summary(rows):
    result = {}
    for arm in ARMS:
        values = [d['values'][arm] for d in rows]
        c = Counter()
        for v in values:
            for k in ('hits', 'expected_mentions', 'errors', 'reference_chars',
                      'raw_hits', 'raw_errors', 'refiner_calls', 'refiner_retries',
                      'replay_checks', 'replay_applied_changes', 'dictionary_excess_mentions',
                      'replay_absent_changes',
                      'dictionary_absent_mentions', 'prompt_hits', 'prompt_expected',
                      'outside_prompt_hits', 'outside_prompt_expected'):
                c[k] += v[k]
        c['samples'] = len(rows)
        result[arm] = dict(c)
        result[arm].update(entity_recall=c['hits']/c['expected_mentions'] if c['expected_mentions'] else None,
                           CER=c['errors']/c['reference_chars'],
                           raw_CER=c['raw_errors']/c['reference_chars'],
                           audio_seconds=sum(d['audio_seconds'] for d in rows),
                           total_wall_seconds=sum(v['timing']['total_seconds'] for v in values),
                           replay_sample_count=sum(v['replay_checks'] > 0 for v in values),
                           replay_changed_samples=sum(v['replay_applied_changes'] > 0 for v in values),
                           context_guard_fallback_samples=sum(v['context_guard_fallback'] for v in values),
                           entity_error_samples=sum(v['hits'] < v['expected_mentions'] for v in values),
                           dictionary_absent_samples=sum(v['dictionary_absent_mentions'] > 0 for v in values))
        for key in ('total_seconds', 'session_setup_seconds', 'audio_stream_seconds', 'tail_seconds'):
            result[arm][key] = stats([v['timing'][key] for v in values])
        result[arm]['RTF'] = stats([v['timing']['total_seconds']/d['audio_seconds'] for d,v in zip(rows,values)])
    changes = Counter()
    for d in rows:
        a, b = (d['values'][arm] for arm in ARMS)
        changes['audio_more_entity_hits'] += b['hits'] > a['hits']
        changes['context_more_entity_hits'] += a['hits'] > b['hits']
        changes['audio_lower_CER_samples'] += b['errors'] < a['errors']
        changes['context_lower_CER_samples'] += a['errors'] < b['errors']
        changes['equal_CER_samples'] += a['errors'] == b['errors']
        changes['audio_faster_samples'] += b['timing']['total_seconds'] < a['timing']['total_seconds']
        changes['context_faster_samples'] += a['timing']['total_seconds'] < b['timing']['total_seconds']
    delta=np.array([d['values']['audio']['timing']['total_seconds']-d['values']['context']['timing']['total_seconds'] for d in rows])
    result['paired'] = dict(changes, audio_minus_context_seconds=stats(delta))
    # Bootstrap paired recordings, preserving all words/mentions of each recording.
    if len(rows) >= 2:
        rng=np.random.default_rng(20261013)
        matrix=np.array([[d['values']['audio']['hits']-d['values']['context']['hits'],
                          d['values']['context']['expected_mentions'],
                          d['values']['audio']['errors']-d['values']['context']['errors'],
                          d['values']['context']['reference_chars'], delta[i]] for i,d in enumerate(rows)])
        boots=[]
        for _ in range(2000):
            sample=matrix[rng.integers(0,len(rows),len(rows))];totals=sample.sum(axis=0)
            boots.append([totals[0]/totals[1] if totals[1] else np.nan,totals[2]/totals[3],totals[4]/len(rows)])
        boots=np.asarray(boots)
        intervals=[np.percentile(column[np.isfinite(column)],[2.5,97.5]).tolist()
                   if np.isfinite(column).any() else None for column in boots.T]
        result['paired']['audio_minus_context_95pct_bootstrap_CI'] = dict(zip(('entity_recall','CER','mean_seconds'),intervals))
    return result


def main(partial=False):
    inputs=[json.loads(l) for l in (OUT/'inference_inputs.jsonl').open()]
    sample_by_id={s['id']:s for s in inputs}
    refs=json.loads((OUT/'references.json').read_text())['samples']
    meta=json.loads((OUT/'library_metadata.json').read_text())['libraries']
    protocol=json.loads((OUT/'protocol.json').read_text())
    worker=json.loads((OUT/'worker.json').read_text())
    assert itn.rule_pack==worker['health']['zh_itn_rule_pack']
    assert digest(OUT/'inference_inputs.jsonl')==protocol['input_sha256']
    for key,m in meta.items():assert digest(m['snapshot_path'])==protocol['library_hashes'][key]
    assert set(refs)==set(sample_by_id) and len(sample_by_id)==len(inputs)
    patterns={}
    for key,m in meta.items():
        with sqlite3.connect('file:'+m['snapshot_path']+'?mode=ro',uri=True) as db:
            words={norm(r[0]) for r in db.execute('SELECT canonical_text FROM protected_entities WHERE enabled=1')}
        words.discard('')
        patterns[key]=re.compile('|'.join(map(re.escape,sorted(words,key=lambda w:(-len(w),w)))))
    data={}
    # A live partial audit may observe the last append before its newline.
    for line in (OUT/'results.jsonl').read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            assert partial, 'incomplete final result row'
            break
        row=json.loads(line);key=(row['id'],row['arm']);assert key not in data
        assert row['pcm_sha256']==sample_by_id[row['id']]['pcm_sha256']
        assert row['transport']['frames']==row['transport']['acks'] and row['transport']['max_outstanding']<=4
        assert row['result']['event']=='final' and 'clean_text' in row['result']
        ready=row['ready'];m=meta[row['library_key']]
        assert ready['entity_asr_context_requested']==(row['arm']=='context')
        if row['arm']=='context':
            assert ready['entity_asr_context_applied']==bool(m['context16_terms'])
            assert ready['entity_asr_context_term_count']==len(m['context16_terms'])
            expected_digest=m['context16_sha256'][:16] if m['context16_text'] else None
            assert ready['entity_asr_context_digest']==expected_digest
        data[key]=row
    if not partial:assert len(data)==len(inputs)*2
    ids=[s['id'] for s in inputs if all((s['id'],a) in data for a in ARMS)]
    assert ids
    details=[]
    for sid in ids:
        sample=sample_by_id[sid];label=refs[sid];ref=norm(label['reference'])
        assert ref
        gold=Counter(norm(e['canonical']) for e in label['gold_entities'])
        assert all(w and ref.count(w)>=n for w,n in gold.items()), ('gold absent from reference',sid,gold,ref)
        prompt={norm(w) for w in meta[sample['library_key']]['context16_terms']}
        pattern=patterns[sample['library_key']];refwords=Counter(pattern.findall(ref));values={}
        for arm in ARMS:
            row=data[sid,arm];r=row['result'];text=norm(r['clean_text']);raw=norm(r['raw_text'])
            words=Counter(pattern.findall(text));audio=r.get('entity_audio_repair',{})
            assert r['entity_asr_context_requested']==(arm=='context') and r['entity_audio_repair_requested']==(arm=='audio')
            hits={w:min(n,text.count(w)) for w,n in gold.items()}
            values[arm]={'text':r['clean_text'],'raw_text':r['raw_text'],
                'hits':sum(hits.values()),'expected_mentions':sum(gold.values()),
                'hit_by_entity':hits,
                'errors':Levenshtein.distance(ref,text),'reference_chars':len(ref),
                'raw_hits':sum(min(n,raw.count(w)) for w,n in gold.items()),
                'raw_errors':Levenshtein.distance(ref,raw),
                'prompt_hits':sum(n for w,n in hits.items() if w in prompt),
                'prompt_expected':sum(n for w,n in gold.items() if w in prompt),
                'outside_prompt_hits':sum(n for w,n in hits.items() if w not in prompt),
                'outside_prompt_expected':sum(n for w,n in gold.items() if w not in prompt),
                'dictionary_excess_mentions':sum((words-refwords).values()),
                'dictionary_absent_mentions':sum(n for w,n in words.items() if w not in ref),
                'dictionary_absent_words':{w:n for w,n in words.items() if w not in ref},
                'refiner_calls':r['session_refiner_call_count'],
                'refiner_retries':r['session_refiner_retry_call_count'],
                'replay_checks':audio.get('checks',0),
                'replay_applied_changes':len(audio.get('changes',[])),
                'replay_absent_changes':sum(norm(c['canonical']) not in ref for c in audio.get('changes',[])),
                'replay_absent_change_details':[c for c in audio.get('changes',[]) if norm(c['canonical']) not in ref],
                'replay_reason':audio.get('reason'),
                'context_guard_fallback':bool(r.get('entity_asr_context_guard',{}).get('fallback_active')),
                'timing':row['timing'],'trace_id':row['trace_id']}
        details.append(dict(id=sid,corpus=sample['corpus'],cohort=sample['cohort'],library_key=sample['library_key'],
                            topics=sample['topics'],audio_seconds=sample['duration_seconds'],reference=label['reference'],
                            gold_entities=label['gold_entities'],values=values))
    groups=defaultdict(list)
    for d in details:
        for key in ('all','corpus:'+d['corpus'],'cohort:'+d['cohort'],'library:'+d['library_key']):groups[key].append(d)
    report={key:summary(rows) for key,rows in groups.items()}
    type_scores={}
    for typ in sorted({e['entity_type'] for d in details for e in d['gold_entities']}):
        scores={a:Counter() for a in ARMS}
        for d in details:
            expected=Counter(norm(e['canonical']) for e in d['gold_entities'] if e['entity_type']==typ)
            for a in ARMS:
                text=norm(d['values'][a]['text']);scores[a].update(expected=sum(expected.values()),hits=sum(min(n,text.count(w)) for w,n in expected.items()))
        type_scores[typ]={a:dict(v,recall=v['hits']/v['expected']) for a,v in scores.items()}
    result={'complete':len(ids)==len(inputs),'completed_pairs':len(ids),'planned_pairs':len(inputs),
            'normalization':'same conservative zh-itn on reference and output, OpenCC t2s, NFKC/casefold, strip P/Z/C; same as prior full benchmark',
            'entity_recall_scope':'canonical substring occurrence capped by annotated reference count; not identity verification or general NER precision',
            'dictionary_absent_scope':'longest non-overlapping canonical library matches absent anywhere from complete normalized reference; diagnostic, not manual hallucination labels',
            'timing_scope':protocol['timing_scope'],'groups':report,'entity_types':type_scores}
    for key,m in meta.items():assert digest(m['snapshot_path'])==protocol['library_hashes'][key]
    for name,expected in protocol['source_hashes'].items():assert digest(OUT/'source_snapshot'/name)==expected
    prefix='partial_' if partial else ''
    (OUT/(prefix+'summary.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/(prefix+'comparisons.jsonl')).open('w') as f:
        for d in details:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','completed_pairs','planned_pairs')},ensure_ascii=False))
    print(json.dumps(report['all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args();main(args.partial)
