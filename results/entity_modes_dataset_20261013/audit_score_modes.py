"""Independent offline cross-check of score_modes.py against a saved result cohort.

This script reads frozen inputs/references, the saved partial comparisons/summary,
result rows, and SQLite snapshots only. It does not import or invoke score_modes.py.
Default mode audits the saved partial cohort; --full audits final summary/comparisons.
Run with the experiment's ASR environment because OpenCC and the zh-ITN bridge are
needed to reproduce the documented text normalization; this performs no inference.
"""
from collections import Counter
import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import sys
import unicodedata
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import opencc
from system.zh_itn_bridge import ZhITNNormalizer

CONVERTER=opencc.OpenCC('t2s')
ITN=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
ARMS=('context','audio')

def normalize(text):
    text=CONVERTER.convert(ITN.normalize(text).text)
    text=unicodedata.normalize('NFKC',text).casefold()
    return ''.join(c for c in text if unicodedata.category(c)[0] not in {'P','Z','C'})

def edit_distance(a,b):
    # Two-row Wagner-Fischer implementation, intentionally independent of RapidFuzz.
    if len(a)<len(b): a,b=b,a
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]

def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare(a,b,tol=1e-12):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return abs(float(a)-float(b))<=tol
    return a==b

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--full',action='store_true',help='audit summary.json/comparisons.jsonl instead of the saved partial snapshot')
args=parser.parse_args()
scope='full' if args.full else 'partial'
summary_path=OUT/('summary.json' if args.full else 'partial_summary.json')
details_path=OUT/('comparisons.jsonl' if args.full else 'partial_comparisons.jsonl')

inputs=[json.loads(line) for line in (OUT/'inference_inputs.jsonl').read_text(encoding='utf-8').splitlines() if line]
sample_by_id={x['id']:x for x in inputs}
refs=load_json(OUT/'references.json')['samples']
meta=load_json(OUT/'library_metadata.json')['libraries']
saved_summary=load_json(summary_path)
details_bytes=details_path.read_bytes()
saved_details=[json.loads(line) for line in details_bytes.decode('utf-8').splitlines() if line]
ids=[x['id'] for x in saved_details]
assert len(ids)==len(set(ids))==saved_summary['completed_pairs']
assert set(ids)<=set(sample_by_id)==set(refs)
if args.full:
    assert saved_summary['complete'] and saved_summary['planned_pairs']==len(inputs)
    assert len(ids)==len(inputs)
else:
    assert not saved_summary['complete'] or len(ids)==len(inputs)
id_set_sha256=hashlib.sha256('\n'.join(sorted(ids)).encode('utf-8')).hexdigest()

results={}
for line in (OUT/'results.jsonl').read_text(encoding='utf-8').splitlines():
    if not line: continue
    row=json.loads(line)
    if row['id'] not in set(ids): continue
    key=(row['id'],row['arm'])
    assert key not in results, f'duplicate result row: {key}'
    assert row['pcm_sha256']==sample_by_id[row['id']]['pcm_sha256']
    results[key]=row
assert set(results)=={(sid,arm) for sid in ids for arm in ARMS}
selected_rows='\n'.join(json.dumps(results[key],ensure_ascii=False,sort_keys=True,separators=(',',':')) for key in sorted(results))
selected_rows_sha256=hashlib.sha256(selected_rows.encode('utf-8')).hexdigest()

patterns={}
for key,entry in meta.items():
    conn=sqlite3.connect(f"file:{entry['snapshot_path']}?mode=ro",uri=True)
    try:
        words={normalize(row[0]) for row in conn.execute('SELECT canonical_text FROM protected_entities WHERE enabled=1')}
    finally:
        conn.close()
    words.discard('')
    patterns[key]=re.compile('|'.join(re.escape(w) for w in sorted(words,key=lambda w:(-len(w),w))))

per_id={}
for sid in ids:
    sample=sample_by_id[sid]; label=refs[sid]
    ref=normalize(label['reference'])
    assert ref
    gold=Counter(normalize(e['canonical']) for e in label['gold_entities'])
    assert all(w and ref.count(w)>=n for w,n in gold.items()), ('reference/gold inconsistency',sid)
    prompt={normalize(w) for w in meta[sample['library_key']]['context16_terms']}
    pattern=patterns[sample['library_key']]
    refwords=Counter(pattern.findall(ref))
    arm_values={}
    for arm in ARMS:
        row=results[sid,arm]; r=row['result']; assert r['event']=='final'
        text=normalize(r['clean_text']); raw=normalize(r['raw_text'])
        words=Counter(pattern.findall(text)); audio=r.get('entity_audio_repair') or {}
        assert r['entity_asr_context_requested']==(arm=='context')
        assert r['entity_audio_repair_requested']==(arm=='audio')
        hit_by={w:min(n,text.count(w)) for w,n in gold.items()}
        raw_hit_by={w:min(n,raw.count(w)) for w,n in gold.items()}
        replay_absent_change_details=[c for c in audio.get('changes',[]) if normalize(c['canonical']) not in ref]
        arm_values[arm]={
            'hits':sum(hit_by.values()),'expected_mentions':sum(gold.values()),
            'errors':edit_distance(ref,text),'reference_chars':len(ref),
            'raw_hits':sum(raw_hit_by.values()),'raw_errors':edit_distance(ref,raw),
            'refiner_calls':r['session_refiner_call_count'],
            'refiner_retries':r['session_refiner_retry_call_count'],
            'replay_checks':audio.get('checks',0),
            'replay_applied_changes':len(audio.get('changes',[])),
            'replay_absent_changes':len(replay_absent_change_details),
            'replay_absent_change_details':replay_absent_change_details,
            'dictionary_excess_mentions':sum((words-refwords).values()),
            'dictionary_absent_mentions':sum(n for w,n in words.items() if w not in ref),
            'prompt_hits':sum(n for w,n in hit_by.items() if w in prompt),
            'prompt_expected':sum(n for w,n in gold.items() if w in prompt),
            'outside_prompt_hits':sum(n for w,n in hit_by.items() if w not in prompt),
            'outside_prompt_expected':sum(n for w,n in gold.items() if w not in prompt),
            'timing':row['timing'],
        }
    per_id[sid]=arm_values

# Compare per-sample core values to the scorer's saved comparison rows.
detail_mismatches=[]
for d in saved_details:
    sid=d['id']
    for arm in ARMS:
        actual=per_id[sid][arm]; saved=d['values'][arm]
        for field in actual:
            if field=='timing':
                for tkey,tval in actual[field].items():
                    if not compare(tval,saved[field][tkey]):
                        detail_mismatches.append((sid,arm,'timing.'+tkey,tval,saved[field][tkey]))
            elif not compare(actual[field],saved.get(field)):
                detail_mismatches.append((sid,arm,field,actual[field],saved.get(field)))

# Independently aggregate the specific metrics requested and compare with partial summary.
all_saved=saved_summary['groups']['all']
summary_mismatches=[]
independent={}
for arm in ARMS:
    vals=[per_id[sid][arm] for sid in ids]
    expected={k:sum(v[k] for v in vals) for k in ('hits','expected_mentions','errors','reference_chars','refiner_calls','refiner_retries')}
    times=[v['timing']['total_seconds'] for v in vals]
    expected.update(mean_total_seconds=statistics.mean(times),median_total_seconds=statistics.median(times),
                    CER=expected['errors']/expected['reference_chars'],entity_recall=expected['hits']/expected['expected_mentions'])
    for saved_key,calc in [('hits',expected['hits']),('expected_mentions',expected['expected_mentions']),('errors',expected['errors']),('reference_chars',expected['reference_chars']),('refiner_calls',expected['refiner_calls']),('refiner_retries',expected['refiner_retries']),('CER',expected['CER']),('entity_recall',expected['entity_recall']),('total_seconds.mean',expected['mean_total_seconds']),('total_seconds.median',expected['median_total_seconds'])]:
        if saved_key.startswith('total_seconds.'):
            observed=all_saved[arm]['total_seconds'][saved_key.split('.',1)[1]]
        else: observed=all_saved[arm].get(saved_key)
        if not compare(calc,observed): summary_mismatches.append((arm,saved_key,calc,observed))
    # The distinct-dictionary-missing metric counts terms absent anywhere in reference;
    # the excess metric also catches over-counts of terms that do appear.
    expected['dictionary_absent_mentions']=sum(v['dictionary_absent_mentions'] for v in vals)
    expected['dictionary_excess_mentions']=sum(v['dictionary_excess_mentions'] for v in vals)
    expected['replay_absent_changes']=sum(v['replay_absent_changes'] for v in vals)
    for k in ('dictionary_absent_mentions','dictionary_excess_mentions','replay_absent_changes'):
        if expected[k]!=all_saved[arm][k]:summary_mismatches.append((arm,k,expected[k],all_saved[arm][k]))
    independent[arm]=expected

report={
    'scope':scope,
    'summary_completed_pairs':saved_summary['completed_pairs'],
    'audited_id_count':len(ids),
    'id_set_sha256':id_set_sha256,
    'comparison_snapshot_sha256':hashlib.sha256(details_bytes).hexdigest(),
    'summary_snapshot_sha256':file_digest(summary_path),
    'selected_result_rows_sha256':selected_rows_sha256,
    'available_current_result_rows_for_these_ids':len(results),
    'independent_metrics':independent,
    'independent_replay_absent_change_details':[
        {'id':sid,'observed':change.get('observed'),'canonical':change.get('canonical'),'reference':refs[sid]['reference']}
        for sid in ids for change in per_id[sid]['audio']['replay_absent_change_details']
    ],
    'per_sample_mismatches':detail_mismatches,
    'summary_mismatches':summary_mismatches,
    'status':'PASS' if not detail_mismatches and not summary_mismatches else 'MISMATCH',
    'notes':[
        f'The audit ID set is exactly {details_path.name}, not all rows currently appended to results.jsonl.',
        'Gold multiplicity is recomputed with Counter(canonical) and each per-string hit is capped at that multiplicity.',
        'Dictionary absence is recomputed against the full normalized reference: a recognized library canonical is absent only when that string occurs nowhere in the reference. Excess mentions separately counts output-library matches beyond reference-library match multiplicity.',
        'CER uses an independent two-row edit-distance implementation after the documented zh-ITN, OpenCC t2s, NFKC/casefold, and P/Z/C removal pipeline.',
    ],
}
output_path=OUT/f'audit_score_modes_report_{scope}_{len(ids)}_{id_set_sha256[:12]}.json'
report['report_path']=str(output_path)
output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ('summary_completed_pairs','audited_id_count','available_current_result_rows_for_these_ids','independent_metrics','status','per_sample_mismatches','summary_mismatches')},ensure_ascii=False,indent=2))
print(json.dumps({k:report[k] for k in ('scope','id_set_sha256','comparison_snapshot_sha256','summary_snapshot_sha256','selected_result_rows_sha256','report_path')},ensure_ascii=False,indent=2))
if report['status']!='PASS': raise SystemExit(1)
