"""Independent full raw-ASR scoring audit, without importing its scorer."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics
import sys
import unicodedata

import opencc

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(OUT.parent/'source_snapshot'))
from system.zh_itn_bridge import ZhITNNormalizer
cc=opencc.OpenCC('t2s')
itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')

@lru_cache(maxsize=20000)
def norm(s):
    s=unicodedata.normalize('NFKC',cc.convert(itn.normalize(s).text)).casefold()
    return ''.join(c for c in s if unicodedata.category(c)[0] not in {'P','Z','C'})

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def main():
    rows=[json.loads(l) for l in (OUT/'results.jsonl').open()]
    refs=json.loads((OUT.parent/'references.json').read_text())['samples']
    saved=json.loads((OUT/'summary.json').read_text())
    assert saved['complete'] and len(rows)==len({r['id'] for r in rows})==2506
    assert {r['id'] for r in rows}==set(refs)
    total=Counter();mismatch=[];bytype={};times=[]
    details={r['id']:r for r in [json.loads(l) for l in (OUT/'paired_details.jsonl').open()]}
    for r in rows:
        label=refs[r['id']];ref=norm(label['reference']);raw=r['result']['asr_payload']['text']
        assert raw==r['result']['raw_text']==r['result']['clean_text']
        assert not r['ready']['entity_asr_context_applied'] and not r['ready']['postprocessing_enabled']
        gold=Counter(norm(e['canonical']) for e in label['gold_entities']);text=norm(raw)
        v=dict(hits=sum(min(n,text.count(w)) for w,n in gold.items()),expected_mentions=sum(gold.values()),
               errors=distance(ref,text),reference_chars=len(ref))
        for key,value in v.items():
            if details[r['id']]['values']['pure_asr_raw'][key]!=value:mismatch.append((r['id'],key))
        total.update(v);times.append(r['timing']['total_seconds'])
        for typ in {e['entity_type'] for e in label['gold_entities']}:
            target=Counter(norm(e['canonical']) for e in label['gold_entities'] if e['entity_type']==typ)
            bytype.setdefault(typ,Counter()).update(hits=sum(min(n,text.count(w)) for w,n in target.items()),expected_mentions=sum(target.values()))
    a=saved['groups']['all']['metrics']['pure_asr_raw']
    assert not mismatch and all(a[k]==v for k,v in total.items())
    for typ,t in bytype.items():
        b=saved['entity_types'][typ]['pure_asr_raw'];assert all(b[k]==v for k,v in t.items())
    timing=saved['groups']['all']['run_timing']['pure_asr']['total_seconds']
    assert abs(timing['mean']-statistics.mean(times))<1e-12
    assert abs(timing['median']-statistics.median(times))<1e-12
    record=dict(status='PASS',samples=2506,per_id_mismatches=mismatch,metrics=dict(total),
                entity_types={k:dict(v) for k,v in bytype.items()},mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),
                summary_sha256=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),
                results_sha256=hashlib.sha256((OUT/'results.jsonl').read_bytes()).hexdigest(),
                method='Independent Unicode normalizer and two-row edit distance; no inference or scorer import')
    (OUT/'audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
