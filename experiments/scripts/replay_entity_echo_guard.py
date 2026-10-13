"""Compare old and audio-verified responses with the same real Refiner."""
from collections import Counter
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import argparse

import compare_entity_methods as framework
from benchmark_multimed_entities import MODEL,transcript,require_complete,digest
from evaluate_entity_library import ROOT,write_json,normalized
from evaluate_medical_entities import lexicon_occurrences
from rapidfuzz.distance import Levenshtein

OUT=ROOT/'results/entity_echo_guard_20261010'
OLD=ROOT/'results/entity_methods_comparison_20261009/medical'


def prepare():
    prior=json.loads((OLD/'protocol.json').read_text())
    for name,source in [('legacy',OLD/'asr_on.jsonl'),('guarded',OUT/'paired/guarded.jsonl')]:
        folder=OUT/'web'/name;folder.mkdir(parents=True,exist_ok=True);(folder/'service_logs').mkdir(exist_ok=True)
        library=folder/'library';library.mkdir(exist_ok=True)
        for filename in ['entities.db','entities.json','sample_entities.json']:
            if not (library/filename).exists():shutil.copy2(OLD/'library'/filename,library/filename)
        if not (folder/'asr_on.jsonl').exists():shutil.copy2(source,folder/'asr_on.jsonl')
        if not (folder/'protocol.json').exists():
            p=dict(prior)
            p['library_hashes']={f:digest(library/f) for f in ['entities.db','entities.json']}
            p['source_hashes']={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'system').rglob('*.py')}
            p['cohort']='Same 225 audio and 16 terms; only the two verified echo inputs are replaced'
            write_json(folder/'protocol.json',p)


def run():
    prepare();framework.OUT=OUT/'web'
    settings=SimpleNamespace(output=OUT/'web/legacy',library=OUT/'web/legacy/library',model=MODEL,
        device='cuda:0',asr_url=framework.ASR,mode='streaming',chunk_seconds=2.0)
    runner=framework.FactorialRunner(settings)
    original=runner.web.create_app
    def app_factory(*args,**kwargs):
        # Audio protection was actually executed by the guarded decoder.
        # Disable a second audio verification during fixed-response replay.
        kwargs['entity_asr_echo_guard']=False
        return original(*args,**kwargs)
    runner.web.create_app=app_factory
    try:
        for condition in ['legacy','guarded']:runner.run_cell(condition,'combined')
    finally:runner.web.create_app=original


def report():
    from system.zh_itn_bridge import ZhITNNormalizer
    itn=ZhITNNormalizer();cache={}
    def norm(t):
        if t not in cache:cache[t]=normalized(itn.normalize(t).text)
        return cache[t]
    p=json.loads((OLD/'protocol.json').read_text());ids=[s['id'] for s in p['samples']]
    rows={c:require_complete(OUT/'web'/c/'combined.jsonl',ids) for c in ['legacy','guarded']}
    gt=json.loads((OLD/'library/sample_entities.json').read_text());names=[d['canonical_text'] for d in json.loads((OLD/'library/entities.json').read_text())]
    summaries={};details=[]
    for condition,records in rows.items():
        s=Counter()
        for rid,r in records.items():
            text=transcript(r,'clean_text');ref=norm(r['reference']);expected=Counter(e['text'] for e in gt[rid]);have=Counter(e['text'] for e in lexicon_occurrences(text,names))
            s.update(samples=1,reference_chars=len(ref),character_errors=Levenshtein.distance(ref,norm(text)),
                expected_mentions=sum(expected.values()),recovered_mentions=sum(min(c,have[n]) for n,c in expected.items()),
                unreferenced_mentions=sum(max(0,c-expected[n]) for n,c in have.items()),
                instruction_echo_samples=int('参考术语仅用于辅助拼写' in text),refiner_calls=len(r['model_calls']))
        summaries[condition]=dict(s,CER=s['character_errors']/s['reference_chars'])
    for rid in ids:
        a,b=rows['legacy'][rid],rows['guarded'][rid]
        assert a['pcm_sha256']==b['pcm_sha256']
        before,after=transcript(a,'clean_text'),transcript(b,'clean_text')
        if before!=after:details.append(dict(id=rid,reference=a['reference'],before=before,after=after))
    write_json(OUT/'web/summary.json',summaries);write_json(OUT/'web/changes.json',details)
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run();report()
    else:report()
