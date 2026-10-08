"""Prepare an isolated acoustic verifier with strict source/PCM provenance."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

from evaluate_entity_library import ROOT,read_jsonl,write_json,audio_bytes
from evaluate_entity_ranking import OUT,sets
from system.entity_boundaries import _tokens
import validate_entity_verifier as evaluator

AUDIO=OUT/'audio_probe'


def candidates_for(row):
    # Independent lexical boundaries: whole tokens and adjacent tokens only.
    # Never populate the tokenizer with a candidate or gold entity spelling.
    tokens=_tokens(row['source']);spans=set()
    for i,(_,a,b) in enumerate(tokens):
        for j in range(i,min(len(tokens),i+3)):
            end=tokens[j][2]
            if 2<=end-a<=16:spans.add((a,end))
    return [c for c in row['candidates'] if (c['start'],c['end']) in spans and
            c['boundary_issue'] in {None,'short_chinese_span_requires_alias'} and
            len(c['observed'])==len(c['canonical'])]


def prepare(stage):
    rng=random.Random(20261006);samples=[];selection=[]
    for split in (['news_dev','medical_dev'] if stage=='dev' else ['news_test','clinical_test','mmed_test']):
        path=OUT/f'recall_{split}.jsonl'
        if not path.exists():path=OUT/'round2'/path.name
        rows=read_jsonl(path);originals={r['id'] for r in rows}
        if stage=='dev':
            # All audited errors are calibration positives, plus a seeded
            # negative subset. This is not a random-deployment estimate.
            positive={r['id'] for r in rows if any(t['category']=='text_correctable_candidate' for t in r['targets'])}
            pool=sorted(originals-positive);rng.shuffle(pool)
            chosen=positive|set(pool[:24]);rows=[r for r in rows if r['id'] in chosen]
        source,_,_=sets()[split];lookup={(r['id'],r['variant']):r for r in read_jsonl(source)}
        for row in rows:
            original=lookup[row['id'],row['variant']]
            snr=original.get('noise_snr_db',10.0)
            if snr is None:snr=10.0
            seed=20261006 if split in {'news_test','mmed_test'} else 20261003
            pcm,_=audio_bytes(original,'clean' if row['variant']=='clean' else 'noise',SimpleNamespace(seed=seed,noise_snr=snr))
            if hashlib.sha256(pcm).hexdigest()!=original['pcm_sha256']:raise ValueError('PCM provenance mismatch: '+original['id']+':'+original['variant'])
            pcm_path=AUDIO/'pcm'/f"{row['id']}_{row['variant']}.pcm";pcm_path.parent.mkdir(parents=True,exist_ok=True);pcm_path.write_bytes(pcm)
            item=dict(id=row['id'],variant=row['variant'],split=split,source=row['source'],candidates=candidates_for(row),
                      pcm_path=str(pcm_path),pcm_sha256=original['pcm_sha256'])
            samples.append(item);selection.append(dict(row,candidates=item['candidates'],split=split))
    write_json(AUDIO/f'{stage}_manifest.json',dict(seed=20261006,stage=stage,samples=samples,
             candidate_policy='Independent tokenizer boundaries; 1-3 adjacent tokens; equal-length CJK registered normalized candidates; all scores, no reference in verifier',
             selection='Calibration: all audited errors plus 24 seeded negative recordings/domain' if stage=='dev' else 'All existing holdout versions; no error-based selection'))
    write_json(AUDIO/f'{stage}_evaluation_rows.json',selection)
    print(Counter(r['split'] for r in samples),'candidates',sum(len(r['candidates']) for r in samples),flush=True)


def convert(stage):
    records=read_jsonl(AUDIO/f'{stage}_scores.jsonl');scored={(r['id'],r['variant']):r for r in records}
    bysplit={}
    for row in json.loads((AUDIO/f'{stage}_evaluation_rows.json').read_text()):
        r=scored.get((row['id'],row['variant']))
        if r is None:raise ValueError('Audio score missing')
        if [c['candidate'] for c in r['candidates']]!=row['candidates']:raise ValueError('Candidate provenance changed')
        output=dict(row,scores=[c['score'] for c in r['candidates']],verifier_ms=r['elapsed_seconds']*1000)
        bysplit.setdefault(row['split'],[]).append(output)
    for split,rows in bysplit.items():
        (AUDIO/f'scored_{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def calibrate():
    evaluator.OUT=AUDIO;trials=[];chosen={}
    for domain,split in [('general','news_dev'),('medical','medical_dev')]:
        for threshold in [.75,.85,.90,.95,.97,.98,.99,.995,.997,.999,.9995]:
            report=evaluator.evaluate(split,threshold,.25,save=False)
            trial=dict(domain=domain,threshold=threshold,counts=dict(report['counts']));trials.append(trial)
        safe=[t for t in trials if t['domain']==domain and t['counts']['regressed_samples']==0 and
              t['counts']['applied_edits']>0 and t['counts']['applied_edits']==t['counts']['exact_aligned_correct_edits']]
        chosen[domain]=max(safe,key=lambda t:t['counts']['additional_repaired']) if safe else None
    write_json(AUDIO/'policy.json',dict(trials=trials,chosen=chosen,
         note='Sigmoid likelihood ratio is a ranking score, not a calibrated correctness probability; require independent heldout validation'))
    print(chosen,flush=True)


def evaluate_test():
    evaluator.OUT=AUDIO;policy=json.loads((AUDIO/'policy.json').read_text())['chosen']
    for split in ['news_test','clinical_test','mmed_test']:
        domain=sets()[split][2];selected=policy[domain]
        # A domain with no safe development edits stays disabled.
        threshold=selected['threshold'] if selected else 1.0
        report=evaluator.evaluate(split,threshold,enabled=selected is not None);print(split,dict(report['counts']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare-dev','prepare-test','convert-dev','convert-test','calibrate','evaluate-test']);a=p.parse_args()
    if a.stage.startswith('prepare-'):prepare(a.stage.split('-')[1])
    elif a.stage.startswith('convert-'):convert(a.stage.split('-')[1])
    elif a.stage=='calibrate':calibrate()
    else:evaluate_test()
