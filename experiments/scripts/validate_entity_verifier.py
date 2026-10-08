"""Freeze an entity-only verifier using calibration, then test real ASR.

Scoring never receives the reference. Metrics retain all reference-alignment
uncertainties and record every edit for inspection, including regressions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,normalized
from evaluate_entity_ranking import OUT,OLD,sets
from system.entity_ranker import EntityPairRanker,apply_verified_choices
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.protection import EntityProtector

MODEL=Path(os.environ.get('ENTITY_VERIFIER_MODEL',str(ROOT/'models/entity_ranker/verifier'))).resolve()


def select():
    choices=[];domain_choices=[]
    for path in sorted(OUT.glob('calibration_epoch*.json')):
        rows=json.loads(path.read_text());epoch=int(path.stem.split('epoch')[1])
        # Scores need not be calibrated probabilities. Choose acceptance by
        # measured calibration precision, grouping identical scores together.
        ordered=sorted(rows,key=lambda r:-r['score']);tp=fp=0;best=None
        for index,row in enumerate(ordered):
            tp+=row['label'];fp+=1-row['label']
            if index+1<len(ordered) and ordered[index+1]['score']==row['score']:continue
            if tp>=100 and tp/(tp+fp)>=.995:best=row['score']
        thresholds={.90,.95,.97,.98,.99,.995,.997,.999,.9995}
        if best is not None:thresholds.add(best)
        for threshold in sorted(thresholds):
            accepted=[r for r in rows if r['score']>=threshold]
            tp=sum(r['label'] for r in accepted);fp=len(accepted)-tp
            choices.append(dict(epoch=epoch,threshold=threshold,true_positive=tp,false_positive=fp,
                                precision=tp/len(accepted) if accepted else None,
                                real_positive=sum(r['label'] for r in accepted if r['kind']=='real_asr')))
        policies={}
        for domain in sorted({r['domain'] for r in rows}):
            ranked=sorted([r for r in rows if r['domain']==domain],key=lambda r:-r['score'])
            tp=fp=0;best_domain=None
            for index,row in enumerate(ranked):
                tp+=row['label'];fp+=1-row['label']
                if index+1<len(ranked) and ranked[index+1]['score']==row['score']:continue
                if tp>=100 and tp/(tp+fp)>=.995:
                    best_domain=dict(threshold=row['score'],true_positive=tp,false_positive=fp,precision=tp/(tp+fp))
            policies[domain]=best_domain
        if all(policies.values()):domain_choices.append(dict(epoch=epoch,domain_policies=policies,
                     true_positive=sum(p['true_positive'] for p in policies.values())))
    safe=[c for c in choices if c['true_positive']>=100 and c['precision']>=.995]
    write_json(OUT/'checkpoint_selection.json',dict(choices=choices,domain_choices=domain_choices,
               rule='Maximize accepted positives with >=99.5% pair precision and >=100 positives separately in EACH domain; calibration only'))
    if not domain_choices:raise ValueError('No calibration checkpoint qualifies in both domains')
    best=max(domain_choices,key=lambda c:(c['true_positive'],-c['epoch']))
    best['threshold']=min(p['threshold'] for p in best['domain_policies'].values())
    shutil.copyfile(MODEL/f"verifier_epoch{best['epoch']}.pt",MODEL/'verifier.pt')
    metadata=json.loads((MODEL/'metadata.json').read_text());metadata['selection']=best
    metadata['minimum_margin']=.25;write_json(MODEL/'metadata.json',metadata)
    print(best,flush=True)


def score(args):
    import torch
    torch.set_num_threads(4)
    ranker=EntityPairRanker(MODEL,device=args.device)
    destination=OUT/f'scored_{args.split}.jsonl'
    done={(r['id'],r['variant']) for r in read_jsonl(destination)}
    for i,row in enumerate(read_jsonl(OUT/f'recall_{args.split}.jsonl')):
        if (row['id'],row['variant']) in done:continue
        start=time.perf_counter();probs=ranker.scores(row['source'],row['candidates'])
        append_jsonl(destination,dict(row,scores=probs,verifier_ms=1000*(time.perf_counter()-start),
            checkpoint=ranker.metadata['selection']['epoch']))
        if i%40==0:print(args.split,i+1,len(probs),flush=True)


def baseline_matcher(split):
    _,db,domain=sets()[split]
    if domain=='general':db=OLD/'context_names.db'
    definitions=EntityStore(db).list_entities(domain=domain)
    return EntityCandidateMatcher(definitions,selected_domain=domain,mode='auto'),EntityProtector(definitions)


def evaluate(split,threshold,margin=.25,save=True,enabled=True):
    matcher,protector=baseline_matcher(split);counts=Counter();edits=[];details=[]
    changed_recordings=set();changed_references=set();changed_speakers=set()
    for row in read_jsonl(OUT/f'scored_{split}.jsonl'):
        source=row['source'];ref=normalized(row['reference'])
        old=matcher.match(source,allow_auto=True,blocked_spans=[(s.start,s.end) for s in protector.protect(source).spans])
        old_spans=[(m.start,m.end) for m in old.auto_matches]
        baseline=source
        for m in sorted(old.auto_matches,key=lambda m:m.start,reverse=True):baseline=baseline[:m.start]+m.canonical+baseline[m.end:]
        available=[(c,s) for c,s in zip(row['candidates'],row['scores']) if enabled and not any(c['start']<b and a<c['end'] for a,b in old_spans)]
        candidates=[c for c,s in available];scores=[s for c,s in available]
        _,applied=apply_verified_choices(source,candidates,scores,threshold=threshold,minimum_margin=margin)
        output=source
        replacements=[dict(start=m.start,end=m.end,canonical=m.canonical) for m in old.auto_matches]+applied
        for c in sorted(replacements,key=lambda c:c['start'],reverse=True):output=output[:c['start']]+c['canonical']+output[c['end']:]
        before=Levenshtein.distance(ref,normalized(baseline));after=Levenshtein.distance(ref,normalized(output))
        counts['samples']+=1;counts['reference_characters']+=len(ref);counts['baseline_character_errors']+=before;counts['verifier_character_errors']+=after
        counts['improved_samples']+=after<before;counts['regressed_samples']+=after>before
        counts['applied_edits']+=len(applied)
        if applied:
            changed_recordings.add(row['id']);changed_references.add(ref);changed_speakers.add(row['speaker'])
        for c in applied:
            exact=any(t['canonical']==c['canonical'] and t['start']==c['start'] and t['end']==c['end'] for t in row['targets'])
            counts['exact_aligned_correct_edits']+=exact
            edits.append(dict(id=row['id'],variant=row['variant'],source=source,reference=row['reference'],output=output,
                              before_errors=before,after_errors=after,exact_aligned_target=exact,**c))
        for t in row['targets']:
            if t['category']=='text_correctable_candidate':
                counts['eligible_occurrences']+=1
                counts['baseline_repaired']+=any(m.canonical==t['canonical'] and m.start==t['start'] and m.end==t['end'] for m in old.auto_matches)
                counts['additional_repaired']+=any(c['canonical']==t['canonical'] and c['start']==t['start'] and c['end']==t['end'] for c in applied)
        if applied:details.append(dict(id=row['id'],variant=row['variant'],baseline=baseline,output=output))
    counts['changed_original_recordings']=len(changed_recordings)
    counts['changed_distinct_reference_sentences']=len(changed_references)
    counts['changed_speakers']=len(changed_speakers)
    report=dict(split=split,threshold=threshold,minimum_margin=margin,enabled=enabled,counts=counts,edits=edits,
                note='Exact text-aligned edits require audio review in nonverbatim medical captions. No Refiner was called or changed.')
    if save:write_json(OUT/f'verifier_{split}_evaluation.json',report)
    return report


def calibrate_dev():
    metadata=json.loads((MODEL/'metadata.json').read_text());trials=[];chosen={}
    for domain,split in [('general','news_dev'),('medical','medical_dev')]:
        threshold=metadata['selection']['domain_policies'][domain]['threshold'];domain_trials=[]
        for value in sorted({threshold,.95,.97,.98,.99,.995,.997,.999,.9995}):
            if value<threshold:continue
            counts=evaluate(split,value,save=False)['counts']
            trial=dict(domain=domain,threshold=value,counts=dict(counts));trials.append(trial);domain_trials.append(trial)
        qualified=[t for t in domain_trials if t['counts']['regressed_samples']==0 and t['counts']['applied_edits']>0 and
                   t['counts']['exact_aligned_correct_edits']==t['counts']['applied_edits']]
        chosen[domain]=max(qualified,key=lambda t:t['counts']['additional_repaired']) if qualified else None
    write_json(OUT/'development_policy.json',dict(trials=trials,chosen=chosen))
    metadata['development_policy']=chosen;write_json(MODEL/'metadata.json',metadata)
    print('Development policy',chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','score','calibrate','evaluate']);p.add_argument('--split',choices=list(sets()),default='news_dev');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage=='select':select()
    elif a.stage=='score':score(a)
    elif a.stage=='calibrate':calibrate_dev()
    else:
        metadata=json.loads((MODEL/'metadata.json').read_text());domain=sets()[a.split][2]
        policies=metadata.get('development_policy') or {};policy=policies.get(domain) if 'domain_policies' in metadata['selection'] else policies
        fallback=metadata['selection'].get('domain_policies',{}).get(domain,{}).get('threshold',metadata['selection']['threshold'])
        threshold=policy['threshold'] if policy else fallback
        report=evaluate(a.split,threshold);print(dict(report['counts']),flush=True)
