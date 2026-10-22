"""Acoustic CTC hypotheses plus fixed Qwen primary transcript quality replay.

Reference labels are used only in the explicitly separate tune/report stages.
Fusion requires both lexical retrieval and localized CTC acoustic evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from system.entity_audio_repair import GenericEntitySelector,_short_expansion_supported,verify_entity_audio_changes
from system.entity_store import EntityDefinition


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []


def definitions(path):
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        return tuple(EntityDefinition(r['id'],r['canonical_text'],r['entity_type'],r['domain'],
            r['normalization_policy'],r['priority'],()) for r in db.execute(
                'SELECT * FROM protected_entities WHERE enabled=1 ORDER BY id'))


@lru_cache(maxsize=128)
def selector(defs):return GenericEntitySelector(defs)


def decode_tokens(tokens,ids):
    data=bytearray()
    for index in ids:
        token=tokens[index]
        if token.startswith('<0x') and token.endswith('>'):
            data.append(int(token[3:-1],16))
        else:data.extend(token.replace('▁',' ').encode())
    return data.decode('utf-8',errors='replace').strip()


def fuse(text,ctc,defs,tokens,minimum_gap=-1.,minimum_margin=.04,available=None,final=True):
    """Keep Qwen text except bounded spans acoustically supported by the CTC graph."""
    if available is None:available=ctc['audio_seconds']
    ids=ctc['ctc_tokens'];frames=ctc['ctc_token_frames']
    hypotheses=ctc['hypotheses']
    if not final:
        visible=[e for e in ctc['events'] if not e['after_input_finished']
                 and e['audio_seconds']<=available+1e-6]
        if not visible:return text,[]
        ids=visible[-1]['ctc_tokens'];frames=visible[-1]['ctc_frame_times']
        # Do not use a hypothesis first obtained from the end-of-input flush
        # when replaying the last ordinary Qwen partial at the same audio time.
        hypotheses=[h for event in visible for h in event['hypotheses']]
    evidence=defaultdict(list)
    for hyp in hypotheses:
        if hyp['acoustic_gap_per_token']<minimum_gap or hyp['available_audio_seconds']>available+1e-6:continue
        if hyp['end_seconds']>available+.1:continue
        evidence[hyp['word']].append(hyp)
    if not evidence:return text,[]
    eligible=tuple(d for d in defs if d.canonical_text in evidence)
    if not eligible:return text,[]
    candidates,_=selector(eligible).select(text)
    candidates=[c for c in candidates if c['retrieval_score']>=.63 and _short_expansion_supported(c)]
    accepted=[]
    for c in candidates:
        confirmations=[]
        for hyp in evidence[c['canonical']]:
            # Construct a local CTC transcript from that exact acoustic frame
            # interval. Generic anchors must agree with the original Qwen text.
            indices=[i for i,f in enumerate(frames) if hyp['start_frame']<=f<=hyp['end_frame']]
            if not indices:continue
            lo,hi=indices[0],indices[-1]+1
            acoustic_text=decode_tokens(tokens,ids[:lo])+c['canonical']+decode_tokens(tokens,ids[hi:])
            _,changes=verify_entity_audio_changes(text,acoustic_text,[c])
            if changes:confirmations.append(hyp)
        if confirmations:
            best=max(confirmations,key=lambda h:h['acoustic_gap_per_token'])
            accepted.append(dict(c,ctc_evidence=best,
                fused_score=c['retrieval_score']+.1*best['acoustic_gap_per_token']))
    chosen=[]
    for c in sorted(accepted,key=lambda c:-c['fused_score']):
        rivals=[x for x in accepted if x['canonical']!=c['canonical'] and
                (x['start'],x['end'])==(c['start'],c['end'])]
        if rivals and c['fused_score']-max(x['fused_score'] for x in rivals)<minimum_margin:continue
        if any(c['start']<x['end'] and x['start']<c['end'] for x in chosen):continue
        chosen.append(c)
    corrected=text
    for c in sorted(chosen,key=lambda c:-c['start']):
        corrected=corrected[:c['start']]+c['canonical']+corrected[c['end']:]
    return corrected,chosen


def token_table(out):
    path=out/'models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30/tokens.txt'
    table={int(s.rsplit(maxsplit=1)[1]):s.rsplit(maxsplit=1)[0] for s in path.read_text().splitlines()}
    return [table[i] for i in range(2000)]


def collect(out,split,minimum_gap,margin,limit=None):
    inputs=[s for s in rows(out/'inputs.jsonl') if split=='all' or s['split']==split]
    if limit:inputs=inputs[:limit]
    ctc={r['id']:r for r in rows(out/'ctc_results.jsonl')}
    qwen={r['id']:r for r in rows(out/'qwen_primary_streams.jsonl')}
    tokens=token_table(out);defs={};dest=out/'hybrid_results.jsonl'
    done={(r['id'],r['minimum_gap'],r['minimum_margin']) for r in rows(dest)}
    count=0
    for s in inputs:
        key=s['id'],minimum_gap,margin
        if key in done:continue
        if s['id'] not in ctc or s['id'] not in qwen:continue
        if s['library_path'] not in defs:defs[s['library_path']]=definitions(s['library_path'])
        d=defs[s['library_path']];net=ctc[s['id']];raw=qwen[s['id']];events=[];audio=0.;before=time.perf_counter()
        for chunk in raw['chunk_payloads']:
            audio+=chunk['data_bytes']/64000
            text=chunk['payload'].get('text','')
            # Replaying actual Qwen events preserves when the text was first
            # available. CTC evidence is limited by the same received audio.
            corrected,changes=fuse(text,net,d,tokens,minimum_gap,margin,available=audio,final=False)
            events.append(dict(audio_seconds=audio,text=corrected,changes=changes))
        original=raw['final_payload'].get('text','')
        corrected,changes=fuse(original,net,d,tokens,minimum_gap,margin)
        record=dict(id=s['id'],split=s['split'],domain=s['domain'],pcm_sha256=s['pcm_sha256'],
            audio_seconds=s['duration_seconds'],minimum_gap=minimum_gap,minimum_margin=margin,
            primary_text=original,text=corrected,changes=changes,events=events,
            fusion_compute_seconds=time.perf_counter()-before,ctc_compute_seconds=net['compute_seconds'],
            ctc_finish_seconds=net['finish_seconds'],timing_scope='Cached actual Qwen primary + actual incremental CTC quality replay; not concurrent latency',
            qwen_stream_source=raw['source'])
        with dest.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        count+=1
    print('FUSED',split,count,'gap',minimum_gap,'margin',margin,flush=True)


def tune(out):
    # All selection labels are confined to predeclared source-disjoint dev.
    from score_current_entity_challenge import add_text_score
    refs=json.loads((out/'references.json').read_text())['samples']
    qwen={r['id']:r for r in rows(out/'qwen_primary_streams.jsonl')}
    dev={s['id'] for s in rows(out/'inputs.jsonl') if s['split']=='dev'}
    native=[r for r in rows(out/'native_results.jsonl') if r['id'] in dev]
    hybrid=[r for r in rows(out/'hybrid_results.jsonl') if r['id'] in dev]
    def metric(data,textfn):
        totals=Counter()
        ids=set()
        for r in data:
            rid=r['id'];assert rid not in ids;ids.add(rid)
            label=refs[rid];v=add_text_score(label['reference'],textfn(r),label['gold_entities'])
            totals.update(samples=1,hits=v['entity']['hits'],mentions=v['entity']['expected'],
                          errors=v['cer_errors'],chars=v['cer_reference_chars'])
        return dict(totals,recall=totals['hits']/totals['mentions'] if totals['mentions'] else 0.,
                    CER=totals['errors']/totals['chars'] if totals['chars'] else 0.,complete=ids==dev)
    base=metric([dict(id=rid) for rid in dev],lambda r:qwen[r['id']]['final_payload']['text'])
    groups=defaultdict(list)
    for r in native:groups[(r['condition'],r.get('score'))].append(r)
    native_scores={str(k):metric(v,lambda r:r['final']['text']) for k,v in groups.items()}
    beam=native_scores.get(str(('beam',None)))
    hot=[(k,metric(v,lambda r:r['final']['text'])) for k,v in groups.items() if k[0]=='hotword']
    hot=[(k,v) for k,v in hot if v['complete']]
    if not beam or not beam['complete'] or not hot:raise ValueError('Native dev not complete')
    # Preregistered rule: maximize recall while not increasing dev CER over
    # unbiased beam. If none qualifies choose the lowest-CER setting explicitly.
    safe=[x for x in hot if x[1]['errors']<=beam['errors']]
    native_best=max(safe,key=lambda x:(x[1]['hits'],-x[1]['errors'],-x[0][1])) if safe else min(hot,key=lambda x:(x[1]['errors'],-x[1]['hits'],x[0][1]))
    hg=defaultdict(list)
    for r in hybrid:hg[(r['minimum_gap'],r['minimum_margin'])].append(r)
    hybrid_scores=[(k,metric(v,lambda r:r['text'])) for k,v in hg.items()]
    hybrid_scores=[x for x in hybrid_scores if x[1]['complete']]
    if not hybrid_scores:raise ValueError('Hybrid dev not complete')
    hs=[x for x in hybrid_scores if x[1]['errors']<=base['errors']]
    hybrid_best=max(hs,key=lambda x:(x[1]['hits'],-x[1]['errors'],x[0][0],x[0][1])) if hs else min(hybrid_scores,key=lambda x:(x[1]['errors'],-x[1]['hits']))
    selection=dict(dev_samples=len(dev),selection_uses_test_labels=False,
        rule='Maximize dev target hits with no dev character-error increase over corresponding unbiased baseline; then fewer char errors, more conservative hyperparameter. Fallback lowest CER if none qualifies.',
        qwen_dev=base,native_dev=native_scores,hybrid_dev={str(k):v for k,v in hybrid_scores},
        native_hotword_score=native_best[0][1],hybrid_minimum_gap=hybrid_best[0][0],hybrid_minimum_margin=hybrid_best[0][1],
        native_selected=native_best[1],hybrid_selected=hybrid_best[1])
    path=out/'selection.json'
    if path.exists():assert json.loads(path.read_text())==selection,'Selection already frozen'
    else:path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(selection,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['collect','tune'])
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--split',default='dev',choices=['dev','test','control','all'])
    p.add_argument('--minimum-gap',type=float,default=-1.);p.add_argument('--margin',type=float,default=.04);p.add_argument('--limit',type=int)
    a=p.parse_args();out=a.run_dir.resolve()
    if a.stage=='collect':collect(out,a.split,a.minimum_gap,a.margin,a.limit)
    else:tune(out)
