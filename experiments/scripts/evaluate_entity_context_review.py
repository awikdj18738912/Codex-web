"""Offline context review, never deployed or supplied target references.

Leave-one-recording-out profiles use other glossary evidence recordings. These
are observed contexts, not externally verified identity descriptions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_context_confirmation_20261012'
BASE=ROOT/'results/large_entity_benchmark_v1_20261010'
sys.path.insert(0,str(ROOT))
PROMPT='''你审核语音转录中的实体替换，不改写原句，不生成新实体。
目标候选和其他候选都不是已确认事实；字形相似、词库中存在、另一个句子出现过该名称，都不能单独证明本句提到了它。
独立资料是其他录音的上下文，不一定提供完整身份；不得编造额外背景。
SUPPORT：当前语境和资料明确支持目标候选且没有冲突。
REJECT：资料与当前语境明确冲突，或目标候选不适合这个位置，并有具体依据。
UNCERTAIN：证据不足、有歧义、或资料无法确定身份。缺少资料不能当成反证。
只输出一个JSON对象：{"decision":"SUPPORT或REJECT或UNCERTAIN","evidence_ids":["资料ID"],"reason":"简短理由"}。'''


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (OUT/'pilot_plan.json').exists():raise FileExistsError('Pilot already frozen')
    entities={e['canonical_text']:e for e in json.loads((BASE/'dataset/entities.json').read_text())}
    samples={r['id']:r for r in read(BASE/'dataset/samples.jsonl')}
    for folder in ('holdout','holdout_news','holdout_v2'):
        samples.update({r['id']:r for r in read(ROOT/'results/entity_confirmation_20261011'/folder/'samples.jsonl')})
    rows=read(OUT/'applied_change_candidates.jsonl');dedup={}
    for row in rows:
        key=(row['sample_id'],row['start'],row['end'],row['canonical'])
        if key not in dedup or row['condition']=='proposed':dedup[key]=row
    # Known failures are development diagnostics, not an independent sample.
    diagnostic_ids={'leb1_38459bb96fdd7d36ecc1','leb1_6c86c4a79561e0611e5e',
        'leb1_8c2510722ec5a19f9656','leb1_d39d7292a9660cc1eef6',
        'ec2_6d69ea498293fa9cf0d8','ec2_11a7a30fcdc77067553b','ec2_8b59e1e5083bb9455821'}
    ordered=sorted(dedup.values(),key=lambda r:hashlib.sha256((r['sample_id']+r['canonical']).encode()).hexdigest())
    selected=[r for r in ordered if r['sample_id'] in diagnostic_ids]
    selected += [r for r in ordered if r['sample_id'] not in diagnostic_ids][:64-len(selected)]
    inputs=[];labels=[]
    for row in selected:
        sid=row['sample_id'];source=row['source_text'];start,end=row['start'],row['end']
        assert source[start:end]==row['observed']
        canonicals=list(dict.fromkeys([row['canonical']]+[c['canonical'] for c in row['overlap_candidates']
            if c.get('same_exact_span')]))[:5]
        profiles=[]
        for canonical in canonicals:
            entity=entities[canonical];evidence=[]
            ids=sorted({e['sample_id'] for e in entity.get('evidence',[]) if e['sample_id']!=sid})
            for eid in ids:
                sample=samples[eid]
                if sample['audio_sha256']==samples[sid]['audio_sha256']:continue
                reference=sample['reference']
                if canonical not in reference:continue
                pos=reference.find(canonical);snippet=reference[max(0,pos-40):pos+len(canonical)+40]
                evidence.append(dict(evidence_id=eid,source=sample.get('source',''),text=snippet,
                                     source_audio_sha256=sample['audio_sha256']))
                if len(evidence)==3:break
            profiles.append(dict(canonical=canonical,registered_type=entity['entity_type'],
                                 evidence=evidence,profile_complete=False))
        input_id=hashlib.sha256(f'{sid}:{start}:{end}:{row["canonical"]}'.encode()).hexdigest()[:20]
        inputs.append(dict(id=input_id,sample_id=sid,source_text=source,
            context_with_slot=source[:start]+'【待核对】'+source[end:],observed=row['observed'],
            target_candidate=row['canonical'],profiles=profiles,
            target_has_independent_evidence=bool(profiles[0]['evidence'])))
        # Target reference is opened only here for a separate scoring sidecar;
        # nothing in it is added to the inference payload above.
        reference=samples[sid]['reference']
        labels.append(dict(id=input_id,sample_id=sid,reference=reference,
            canonical_present_in_reference=row['canonical'] in reference,
            diagnostic_only=sid in diagnostic_ids,
            already_vetoed_by_deployed_rule=sid=='ec2_6d69ea498293fa9cf0d8'))
    (OUT/'pilot_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
    (OUT/'pilot_labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in labels))
    write(OUT/'pilot_plan.json',dict(samples=len(inputs),development_only=True,
        choice='All seven known absent-canonical diagnostics plus deterministic hash controls; no claim of random population accuracy',
        target_reference_at_inference=False,input_sha256=digest(OUT/'pilot_inputs.jsonl'),
        prompt=PROMPT,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        model='/home/aim0/data/models/ASR/AgenticASR-Refiner',
        conditions=['explicit_review','native_patch_with_background'],
        evidence_policy='Exclude target recording, use at most3 other recording snippets; not externally verified identity profiles'))
    print('PREPARED',len(inputs),'with target evidence',sum(r['target_has_independent_evidence'] for r in inputs),flush=True)


def parse_review(text,completed,valid_ids):
    if not completed:return dict(decision='UNCERTAIN',valid=False,reason='incomplete_generation')
    try:
        value=json.loads(text)
        assert value['decision'] in ('SUPPORT','REJECT','UNCERTAIN')
        assert isinstance(value.get('evidence_ids',[]),list)
        assert set(value.get('evidence_ids',[]))<=valid_ids
        return dict(decision=value['decision'],valid=True,reason=value.get('reason',''),evidence_ids=value.get('evidence_ids',[]))
    except (ValueError,TypeError,KeyError,AssertionError):
        return dict(decision='UNCERTAIN',valid=False,reason='invalid_review_response')


def infer(device):
    from system.live_qwen_refiner import TransformersRefiner
    plan=json.loads((OUT/'pilot_plan.json').read_text());assert digest(OUT/'pilot_inputs.jsonl')==plan['input_sha256']
    destination=OUT/'pilot_model_outputs.jsonl';done={(r['id'],r['condition']) for r in read(destination)} if destination.exists() else set()
    model=TransformersRefiner(Path(plan['model']),device,160);model._torch.set_num_threads(4)
    for row in read(OUT/'pilot_inputs.jsonl'):
        payload={k:row[k] for k in ('context_with_slot','observed','target_candidate','profiles')}
        valid_ids={e['evidence_id'] for p in row['profiles'] for e in p['evidence']}
        for condition in plan['conditions']:
            if (row['id'],condition) in done:continue
            started=time.perf_counter()
            if condition=='explicit_review':
                text,latency,completed=model._review_messages_with_status(plan['prompt'],json.dumps(payload,ensure_ascii=False))
                result=parse_review(text,completed,valid_ids)
            else:
                text,latency=model.refine(row['source_text'],entity_candidates=[dict(
                    observed=row['observed'],canonical=row['target_candidate'],
                    background=json.dumps(row['profiles'],ensure_ascii=False))])
                completed=None
                try:
                    patch=json.loads(text)
                    support=(patch.get('source')==row['observed'] and patch.get('target')==row['target_candidate'])
                    if 'edits' in patch:
                        support=any(e.get('source')==row['observed'] and e.get('target')==row['target_candidate'] for e in patch['edits'])
                    result=dict(decision='SUPPORT' if support else 'UNCERTAIN',valid=True,reason='native_patch')
                except (ValueError,TypeError,AttributeError):result=dict(decision='UNCERTAIN',valid=False,reason='non_json_patch')
            record=dict(id=row['id'],condition=condition,output=text,parsed=result,completed=completed,
                        model_latency_ms=latency,wall_seconds=time.perf_counter()-started)
            with destination.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print('MODEL',row['id'],condition,result['decision'],result['valid'],flush=True)
    print('PILOT COMPLETE',flush=True)


def report():
    labels={r['id']:r for r in read(OUT/'pilot_labels.jsonl')};inputs={r['id']:r for r in read(OUT/'pilot_inputs.jsonl')}
    outputs=read(OUT/'pilot_model_outputs.jsonl');assert len(outputs)==len(labels)*2
    summary={};diagnostics=[]
    for row in outputs:
        label=labels[row['id']];value=row['parsed'];word_ok=label['canonical_present_in_reference']
        c=summary.setdefault(row['condition'],Counter())
        c.update(cases=1,format_valid=value['valid'],reference_present=word_ok,
                 reference_absent=not word_ok,support_correct=word_ok and value['decision']=='SUPPORT',
                 reject_absent=not word_ok and value['decision']=='REJECT',
                 reject_correct=word_ok and value['decision']=='REJECT',
                 uncertain=value['decision']=='UNCERTAIN',no_target_evidence=not inputs[row['id']]['target_has_independent_evidence'])
        if label['diagnostic_only']:diagnostics.append(dict(label,input=inputs[row['id']],result=row))
    write(OUT/'pilot_summary.json',summary);write(OUT/'pilot_diagnostics.json',diagnostics)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','report']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='infer':infer(a.device)
    else:report()
