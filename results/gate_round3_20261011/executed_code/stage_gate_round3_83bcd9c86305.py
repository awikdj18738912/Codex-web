"""Freeze reused train/development ASR sources; exclude all consumed test families."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import synthetic_gate_data as data


def stage(output:Path):
    if output.exists():raise FileExistsError('Use a new round-three directory')
    old=ROOT/'results/qwen_tts_asr_gpu_pilot_20261010'
    splitroot=ROOT/'results/learned_gate_weak_prototype_20261010'
    groups={p:{r['split_group'] for r in data.read_rows(splitroot/f'{p}.jsonl')} for p in ['train','validation','test']}
    test=ROOT/'results/gate_round2_20261010/test_data'
    excluded=groups['test']|{r['template_group'] for r in data.seeds(test)}
    output.mkdir(parents=True);counts={};sources={}
    for partition in ['train','dev']:
        dest=output/f'{partition}_data';dest.mkdir();allrows=[];allseeds={}
        selections=[(old,'old_train' if partition=='train' else 'old_validation')]
        if partition=='train':selections.append((ROOT/'results/gate_template_expansion_20261010','new_train'))
        else:selections.extend([(ROOT/'results/gate_round2_20261010/dev_data','new_dev'),
                                (ROOT/'results/gate_round2_20261010/risk_dev_data','risk_dev')])
        for source,cohort in selections:
            seeds={r['seed_id']:r for r in data.seeds(source)}
            rows=data.read_rows(source/'asr.jsonl')
            if source==old:
                permitted=groups['train' if partition=='train' else 'validation']
                rows=[r for r in rows if r['template_group'] in permitted]
            if any(r['template_group'] in excluded for r in rows):raise ValueError('Consumed test family leaked')
            sources[str(source/'asr.jsonl')]=data.file_hash(source/'asr.jsonl')
            for r in rows:
                if not Path(r['audio_path']).is_file():raise FileNotFoundError(r['audio_path'])
                s=dict(seeds[r['seed_id']]);s.pop('seed_sha256')
                s.update(origin_seed_sha256=r['seed_sha256'],synthetic_split=partition,round3_cohort=cohort)
                s['seed_sha256']=data.digest(s);allseeds[s['seed_id']]=s
                row=dict(r,synthetic_split=partition,seed_sha256=s['seed_sha256'],round3_cohort=cohort)
                allrows.append(row)
        for s in allseeds.values():data.append_row(dest/'texts.jsonl',s)
        for r in allrows:data.append_row(dest/'asr.jsonl',r)
        data.write_json(dest/'manifest.json',dict(count=len(allseeds),seed=20261011,texts_sha256=data.file_hash(dest/'texts.jsonl'),
            synthetic_splits={partition:len(allseeds)},scope='reused_ASR_partition_with_new_teacher_no_new_audio_claim'))
        counts[partition]=dict(seeds=len(allseeds),asr_rows=len(allrows),templates=len({r['template_group'] for r in allrows}),
            by_cohort=dict(Counter(r['round3_cohort'] for r in allrows)))
    train=data.read_rows(output/'train_data/asr.jsonl');dev=data.read_rows(output/'dev_data/asr.jsonl')
    for key in ['recording_group','template_group','pcm_sha256']:
        if {r[key] for r in train}&{r[key] for r in dev}:raise ValueError('Cross-partition '+key)
    report=dict(counts=counts,source_hashes=sources,excluded_test_templates=sorted(excluded),
        old_development_is_reused_not_fresh=True,clean_noise_share_recording_group=True,
        source_labels_not_reused=True,seed_hash_changed_only_for_declared_partition_provenance=True)
    data.write_json(output/'staging_manifest.json',report)
    data.write_json(output/'job_state.json',dict(status='staged',stage='new_teacher_pair_preparation'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(stage(a.output),ensure_ascii=False,indent=2))
