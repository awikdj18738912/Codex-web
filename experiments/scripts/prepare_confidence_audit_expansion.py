"""Freeze diverse audio clips before expanded ASR/Refiner diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.benchmark_fixed_gate_pipeline import sha,write


def spaced_indices(size,count):
    if count<1 or size<count:
        raise ValueError("Insufficient clips for declared source quota")
    if count==1:
        return [0]
    indices=[round(index*(size-1)/(count-1)) for index in range(count)]
    if len(set(indices))!=count:
        raise ValueError("Nonunique stratified temporal selection")
    return indices


def prepare(args):
    import soundfile as sf

    args.output.mkdir(parents=True,exist_ok=False)
    with args.pool.open() as stream:
        rows=[json.loads(line) for line in stream]
    clips={}
    for row in rows:
        path=row['audio_path']; provenance=row['audio_provenance']
        entry=dict(audio_path=path,audio_sha256=row['audio_sha256'],recording_group=row['recording_group'],
            category=row['category'],audio_provenance=provenance)
        if path in clips and clips[path]!=entry:
            raise ValueError('Conflicting clip provenance')
        clips[path]=entry
    groups=defaultdict(list)
    for clip in clips.values():
        if clip['audio_provenance']['duration_seconds']>=10:
            groups[Path(clip['audio_provenance']['original_path']).name].append(clip)
    ordered=sorted(groups,key=lambda name:(not name.startswith('测试音频'),name))
    cases=[];group_plan=[]
    for name in ordered:
        pool=sorted(groups[name],key=lambda clip:clip['audio_provenance']['clip_start_seconds'])
        quota=2 if name=='测试音频1.mp3' else 8 if name=='测试音频5.mp3' or not name.startswith('测试音频') else 6
        chosen=[pool[index] for index in spaced_indices(len(pool),quota)]
        prior_end=-1.
        for clip in chosen:
            path=Path(clip['audio_path']);p=clip['audio_provenance']
            if sha(path)!=clip['audio_sha256']:
                raise ValueError('Selected audio changed: '+str(path))
            original=Path(p['original_path'])
            if sha(original)!=p['original_sha256']:
                raise ValueError('Original recording changed')
            info=sf.info(path)
            if info.samplerate!=16000 or info.channels!=1:
                raise ValueError('Existing clips must be 16k mono')
            if p['clip_start_seconds']<prior_end:
                raise ValueError('Selected audio overlap in original recording')
            used_seconds=min(info.duration,30.)
            prior_end=p['clip_start_seconds']+used_seconds
            cases.append(dict(case=len(cases),**clip,used_max_seconds=30.,
                original_used_start_seconds=p['clip_start_seconds'],
                original_used_end_seconds=prior_end,expected_used_audio_seconds=used_seconds,
                speaker_identity='unverified',already_consumed_development=True,
                selection='fixed per-original quotas with temporal spacing; no score, edit or model-output selection',
                audio_checked=False,human_audio_gold=False,training_eligible=False,independent_acceptance_eligible=False))
        group_plan.append(dict(original_name=name,recording_group=chosen[0]['recording_group'],
                               available_clips=len(pool),selected_clips=len(chosen)))
    if len(group_plan)!=6 or len(cases)!=36:
        raise ValueError('Expected six original files and 36 clips')
    plan=dict(scope='expanded_consumed_audio_development_diagnostic_not_independent_acceptance',
        cases=cases,groups=group_plan,total_clips=len(cases),original_file_groups=len(group_plan),
        planned_audio_seconds=sum(c['expected_used_audio_seconds'] for c in cases),
        target_window_count='approximately 150; actual punctuation count determined by fresh ASR',
        source_pool=str(args.pool.resolve()),source_pool_sha256=sha(args.pool),
        no_new_model_training=True,no_threshold_search=True,human_audio_gold=0,
        limitations=['Six original file groups are not six verified independent speakers',
                     'Test clips may share dubbed/video material; generation origin unverified',
                     'Previously consumed audio supports diagnostics, not fresh acceptance',
                     'Thirty-second excerpts with fresh single-utterance states do not reproduce live VAD history'])
    write(args.output/'plan.json',plan)
    (args.output/'executed_code').mkdir()
    (args.output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key:plan[key] for key in ['total_clips','original_file_groups','planned_audio_seconds','groups']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,default=ROOT/'results/audio_edit_review_20261013_v2/all_candidates.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args())
