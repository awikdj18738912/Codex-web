"""Separate model behavior from published-reference fidelity evaluation."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.benchmark_fixed_gate_pipeline import sha,write
from system.zh_itn_bridge import ZhITNNormalizer


def lexical_content(text,normalizer):
    # Both sides use identical rule formatting. Retained digit values are not
    # corrected by a reference, only deterministically normalized.
    normalized=normalizer.normalize(text).text
    return ''.join(char.casefold() for char in normalized
                   if not char.isspace() and unicodedata.category(char)[0] in {'L','N','M'})


def edit_distance(left,right):
    previous=list(range(len(right)+1))
    for i,char in enumerate(left,1):
        current=[i]
        for j,other in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(char!=other)))
        previous=current
    return previous[-1]


def summarize(directory,output,zh_itn):
    normalizer=ZhITNNormalizer(zh_itn)
    with (directory/'sessions.jsonl').open() as stream:
        sessions=[json.loads(line) for line in stream]
    source_cases=json.loads((directory/'sources.json').read_text())
    sources={case['case']:case for case in source_cases}
    windows=[];counter=Counter();reference_metrics=defaultdict(Counter)
    ref_items=[];groups=defaultdict(dict)
    for row in sessions:
        groups[(row['case'],row['repetition'])][row['policy']]=row
        if row['repetition']!=0:
            continue
        source=sources[row['case']]
        plan=source['audio_provenance'].get('source_plan_case',{})
        reference=plan.get('reference_text')
        if reference is not None:
            ref=lexical_content(reference,normalizer)
            before=lexical_content(row['raw_text'],normalizer)
            after=lexical_content(row['final_text'],normalizer)
            old_errors,new_errors=edit_distance(ref,before),edit_distance(ref,after)
            stats=reference_metrics[row['policy']]
            stats['cases']+=1;stats['reference_characters']+=len(ref)
            stats['raw_asr_errors']+=old_errors;stats['final_errors']+=new_errors
            stats['improved_cases']+=new_errors<old_errors
            stats['regressed_cases']+=new_errors>old_errors
            stats['unchanged_cases']+=new_errors==old_errors
            ref_items.append(dict(case=row['case'],policy=row['policy'],speaker_id=plan.get('speaker_id'),
                reference=reference,raw_text=row['raw_text'],final_text=row['final_text'],
                reference_characters=len(ref),raw_asr_errors=old_errors,final_errors=new_errors,
                comparison='published_read_speech_fidelity_not_cleanup_quality',project_audio_checked=False,
                source_reference_from_publisher=True,training_eligible=False))
        if row['policy']!='all_refine':
            continue
        by_source={profile['text']:profile for profile in source['profiles']}
        for segment in row['window_segments']:
            outcome=segment.get('outcome')
            category=('rejected_fallback' if outcome=='rejected'
                else 'unassessed_pipeline_status' if outcome!='accepted'
                else 'accepted_changed' if segment.get('output')!=segment.get('baseline')
                else 'accepted_unchanged')
            counter[category]+=1
            profile=by_source[segment['source']]
            windows.append(dict(case=row['case'],source=segment['source'],baseline=segment['baseline'],
                final_output=segment.get('output'),classification=category,outcome=segment.get('outcome'),
                reject_reasons=segment.get('reject_reasons',[]),model_calls=segment.get('model_calls',[]),
                score=profile['confidence'],audio_provenance=source['audio_provenance'],
                cleanup_correctness='unverified',audio_checked=False,human_audio_gold=False,
                training_eligible=False,independent_acceptance_eligible=False))
    group_summaries=defaultdict(lambda:defaultdict(Counter))
    for row in sessions:
        plan=sources[row['case']]['audio_provenance'].get('source_plan_case',{})
        group=plan.get('dependency_group',plan.get('recording_group','unknown'))
        group_summaries[group][row['policy']]['calls']+=row['stats']['call_count']
        group_summaries[group][row['policy']]['sessions']+=1
    policy_pairs={}
    for policy in sorted({row['policy'] for row in sessions}-{'native','all_refine'}):
        pairs=list(groups.values())
        policy_pairs[policy]=dict(pairs=len(pairs),
            identical_final_pairs=sum(g[policy]['final_text']==g['native']['final_text'] for g in pairs),
            different_final_pairs=sum(g[policy]['final_text']!=g['native']['final_text'] for g in pairs),
            added_calls=sum(g[policy]['stats']['call_count']-g['native']['stats']['call_count'] for g in pairs))
    fidelity={policy:{**dict(values),
        'raw_asr_cer':values['raw_asr_errors']/values['reference_characters'] if values['reference_characters'] else None,
        'final_cer':values['final_errors']/values['reference_characters'] if values['reference_characters'] else None}
        for policy,values in reference_metrics.items()}
    summary=dict(scope='expanded_model_behavior_audit_with_separate_published_read_speech_reference_panel',
        source_cases=len(source_cases),unique_all_refine_windows=len(windows),
        all_refine_classifications=dict(counter),candidate_pairs=policy_pairs,
        public_read_speech_fidelity=fidelity,recording_groups={g:{p:dict(v) for p,v in policies.items()}
                                                            for g,policies in group_summaries.items()},
        project_human_audio_gold=0,cleanup_gold=0,training_eligible=0,production_eligible=False,
        limitations=['Model edit acceptance is not audio correctness or positive cleanup gain',
                     'Published read-speech reference evaluates spelling/content fidelity, not spontaneous speech cleanup',
                     'Reference/CER normalization ignores punctuation and uses the same numeric rule formatting on both sides',
                     'Archive speaker ID is grouping metadata; ASR pretraining overlap unknown',
                     'Original project source recordings are consumed and some share potentially related video material'],
        source_sha256={name:sha(directory/name) for name in ['sources.json','sessions.jsonl','summary.json']})
    output.mkdir(parents=True,exist_ok=False)
    write(output/'summary.json',summary)
    write(output/'windows.json',windows)
    write(output/'published_reference_fidelity.json',ref_items)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['recording_groups','source_sha256']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--zh-itn',type=Path,default=ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    args=parser.parse_args();summarize(args.input,args.output,args.zh_itn)
