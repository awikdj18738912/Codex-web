"""New-teacher operation effects; scripted weak supervision, never audio gold."""
from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as data
from experiments.scripts.audit_gate_failures import projected_cleanup
from experiments.scripts.train_gate_round2 import load_records,check_partition_links


def operation_label(strict,baseline,refined,spoken,intended):
    if strict!='uncertain':return strict,'strict_full_text_effect'
    if projected_cleanup(spoken,intended,baseline)==refined and baseline!=refined:
        return 'benefit','known_script_cleanup_projected_preserving_residual_error'
    edits=[op for op in SequenceMatcher(None,baseline,refined,autojunk=False).get_opcodes() if op[0]!='equal']
    if len(edits)==1:
        before,after=c.edit_distance(baseline,intended),c.edit_distance(refined,intended)
        if after<before:return 'benefit','single_lexical_edit_reduces_script_distance'
        if after>before:return 'harm','single_lexical_edit_increases_script_distance'
    return 'uncertain','ambiguous_or_multiple_lexical_edits'


def assemble(root:Path):
    from system.zh_itn_bridge import ZhITNNormalizer
    out=root/'dataset';out.mkdir(exist_ok=True);fingerprints=[];report={}
    partitions={};alloutcomes={}
    for partition in ['train','dev']:
        source=root/f'{partition}_data';rows,outcomes,teachers=load_records(source,partition)
        fingerprints.extend(teachers.values());seeds={r['seed_id']:r for r in data.seeds(source)}
        folders={};values=[]
        for row in rows:
            name='pairs_full' if row['pair_set']=='window_full' else 'pairs_review_full'
            if name not in folders:
                f=source/name;norm=ZhITNNormalizer(f/'native/libzh_itn_bridge.so')
                folders[name]=({r['case_id']:r for r in c.read_rows(f/'cases.jsonl')},
                               {r['case_id']:r for r in c.read_rows(f/'pairs.jsonl')},norm)
            cases,pairs,norm=folders[name];case=cases[row['case_id']];pair=pairs[row['case_id']]
            normalize=lambda s:data.lexical(norm.normalize(s).text)
            baseline,refined=[normalize(pair['arms'][p]['comparison_text']) for p in ['rules','always']]
            meta=case['synthetic_metadata'];spoken,intended=normalize(meta['spoken_text']),normalize(meta['intended_text'])
            label,basis=operation_label(row['label'],baseline,refined,spoken,intended)
            row=dict(row,strict_label=row['label'],label=label,operation_label_basis=basis,
                eligible_for_weak_pretraining=label!='uncertain',validation_cohort=seeds[row['seed_id']]['round3_cohort'],
                reference_scope='unverified_script_operation_effect_not_real_audio_gold')
            # The actual Web adapter has no left-context argument for window
            # routing. Match its deployable view rather than a richer offline
            # feature that the streaming caller cannot reliably provide.
            if row['input']['task_kind']=='window_refine':row['input']=dict(row['input'],available_context='')
            for arm,value in outcomes[row['case_id']]['arms'].items():
                text=normalize(pair['arms'][arm]['comparison_text'])
                effect=value['effect'];value['strict_effect']=effect
                value['effect'],_=operation_label(effect,baseline,text,spoken,intended)
            values.append(row)
        partitions[partition]=values;alloutcomes.update(outcomes)
        report[partition]=dict(cases=len(values),groups=len({r['recording_group'] for r in values}),
            strict_labels=dict(Counter(r['strict_label'] for r in values)),operation_labels=dict(Counter(r['label'] for r in values)),
            native_actions=dict(Counter(r['native_action'] for r in values)),changed_labels=sum(r['strict_label']!=r['label'] for r in values))
        (out/f'{partition}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    if any(fp!=fingerprints[0] for fp in fingerprints):raise ValueError('New-teacher signature mismatch')
    # Input-equivalent states across groups can arise from ASR errors. Keep
    # the established template split and remove training collisions with dev;
    # never alter the development partition to increase a score.
    from experiments.scripts.train_gate_round2 import signature
    dev_inputs={signature(r) for r in partitions['dev']}
    removed=[r for r in partitions['train'] if signature(r) in dev_inputs]
    partitions['train']=[r for r in partitions['train'] if signature(r) not in dev_inputs]
    check_partition_links(dict(train=partitions['train'],validation=partitions['dev']))
    (out/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in partitions['train']))
    c.write_json(out/'outcomes.json',alloutcomes)
    c.write_json(out/'manifest.json',dict(partitions=report,teacher=fingerprints[0],
        removed_training_input_collisions=[r['case_id'] for r in removed],train_after_collision_removal=len(partitions['train']),
        window_features='raw_and_rules_without_offline_only_left_context',
        labels='strict_exact_plus_single_edit_distance_or_known_script_cleanup',human_gold=0,audio_script_verified=False,
        old_labels_not_used=True,consumed_test_not_used=True,old_dev_reused_not_independent=True))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(assemble(a.root),ensure_ascii=False,indent=2))
