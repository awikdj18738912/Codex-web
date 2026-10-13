"""Propose local operation supervision without rewriting full-text weak labels.

Only training-source pairs are exported. These are old-teacher, unverified
script-derived proposals, not current-teacher outcomes or risk-calibration gold.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as collector, synthetic_gate_data as data
from experiments.scripts.audit_gate_failures import projected_cleanup


def operation_effect(full_label,spoken,intended,baseline,refined):
    projected=projected_cleanup(spoken,intended,baseline)
    if full_label=='uncertain' and projected is not None and projected==refined and baseline!=refined:
        return 'benefit','script_deletion_projected_through_exact_alignment_unverified_audio',projected
    return full_label,'original_full_text_effect',projected


def export(source: Path,output: Path):
    from system.zh_itn_bridge import ZhITNNormalizer
    rows=[];signatures={}
    for name in ['pairs_full','pairs_review_full']:
        folder=source/name;collector.verify_frozen(folder)
        labels=data.read_rows(folder/'weak_gate_labels.jsonl')
        if any(r['split']!='train' for r in labels):raise ValueError('Operation proposals accept training-source pairs only')
        cases={r['case_id']:r for r in data.read_rows(folder/'cases.jsonl')}
        pairs={r['case_id']:r for r in data.read_rows(folder/'pairs.jsonl')}
        norm=ZhITNNormalizer(folder/'native/libzh_itn_bridge.so')
        normalize=lambda s:data.lexical(norm.normalize(s).text)
        signatures[name]=dict(protocol_sha256=collector.file_hash(folder/'protocol.json'),
            pair_sha256=collector.file_hash(folder/'pairs.jsonl'),label_sha256=collector.file_hash(folder/'weak_gate_labels.jsonl'))
        for row in labels:
            case=cases[row['case_id']];pair=pairs[row['case_id']];meta=case['synthetic_metadata']
            baseline,refined=[normalize(pair['arms'][p]['comparison_text']) for p in ['rules','always']]
            effect,basis,projected=operation_effect(row['label'],normalize(meta['spoken_text']),
                normalize(meta['intended_text']),baseline,refined)
            rows.append(dict(case_id=row['case_id'],recording_group=row['recording_group'],split='train',
                split_group=row['split_group'],input=row['input'],source_full_label=row['label'],
                operation_effect_proposal=effect,proposal_basis=basis,projected_cleanup=projected,
                differs_from_full_label=effect!=row['label'],human_gold=False,audio_script_verified=False,
                eligible_for_real_risk_calibration=False,automatic_training_enabled=False,
                teacher_scope='original_frozen_pair_teacher_before_repetition_preservation_optimization'))
    output.mkdir(parents=True,exist_ok=True)
    (output/'operation_effect_proposals_train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary=dict(records=len(rows),original_labels=dict(Counter(r['source_full_label'] for r in rows)),
        proposed_labels=dict(Counter(r['operation_effect_proposal'] for r in rows)),
        changed_cases=[r['case_id'] for r in rows if r['differs_from_full_label']],
        source_fingerprints=signatures,source=str(source.resolve()),test_or_dev_included=False,
        scope='training_source_operation_proposals_only_not_relabelled_gold_or_current_teacher',
        automatic_training_enabled=False,
        next_step='Recapture outcomes with the new teacher; do not mix teacher versions or use proposals as audio gold.')
    collector.write_json(output/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(export(args.source,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
