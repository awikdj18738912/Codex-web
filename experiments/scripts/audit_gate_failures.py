"""Post-test diagnosis, never training labels or acceptance evidence."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts import collect_gate_pairs as collector, synthetic_gate_data as data


def projected_cleanup(spoken, intended, baseline):
    """Project script deletions through exact aligned spans, preserving other errors.

    Ambiguous alignments and non-deletion edits are deliberately left unresolved.
    This is diagnostic evidence only: it cannot establish the spoken audio truth.
    """
    edits = [op for op in SequenceMatcher(None, spoken, intended, autojunk=False).get_opcodes()
             if op[0] != 'equal']
    if not edits or any(op[0] != 'delete' for op in edits):
        return None
    align = SequenceMatcher(None, spoken, baseline, autojunk=False).get_matching_blocks()
    deletions = []
    for _, start, end, _, _ in edits:
        matches = [b.b + start - b.a for b in align if b.a <= start and end <= b.a + b.size]
        if len(matches) != 1:
            return None
        deletions.append((matches[0], matches[0] + end - start))
    result = baseline
    for start, end in sorted(deletions, reverse=True):
        result = result[:start] + result[end:]
    return result


def duplicate_repairs(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['baseline_normalized'] != row['always_normalized']:
            groups[(row['recording_group'], row['baseline_normalized'], row['always_normalized'])].append(row)
    return [dict(recording_group=key[0], baseline=key[1], refined=key[2],
                 case_ids=[r['case_id'] for r in values], labels=[r['weak_label'] for r in values])
            for key, values in groups.items() if len({r['task_kind'] for r in values}) > 1]


def audit_pack(source, name, predictions):
    from system.zh_itn_bridge import ZhITNNormalizer
    rows = []
    for set_name in ['pairs_full', 'pairs_review_full']:
        folder = source / set_name
        collector.verify_frozen(folder)
        cases = {r['case_id']: r for r in data.read_rows(folder / 'cases.jsonl')}
        pairs = {r['case_id']: r for r in data.read_rows(folder / 'pairs.jsonl')}
        normalizer = ZhITNNormalizer(folder / 'native/libzh_itn_bridge.so')
        normalize = lambda s: data.lexical(normalizer.normalize(s).text)
        for label in data.read_rows(folder / 'weak_gate_labels.jsonl'):
            case, pair = cases[label['case_id']], pairs[label['case_id']]
            baseline, always, current = [pair['arms'][k]['comparison_text'] for k in ['rules','always','current']]
            meta = case['synthetic_metadata']
            before, after, target, spoken = map(normalize, [baseline,always,meta['intended_text'],meta['spoken_text']])
            recomputed, _ = data.weak_label(baseline,always,meta['intended_text'],lambda s:normalizer.normalize(s).text)
            if recomputed != label['label']:
                raise RuntimeError('Weak label drift: ' + label['case_id'])
            native = case['original_gate']['action'] if case['task_kind']=='window_refine' else 'refine'
            student = predictions.get(label['case_id'])
            tags = []
            if label['label']=='benefit' and native!='refine': tags.append('native_gate_missed_beneficial_operation')
            if label['label']=='benefit' and native=='refine' and student and student['action']!='refine':
                tags.append('student_vetoed_beneficial_operation')
            if label['label']=='harm' and spoken==target: tags.append('script_preserves_content_refiner_changes_it')
            if before!=target and before==after: tags.append('neutral_operation_but_transcript_not_exact')
            projected = projected_cleanup(spoken,target,before)
            if label['label']=='uncertain' and projected is not None and projected==after and before!=after:
                tags.append('partial_cleanup_success_with_unrelated_residual_error')
            if label['label']=='uncertain' and before!=spoken and before!=target:
                tags.append('audio_or_asr_or_reference_mismatch_unresolved')
            rows.append(dict(pack=name,case_id=label['case_id'],task_kind=case['task_kind'],
                recording_group=case['recording_group'],template_group=meta['template_group'],
                weak_label=label['label'],label_basis=label['label_basis'],native_action=native,
                native_reasons=case.get('original_gate',{}).get('reasons',[]),student_decision=student,
                raw=case['raw_text_full'],baseline=baseline,current=current,always=always,
                spoken_script=meta['spoken_text'],intended_script=meta['intended_text'],
                baseline_normalized=before,always_normalized=after,projected_cleanup=projected,
                tags=tags,human_gold=False,audio_script_verified=False))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    trial=ROOT/'results/gate_round2_20261010'
    predictions={r['case_id']:r for r in data.read_rows(trial/'training/test_predictions.jsonl')
                 if r['model']=='encoder_weighted' and r['route']=='native_veto'}
    packs={'test':trial/'test_data','new_dev':trial/'dev_data','risk_dev':trial/'risk_dev_data',
           'new_train':ROOT/'results/gate_template_expansion_20261010'}
    rows=[];summary={}
    for name,path in packs.items():
        values=audit_pack(path,name,predictions if name=='test' else {})
        rows.extend(values)
        summary[name]=dict(cases=len(values),recordings=len({r['recording_group'] for r in values}),
            labels=dict(Counter(r['weak_label'] for r in values)),tags=dict(Counter(t for r in values for t in r['tags'])),
            benefit_recordings=len({r['recording_group'] for r in values if r['weak_label']=='benefit'}),
            duplicate_cross_task_repairs=duplicate_repairs(values))
    collector.write_json(args.output/'pair_audit_summary.json',dict(packs=summary,
        scope='post_test_diagnostic_not_new_acceptance_test',new_training_labels=False,
        policy_frozen_sha256=collector.file_hash(trial/'training/selection.json'),
        warning='Decision pairs cannot estimate sequential final-output utility; references are unverified synthesis scripts.'))
    (args.output/'case_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
