"""Reconstruct policy summaries from immutable completed-session records."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.benchmark_confidence_rule_gate import text_differences
from experiments.scripts.benchmark_fixed_gate_pipeline import sha,write


def summarize_records(records,configuration):
    expected=tuple(configuration['policies'])
    groups=defaultdict(dict)
    for row in records:
        group=groups[(row['case'],row['repetition'])]
        if row['policy'] in group:
            raise ValueError('Duplicate policy session')
        group[row['policy']]=row
    if not groups or 'native' not in expected:
        raise ValueError('Native paired sessions required')
    for group in groups.values():
        if set(group)!=set(expected):
            raise ValueError('Incomplete policy pair')
        if not all(row['windows']==group['native']['windows']
            and row['window_pipeline']==group['native']['window_pipeline'] for row in group.values()):
            raise ValueError('Source window mismatch')
    policy_summary={};differences=[]
    for policy in expected:
        selected=[row for row in records if row['policy']==policy]
        policy_summary[policy]=dict(sessions=len(selected),calls=sum(row['stats']['call_count'] for row in selected),
            refiner_wall_ms=sum(row['stats']['completed_wall_latency_ms'] for row in selected),
            chain_total_seconds=sum(row['elapsed_seconds'] for row in selected),
            chain_median_seconds=statistics.median(row['elapsed_seconds'] for row in selected),
            identical_final_pairs_vs_native=sum(group[policy]['final_text']==group['native']['final_text'] for group in groups.values()),
            gate_change_reasons=dict(Counter(a['reason'] for row in selected for a in row.get('gate_audit',[]) if a['changed'])))
        if policy=='native':continue
        for (case,repetition),group in groups.items():
            if group[policy]['final_text']!=group['native']['final_text']:
                differences.append(dict(case=case,repetition=repetition,policy=policy,
                    changes=text_differences(group['native']['final_text'],group[policy]['final_text']),
                    correctness='unverified',audio_checked=False))
    return dict(scope=configuration['scope'],matched_windows=True,
        cases=len({row['case'] for row in records}),sessions=len(records),pairs_per_policy=len(groups),
        policies=policy_summary,calibrated=False,production_eligible=False,
        quality_improvement_not_established=True,whole_audio_latency_not_measured=True,
        output_difference_records=len(differences)),differences


def run(args):
    config=json.loads((args.input/'configuration.json').read_text())
    with (args.input/'sessions.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    summary,differences=summarize_records(rows,config)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'summary.json',summary)
    write(args.output/'output_differences.json',differences)
    write(args.output/'provenance.json',dict(original_outputs_preserved=True,
        sessions_sha256=sha(args.input/'sessions.jsonl'),configuration_sha256=sha(args.input/'configuration.json'),
        reason='Derived summary variable-name conflict; actual replay session logs complete',
        source=str(args.input.resolve())))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
