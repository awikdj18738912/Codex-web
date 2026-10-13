"""Keep train/dev separate, including shared clean/noise recording groups."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.gate_terminal_gain_pilot import view_id
from experiments.scripts.summarize_terminal_gain_pilot import strategy_summary,choose_posthoc


def conservative_target(effect):
    if effect['label']=='uncertain' or not effect['native_control_stable']:return None
    if effect['label']=='harm':return 'skip_allowed'
    if effect['label']=='benefit':return 'refine_required'
    return 'skip_allowed' if effect['skip_has_compute_advantage_proxy'] else 'refine_required'


def assemble(root):
    output=root/'dataset'
    if output.exists():raise FileExistsError('Do not overwrite assembled labels')
    allrows={};records={};summaries={};provenance={}
    pilot_ids={view_id(r) for r in d.read_rows(ROOT/'results/gate_terminal_gain_pilot_20261011/sources.jsonl')}
    for split in ('train','dev'):
        folder=root/f'{split}_collection';c.verify_frozen(folder/'runtime')
        manifest=json.loads((folder/'manifest.json').read_text())
        cfg=json.loads((folder/'configuration.json').read_text())
        assert manifest['source_rows_sha256']==c.file_hash(folder/'sources.jsonl')
        assert cfg['manifest_sha256']==c.file_hash(folder/'manifest.json')
        assert manifest['adapter_sha256']==c.file_hash(folder/'executed_code/gate_terminal_gain_pilot.py')
        values=d.read_rows(folder/'recordings.jsonl');sources=d.read_rows(folder/'sources.jsonl')
        assert len(values)==len(sources)==manifest['recordings']
        assert {view_id(r) for r in values}=={view_id(r) for r in sources}
        assert all(r['synthetic_split']==split for r in values)
        records[split]=values;labels=[]
        by_view={view_id(r):r for r in sources}
        for row in values:
            source=by_view[view_id(row)]
            native=row['native'];native_routes={r['key']:r['action'] for r in native['routes']}
            for effect in row['effects']:
                skip=row['counterfactuals'][effect['window_key']]
                routed={r['key']:r['action'] for r in skip['routes']}
                assert native_routes.keys()==routed.keys()
                assert [k for k in native_routes if native_routes[k]!=routed[k]]==[effect['window_key']]
                assert routed[effect['window_key']]=='keep'
                assert native['final']['raw_text']==skip['final']['raw_text']==row['raw_text']
                cost_keys=('refiner_wall_ms','plausibility_wall_ms')
                native_time=sum(native['cost'][k] for k in cost_keys)
                repeat_time=sum(row['native_repeat']['cost'][k] for k in cost_keys)
                skip_time=sum(skip['cost'][k] for k in cost_keys)
                labels.append(dict(case_id=c.digest([split,view_id(row),effect['window_key']])[:24],
                    recording_view_id=view_id(row),seed_id=row['seed_id'],recording_group=row['recording_group'],
                    variant=row['variant'],pcm_sha256=row['pcm_sha256'],split=split,split_group=row['template_group'],
                    cohort=source['round3_cohort'],input=effect['input'],input_signature=effect['window_key'],
                    label=effect['label'],action_target=conservative_target(effect),native_action='refine',
                    terminal_effect=effect,seen_in_pilot=view_id(row) in pilot_ids,
                    native_cost=native['cost'],veto_cost=skip['cost'],
                    measured_model_time_saving_ms=native_time-skip_time,
                    saving_against_faster_native_control_ms=min(native_time,repeat_time)-skip_time,
                    native_final_distance=native['intended_distance'],veto_final_distance=skip['intended_distance'],
                    human_gold=False,audio_script_verified=False,eligible_for_real_risk_calibration=False))
        allrows[split]=labels
        physical=d.read_rows(folder/'arms.jsonl')
        summaries[split]=dict(source_views=len(values),source_groups=len({r['recording_group'] for r in values}),
            templates=len({r['template_group'] for r in values}),
            recording_views_with_window=sum(bool(r['effects']) for r in values),labels=dict(Counter(r['label'] for r in labels)),
            action_targets=dict(Counter(r['action_target'] for r in labels)),window_examples=len(labels),
            native_controls_stable=sum(r['native_control_stable'] for r in values),
            physical_snapshots=len(physical),physical_refiner_calls=sum(len(r['model_calls']) for r in physical),
            physical_plausibility_calls=sum(len(r['plausibility_calls']) for r in physical),
            native=strategy_summary(values,lambda r:r['native']),skip_all=strategy_summary(values,lambda r:r['skip_all']),
            posthoc=strategy_summary(values,choose_posthoc))
        provenance[split]=dict(recordings_sha256=c.file_hash(folder/'recordings.jsonl'),
            source_rows_sha256=manifest['source_rows_sha256'],runtime_sha256=manifest['runtime_protocol_sha256'])
    for key in ('recording_group','split_group','pcm_sha256'):
        if {r[key] for r in allrows['train']}&{r[key] for r in allrows['dev']}:raise AssertionError('Leakage: '+key)
    dev_inputs={r['input_signature'] for r in allrows['dev']}
    overlapped=[r for r in allrows['train'] if r['input_signature'] in dev_inputs]
    fitting=[r for r in allrows['train'] if r['input_signature'] not in dev_inputs]
    by_input=defaultdict(set)
    for row in fitting:
        if row['action_target'] is not None:by_input[row['input_signature']].add(row['action_target'])
    conflicting={k for k,v in by_input.items() if len(v)>1}
    fitting=[r for r in fitting if r['input_signature'] not in conflicting and r['action_target'] is not None]
    output.mkdir()
    for name,rows in [('train_all',allrows['train']),('train',fitting),('dev',allrows['dev']),('train_input_overlap_excluded',overlapped)]:
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    c.write_json(output/'recordings.json',records)
    c.write_json(output/'manifest.json',dict(summaries=summaries,provenance=provenance,
        fit_rows=len(fitting),fit_targets=dict(Counter(r['action_target'] for r in fitting)),
        fit_benefit_groups=len({r['recording_group'] for r in fitting if r['label']=='benefit'}),
        fit_benefit_templates=len({r['split_group'] for r in fitting if r['label']=='benefit'}),
        input_overlap_train_rows_excluded=len(overlapped),conflicting_train_inputs_excluded=len(conflicting),
        train_dev_exact_input_disjoint=True,development_is_reused_not_fresh=True,
        seen_pilot_train_views=sum(view_id(r) in pilot_ids for r in records['train']),
        pilot_not_merged_as_additional_data=True,human_gold=False,scope='frozen_ASR_cold_final_snapshot_terminal_effects',
        action_target='benefit or neutral without conservative cost advantage requires window; harm or cheaper neutral may veto',
        files_sha256={p.name:c.file_hash(p) for p in output.glob('*.json*')}))
    print(json.dumps(json.loads((output/'manifest.json').read_text()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();assemble(a.root.resolve())
