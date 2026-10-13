"""Freeze all native-window opportunities in existing train/dev partitions."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.gate_terminal_gain_pilot import view_id


def eligible_views(rows,cases,partition):
    if any(r['synthetic_split']!=partition for r in rows):raise ValueError('Partition metadata mismatch')
    opportunities={(r['recording_group'],r['variant']) for r in cases
                   if r.get('original_gate',{}).get('action')=='refine'}
    selected=[r for r in rows if (r['recording_group'],r['variant']) in opportunities]
    if len({view_id(r) for r in selected})!=len(selected):raise ValueError('Duplicate recording view')
    return selected


def prepare(output,source,runtime):
    if output.exists():raise FileExistsError('Do not overwrite a prior collection')
    c.verify_frozen(runtime)
    partitions={};inputs={}
    for split in ('train','dev'):
        folder=source/f'{split}_data';rows=d.read_rows(folder/'asr.jsonl')
        cases=d.read_rows(folder/'pairs_full/cases.jsonl')
        partitions[split]=eligible_views(rows,cases,split)
        inputs[split]=dict(source=str(folder),asr_sha256=c.file_hash(folder/'asr.jsonl'),
            sampling_inputs_sha256=c.file_hash(folder/'pairs_full/cases.jsonl'),
            total_source_views=len(rows),selected_views=len(partitions[split]),
            variants=dict(Counter(r['variant'] for r in partitions[split])))
    for key in ('recording_group','template_group','pcm_sha256','seed_id'):
        if {r[key] for r in partitions['train']}&{r[key] for r in partitions['dev']}:raise ValueError('Train/dev overlap: '+key)
    excluded=set(json.loads((source/'staging_manifest.json').read_text())['excluded_test_templates'])
    test_hashes={}
    for path in (ROOT/'results').rglob('test_consumed.json'):
        texts=path.parent/'texts.jsonl'
        if texts.exists():
            excluded.update(r['template_group'] for r in d.read_rows(texts) if r.get('synthetic_split')=='test')
            test_hashes[str(texts)]=c.file_hash(texts)
    if any(r['template_group'] in excluded for rows in partitions.values() for r in rows):raise ValueError('Consumed test template reused')
    output.mkdir(parents=True);(output/'logs').mkdir();(output/'executed_code').mkdir()
    runner=ROOT/'experiments/scripts/gate_terminal_gain_pilot.py'
    for split,rows in partitions.items():
        dest=output/f'{split}_collection';dest.mkdir();(dest/'logs').mkdir();(dest/'executed_code').mkdir()
        shutil.copytree(runtime,dest/'runtime')
        for row in rows:
            if c.file_hash(Path(row['audio_path']))!=row['audio_sha256']:raise ValueError('Audio changed')
            c.append_row(dest/'sources.jsonl',row)
        (dest/'executed_code'/runner.name).write_bytes(runner.read_bytes())
        c.write_json(dest/'manifest.json',dict(recordings=len(rows),role=f'{split}_terminal_gain_expansion',
            source_rows_sha256=c.file_hash(dest/'sources.jsonl'),adapter_sha256=c.file_hash(runner),
            runtime_protocol_sha256=c.file_hash(dest/'runtime/protocol.json'),
            source_provenance=inputs[split],templates=len({r['template_group'] for r in rows}),
            sampling='all available native REFINE recording views; no outcome-based sampling',
            synthetic_split=split,human_gold=False,audio_script_verified=False,confidence='disabled',
            scope='actual_WebSocket_final_snapshot_with_frozen_recorded_ASR_not_live_streaming_latency'))
    (output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    c.write_json(output/'manifest.json',dict(partitions=inputs,excluded_test_templates=sorted(excluded),
        consumed_texts_sha256=test_hashes,train_dev_disjoint=True,development_is_reused_not_fresh=True,
        sampler_sha256=c.file_hash(Path(__file__)),runtime_protocol_sha256=c.file_hash(runtime/'protocol.json'),
        pilot_reexecuted_not_merged=True,original_gate_unchanged=True,created_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args();prepare(a.output.resolve(),a.source.resolve(),a.runtime.resolve())
