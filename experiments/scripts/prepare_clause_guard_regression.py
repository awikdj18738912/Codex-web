"""Replay consumed ASR texts through identical old/new cold finalizers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d


def prepare(root):
    old=ROOT/'results/gate_linear_holdout_20261012'
    sources={r['seed_id']:r for r in d.read_rows(old/'test_data/tts.jsonl')+d.read_rows(old/'human_sources.jsonl')}
    distinct={}
    for row in d.read_rows(old/'streaming/sessions.jsonl'):
        distinct.setdefault((row['seed_id'],row['raw_text']),row)
    rows=[];seen={}
    for (seed,raw),previous in distinct.items():
        original=sources[seed];index=seen.get(seed,0);seen[seed]=index+1
        rows.append(dict(seed_id=seed,recording_group='consumed_holdout:'+seed,variant=f'ASR_view_{index}',
            synthetic_split='regression',template_group=original['template_group'],category=original['category'],
            domain='general',audio_path=original['audio_path'],audio_sha256=original['audio_sha256'],pcm_sha256=previous['pcm_sha256'],
            raw_text=raw,intended_text=original['intended_text'],spoken_text=original['spoken_text'],
            original_human_gold=original.get('human_gold',False),
            final_payload=dict(text=raw,language='Chinese',completed_segments=[dict(segment_id=1,text=raw)])))
    runner=ROOT/'experiments/scripts/gate_terminal_gain_pilot.py'
    for version,runtime in [('old',old/'runtime'),('new',root/'runtime')]:
        dest=root/f'{version}_regression'
        if dest.exists():raise FileExistsError('Never overwrite an executed replay')
        dest.mkdir();(dest/'logs').mkdir();(dest/'executed_code').mkdir()
        c.verify_frozen(runtime);shutil.copytree(runtime,dest/'runtime')
        for row in rows:c.append_row(dest/'sources.jsonl',row)
        (dest/'executed_code'/runner.name).write_bytes(runner.read_bytes())
        c.write_json(dest/'manifest.json',dict(recordings=len(rows),source_rows_sha256=c.file_hash(dest/'sources.jsonl'),
            adapter_sha256=c.file_hash(runner),runtime_protocol_sha256=c.file_hash(dest/'runtime/protocol.json'),
            role='consumed_ASR_regression',scope='cold_final_snapshot_matched_old_new_not_streaming_accuracy_or_latency',
            original_audio_clips=len(sources),different_raw_ASR_views=len(rows),synthetic_split='regression'))
    c.write_json(root/'regression_manifest.json',dict(views=len(rows),clips=len(sources),
        old_streaming_sha256=c.file_hash(old/'streaming/sessions.jsonl'),
        same_source_sha256=c.file_hash(root/'old_regression/sources.jsonl')))
    print('Staged',len(rows),'recorded ASR views')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root.resolve())
