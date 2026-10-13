"""Fork a frozen replay runtime for development ablations without editing it."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as collector


def fork(source: Path, output: Path, modules: list[str]):
    source,output=source.resolve(),output.resolve()
    if output.exists():raise ValueError('Destination already exists; do not mutate executed bundles')
    parent=source/'pairs_full';protocol=collector.verify_frozen(parent)
    allowed={'refinement_guard.py','deterministic_cleanup.py','repetition_preservation.py','refinement_protocol.py','web_app.py'}
    if not set(modules).issubset(allowed):raise ValueError('Unsupported runtime patch')
    changes={}
    for name in modules:
        path=ROOT/'system'/name
        if not path.is_file():raise ValueError('Missing patch: '+name)
        changes[name]=dict(parent_sha256=collector.file_hash(parent/'frozen/system'/name) if (parent/'frozen/system'/name).exists() else None,
                           patched_sha256=collector.file_hash(path))
    output.mkdir(parents=True)
    shutil.copy2(source/'asr.jsonl',output/'asr.jsonl')
    target=output/'pairs_full';target.mkdir()
    for name in ['frozen','native','inputs']:
        shutil.copytree(parent/name,target/name)
    for name in ['cases.jsonl','entities_snapshot.db','run_configuration.json']:
        shutil.copy2(parent/name,target/name)
    for name in modules:
        relative='frozen/system/'+name
        shutil.copy2(ROOT/'system'/name,target/relative)
        protocol['frozen_hashes'][relative]=changes[name]['patched_sha256']
    protocol['development_patch']=dict(parent_protocol_sha256=collector.file_hash(parent/'protocol.json'),changes=changes,
        scope='development_ablation_on_consumed_data_not_independent_test',original_policy_and_weights_unchanged=True)
    collector.write_json(target/'protocol.json',protocol)
    collector.write_json(output/'fork_manifest.json',protocol['development_patch'])
    collector.verify_frozen(target)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--modules',nargs='+',required=True)
    args=parser.parse_args();print(fork(args.source,args.output,args.modules))


if __name__=='__main__':main()
