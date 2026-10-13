"""Copy a frozen split into its own audio workspace; test requires frozen policy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.scripts import synthetic_gate_data as data


def stage(source, output, split, freeze_selection=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite a split pack")
    if split not in {'train','dev','test'}:
        raise ValueError('Unknown split')
    policy_hash=None
    if split=='test':
        if freeze_selection is None or not freeze_selection.is_file():
            raise ValueError('Test staging requires a round-two frozen selection')
        frozen=json.loads(freeze_selection.read_text())
        if frozen.get('event')!='round2_policy_frozen_before_test_generation':
            raise ValueError('Policy was not frozen before test generation')
        policy_hash=data.file_hash(freeze_selection)
    records=[r for r in data.seeds(source) if r['synthetic_split']==split]
    if not records: raise ValueError('Empty requested split')
    output.mkdir(parents=True)
    for row in records: data.append_row(output/'texts.jsonl',row)
    manifest=dict(schema_version=1,count=len(records),seed=json.loads((source/'manifest.json').read_text())['seed'],
                  texts_sha256=data.file_hash(output/'texts.jsonl'),source=str(source.resolve()),
                  source_manifest_sha256=data.file_hash(source/'manifest.json'),synthetic_splits={split:len(records)},
                  requested_split=split,policy_sha256_before_test_generation=policy_hash,
                  freeze_selection=str(freeze_selection.resolve()) if freeze_selection else None,
                  generator_sha256=data.file_hash(Path(__file__)),
                  reference_role='unverified_synthesis_scripts_not_human_gold')
    data.write_json(output/'manifest.json',manifest)
    (output/'preparation_sources').mkdir()
    (output/'preparation_sources'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/gate_template_expansion_20261010')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','dev','test'],required=True)
    p.add_argument('--freeze-selection',type=Path)
    a=p.parse_args()
    print(json.dumps(stage(a.source.resolve(),a.output.resolve(),a.split,a.freeze_selection),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
