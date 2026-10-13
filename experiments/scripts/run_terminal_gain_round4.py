"""Run train then development collection sequentially on one GPU."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();root=a.root.resolve()
    state=dict(status='running',stage='train',device=a.device,owned_experiment_only=True)
    c.write_json(root/'collection_job_state.json',state)
    try:
        for split in ('train','dev'):
            state['stage']=split;c.write_json(root/'collection_job_state.json',state)
            with (root/'logs'/f'{split}_collection.log').open('a') as log:
                subprocess.run([sys.executable,'-u','experiments/scripts/gate_terminal_gain_pilot.py','run',
                    '--root',str(root/f'{split}_collection'),'--device',a.device],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='complete',stage='train_and_dev_counterfactuals_complete')
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:c.write_json(root/'collection_job_state.json',state)


if __name__=='__main__':main()
