"""Run matched regression then new natural speech, without changing policy."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();root=a.root.resolve();state=dict(status='running',stage='old_regression')
    try:
        for version in ('old','new'):
            state['stage']=version+'_regression';c.write_json(root/'validation_job_state.json',state)
            with (root/'logs'/f'{version}_regression.log').open('a') as log:
                subprocess.run([sys.executable,'-u','experiments/scripts/gate_terminal_gain_pilot.py','run',
                    '--root',str(root/f'{version}_regression'),'--device',a.device],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        state['stage']='natural_streaming';c.write_json(root/'validation_job_state.json',state)
        with (root/'logs/natural_streaming.log').open('a') as log:
            subprocess.run([sys.executable,'-u','experiments/scripts/stream_linear_gate_holdout.py',
                '--root',str(root/'natural_holdout'),'--device',a.device,'--repetitions','2'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='complete',stage='regression_and_natural_streaming_complete')
    except BaseException as error:state.update(status='failed',error=repr(error));raise
    finally:c.write_json(root/'validation_job_state.json',state)


if __name__=='__main__':main()
