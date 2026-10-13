"""Resumable new-teacher pair collection; no TTS/ASR service mutation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    root=args.root.resolve();logs=root/'logs';logs.mkdir(exist_ok=True)
    state=dict(status='running',stage='prepare',started_unix=time.time(),driver_pid=__import__('os').getpid(),completed=[])
    def save():
        state['updated_unix']=time.time();tmp=root/'job_state.tmp';c.write_json(tmp,state);tmp.replace(root/'job_state.json')
    save()
    snapshot=root/'executed_code';snapshot.mkdir(exist_ok=True)
    for name in ['run_gate_round3_collection.py','collect_gate_pairs.py','synthetic_gate_data.py']:
        (snapshot/name).write_bytes((ROOT/'experiments/scripts'/name).read_bytes())
    parent=ROOT/'results/gate_round2_20261010/test_data/pairs_full'
    tasks=[(partition,name,windows,reviews) for partition in ['train','dev']
           for name,windows,reviews in [('pairs_full',10000,0),('pairs_review_full',0,10000)]]
    def run(stage,command):
        state.update(stage=stage);save();print('STAGE '+stage,flush=True)
        with (logs/(stage+'.log')).open('a') as stream:
            subprocess.run([sys.executable,'-u',*command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(stage);save()
    try:
        for partition,name,windows,reviews in tasks:
            source=root/f'{partition}_data';out=source/name
            if not (out/'protocol.json').exists():
                run(partition+'_'+name+'_prepare',['experiments/scripts/collect_gate_pairs.py','prepare','--output',str(out),
                    '--sources',str(source/'asr.jsonl'),'--no-demo-sources','--window-count',str(windows),'--review-count',str(reviews),
                    '--entity-db',str(parent/'entities_snapshot.db'),'--library',str(parent/'native/libzh_itn_bridge.so')])
            c.verify_frozen(out)
        signatures=[{k:v for k,v in c.verify_frozen(root/f'{p}_data'/n)['frozen_hashes'].items() if k.startswith('frozen/system/')}
                    for p,n,_,_ in tasks]
        if any(s!=signatures[0] for s in signatures):raise RuntimeError('New-teacher runtimes differ')
        for partition,name,_,_ in tasks:
            source=root/f'{partition}_data';out=source/name
            if {r['case_id'] for r in c.read_rows(out/'cases.jsonl')}!={r['case_id'] for r in c.read_rows(out/'pairs.jsonl')}:
                run(partition+'_'+name+'_run',['experiments/scripts/collect_gate_pairs.py','run','--output',str(out),'--device',args.device,'--pause','0'])
            if not (out/'weak_gate_labels.jsonl').exists():
                run(partition+'_'+name+'_label',['experiments/scripts/synthetic_gate_data.py','label','--output',str(source),'--pairs-dir',str(out)])
        for partition in ['train','dev']:
            source=root/f'{partition}_data';rows=[]
            for name,key in [('pairs_full','window_full'),('pairs_review_full','review_full')]:
                rows.extend(dict(r,pair_set=key) for r in c.read_rows(source/name/'weak_gate_labels.jsonl'))
            (source/f'weak_gate_labels_{partition}_all.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            c.write_json(source/'job_state.json',dict(status='complete',pairs=len(rows),new_teacher=True))
        state.update(status='complete',stage='new_teacher_collection_complete');save()
    except BaseException as error:
        state.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
