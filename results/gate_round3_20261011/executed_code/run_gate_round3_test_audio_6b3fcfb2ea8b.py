"""Own a temporary GPU TTS helper, then reuse the existing ASR service."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c
from experiments.scripts.run_gate_template_pilot import stop_owned_service


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--minimum-free-gib',type=float,default=6.0);a=p.parse_args()
    root=a.root.resolve();source=root/'test_data';logs=root/'logs';selection=root/'training/selection.json'
    if (source/'job_state.json').exists():c.append_row(source/'job_history.jsonl',json.loads((source/'job_state.json').read_text()))
    policy_hash=c.file_hash(selection);manifest=json.loads((source/'manifest.json').read_text())
    if manifest['policy_sha256_before_test_generation']!=policy_hash:raise RuntimeError('Frozen policy changed')
    state=dict(status='running',stage='TTS_start',policy_sha256=policy_hash,owned_helper_only=True,minimum_free_gib=a.minimum_free_gib);service=None
    def save():c.write_json(source/'job_state.json',state)
    def run(stage,args):
        state['stage']=stage;save();print('STAGE '+stage,flush=True)
        with (logs/(stage+'.log')).open('a') as stream:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        if len(c.read_rows(source/'tts.jsonl'))<manifest['count']:
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',8011))==0:raise RuntimeError('TTS port occupied; no unrelated helper will be stopped')
            stream=(logs/'fresh_tts_service.log').open('a')
            service=subprocess.Popen(['bash','scripts/qwen_tts_service.sh','--device','cuda:0','--minimum-free-gib',str(a.minimum_free_gib)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['owned_tts_pid']=service.pid;save()
            for _ in range(300):
                if service.poll() is not None:raise RuntimeError('Owned TTS service exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8011/health',timeout=1) as response:
                        health=json.load(response)
                        if health.get('ok') or health.get('status')=='ok':break
                except Exception:pass
                time.sleep(1)
            else:raise RuntimeError('Owned TTS service did not become ready')
            run('fresh_synthesize',['experiments/scripts/synthetic_gate_data.py','synthesize','--output',str(source),'--split','test'])
            state['helper_shutdown']=stop_owned_service(service);service=None;stream.close();save()
        run('fresh_decode',['experiments/scripts/synthetic_gate_data.py','decode','--output',str(source),'--split','test','--variants','clean'])
        if c.file_hash(selection)!=policy_hash:raise RuntimeError('Policy changed after test started')
        state.update(status='complete',stage='fresh_audio_and_ASR_complete');save()
    except BaseException as error:
        state.update(status='failed',error=repr(error));save();raise
    finally:
        if service is not None:state['helper_shutdown']=stop_owned_service(service);save()


if __name__=='__main__':main()
