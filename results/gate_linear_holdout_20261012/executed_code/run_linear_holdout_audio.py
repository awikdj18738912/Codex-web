"""Generate only new TTS audio; the test runner performs the first ASR decode."""
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
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.run_gate_template_pilot import stop_owned_service


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    manifest=json.loads((root/'test_data/manifest.json').read_text());selection=root/'selection.json'
    if c.file_hash(selection)!=manifest['policy_sha256_before_test_generation']:raise ValueError('Policy drift')
    process=None;state=dict(status='running',stage='new_TTS_audio_only',owned_helper_only=True)
    c.write_json(root/'audio_job_state.json',state)
    try:
        if len(d.read_rows(root/'test_data/tts.jsonl'))<manifest['count']:
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',8011))==0:raise RuntimeError('TTS port busy; will not stop unrelated service')
            with (root/'logs/tts_service.log').open('a') as log:
                process=subprocess.Popen(['bash','scripts/qwen_tts_service.sh','--device','cuda:0'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['owned_tts_pid']=process.pid;c.write_json(root/'audio_job_state.json',state)
                for _ in range(180):
                    if process.poll() is not None:raise RuntimeError('Owned TTS helper exited')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8011/health',timeout=1) as response:
                            if json.load(response).get('status')=='ok':break
                    except Exception:pass
                    time.sleep(1)
                else:raise TimeoutError('Owned TTS did not become ready')
                with (root/'logs/synthesize.log').open('a') as stream:
                    subprocess.run([sys.executable,'-u','experiments/scripts/synthetic_gate_data.py','synthesize',
                        '--output',str(root/'test_data'),'--split','test'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if c.file_hash(selection)!=manifest['policy_sha256_before_test_generation']:raise ValueError('Frozen policy changed')
        state.update(status='complete',stage='audio_complete_no_test_ASR_yet')
    except BaseException as error:state.update(status='failed',error=repr(error));raise
    finally:
        if process is not None:state['helper_shutdown']=stop_owned_service(process)
        c.write_json(root/'audio_job_state.json',state)


if __name__=='__main__':main()
