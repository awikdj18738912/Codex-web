"""Reload only the two local Web processes with their existing argv/env."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT=Path('/home/aim0/桌面/tan/ASR/Codex-web')
OUT=ROOT/'results/entity_local_replay_20261012'


def process_for(port):
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:args=[a.decode() for a in (entry/'cmdline').read_bytes().split(b'\0') if a]
        except (OSError,UnicodeError):continue
        if 'system.web_app' in args and '--port' in args and args[args.index('--port')+1]==str(port):
            raw=(entry/'environ').read_bytes()
            environment={s.split(b'=',1)[0].decode():s.split(b'=',1)[1].decode() for s in raw.split(b'\0') if b'=' in s}
            return int(entry.name),args,environment
    raise RuntimeError(f'No existing Web worker on port{port}')


def probe(port):
    import websocket
    socket=websocket.create_connection(f'ws://127.0.0.1:{port}/ws/stream?domain=general&entity_audio_repair=1&entity_asr_context=0',timeout=20,http_no_proxy=['127.0.0.1','localhost'])
    try:
        ready=json.loads(socket.recv())
        status=ready.get('entity_audio_repair') or {}
        assert ready['event']=='ready' and status.get('local_replay_rule')=='segment_windows_v1'
        assert status.get('local_replay_enabled') is True
        return status
    finally:socket.close()


def main():
    results=[]
    for port in (8081,8051):
        pid,args,environment=process_for(port)
        record={'port':port,'previous_pid':pid,'argv_preserved':True,
                'argv_sha256':hashlib.sha256(json.dumps(args).encode()).hexdigest(),
                'cuda_visible_devices':environment.get('CUDA_VISIBLE_DEVICES')}
        try:
            status=probe(port)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:health=json.load(response)
            record.update(pid=pid,health=health,entity_ready_status=status,already_loaded=True)
            results.append(record)
            (OUT/'deployment.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
            print('ALREADY_READY',port,'pid',pid,flush=True)
            continue
        except Exception:pass
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+15
        while Path(f'/proc/{pid}').exists() and time.monotonic()<deadline:
            try:
                if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
            except OSError:break
            time.sleep(.2)
        log=(OUT/f'deployed_{port}.log').open('ab')
        process=subprocess.Popen(args,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=process.pid
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError(f'Reloaded Web exited on port{port}; see local log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:health=json.load(response)
                if health.get('ok'):break
            except Exception:pass
            time.sleep(.5)
        else:raise RuntimeError(f'Web readiness timeout on port{port}')
        record['health']=health
        record['entity_ready_status']=probe(port)
        results.append(record)
        (OUT/'deployment.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print('READY',port,'pid',process.pid,'rule',record['entity_ready_status']['local_replay_rule'],flush=True)


if __name__=='__main__':main()
