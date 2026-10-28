"""Release GPU1 compute jobs at 09:00 Asia/Shanghai; default is read-only.

Stops owning user services or Docker containers when identifiable, then signals
remaining compute PIDs. Never targets graphics processes or broad user slices.
Other users' jobs require permissions the current user might not possess.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/aim0/桌面/tan/ASR/Codex-web')
LOG_DIR = ROOT / 'results/gpu1_daily_stop'
TZ = ZoneInfo('Asia/Shanghai')
UID = os.getuid()


def log(event, **details):
    entry = {'time': datetime.now(TZ).isoformat(), 'event': event, **details}
    line = json.dumps(entry, ensure_ascii=False)
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def command(args, timeout=45):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit={result.returncode}: {result.stderr.strip()[:300]}')
    return result.stdout


def gpu_pids():
    output = command(['/usr/bin/nvidia-smi', '-i', '1',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    return sorted({int(line.strip()) for line in output.splitlines() if line.strip().isdigit()})


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text()
        # Fields after the command's closing ')' begin at field 3; starttime=22.
        started = stat[stat.rfind(')') + 2:].split()[19]
        return {'pid': pid, 'started': started, 'uid': proc.stat().st_uid,
            'command': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')[:400],
            'cgroup': (proc / 'cgroup').read_text()}
    except FileNotFoundError:
        return None


def stop_owner(info):
    cgroup = info['cgroup']
    # Only a dedicated app service owned by this user, never the login manager,
    # session scope, desktop scope, user@ service, or the overall user slice.
    prefix = f'/user.slice/user-{UID}.slice/user@{UID}.service/app.slice/'
    for line in cgroup.splitlines():
        path = line.split(':', 2)[-1]
        if path.startswith(prefix):
            leaf = path[len(prefix):].split('/')[0]
            if leaf.endswith('.service'):
                log('stop_user_service', unit=leaf, gpu_pid=info['pid'])
                command(['/usr/bin/systemctl', '--user', 'stop', leaf], timeout=100)
                return
    container = re.search(r'(?:docker-|/docker/)([0-9a-f]{64})(?:\.scope)?', cgroup)
    if container:
        log('stop_container', container=container.group(1), gpu_pid=info['pid'])
        command(['/usr/bin/docker', 'stop', '--time', '15', container.group(1)], timeout=45)


def send(info, sig):
    now = identity(info['pid'])
    if now is None or now['started'] != info['started']:
        return
    try:
        os.kill(info['pid'], sig)
        log('signal', pid=info['pid'], signal=sig.name)
    except ProcessLookupError:
        pass
    except PermissionError:
        log('permission_denied', pid=info['pid'], uid=info['uid'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    now = datetime.now(TZ)
    # A manual invocation outside the scheduled window cannot terminate jobs.
    if args.execute and not (now.hour == 9 and now.minute < 10):
        log('refused_outside_0900_window', local_time=now.isoformat())
        return 2
    os.environ.setdefault('XDG_RUNTIME_DIR', f'/run/user/{UID}')
    os.environ.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{UID}/bus')
    pids = gpu_pids()
    log('inventory', gpu=1, execute=args.execute,
        processes=[identity(pid) for pid in pids])
    if not args.execute:
        return 0
    # Recheck to catch a supervisor restarting a child during shutdown.
    for sweep in range(3):
        pids = gpu_pids()
        if not pids:
            log('gpu1_compute_idle')
            return 0
        infos = [info for pid in pids if (info := identity(pid)) is not None]
        for info in infos:
            try:
                stop_owner(info)
            except Exception as error:
                log('owner_stop_failed', pid=info['pid'], error=str(error))
            send(info, signal.SIGTERM)
        until = time.monotonic() + 15
        while time.monotonic() < until and gpu_pids():
            time.sleep(1)
        for pid in gpu_pids():
            info = identity(pid)
            if info is not None:
                send(info, signal.SIGKILL)
        time.sleep(2)
    remaining = gpu_pids()
    log('finished', remaining_gpu1_compute_pids=remaining)
    return 1 if remaining else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        log('failed', error=str(error))
        raise
