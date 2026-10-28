"""Drain the legacy client at a record-batch boundary; never signal a GPU server."""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
OLD = P.parent / 'refiner_dataset_repair_20261023'
STORE = Path('/home/aim0/data/ASR/dataset_error_loop_20261024')
STORE.mkdir(parents=True, exist_ok=True)


def args(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    except OSError:
        return []


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0] != 'Z'
    except OSError:
        return False


def matching(filename):
    target = str(OLD / filename).encode()
    return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()
            and any(a == target or a == str(Path('results/refiner_dataset_repair_20261023') / filename).encode()
                    for a in args(p.name))]


state = {'at': datetime.now().astimezone().isoformat(), 'gpu_servers_restarted': False}
parents = matching('run_batches.py')
if len(parents) > 1:
    raise RuntimeError('multiple legacy dispatchers; inspect before handover')
for pid in matching('export_when_done.py'):
    os.kill(pid, signal.SIGTERM)
    state.setdefault('old_export_watchers_stopped', []).append(pid)
for pid in parents:
    os.kill(pid, signal.SIGSTOP)
    children_path = Path(f'/proc/{pid}/task/{pid}/children')
    children = [int(x) for x in children_path.read_text().split()]
    state.update(old_dispatcher=pid, draining_workers=children, status='waiting_for_current_batch')
    (STORE / 'handover.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(json.dumps(state, ensure_ascii=False), flush=True)
    while any(alive(child) for child in children):
        time.sleep(5)
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if alive(pid):
        raise RuntimeError('legacy dispatcher did not exit')
if not (STORE / 'manifest.json').exists() and (STORE / 'results.jsonl').exists():
    # Deployment prompts evolved during the small live pilot; do not silently
    # inherit its candidate decisions under the finalized script provenance.
    archive = STORE / ('predeployment_pilot_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    archive.mkdir()
    for name in ['results.jsonl', 'progress.json', '错误实例对照表.tsv']:
        src = STORE / name
        if src.exists():
            shutil.move(str(src), str(archive / name))
    state['pilot_outcomes_archived_and_queued_for_reaudit'] = str(archive)
state.update(status='catalog_loop_started', started_at=datetime.now().astimezone().isoformat())
(STORE / 'handover.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
print(json.dumps(state, ensure_ascii=False), flush=True)
subprocess.run([sys.executable, str(P / 'run_error_loop.py'), '--concurrency-per-gpu', '16'], check=True)
subprocess.run([sys.executable, str(P / 'export_candidates.py')], check=True)
