"""Resume after the second epoch checkpoint and use a larger evaluation batch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = ROOT / 'logs/qwen25_05b_refiner_reviewed_v5/run.json'
LOG_DIR = ROOT / 'logs/qwen25_05b_refiner_reviewed_v5'
BASE_CONFIG = ROOT / 'experiments/llamafactory/qwen25_05b_refiner_reviewed_v5.yaml'
RESUME_CONFIG = ROOT / 'experiments/llamafactory/qwen25_05b_refiner_reviewed_v5_resume.yaml'
GPU_UUID = 'GPU-c3da2483-42aa-6594-0038-5deef8baef11'
CHECKPOINT = Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_reviewed_v5_full/checkpoint-5306')
EXPECTED_STEP = 5306


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    run = json.loads(RUN_PATH.read_text())
    if run.get('status') != 'running' or run.get('gpu_index') != 0:
        raise RuntimeError('Expected the active five-epoch GPU0 training run.')
    report_path = ROOT / 'results/qwen25_05b_refiner_preparation_20261021/preparation_report.json'
    report = json.loads(report_path.read_text())
    base = yaml.safe_load(BASE_CONFIG.read_text())
    if base['num_train_epochs'] != 5 or base['output_dir'] != run['output_dir']:
        raise RuntimeError('The base training config no longer matches the active run.')
    if report['config_sha256'] != hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest():
        raise RuntimeError('Base config hash mismatch.')
    data = Path(base['dataset_dir'])
    export = json.loads((data / 'export_report.json').read_text())
    if report['source_export_report_sha256'] != hashlib.sha256((data / 'export_report.json').read_bytes()).hexdigest():
        raise RuntimeError('Dataset export report changed.')
    for name, item in export['files'].items():
        h = hashlib.sha256((data / name).read_bytes()).hexdigest()
        if h != item['sha256']:
            raise RuntimeError('Dataset changed: ' + name)
    cache = Path(base['tokenized_path'])
    for name, expected in report['tokenized_files'].items():
        if hashlib.sha256((cache / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Token cache changed: ' + name)
    # Write a separate continuation config; the currently running process has
    # already read the original YAML and will finish its first evaluation as-is.
    resumed = dict(base)
    resumed['per_device_eval_batch_size'] = 4
    resumed['resume_from_checkpoint'] = str(CHECKPOINT)
    RESUME_CONFIG.write_text(
        '# Resume from the completed first epoch; larger eval batches only.\n'
        + yaml.safe_dump(resumed, allow_unicode=True, sort_keys=False)
    )
    checkpoint_deadline = time.monotonic() + 5 * 3600
    while not CHECKPOINT.exists() and time.monotonic() < checkpoint_deadline:
        # Do not disturb the original run if it has not safely checkpointed.
        alive = subprocess.run(['kill', '-0', str(run['pid'])], check=False).returncode == 0
        if not alive:
            raise RuntimeError(f'Original trainer exited before producing checkpoint-{EXPECTED_STEP}.')
        time.sleep(2)
    if not CHECKPOINT.exists():
        raise RuntimeError(f'Timed out waiting for checkpoint-{EXPECTED_STEP}; original training remains untouched.')
    trainer_state = CHECKPOINT / 'trainer_state.json'
    required = ['model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']
    signature = None
    stable = 0
    for _ in range(180):
        if any(not (CHECKPOINT / name).is_file() for name in required):
            time.sleep(0.5)
            continue
        try:
            saved_step = json.loads(trainer_state.read_text()).get('global_step')
        except (json.JSONDecodeError, OSError):
            time.sleep(0.5)
            continue
        if saved_step != EXPECTED_STEP:
            time.sleep(0.5)
            continue
        state = tuple((f.name, f.stat().st_size) for f in sorted(CHECKPOINT.iterdir()) if f.is_file())
        if state == signature:
            stable += 1
        else:
            signature, stable = state, 0
        if stable >= 3:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f'Checkpoint-{EXPECTED_STEP} files did not become stable; leaving current training untouched.')

    # The durable checkpoint is now verified, so stop the old batch-1 run
    # before it can consume meaningful work in epoch two.
    ps = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,stat=,args='], text=True)
    candidates = []
    for line in ps.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4 and 'llamafactory.cli' in fields[3] and str(BASE_CONFIG) in fields[3]:
            candidates.append((int(fields[0]), int(fields[1]), fields[2]))
    main_candidates = [item for item in candidates if item[0] not in {x[1] for x in candidates}]
    main_pid = main_candidates[0][0] if main_candidates else None
    if len(main_candidates) > 1:
        raise RuntimeError('Multiple active original trainers found; checkpoint is safe, none was stopped.')
    # If the previous launcher already exited, the complete checkpoint is
    # sufficient: continue from it without trying to signal a stale parent PID.
    if main_pid is not None:
        os.kill(main_pid, signal.SIGINT)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            current = subprocess.check_output(['ps', '-p', str(main_pid), '-o', 'stat='], text=True).strip()
            if not current or current.startswith('Z'):
                break
            time.sleep(1)
        else:
            # The checkpoint is durable; terminate only this exact trainer process.
            os.kill(main_pid, signal.SIGTERM)
            time.sleep(2)
    if not CHECKPOINT.exists() or json.loads(trainer_state.read_text()).get('global_step') != EXPECTED_STEP:
        raise RuntimeError(f'Checkpoint-{EXPECTED_STEP} changed unexpectedly.')

    deadline = time.monotonic() + 180
    free_gib = 0.0
    while time.monotonic() < deadline:
        free = subprocess.check_output([
            'nvidia-smi', '--id=' + GPU_UUID, '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'
        ], text=True).strip()
        free_gib = int(free) / 1024
        if free_gib >= 18:
            break
        time.sleep(3)
    if free_gib < 18:
        raise RuntimeError(f'GPU0 has only {free_gib:.1f} GiB free after stop; checkpoint remains safe.')

    train_log = LOG_DIR / 'train_resume_eval_batch4.log'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    python = ROOT / '.runtime/llamafactory-env/bin/python'
    with train_log.open('ab', buffering=0) as stream:
        resumed_process = subprocess.Popen(
            [str(python), '-u', '-c', 'from llamafactory.cli import main; main()', 'train', str(RESUME_CONFIG)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
    run.update(status='running', previous_trainer_pid=main_pid, pid=resumed_process.pid,
        resumed_from=str(CHECKPOINT), eval_batch_size=4, resume_config=str(RESUME_CONFIG),
        resume_log=str(train_log), resumed_at=datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        verified_checkpoint_step=EXPECTED_STEP)
    write_json(RUN_PATH, run)
    report.update(resume_config=str(RESUME_CONFIG), eval_batch_size_after_epoch1=4)
    write_json(report_path, report)
    print(json.dumps(run, ensure_ascii=False, indent=2), flush=True)
    # Confirm that the resumed trainer loads its checkpoint and advances.
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if resumed_process.poll() is not None:
            raise RuntimeError(f'Resumed trainer exited with {resumed_process.returncode}; inspect {train_log}.')
        if train_log.exists():
            tail = train_log.read_text(errors='replace')[-12000:]
            if any(marker in tail for marker in ('Resuming training from checkpoint',
                'Loading model from checkpoint', 'Continuing training from checkpoint',
                'Will skip the first', '***** Running training *****')):
                break
        time.sleep(5)
    else:
        raise RuntimeError('Resumed trainer did not report startup within five minutes.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        error = dict(status='resume_failed', error=f'{type(exc).__name__}: {exc}',
            timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
        write_json(LOG_DIR / 'resume_supervisor_error.json', error)
        raise
