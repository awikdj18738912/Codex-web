"""Build a capped first experiment from an immutable pool snapshot, then fill gaps."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import requests

import generate_refiner_100k as base
from generate_balanced_refiner_gaps import BalancedGapRunner, file_hash
from generate_refiner_multigpu import EndpointTransport
from select_balanced_refiner_data import CATEGORIES, public_label_evidence


CAP = 2941


class First50kRunner(BalancedGapRunner):
    def status(self, *args, **kwargs):
        super().status(*args, **kwargs)
        path = self.out/'运行状态.md'
        text = path.read_text().replace('# 八类均衡补缺：生成状态', '# 首轮约5万条：生成状态')
        text = text.replace('一类达到目标后自动跳过，继续其余未满类别。A01/C02/C03/C04先处理有明确标签的公开数据。',
            '已筛选通过的公开数据优先复用；其余缺口合成补齐。未完成语义审核的公开候选不计入配额。各类足量即停止新增。')
        temp = path.with_suffix('.md.tmp')
        temp.write_text(text)
        temp.replace(path)


def capped_seeds(rows):
    """Prefer admitted public records; select synthetic groups deterministically."""
    counts = Counter()
    def order(row):
        p = json.loads(row['payload_json'])
        public = p.get('dataset') != 'Gemma4'
        group = str(row['group_id'] or row['id'])
        return (not public, hashlib.sha256(group.encode()).hexdigest(), row['id'])
    for raw in sorted(rows, key=order):
        if raw['status'] != 'accepted':
            continue
        c = raw['category']
        if c not in CATEGORIES:
            raise ValueError('unknown_category')
        if counts[c] >= CAP:
            continue
        previous = json.loads(raw['payload_json'])
        row = dict(previous)
        row.update(id=raw['id'], category=c, group_id=raw['group_id'],
                   input=raw['source'], reference=raw['target'],
                   selection_state='selected_pending_human_review',
                   human_review_status='pending', training_ready=False,
                   prior_ledger_payload=previous)
        if row.get('dataset') != 'Gemma4' and public_label_evidence(row) is None:
            raise ValueError('public_seed_missing_explicit_native_label:'+raw['id'])
        if row.get('dataset') is None:
            raise ValueError('missing_dataset')
        counts[c] += 1
        yield row


def prepare(source, inputs):
    inputs.mkdir(parents=True, exist_ok=False)
    snapshot = inputs/'source_pool.sqlite3'
    with sqlite3.connect('file:'+str(source.resolve())+'?mode=ro', uri=True) as src:
        with sqlite3.connect(snapshot) as dst:
            src.backup(dst)
    with sqlite3.connect('file:'+str(snapshot)+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM samples WHERE status='accepted'").fetchall()
    counts = Counter()
    sources = Counter()
    selected = inputs/'selected.pending_review.jsonl'
    with selected.open('w') as stream:
        for row in capped_seeds(rows):
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[row['category']] += 1
            sources[row['dataset']] += 1
    (inputs/'empty_d02.jsonl').write_text('')
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),
               'source_ledger':str(source.resolve()), 'snapshot_sha256':file_hash(snapshot),
               'selected_sha256':file_hash(selected), 'target_per_class':CAP,
               'target_total':CAP*len(CATEGORIES), 'selected_total':sum(counts.values()),
               'by_category':{c:{'selected':counts[c], 'remaining':CAP-counts[c]}
                              for c in CATEGORIES}, 'by_source':dict(sources),
               'human_reviewed_total':0, 'training_ready':False,
               'selection':'original admitted public labels first; deterministic group order',
               'source_ledger_modified':False}
    base.atomic_json(inputs/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prepare-from', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gpu-ids', choices=['0', '0,1'], default='0,1')
    parser.add_argument('--workers', type=int, choices=[16, 32], default=32)
    args = parser.parse_args()
    if args.prepare_from:
        print(json.dumps(prepare(args.prepare_from, args.inputs), ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    summary = json.loads((args.inputs/'summary.json').read_text())
    selected = args.inputs/'selected.pending_review.jsonl'
    if file_hash(selected) != summary['selected_sha256']:
        raise ValueError('frozen_seed_input_changed')
    active = [c for c in CATEGORIES if summary['by_category'][c]['remaining']]
    args.selected = selected
    args.d02_snapshot = args.inputs/'empty_d02.jsonl'
    args.public_root = Path('/home/aim0/data/datasets/ASR/public_refiner_sources_20261025')
    args.fcgec_test = Path('/home/aim0/data/datasets/ASR/refiner_balanced_17x5882_20261025/heldout_sources/FCGEC_test.json')
    args.api = 'http://127.0.0.1:8014/v1'
    args.model = 'gemma4-31b-teacher'
    args.per_server_limit = args.workers
    args.gpu_ids = [int(v) for v in args.gpu_ids.split(',')]
    args.batch_size = 4
    args.target_per_class = CAP
    args.seed = 2026102601
    args.request_timeout = 900
    args.stop_after = 0
    args.categories = ','.join(active)
    requests.get('http://127.0.0.1:8014/health', timeout=10).raise_for_status()
    models = requests.get(args.api+'/models', timeout=10)
    models.raise_for_status()
    if args.model not in {r['id'] for r in models.json()['data']}:
        raise ValueError('teacher_model_name_mismatch')
    runner = First50kRunner(args)
    # Keep IDs distinct from the previous pool, including its reserved examples.
    with runner.ledger._connect() as db:
        db.execute("UPDATE counters SET value=max(value,2000000) WHERE key='sequence'")
    manifest = {'entrypoint':str(Path(__file__).resolve()), 'sha256':file_hash(Path(__file__)),
                'config_hash':runner.config_hash, 'input_summary':summary}
    manifest_path = runner.out/'first_50k_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('50k_entrypoint_or_inputs_changed')
    base.atomic_json(manifest_path, manifest)
    (runner.out/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    runtime = {'started_utc':datetime.now(timezone.utc).isoformat(),
               'config_hash':runner.config_hash, 'active_gpu_ids':args.gpu_ids,
               'workers':args.workers, 'per_server_limit':args.per_server_limit,
               'active_categories':active, 'target_per_class':CAP,
               'mixed_dataset_target':CAP*len(CATEGORIES),
               'selected_seed_total':runner.initial_count, 'training_started':False,
               'public_label_policy':'explicit_matching_original_subclass_label_required'}
    base.atomic_json(runner.out/'runtime_current.json', runtime)
    with (runner.out/'runtime_history.jsonl').open('a') as stream:
        stream.write(json.dumps(runtime, ensure_ascii=False)+'\n')
    base.requests = EndpointTransport(args.api, [args.api], args.per_server_limit,
        source_hash=file_hash(Path(__file__)), request_timeout=args.request_timeout)
    print(json.dumps(runtime, ensure_ascii=False), flush=True)
    runner.run()


if __name__ == '__main__':
    main()
