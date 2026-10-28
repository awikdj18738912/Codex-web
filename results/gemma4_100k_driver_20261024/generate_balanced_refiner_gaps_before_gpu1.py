"""Resume audited coverage gaps with explicit balanced quotas and provenance.

Uses a new ledger, retaining the original generated and public-source records.
Selected public seeds require explicit native subclass evidence. Imported seeds
and newly model-audited examples are both pending human review, not gold labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

import requests

import generate_refiner_100k as base
from generate_gemma4_selfgen_pilot import audit_pass
from generate_refiner_multigpu import EndpointTransport
from refiner_scoped_runner import ScopedRunner
from select_balanced_refiner_data import (
    CATEGORIES, NAMES, dedup_text, heldout_texts, normalized, public_label_evidence,
)
from selfgen_ledger import Ledger, normalize_source


DEFAULT_ACTIVE = ['A02', 'A03', 'A04', 'C01', 'D01', 'E03', 'E05', 'E06']


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


class BalancedGapRunner(ScopedRunner):
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out/'runner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.categories = args.categories.split(',')
        if len(set(self.categories)) != len(self.categories) or set(self.categories)-set(CATEGORIES):
            raise ValueError('invalid_generation_categories')
        self.quotas = {c: args.target_per_class for c in CATEGORIES}
        self.catalog = base.strict_json(base.CATALOG_PATH.read_text())
        model = base.strict_json((base.MODEL_DIR/'download_manifest.json').read_text())
        code = [Path(__file__), Path(base.__file__), Path(__file__).with_name('selfgen_ledger.py'),
                Path(__file__).with_name('refiner_scoped_runner.py'),
                Path(__file__).with_name('generate_refiner_multigpu.py'),
                Path(__file__).with_name('generate_gemma4_selfgen_pilot.py'),
                Path(__file__).with_name('select_balanced_refiner_data.py'),
                Path(__file__).with_name('fcgec_public_adapter.py')]
        self.config = {
            'version': 'balanced-gap-native-labels-v1', 'quotas': self.quotas,
            'generation_categories': self.categories, 'seed': args.seed,
            'api': args.api, 'model': args.model, 'model_revision': model['revision'],
            'quantization': 'QAT W4A16 compressed-tensors', 'rules': base.RULES,
            'generation_system': base.GEN_SYSTEM, 'audit_system': base.AUDIT_SYSTEM,
            'source_hashes': {str(path.resolve()): file_hash(path) for path in code},
            'catalog_sha256': file_hash(base.CATALOG_PATH),
            'seed_inputs': {str(path.resolve()): file_hash(path)
                            for path in (args.selected, args.d02_snapshot)},
            'heldout_inputs': {str(path.resolve()): file_hash(path) for path in (
                args.public_root/'CS2W/val.jsonl', args.public_root/'CS2W/test.jsonl',
                args.public_root/'FCGEC/FCGEC_valid.json', args.fcgec_test)},
            'public_admission_policy': 'explicit_matching_original_subclass_label_required',
            'human_review_status': 'pending', 'training_started': False,
            'near_duplicate_jaccard': .85,
            'note': 'Seeds are prior selected pending-human-review candidates, not model-generated anew. '
                    'Quality evidence and all original source IDs are retained. '
                    'No WenetSpeech-Formal or official heldout material is used for generation.',
        }
        self.config_hash = base.stable_hash(self.config)
        config_path = self.out/'config.json'
        if config_path.exists():
            if base.strict_json(config_path.read_text()) != self.config:
                raise ValueError('config_changed_refusing_implicit_resume')
        else:
            if any(p.name != 'runner.lock' for p in self.out.iterdir()):
                raise ValueError('nonempty_destination_without_config')
            base.atomic_json(config_path, self.config)
            (self.out/'seed_catalog.json').write_bytes(base.CATALOG_PATH.read_bytes())
            snapshots = self.out/'code_snapshot'; snapshots.mkdir()
            for path in code:
                (snapshots/path.name).write_bytes(path.read_bytes())
        self.ledger = Ledger(self.out/'ledger.sqlite3', self.quotas, self.config_hash)
        self.started = time.monotonic()
        self.last_status = 0.0
        self.failure_streak = 0
        self.audit_deferred = False
        self.stop_event = threading.Event()
        self.near = base.NearIndex()
        self.heldout = base.NearIndex()
        self.heldout_exact = set()
        print('Building official heldout exclusion index', flush=True)
        for identifier, text in heldout_texts(args.public_root, args.fcgec_test):
            self.heldout_exact.add(normalized(text))
            self.heldout.add(identifier, dedup_text(text))
        if not (self.out/'seed_import_summary.json').exists():
            self.import_seeds()
        for row in self.ledger.iter_accepted():
            self.index_pair(row['id'], row['source'], row['target'])
        stats = self.ledger.stats()
        self.initial_count = stats['total']['accepted']
        self.initial_scope_count = sum(stats[c]['accepted'] for c in self.categories)
        self.last_export = self.initial_count
        self.scope_target = sum(self.quotas[c] for c in self.categories)
        self.initial_stats = stats

    def index_pair(self, identifier, source, target):
        self.near.add(identifier+'/input', dedup_text(source))
        self.near.add(identifier+'/target', dedup_text(target))

    def pair_exclusion(self, source, target):
        if normalized(source) in self.heldout_exact or normalized(target) in self.heldout_exact:
            return 'official_heldout_exact_overlap'
        if self.heldout.match(dedup_text(source)) or self.heldout.match(dedup_text(target)):
            return 'official_heldout_lexical_near_overlap'
        if self.near.match(dedup_text(source)) or self.near.match(dedup_text(target)):
            return 'lexical_near_duplicate_of_seed_or_previous_candidate'
        return None

    def import_seeds(self):
        """Bulk insert deterministic frozen seeds; restart safely after interruption."""
        accepted = Counter()
        rejected = Counter()
        seen = set()
        source_targets = {}
        records = []
        print('Importing frozen selected candidates', flush=True)
        for line in self.args.selected.open():
            row = json.loads(line)
            category = row['category']
            if category not in self.quotas:
                raise ValueError('unknown_seed_category')
            if row['dataset'] != 'Gemma4' and public_label_evidence(row) is None:
                raise ValueError('public_seed_missing_explicit_native_label:'+row['id'])
            if row['selection_state'] != 'selected_pending_human_review':
                raise ValueError('seed_not_selected_pending_review')
            if accepted[category] >= self.quotas[category]:
                raise ValueError('selected_seed_exceeds_balanced_cap')
            key = normalize_source(row['input'])
            if key in source_targets:
                raise ValueError('duplicate_or_conflicting_selected_seed_source')
            source_targets[key] = row['reference']
            row.update(origin='imported_prior_selected_candidate',
                       imported_from=str(self.args.selected.resolve()),
                       original_quality_basis=row.get('quality_basis'),
                       current_balanced_config_hash=self.config_hash,
                       human_review_status='pending', training_ready=False)
            records.append((row['id'], category, row['group_id'], row['input'], key,
                row['reference'], json.dumps(row,ensure_ascii=False), 'accepted'))
            seen.add(row['id']); accepted[category] += 1
            self.index_pair(row['id'], row['input'], row['reference'])
        print('Importing completed D02 supplement up to balanced cap', flush=True)
        for line in self.args.d02_snapshot.open():
            raw = json.loads(line)
            identifier = 'Gemma4:'+raw['id']
            if identifier in seen:
                continue
            if accepted['D02'] >= self.quotas['D02']:
                rejected['D02_quota_excess_reserved_in_original_ledger'] += 1
                continue
            source, target, payload = raw['source'], raw['target'], raw['payload']
            if raw['category'] != 'D02' or not audit_pass(payload.get('audit',{}), target):
                rejected['D02_saved_audit_does_not_pass'] += 1
                continue
            rebuilt = base.construct_validate(payload['raw_teacher_sample'], 'D02', payload['plan'])
            if rebuilt['input'] != source or rebuilt['reference'] != target:
                rejected['D02_pair_reconstruction_mismatch'] += 1
                continue
            key = normalize_source(source)
            if key in source_targets:
                rejected['D02_duplicate_or_conflicting_input'] += 1
                continue
            reason = self.pair_exclusion(source,target)
            if reason:
                rejected[reason] += 1
                continue
            row = {'id':identifier,'category':'D02','dataset':'Gemma4',
                   'group_id':'Gemma4:'+raw['group_id'],'input':source,'reference':target,
                   'origin':'imported_completed_D02_supplement','legacy_payload':payload,
                   'imported_from':str(self.args.d02_snapshot.resolve()),
                   'quality_basis':'saved_model_audit_and_program_revalidation',
                   'human_review_status':'pending','training_ready':False,
                   'current_balanced_config_hash':self.config_hash}
            records.append((identifier,'D02',row['group_id'],source,key,target,
                            json.dumps(row,ensure_ascii=False),'accepted'))
            source_targets[key]=target;seen.add(identifier);accepted['D02']+=1
            self.index_pair(identifier,source,target)
        with self.ledger._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            now=datetime.now(timezone.utc).isoformat()
            for offset in range(0,len(records),512):
                db.executemany('''INSERT OR IGNORE INTO samples
                    (id,category,group_id,source,source_norm,target,payload_json,status,reason,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,NULL,?,?)''',
                    [record+(now,now) for record in records[offset:offset+512]])
            # New generations never reuse historical generated sequence IDs.
            db.execute("UPDATE counters SET value=max(value,1000000) WHERE key='sequence'")
            db.commit()
        actual = self.ledger.stats()
        if any(actual[c]['accepted'] != accepted[c] for c in self.quotas):
            raise ValueError('seed_import_counts_mismatch')
        base.atomic_json(self.out/'seed_import_summary.json',{
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'imported_by_category':dict(accepted),'imported_total':sum(accepted.values()),
            'D02_supplement_exclusions':dict(rejected),'human_reviewed_total':0,
            'training_ready':False,'original_ledgers_modified':False,
            'seed_source_hashes':self.config['seed_inputs']})
        self.near = base.NearIndex()
        print('Seed import complete',dict(accepted),flush=True)

    def commit_record(self, record):
        if record['verdict'] == 'pending_audit':
            self.ledger.defer_audit(record['id'], record)
            self.audit_deferred=True
            return 'pending_audit'
        source,target=record.get('input',''),record.get('reference','')
        record['origin']='balanced_coverage_gap_synthetic_v1'
        record['dataset']='Gemma4'
        record['quality_basis']='generation_plan_program_checks_and_separate_teacher_audit'
        record['training_ready']=False
        # Preserve same-source conflict detection: do not hide conflicts behind
        # near-duplicate rejection. Ledger invalidates both incompatible targets.
        with self.ledger._connect() as db:
            exact=db.execute("SELECT id,target FROM samples WHERE source_norm=? AND status IN ('accepted','conflict')",(normalize_source(source),)).fetchall()
        if record['verdict']=='passed' and not exact:
            reason=self.pair_exclusion(source,target)
            if reason:
                record.update(verdict='rejected',rejection_reason=reason)
        result=self.ledger.record_candidate(record['id'],record['category'],record['group_id'],
            source,target,record,record['verdict'],record.get('rejection_reason'))
        if result=='accepted':
            self.index_pair(record['id'],source,target)
        return result

    def status(self,state='running_balanced_gaps',error=None,force=False):
        now=time.monotonic()
        if not force and now-self.last_status<15:
            return
        self.last_status=now
        stats=self.ledger.stats()
        accepted=sum(stats[c]['accepted'] for c in self.categories)
        value={'status':state,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'stats':stats,'target':self.scope_target,'target_per_class':self.args.target_per_class,
            'mixed_dataset_target':self.args.target_per_class*17,
            'target_basis':'balanced_selected_seeds_plus_audited_deduplicated_new_candidates',
            'active_categories':self.categories,'gpu_ids':[0],'workers':self.args.workers,
            'scope_machine_passed_pending_human_review':accepted,
            'selected_or_machine_passed_pending_human_review':stats['total']['accepted'],
            'machine_passed_pending_human_review':stats['total']['accepted'],
            'new_since_process_start':stats['total']['accepted']-self.initial_count,
            'accepted_per_hour_this_run':round((accepted-self.initial_scope_count)/max(now-self.started,1)*3600,1),
            'error':error,'training_started':False,'human_reviewed_total':0,
            'config_hash':self.config_hash,
            'remaining_by_active_category':{c:max(0,self.quotas[c]-stats[c]['accepted']) for c in self.categories}}
        base.atomic_json(self.out/'progress.json',value)
        md=['# 八类均衡补缺：生成状态','','> 导入的原生标注候选与新模型审核候选均待人工复核；计数不等于人工合格量。','',
            f'状态：{state}；GPU0单卡{self.args.workers}路；GPU1停用。',
            '当前生成：'+', '.join(self.categories)+'。',
            f'目标每类{self.args.target_per_class}条，导入及新保留共{stats["total"]["accepted"]}条；本次启动后净增{value["new_since_process_start"]}条。',
            '一类达到目标后自动跳过，继续其余未满类别。A01/C02/C03/C04先处理有明确标签的公开数据。','',
            '| 小类 | 名称 | 目标 | 已保留候选 | 缺口 | 当前路线 |','| --- | --- | ---: | ---: | ---: | --- |']
        for c in CATEGORIES:
            n=stats[c]['accepted']
            route='持续生成补缺' if c in self.categories and n<self.quotas[c] else '已足量，停止新增' if n>=self.quotas[c] else '公开明确标签候选先筛选'
            md.append(f'| {c} | {NAMES[c]} | {self.quotas[c]} | {n} | {max(0,self.quotas[c]-n)} | {route} |')
        md+=['',f'本轮新增速度：{value["accepted_per_hour_this_run"]}条／小时；刚启动时不宜外推。','原始旧账本和严格标签筛选文件保留，新进度写入本目录。未开始微调。','']
        if error: md+=['需要处理：'+error,'']
        temp=self.out/'运行状态.md.tmp';temp.write_text('\n'.join(md));temp.replace(self.out/'运行状态.md')

    def run(self):
        """One coordinator owns every class cap and drains submitted work."""
        import signal
        old={sig:signal.signal(sig,lambda *_:self.stop_event.set())
             for sig in (signal.SIGTERM,signal.SIGINT)}
        self.status(force=True)
        try:
            if not self.retry_deferred():
                return
            index=0;inflight=Counter();pending={}
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                while True:
                    stats=self.ledger.stats()
                    while len(pending)<self.args.workers and not(self.audit_deferred or self.stop_event.is_set()):
                        selected=None
                        for _ in self.categories:
                            category=self.categories[index%len(self.categories)];index+=1
                            if stats[category]['accepted']+inflight[category]<self.quotas[category]:
                                selected=category;break
                        if selected is None:
                            break
                        n=min(self.args.batch_size,self.quotas[selected]-stats[selected]['accepted']-inflight[selected])
                        seq=self.ledger.next_sequence()
                        future=pool.submit(self.batch,selected,seq,n)
                        pending[future]=(selected,n);inflight[selected]+=n
                    if not pending:
                        if self.audit_deferred:
                            self.status('waiting_teacher','Audit candidates saved for retry.',force=True)
                        elif self.stop_event.is_set():
                            self.status('stopped_after_draining',force=True)
                        else:
                            self.export_review(final=True)
                            self.status('balanced_gaps_complete_pending_human_review',force=True)
                        return
                    done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        category,n=pending.pop(future);inflight[category]-=n
                        result=future.result()
                        if result.get('batch_error'):
                            self.failure_streak+=1
                            print('BATCH_ERROR',category,result['batch_error'],flush=True)
                            if self.failure_streak>=8:
                                raise RuntimeError('Repeated API/output failure; inspect teacher')
                            continue
                        self.failure_streak=0
                        for record in result['rows']:
                            disposition=self.commit_record(record)
                            print(record['id'],disposition,record.get('rejection_reason'),flush=True)
                        fresh=self.ledger.stats()[category]
                        initial=self.initial_stats[category]
                        if fresh['total']-initial['total']>=200 and fresh['accepted']==initial['accepted']:
                            raise RuntimeError('No accepted new candidates after 200 attempts in '+category)
                        if fresh['total']-initial['total']>max(200,self.quotas[category]-initial['accepted'])*8 and fresh['accepted']<self.quotas[category]:
                            raise RuntimeError('Attempt budget exceeded in '+category)
                    self.status()
                    if self.ledger.stats()['total']['accepted']-self.last_export>=1000:
                        self.export_review()
        except BaseException as error:
            self.status('needs_attention',str(error)[:400],force=True)
            raise
        finally:
            for sig,handler in old.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selected',type=Path,required=True)
    parser.add_argument('--d02-snapshot',type=Path,required=True)
    parser.add_argument('--public-root',type=Path,required=True)
    parser.add_argument('--fcgec-test',type=Path,required=True)
    parser.add_argument('--api',default='http://127.0.0.1:8014/v1')
    parser.add_argument('--model',default='gemma4-31b-teacher')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--per-server-limit',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--target-per-class',type=int,default=5882)
    parser.add_argument('--seed',type=int,default=2026102502)
    parser.add_argument('--request-timeout',type=int,default=900)
    parser.add_argument('--stop-after',type=int,default=0)
    parser.add_argument('--categories',default=','.join(DEFAULT_ACTIVE))
    args=parser.parse_args()
    if not 1<=args.workers<=32 or not 1<=args.per_server_limit<=32 or not 1<=args.batch_size<=4:
        parser.error('invalid worker/request/batch limits')
    if args.target_per_class!=5882 or args.stop_after<0:
        parser.error('this balanced run requires target-per-class 5882')
    if args.api!='http://127.0.0.1:8014/v1':
        parser.error('only existing GPU0 loopback service is supported')
    requests.get('http://127.0.0.1:8014/health',timeout=10).raise_for_status()
    models=requests.get(args.api+'/models',timeout=10);models.raise_for_status()
    if args.model not in {r['id'] for r in models.json()['data']}:
        raise ValueError('teacher_model_name_mismatch')
    runner=BalancedGapRunner(args)
    runtime={'started_utc':datetime.now(timezone.utc).isoformat(),
        'config_hash':runner.config_hash,'active_gpu_ids':[0],'workers':args.workers,
        'per_server_limit':args.per_server_limit,'active_categories':runner.categories,
        'target_per_class':5882,'scope_maximum':runner.scope_target,
        'selected_seed_total':runner.initial_count,'phase_stop_after_additional':None,
        'public_label_policy':'explicit_matching_original_subclass_label_required',
        'source_ledgers_unchanged':True,'training_started':False}
    base.atomic_json(runner.out/'runtime_current.json',runtime)
    with (runner.out/'runtime_history.jsonl').open('a') as stream:
        stream.write(json.dumps(runtime,ensure_ascii=False)+'\n')
    base.requests=EndpointTransport(args.api,[args.api],args.per_server_limit,
        source_hash=file_hash(Path(__file__)),request_timeout=args.request_timeout)
    print(json.dumps(runtime,ensure_ascii=False),flush=True)
    runner.run()


if __name__=='__main__':
    main()
