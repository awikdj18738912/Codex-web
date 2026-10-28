"""Select coverage gaps without changing existing candidate provenance or labels."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import signal
import threading
import time

import generate_refiner_100k as base


class ScopedRunner(base.Runner):
    def __init__(self, args):
        super().__init__(args)
        self.categories = list(dict.fromkeys(args.categories.split(',')))
        if not self.categories or set(self.categories) - set(base.QUOTAS):
            raise ValueError('unknown_or_empty_generation_categories')
        stats = self.ledger.stats()
        self.initial_scope_count = sum(stats[c]['accepted'] for c in self.categories)
        self.scope_target = sum(base.QUOTAS[c] for c in self.categories)
        self.stop_event = threading.Event()

    def status(self, state='running_coverage_gaps', error=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_status < 15:
            return
        self.last_status = now
        stats = self.ledger.stats()
        accepted = sum(stats[c]['accepted'] for c in self.categories)
        elapsed = now - self.started
        value = {
            'status': state, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'stats': stats, 'target': self.scope_target,
            'target_basis': 'upper bound inherited for selected gaps; not a fixed final mix allocation',
            'active_categories': self.categories,
            'scope_machine_passed_pending_human_review': accepted,
            'machine_passed_pending_human_review': stats['total']['accepted'],
            'mixed_dataset_target': 100000, 'public_dataset_accepted': None,
            'percent': round(accepted / max(self.scope_target, 1) * 100, 3),
            'run_elapsed_seconds': round(elapsed, 1),
            'accepted_per_hour_this_run': round((accepted-self.initial_scope_count)/max(elapsed, 1)*3600, 1),
            'phase_stop_after_additional': self.args.stop_after or None,
            'error': error, 'training_started': False, 'config_hash': self.config_hash,
        }
        base.atomic_json(self.out / 'progress.json', value)
        md = ['# 公开数据优先：定向补缺运行状态', '',
              '> 机器通过仍待人工复核。原config.json保留历史生成规则和上限，当前范围以runtime_current.json为准。', '',
              f'状态：{state}；仅生成：{", ".join(self.categories)}。',
              f'当前范围已保留{accepted}条；历史全部类别共保留{stats["total"]["accepted"]}条。',
              '混合总规模暂按10万条规划；公开数据合格量尚未确认，不将原始下载条数当成可用量。',
              f'选定类别历史上限合计{self.scope_target}条，并非必须生成的最终数量；公开数据覆盖后应扣减。',
              f'本次新增停止阈值：{self.args.stop_after or "未设阶段阈值"}。', '',
              '| 小类 | 处理路线 | 机器通过 | 拒收 | 待恢复审核 |',
              '| --- | --- | ---: | ---: | ---: |']
        for category, stat in stats.items():
            if category == 'total':
                continue
            route = '本次定向补充' if category in self.categories else '历史保留；暂停新增，优先公开数据'
            md.append(f'| {category} | {route} | {stat["accepted"]} | {stat["rejected"]} | {stat["pending_audit"]} |')
        md += ['', f'本次新增速度：{value["accepted_per_hour_this_run"]}条/小时；刚启动时不宜外推。',
               '未启动训练。人工抽样、公开数据适配和跨来源去重仍须完成。', '']
        if error:
            md += ['需要处理：' + error, '']
        tmp = self.out / '运行状态.md.tmp'
        tmp.write_text('\n'.join(md))
        tmp.replace(self.out / '运行状态.md')

    def run(self):
        # TERM/INT drain existing batches and commit their results before stopping.
        old_handlers = {sig: signal.signal(sig, lambda *_: self.stop_event.set())
                        for sig in (signal.SIGTERM, signal.SIGINT)}
        self.status(force=True)
        try:
            if not self.retry_deferred():
                return
            index = 0
            in_flight = Counter()
            pending = {}
            phase_target = min(self.scope_target, self.initial_scope_count + self.args.stop_after) if self.args.stop_after else self.scope_target
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                while True:
                    stats = self.ledger.stats()
                    accepted = sum(stats[c]['accepted'] for c in self.categories)
                    full = accepted >= self.scope_target
                    stop_phase = accepted >= phase_target
                    while len(pending) < self.args.workers and not (full or stop_phase or self.audit_deferred or self.stop_event.is_set()):
                        selected = None
                        for _ in self.categories:
                            category = self.categories[index % len(self.categories)]
                            index += 1
                            if stats[category]['accepted'] + in_flight[category] < base.QUOTAS[category]:
                                selected = category
                                break
                        if selected is None:
                            break
                        n = min(self.args.batch_size, base.QUOTAS[selected]-stats[selected]['accepted']-in_flight[selected])
                        sequence = self.ledger.next_sequence()
                        future = pool.submit(self.batch, selected, sequence, n)
                        pending[future] = (selected, n)
                        in_flight[selected] += n
                    if not pending:
                        if self.audit_deferred:
                            self.status('waiting_teacher', 'Audit candidates saved for retry.', force=True)
                        elif self.stop_event.is_set():
                            self.status('stopped_after_draining', force=True)
                        else:
                            # Exporting all historical candidates as pending review is safe;
                            # scope completion does not imply the mixed dataset is ready.
                            self.export_review(final=False)
                            self.status('coverage_gaps_stage_finished', force=True)
                        return
                    done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    for future in done:
                        category, n = pending.pop(future)
                        in_flight[category] -= n
                        result = future.result()
                        if result.get('batch_error'):
                            self.failure_streak += 1
                            print('BATCH_ERROR', category, result['batch_error'], flush=True)
                            if self.failure_streak >= 8:
                                raise RuntimeError('Repeated API/output failure; inspect teacher before retry')
                            continue
                        self.failure_streak = 0
                        for record in result['rows']:
                            disposition = self.commit_record(record)
                            print(record['id'], disposition, record.get('rejection_reason'), flush=True)
                        fresh = self.ledger.stats()[category]
                        if fresh['total'] >= 200 and fresh['accepted'] == 0:
                            raise RuntimeError('No accepted examples in ' + category)
                        if fresh['total'] > base.QUOTAS[category]*8 and fresh['accepted'] < base.QUOTAS[category]:
                            raise RuntimeError('Attempt budget exceeded in ' + category)
                    self.status()
                    if self.ledger.stats()['total']['accepted'] - self.last_export >= 1000:
                        self.export_review()
        except BaseException as error:
            self.status('needs_attention', str(error)[:400], force=True)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
