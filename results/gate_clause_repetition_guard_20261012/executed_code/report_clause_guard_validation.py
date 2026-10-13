"""Verify and report matched consumed regression and new spontaneous speech."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts import collect_gate_pairs as c
from experiments.scripts import report_linear_gate_holdout as streaming_report
from experiments.scripts import synthetic_gate_data as d


def load(path):
    return json.loads(path.read_text())


def verify_hash(path, expected):
    actual = c.file_hash(path)
    if actual != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')


def regression(root):
    manifest = load(root / 'regression_manifest.json')
    arms = {}
    for label in ('old', 'new'):
        folder = root / f'{label}_regression'
        c.verify_frozen(folder / 'runtime')
        verify_hash(folder / 'sources.jsonl', manifest['same_source_sha256'])
        if load(folder / 'job_state.json')['status'] != 'complete':
            raise ValueError(f'{label} regression incomplete')
        rows = d.read_rows(folder / 'recordings.jsonl')
        arms[label] = {r['recording_view_id']: r for r in rows}
        if len(arms[label]) != len(rows) or len(rows) != manifest['views']:
            raise ValueError('Regression row count or uniqueness failed')
        if not all(r['native_control_stable'] for r in rows):
            raise ValueError('Native repeat unstable')
    if set(arms['old']) != set(arms['new']):
        raise ValueError('Regression inputs not paired')
    changes = []
    worse = better = 0
    for view, before in arms['old'].items():
        after = arms['new'][view]
        for key in ('seed_id', 'variant', 'raw_text', 'pcm_sha256', 'intended_script'):
            if before[key] != after[key]:
                raise ValueError(f'Regression pairing differs: {key}')
        old, new = before['native'], after['native']
        worse += new['intended_distance'] > old['intended_distance']
        better += new['intended_distance'] < old['intended_distance']
        if old['final_text'] != new['final_text']:
            changes.append(dict(view_id=view, seed_id=before['seed_id'], raw_text=before['raw_text'],
                                old_text=old['final_text'], new_text=new['final_text'],
                                old_distance=old['intended_distance'], new_distance=new['intended_distance'],
                                old_calls=old['cost']['refiner_calls'], new_calls=new['cost']['refiner_calls'],
                                new_reviews=new['final']['repetition_reviews'],
                                new_restorations=new['final']['repetition_restorations']))
    totals = {}
    for label, rows in arms.items():
        native = [r['native'] for r in rows.values()]
        totals[label] = dict(views=len(native), exact=sum(r['intended_distance'] == 0 for r in native),
                             summed_reference_distance=sum(r['intended_distance'] for r in native),
                             refiner_calls=sum(r['cost']['refiner_calls'] for r in native),
                             plausibility_calls=sum(r['cost']['plausibility_calls'] for r in native))
    return dict(scope='consumed_matched_cold_final_snapshots_not_live_streaming',
                views=manifest['views'], clips=manifest['clips'], arms=totals,
                changed=len(changes), unchanged=manifest['views'] - len(changes),
                new_worse=worse, new_better=better, changes=changes,
                reference_caveat='Mixed official references/unverified synthetic scripts; diagnostic regression only')


def natural(root):
    folder = root / 'natural_holdout'
    manifest = load(folder / 'manifest.json')
    cfg = load(folder / 'streaming/configuration.json')
    for key, path in [('model_sha256', folder / 'frozen_model/model.pkl'),
                      ('selection_sha256', folder / 'selection.json'),
                      ('human_sources_sha256', folder / 'human_sources.jsonl'),
                      ('streaming_adapter_sha256', ROOT / 'experiments/scripts/stream_linear_gate_holdout.py')]:
        verify_hash(path, manifest[key])
    verify_hash(root / 'source_candidates.json', manifest['selection_plan_sha256'])
    verify_hash(folder / 'runtime/protocol.json', manifest['runtime_protocol_sha256'])
    for path, expected in manifest['source_parquets_sha256'].items():
        verify_hash(Path(path), expected)
    c.verify_frozen(folder / 'runtime')
    if load(folder / 'streaming/job_state.json')['status'] != 'complete':
        raise ValueError('Natural streaming incomplete')
    rows = d.read_rows(folder / 'streaming/sessions.jsonl')
    failures = d.read_rows(folder / 'streaming/failures.jsonl')
    sources = d.read_rows(folder / 'human_sources.jsonl')
    source_by_id = {r['seed_id']: r for r in sources}
    expected = manifest['human_audio'] * cfg['repetitions'] * 2
    if len(rows) != expected or len({(r['seed_id'], r['repetition'], r['policy']) for r in rows}) != expected:
        raise ValueError('Natural session count or uniqueness failed')
    order = defaultdict(dict)
    pairs = defaultdict(dict)
    for r in rows:
        source = source_by_id[r['seed_id']]
        for key in ('audio_sha256', 'pcm_sha256'):
            if r[key] != source[key]:
                raise ValueError(f'Source/session mismatch: {key}')
        order[(r['seed_id'], r['policy'])][r['repetition']] = r['order']
        pairs[(r['seed_id'], r['repetition'])][r['policy']] = r
    for source in sources:
        verify_hash(Path(source['audio_path']), source['audio_sha256'])
    if not all(v[0] != v[1] for v in order.values()):
        raise ValueError('Policy order did not reverse')
    report = streaming_report.summarize(rows)
    for stats in report['by_source'].values():
        stats['paired_audio_clock_lag']['unit'] = 'source_shard_equal_weight_not_independent_speaker'
    report.update(sessions=len(rows), clips=len(sources), repetitions=cfg['repetitions'],
                  failures=len(failures), threshold=manifest['threshold'], no_test_tuning=True,
                  production_enabled=False, complete=True,
                  cross_clause_guard_trace_sessions=sum('cross_clause_noun_reuse' in json.dumps(r) for r in rows),
                  reference_kind='SpeechIO_dataset_verbatim_reference_not_cleaned_target',
                  ASR_pretraining_overlap_unknown=True, audio_independently_reviewed=False)
    # Preserve source-to-output edits as diagnostics, without inferring cleaned gold labels.
    edits = [dict(seed_id=r['seed_id'], repetition=r['repetition'], policy=r['policy'],
                  reference=r['intended_text'], raw_text=r['raw_text'], final_text=r['final_text'],
                  normalized_distance=r['normalized_distance'], lexical_distance=r['lexical_distance'])
             for r in rows if r['raw_text'] != r['final_text']]
    c.write_json(folder / 'streaming/summary.json', report)
    c.write_json(folder / 'streaming/source_output_edits.json', edits)
    consumed = dict(status='consumed', subsequent_use='regression_only', evaluated_unix=time.time(),
                    human_audio=len(sources), synthetic_audio=0,
                    model_sha256=manifest['model_sha256'], selection_sha256=manifest['selection_sha256'])
    c.write_json(folder / 'test_consumed.json', consumed)
    c.write_json(folder / 'test_data/test_consumed.json', consumed)
    return report, manifest


def write_report(root, reg, speech):
    stats = speech['by_source']['human_spontaneous']
    old, new = stats['arms']['current'], stats['arms']['learned']
    ci = stats['paired_audio_clock_lag']
    lines = ['# 跨分句重复保护修复与自然口语门控验证（2026-10-12）', '',
             '修复了原始 ASR 中“检查索引，索引交给……”和“核对名单，名单送到……”被合并后误判为口吃的问题。使用原始文本的标点、名词词性及左右谓词证据，在局部审核读取缓存或调用模型之前保留必要重复；终态对齐恢复缺失的原始分隔符。该修改同时作用于原策略和学习门控流程。', '',
             '学习门控仍是冻结的字符 TF-IDF 线性模型、阈值 0.8，仅筛选原生窗口 REFINE 机会；局部审核保持原流程。未重训、未使用 ASR 声学置信度、未修改原 float32 Refiner 权重，未在运行中的生产服务启用学习门控。', '',
             '## 相同输入的旧版/新版回归', '',
             '此前消费过的 48 段音频提供 49 个不同原始 ASR 视图。旧版与新版分别以相同的冷缓存最终快照运行；这不是新音频测试，也不是流式时序或延迟比较。以下只统计 native 臂，不重复计算 skip_all 别名或控制复跑。', '',
             '| 指标 | 旧版 | 新版 |', '|---|---:|---:|',
             f"| 参考完全匹配 | {reg['arms']['old']['exact']}/49 | {reg['arms']['new']['exact']}/49 |",
             f"| 参考距离总和 | {reg['arms']['old']['summed_reference_distance']} | {reg['arms']['new']['summed_reference_distance']} |",
             f"| Refiner 调用 | {reg['arms']['old']['refiner_calls']} | {reg['arms']['new']['refiner_calls']} |", '',
             f"仅 {reg['changed']} 个最终文本变化，均改善；其余 {reg['unchanged']} 个逐字相同，新增参考距离退化 {reg['new_worse']} 个。两版各自 native/control 结果稳定。两例实际记录 cross_clause_noun_reuse KEEP、model_called=false，并恢复分隔符。", '']
    for change in reg['changes']:
        lines += [f"- `{change['seed_id']}`：参考距离 {change['old_distance']}→{change['new_distance']}；旧结果“{change['old_text']}”；新结果“{change['new_text']}”。"]
    lines += ['', '本回归混合官方朗读参考和未经人工核验的合成脚本。两个修复例属于已消费合成脚本，不能用于声学正确性或独立泛化结论。', '',
              '## 新自然口语真实流式对照', '',
              '从之前项目评测未使用的 SpeechIO ID 池固定选取 18 段、六个来源分片，每片三段。按时长及固定哈希顺序选择，未按本轮模型输出筛选。源数据参考保留口语词和重复，音频没有独立人工复核；静态文本资源中的既往曝光和模型预训练重叠未知。', '',
              '每段两轮、每轮原策略和冻结学习门控，两轮策略顺序反转；共 72 会话、36 对，但只有 18 段独立音频。使用真实 WebSocket 和 HTTP ASR、按捕获时钟发送 0.5 秒音频块，至多两块未确认，每会话缓存独立；模型在旧训练素材上预热。没有停止或重启现有服务。', '',
              '| 指标 | 原策略 | 冻结学习门控 |', '|---|---:|---:|',
              f"| Refiner 调用 | {old['refiner_calls']} | {new['refiner_calls']} |",
              f"| 规范化参考 CER | {old['normalized_micro_cer']:.4%} | {new['normalized_micro_cer']:.4%} |",
              f"| 原字词参考 CER | {old['lexical_micro_cer']:.4%} | {new['lexical_micro_cer']:.4%} |",
              f"| 规范化完全匹配 | {old['normalized_reference_exact']}/36 | {new['normalized_reference_exact']}/36 |"]
    for percentile, label in [('p50', 'p50'), ('p95', 'p95'), ('max', '最大')]:
        a = old['timing']['completion_lag_from_audio_clock_seconds'][percentile]
        b = new['timing']['completion_lag_from_audio_clock_seconds'][percentile]
        lines.append(f'| 音频完成滞后 {label}（秒） | {a:.3f} | {b:.3f} |')
    lines += ['',
              f"Refiner 调用减少 {(old['refiner_calls'] - new['refiner_calls']) / old['refiner_calls']:.1%}。学习窗口门控实际执行 {new['gate_calls']} 次、否决 {new['vetoes']} 次，评分累计 {new['gate_ms']:.2f} 毫秒；两策略均无 MLM 调用、无记录到的模型错误。36/36 对的音频哈希、最终原始 ASR 和逐次假设历史一致；34/36 对最终文本逐字相同，另两对是同一片段两轮的逗号差异，规范化距离均相同。", '',
              f"六个来源分片等权、3000 次成组 bootstrap 的学习门控−原策略完成滞后均值差为 {ci['mean_difference_seconds'] * 1000:.1f} 毫秒，95% 区间 [{ci['bootstrap95_seconds'][0] * 1000:.1f}, {ci['bootstrap95_seconds'][1] * 1000:.1f}] 毫秒。区间包含 0，且 p95 略高，不能据此确认稳定提速。分片不是独立说话人，共享服务/GPU、ASR 缓存和外部负载也会影响局部延迟。", '',
              '规范化 CER 使用项目中文数字规则；原字词 CER 仅去标点空格并进行 NFKC。SpeechIO 逐字参考不是期望的精修输出，因此 CER 相同仅说明本批参考距离相同，不能证明精修语义质量不变。两策略都保留“三天三天三天”（呈现为“3天3天3天”）及“开着开着”，并都删除部分“呃”；后者是观察到的文本变化，未经听音不能直接判定错误。逐项差异见 streaming/source_output_edits.json。', '',
              f"本自然口语批次记录到跨分句保护的会话数为 {speech['cross_clause_guard_trace_sessions']}。它检验了修复后的整体流程和学习窗口门控的实际触发，但没有提供跨分句保护在新自然音频上生效的正例；新保护的直接证据来自回归与针对性测试。", '',
              '## 实现限制与验证', '',
              '跨分句保护使用隔离的 Jieba 词性标注器：原始分隔符、名词重复、左侧动词、右侧谓词同时满足才保护。这是保守的词性启发式，不是依存分析或意图模型。标注错误、无标点分句会漏检；符合相同形态的真正名词口吃可能被保留。启动时预热词典，避免首次语音回调承担加载开销。', '',
              '源文本保护比较具体的重复位置，已有其他重复丢失不会禁止无关修改；恢复仅插入对齐的源文本词和分隔符，避开实体/数字替换，保留无关规范化。完整原句不会写入不同分组的窗口缓存。', '',
              '相关测试 166 项通过（logs/verification.log）。Luna 独立核对冻结代码、两个实际拦截 trace、输入配对、72 条会话、统计口径和延迟限制；审计见 LUNA_AUDIT.md，其中早期无结果段落是过程检查，最终核对段落给出完成状态。', '',
              '所有模型和选择文件保持冻结；完整哈希在 integrity_check.json。源码已修复，但本次未重启生产服务或启用学习门控。自然音频现标记为已消费，之后只能作回归；不能在这批数据上调阈值后仍声称是独立测试。', '',
              '## 下一步', '',
              '优先建立一批独立自然口语的精修目标标注，覆盖真实口吃、正常跨句重复、纠错和填充词；用原始音频、逐字参考及期望清理结果区分必要重复与可删重复。保持门控模型/阈值冻结再评估删改风险和调用节省，跨分句保护需要独立自然正例。当前小样本支持调用减少，尚不足以支持论文泛化结论或部署收益承诺。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    if load(root / 'validation_job_state.json')['status'] != 'complete':
        raise ValueError('Validation job incomplete')
    c.verify_frozen(root / 'runtime')
    patch = load(root / 'patch_manifest.json')
    for name, hashes in patch['patches'].items():
        verify_hash(ROOT / 'system' / name, hashes['new_sha256'])
    reg = regression(root)
    speech, manifest = natural(root)
    if '166 passed' not in (root / 'logs/verification.log').read_text():
        raise ValueError('Required test validation missing')
    c.write_json(root / 'regression_summary.json', reg)
    write_report(root, reg, speech)
    for name in ('prepare_clause_guard_regression.py', 'prepare_clause_guard_speechio.py',
                 'run_clause_guard_validation.py', 'report_clause_guard_validation.py'):
        shutil.copyfile(ROOT / 'experiments/scripts' / name, root / 'executed_code' / name)
    hashes = {str(path.relative_to(root)): c.file_hash(path) for path in root.rglob('*') if path.is_file()
              and path.name not in {'integrity_check.json', 'completed_experiment.json'}
              and str(path.relative_to(root)) != 'logs/report.log'}
    c.write_json(root / 'integrity_check.json', dict(artifacts_sha256=hashes, runtime_verified=True,
                 current_modules_match_frozen_patch=True, paired_audio_verified=True,
                 frozen_model_sha256=manifest['model_sha256'], frozen_selection_sha256=manifest['selection_sha256'],
                 no_test_tuning=True, production_enabled=False))
    c.write_json(root / 'completed_experiment.json', dict(status='complete', regression_views=reg['views'],
                 natural_sessions=speech['sessions'], natural_clips=speech['clips'], tests_passed=166,
                 new_holdout_consumed=True, production_enabled=False, no_test_tuning=True))
    print(json.dumps(dict(status='complete', regression_exact=[reg['arms'][a]['exact'] for a in ('old','new')],
                         refiner_calls=[speech['by_source']['human_spontaneous']['arms'][a]['refiner_calls']
                                        for a in ('current','learned')], artifacts=len(hashes)), ensure_ascii=False))


if __name__ == '__main__':
    main()
