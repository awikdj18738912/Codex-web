"""Build a descriptive report from the completed AASR-Bench text-only run."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/aasr_bench_refiners_gpu0_20261028'
VARIANTS = ('lora', 'lora_gate', 'full', 'full_gate', 'agentic1b')
NAMES = {
    'lora': 'LoRA 0.5B', 'lora_gate': 'LoRA 0.5B＋门控',
    'full': '全量 0.5B', 'full_gate': '全量 0.5B＋门控',
    'agentic1b': '原配置 AgenticASR 1B',
}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def exact_counts(rows):
    """Report literal agreement against both raw clean and normalized clean."""
    return {
        'n': len(rows),
        'raw_clean_exact': sum(r['output'] == r['original_reference'] for r in rows),
        'normalized_clean_exact': sum(r['output'] == r['reference'] for r in rows),
        'raw_clean_exact_rate': (sum(r['output'] == r['original_reference'] for r in rows) / len(rows)) if rows else None,
        'normalized_clean_exact_rate': (sum(r['output'] == r['reference'] for r in rows) / len(rows)) if rows else None,
    }


def ms_mean(rows, key):
    return statistics.mean(r[key] * 1000 for r in rows) if rows else None


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def main():
    progress = json.loads((OUT / 'progress.json').read_text(encoding='utf-8'))
    if progress.get('phase') != 'complete':
        raise RuntimeError(f"Benchmark is not complete: {progress}")
    manifest = read_jsonl(OUT / 'text_manifest.jsonl')
    predictions = read_jsonl(OUT / 'predictions.jsonl')
    timings = read_jsonl(OUT / 'batch_timings.jsonl')
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert len(manifest) == 510 and len({r['id'] for r in manifest}) == 510
    assert len(predictions) == 2710, len(predictions)
    assert len(timings) == 480, len(timings)

    inputs = {r['id']: r for r in manifest}
    pred_by = {}
    for mode in ('latency', 'throughput'):
        for variant in VARIANTS:
            group = [r for r in predictions if r['mode'] == mode and r['variant'] == variant]
            expected = 32 if mode == 'latency' else 510
            assert len(group) == expected, (mode, variant, len(group))
            assert len({r['id'] for r in group}) == expected
            pred_by[mode, variant] = {r['id']: r for r in group}

    by_timing = defaultdict(list)
    for row in timings:
        by_timing[row['mode'], row['variant']].append(row)
    summary = {
        'benchmark': 'AASR-Bench Chinese oral text-only adaptation',
        'official_rubric_run': False,
        'official_rubric_file': str(OUT / 'rubric_scores_by_variant.json'),
        'dataset': {'n': len(manifest), 'scenes': dict(Counter(r['scene'] for r in manifest)),
                    'passthrough_n': sum(r['oral'] == r['clean'] for r in manifest),
                    'changed_n': sum(r['oral'] != r['clean'] for r in manifest)},
        'pipelines': {}, 'gate_comparisons': {}, 'numeric_guard_losses': {},
        'protocol': {
            'input': 'Pure text: official oral transcript is supplied directly; audio is not transcribed.',
            'quality_warning': 'Raw and rule-normalized exact matches are descriptive only, not official rubric scores.',
            'execution': protocol.get('concurrency'),
            'precision': protocol.get('refiner_dtype'),
            'rule_pack': protocol.get('rule_pack'),
            'lora': protocol.get('lora'),
            'agentic1b': protocol.get('agentic1b_pipeline'),
        },
    }
    rubric_scores_path = OUT / 'rubric_scores_by_variant.json'
    rubric_result = None
    if rubric_scores_path.exists():
        rubric_result = json.loads(rubric_scores_path.read_text(encoding='utf-8'))
        if rubric_result.get('status') == 'complete':
            scores = rubric_result.get('scores', {})
            required_categories = {'content', 'format', 'filter', 'rephrase'}
            for variant in VARIANTS:
                if variant not in scores:
                    raise ValueError(f'Official rubric result is missing variant: {variant}')
                result = scores[variant]
                run = result.get('run', {})
                overall = result.get('overall', {})
                if run.get('completed_sample_count') != 510:
                    raise ValueError(f'{variant}: rubric completed_sample_count is not 510')
                if overall.get('question_count') != 3416:
                    raise ValueError(f'{variant}: overall question_count is not 3416')
                categories = result.get('by_category', {})
                if not required_categories.issubset(categories):
                    raise ValueError(f'{variant}: rubric category summaries are incomplete')
                for category in required_categories:
                    if not isinstance(categories[category].get('normalized_score'), (int, float)):
                        raise ValueError(f'{variant}: missing normalized score for {category}')
            summary['official_rubric_run'] = True
            summary['official_rubric_file'] = str(rubric_scores_path)
            summary['official_rubric_qualification'] = rubric_result.get('qualification')
            summary['judge_protocol'] = rubric_result.get('judge_protocol')
            summary['official_rubric_scores'] = scores
        else:
            summary['official_rubric_status'] = rubric_result.get('status', 'incomplete')

    scene_ids = defaultdict(set)
    for row in manifest:
        scene_ids[row['scene']].add(row['id'])
    throughput = {}
    latency = {}
    for variant in VARIANTS:
        latency_rows = by_timing['latency', variant]
        throughput_rows = by_timing['throughput', variant]
        assert len(latency_rows) == 32 and len(throughput_rows) == 64
        lat_ms = [r['batch_seconds'] * 1000 for r in latency_rows]
        total_batch_s = sum(r['batch_seconds'] for r in throughput_rows)
        bulk = pred_by['throughput', variant]
        passthrough = [r for r in bulk.values() if r['oral'] == r['clean']]
        changed = [r for r in bulk.values() if r['oral'] != r['clean']]
        gate = variant.endswith('_gate')
        quality = {
            'all_510': exact_counts(list(bulk.values())),
            'passthrough_76': exact_counts(passthrough),
            'changed_434': exact_counts(changed),
            'by_scene': {},
        }
        for scene, ids in sorted(scene_ids.items()):
            quality['by_scene'][scene] = exact_counts([bulk[i] for i in ids])
        metric = {
            'latency_n': len(lat_ms),
            'latency_mean_ms': statistics.mean(lat_ms),
            'latency_median_ms': statistics.median(lat_ms),
            'latency_p95_ms': percentile(lat_ms, .95),
            'latency_rule_mean_ms': ms_mean(latency_rows, 'rule_seconds'),
            'latency_gate_mean_ms': ms_mean(latency_rows, 'gate_seconds'),
            'latency_refiner_mean_ms': ms_mean(latency_rows, 'refiner_seconds'),
            'latency_numeric_guard_mean_ms': ms_mean(latency_rows, 'numeric_guard_seconds'),
            'throughput_n': len(bulk),
            'throughput_batches': len(throughput_rows),
            'throughput_batch8_seconds_all_510': total_batch_s,
            'throughput_inputs_per_second': len(bulk) / total_batch_s,
            'throughput_mean_batch_completion_seconds': statistics.mean(r['batch_seconds'] for r in throughput_rows),
            'throughput_refiner_calls': sum(bool(r['route']) for r in bulk.values()),
            'throughput_rule_seconds': sum(r['rule_seconds'] for r in throughput_rows),
            'throughput_gate_seconds': sum(r['gate_seconds'] for r in throughput_rows),
            'throughput_refiner_seconds': sum(r['refiner_seconds'] for r in throughput_rows),
            'throughput_numeric_guard_seconds': sum(r['numeric_guard_seconds'] for r in throughput_rows),
            'quality': quality,
            'numeric_fallback_count': sum(bool(r['numeric_fallback']) for r in bulk.values()),
            'numeric_guard_lost_raw_clean_ids': [r['id'] for r in bulk.values()
                if r['numeric_fallback'] and r['model_output'] == r['original_reference']
                and r['output'] != r['original_reference']],
            'numeric_guard_lost_normalized_clean_ids': [r['id'] for r in bulk.values()
                if r['numeric_fallback'] and r['model_output'] == r['reference']
                and r['output'] != r['reference']],
            'non_eos_count': sum(not r['eos'] for r in bulk.values()),
            'gate_skipped_count': sum(not r['route'] for r in bulk.values()) if gate else None,
        }
        summary['pipelines'][variant] = metric
        latency[variant] = pred_by['latency', variant]
        throughput[variant] = bulk
        summary['numeric_guard_losses'][variant] = {
            'n': metric['numeric_fallback_count'],
            'lost_raw_clean_exact_ids': metric['numeric_guard_lost_raw_clean_ids'],
            'lost_normalized_clean_exact_ids': metric['numeric_guard_lost_normalized_clean_ids'],
        }

    gate_cases = []
    for base in ('lora', 'full'):
        gated_name = base + '_gate'
        plain, gated = throughput[base], throughput[gated_name]
        lost, skip_loss, routed_loss, gained, all_diffs = [], [], [], [], []
        norm_lost, norm_skip_loss, norm_routed_loss = [], [], []
        for ident, row in gated.items():
            old = plain[ident]
            if row['output'] == old['output']:
                continue
            old_exact = old['output'] == old['original_reference']
            new_exact = row['output'] == row['original_reference']
            old_norm_exact = old['output'] == old['reference']
            new_norm_exact = row['output'] == row['reference']
            record = {
                'base': base, 'id': ident, 'scene': row['scene'], 'oral': row['oral'],
                'rule_baseline': row['rule_baseline'], 'original_clean': row['original_reference'],
                'normalized_clean': row['reference'], 'ungated_output': old['output'],
                'gated_output': row['output'], 'gated_route': bool(row['route']),
                'ungated_matches_raw_clean': old_exact, 'gated_matches_raw_clean': new_exact,
                'ungated_matches_normalized_clean': old_norm_exact,
                'gated_matches_normalized_clean': new_norm_exact,
                'classification': ('lost_raw_clean_exact' if old_exact and not new_exact else
                                   'gained_raw_clean_exact' if new_exact and not old_exact else
                                   'different_output_same_exact_status'),
            }
            if old_exact and not new_exact:
                lost.append(ident)
                if not row['route']:
                    record['cause_group'] = 'skip_exact_loss'
                    skip_loss.append(ident)
                else:
                    # A routed difference is associated with the gated batch shape; it is
                    # not evidence of a gate miss because the refiner still ran.
                    record['cause_group'] = 'still_routed_batch_shape_difference'
                    routed_loss.append(ident)
            if old_norm_exact and not new_norm_exact:
                norm_lost.append(ident)
                if not row['route']:
                    norm_skip_loss.append(ident)
                else:
                    norm_routed_loss.append(ident)
            elif new_exact and not old_exact:
                gained.append(ident)
                record['cause_group'] = 'gained_raw_clean_exact'
            else:
                record['cause_group'] = 'other_output_difference'
            all_diffs.append(ident)
            gate_cases.append(record)
        summary['gate_comparisons'][base] = {
            'latency_mean_reduction': 1 - summary['pipelines'][gated_name]['latency_mean_ms'] /
                summary['pipelines'][base]['latency_mean_ms'],
            'throughput_total_time_reduction': 1 - summary['pipelines'][gated_name]['throughput_batch8_seconds_all_510'] /
                summary['pipelines'][base]['throughput_batch8_seconds_all_510'],
            'throughput_speedup': summary['pipelines'][gated_name]['throughput_inputs_per_second'] /
                summary['pipelines'][base]['throughput_inputs_per_second'],
            'raw_clean_exact_losses': lost,
            'losses_due_to_skip': skip_loss,
            'losses_while_still_routed_batch_shape_difference': routed_loss,
            'normalized_clean_exact_losses': norm_lost,
            'normalized_losses_due_to_skip': norm_skip_loss,
            'normalized_losses_while_still_routed_batch_shape_difference': norm_routed_loss,
            'raw_clean_exact_gains': gained,
            'different_output_ids': all_diffs,
            'weak_changed_skipped_count': sum(not r['route'] and r['oral'] != r['clean'] for r in gated.values()),
            'routed_outputs_differ_count': sum(r['route'] and r['output'] != plain[i]['output'] for i, r in gated.items()),
        }

    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'gate_differences.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in gate_cases), encoding='utf-8')
    with (OUT / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['variant', 'variant_name', 'latency_mean_ms', 'latency_median_ms', 'latency_p95_ms',
                  'throughput_batch8_seconds_all_510', 'throughput_inputs_per_second',
                  'throughput_refiner_calls', 'raw_clean_exact_510', 'normalized_clean_exact_510',
                  'raw_clean_exact_passthrough_76', 'normalized_clean_exact_passthrough_76',
                  'numeric_fallback_count', 'gate_skipped_count']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, m in summary['pipelines'].items():
            writer.writerow({'variant': variant, 'variant_name': NAMES[variant],
                'latency_mean_ms': m['latency_mean_ms'], 'latency_median_ms': m['latency_median_ms'],
                'latency_p95_ms': m['latency_p95_ms'],
                'throughput_batch8_seconds_all_510': m['throughput_batch8_seconds_all_510'],
                'throughput_inputs_per_second': m['throughput_inputs_per_second'],
                'throughput_refiner_calls': m['throughput_refiner_calls'],
                'raw_clean_exact_510': m['quality']['all_510']['raw_clean_exact'],
                'normalized_clean_exact_510': m['quality']['all_510']['normalized_clean_exact'],
                'raw_clean_exact_passthrough_76': m['quality']['passthrough_76']['raw_clean_exact'],
                'normalized_clean_exact_passthrough_76': m['quality']['passthrough_76']['normalized_clean_exact'],
                'numeric_fallback_count': m['numeric_fallback_count'],
                'gate_skipped_count': m['gate_skipped_count']})

    lines = [
        '# AASR-Bench 中文集合：五种精修方案报告', '',
        '本报告输入是 AASR-Bench 的纯文本 oral 转录，未送入音频，也未运行 ASR。逐字匹配 clean 仅作描述性对照，不是论文官方 rubric 分数。原始 clean 与同规则库归一化后的 clean 均单独报告。', '',
        f"集合共 {len(manifest)} 条、{len(scene_ids)} 个场景；{summary['dataset']['passthrough_n']} 条 oral 与 clean 相同（passthrough），{summary['dataset']['changed_n']} 条不同。五种方案在 GPU 上串行运行。模型为 BF16 Transformers/PyTorch 推理，不是 vLLM；LoRA 在内存中合并。四个 0.5B 方案使用 native v91 数字规则，原配置 1B 不使用数字规则。", '',
        '## 单句响应时间', '',
        '32 条（5 条 passthrough、27 条有改动）逐条 batch=1。该时间包含规则处理、门控（适用时）、refiner 和数字 guard。', '',
        '| 方案 | n | 平均 ms | 中位数 ms | P95 ms | 规则 ms | 门控 ms | refiner ms | 数字 guard ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in summary['pipelines'].items():
        lines.append(f"| {NAMES[name]} | 32 | {m['latency_mean_ms']:.2f} | {m['latency_median_ms']:.2f} | {m['latency_p95_ms']:.2f} | {m['latency_rule_mean_ms']:.3f} | {m['latency_gate_mean_ms']:.3f} | {m['latency_refiner_mean_ms']:.2f} | {m['latency_numeric_guard_mean_ms']:.3f} |")
    lines += ['', '## Batch 8 吞吐与质量', '',
        '510 条完整集合以 batch size 8 处理，最后一批 6 条，共 64 批。批量吞吐表示整批执行的总时间和句/秒，不能等同单句响应时间；批内输出需等待本批生成结束。', '',
        '| 方案 | 510条总批处理秒数 | 句/秒 | refiner调用 | 原始clean逐字匹配 | 归一化clean逐字匹配 | passthrough原始匹配 | passthrough归一化匹配 | 数字回退 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in summary['pipelines'].items():
        q, p = m['quality']['all_510'], m['quality']['passthrough_76']
        lines.append(f"| {NAMES[name]} | {m['throughput_batch8_seconds_all_510']:.2f} | {m['throughput_inputs_per_second']:.2f} | {m['throughput_refiner_calls']}/510 | {q['raw_clean_exact']}/510 | {q['normalized_clean_exact']}/510 | {p['raw_clean_exact']}/76 | {p['normalized_clean_exact']}/76 | {m['numeric_fallback_count']} |")
    lines += ['', '质量统计是字面匹配，不是官方 rubric 质量分。原始 clean 和规则归一化 clean 都列出，以呈现数字规范化差异。逐场景统计和 434 条非 passthrough 子集见 summary.json。', '',
        '当前全量模型在 `recheck.jsonl:201` 出现重复生成，达到1024个新token上限；加门控与不加门控均发生，最后由数字保护回退为规则基线。该异常的完整耗时计入批量总时间，未删除异常样本，也未用重跑替换。它会拉低批量吞吐，不能仅按平均耗时判断模型速度。', '',
        '1B最终使用 `system.refiner.SYSTEM_PROMPT` 的原始输入配置，允许模型规范数字。此前“数字已由规则处理”的离线CLI提示词不适用于1B原始输入对照，已仅补跑1B；首轮仍归档在 `first_pass_1b_offline_cli_prompt`，四个0.5B方案保持不变。', '',
        '## 门控差异', '',
        '| 基础模型 | 单句均值变化 | 510条批处理总时长变化 | 吞吐加速 | 因 skip 损失原始 clean 匹配 | 仍 route、批形状相关差异造成匹配损失 | 弱标签需修改但被 skip |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for base, m in summary['gate_comparisons'].items():
        lines.append(f"| {NAMES[base]} | {m['latency_mean_reduction']:.1%} | {m['throughput_total_time_reduction']:.1%} | {m['throughput_speedup']:.2f}× | {len(m['losses_due_to_skip'])} | {len(m['losses_while_still_routed_batch_shape_difference'])} | {m['weak_changed_skipped_count']} |")
    lines += ['', '“仍 route”说明该输入确实进入了精修模型。若该输出与无门控结果不同，报告将其单列为门控改变子批形状时观察到的差异，不能称为门控漏检。详细 ID 与输出在 gate_differences.jsonl。', '',
        '本轮门控跳过68/510句：66句为 oral 与 clean 相同的干净样本，另2句（`recheck.jsonl:774`、`recheck.jsonl:619`）主要差在百分比或时间写法。它们仍由规则库处理，不能仅因 oral≠clean 就认定漏掉了必须由精修模型解决的问题。该集合多数句子需清理，门控跳过比例约13.3%，与先前人为50/50集合不同。', '',
        '## 数字 guard 回退', '',
        'numeric_fallback 表示模型正文经数字恢复与完整性检查后回退到 rule_baseline。下表给出回退数量，以及回退前 model_output 命中而最终 output 失去匹配的 ID；正文与 raw_output 保存在逐条预测中。', '',
        '| 方案 | 回退数 | 丢失原始 clean 匹配 | 丢失归一化 clean 匹配 |', '| --- | ---: | ---: | ---: |']
    for name, m in summary['pipelines'].items():
        guard = summary['numeric_guard_losses'][name]
        lines.append(f"| {NAMES[name]} | {guard['n']} | {len(guard['lost_raw_clean_exact_ids'])} | {len(guard['lost_normalized_clean_exact_ids'])} |")
    if summary['official_rubric_run']:
        rubric_scores = summary['official_rubric_scores']
        lines += ['', '## 官方 rubric 评分（本机量化裁判）', '',
            '以下评分使用原始 AASR official rubric 和本机量化 Gemma4-QAT-W4A16 judge，通过仓库官方 benchmark_io/result_store/metrics 解析与汇总。此评测直接输入 oral 文本，不是论文的全音频、全精度复现；不得称为官方论文分数。百分比是 rubric 规范化得分，不是准确率。', '',
            '| 方案 | 总分 | rubric规范化得分 | content | format | filter | rephrase |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for variant in VARIANTS:
            result = rubric_scores[variant]
            overall = result['overall']
            category = result['by_category']
            pct = lambda value: f"{value['normalized_score'] * 100:.2f}%"
            lines.append(f"| {NAMES[variant]} | {overall['total_score']}/{overall['max_possible_score']} ({overall['question_count']}题) | {pct(overall)} | {pct(category['content'])} | {pct(category['format'])} | {pct(category['filter'])} | {pct(category['rephrase'])} |")
        judge_protocol = Path(rubric_result.get('judge_protocol') or OUT / 'judge_protocol.json')
        lines += ['',
            f"评分汇总文件：[rubric_scores_by_variant.json]({rubric_scores_path})；裁判协议：[judge_protocol.json]({judge_protocol})。各方案原始汇总："]
        for variant in VARIANTS:
            lines.append(f"- [{variant}.rubric_summary.json]({OUT / (variant + '.rubric_summary.json')})")
    elif not rubric_scores_path.exists():
        lines += ['', '## 官方 rubric 状态', '',
            '官方 rubric 尚未运行。本次只保留了 original_rubric.json，并未生成 rubric_scores_by_variant.json；因此本报告不声称完成官方评分，也不以 CER 替代 rubric。']
    else:
        lines += ['', '## 官方 rubric 状态', '',
            f"rubric_scores_by_variant.json 状态为 {summary['official_rubric_status']}，五方案完成数或题数尚未全部通过校验；本报告不呈现为完整评分。"]
    lines += ['', '## 测试口径与产物', '',
        '四个 0.5B 方案使用 native `zh-itn-strict-conservative-v91` 数字规则。LoRA checkpoint 在内存合并，模型通过 BF16 PyTorch Transformers 加载；AgenticASR 1B 按原配置直接处理原文，没有数字规则。batch 8 总耗时只表示批处理吞吐，不能解释为单句延迟。', '',
        f'- [逐场景、逐组 summary.json]({OUT / "summary.json"})',
        f'- [summary.csv]({OUT / "summary.csv"})',
        f'- [门控输出差异 JSONL]({OUT / "gate_differences.jsonl"})',
        f'- [510句逐条对照，按场景分组]({OUT / "510句逐条对照.md"})',
        f'- [原始预测与 model_output]({OUT / "predictions.jsonl"})',
        f'- [逐批计时]({OUT / "batch_timings.jsonl"})',
        f'- [原始 rubric]({OUT / "original_rubric.json"})',
        f'- [运行协议]({OUT / "protocol.json"})', '']
    if summary['official_rubric_run']:
        conclusions = [
            '## 本轮结论', '',
            '在本轮510条中文AASR纯文本评估中，原AgenticASR 1B的rubric规范化得分89.98%，高于当前0.5B LoRA＋规则库的62.22%和全量＋规则库的60.83%。此前内部256句上“全量组合较好”的结论不能推广为在AASR上优于1B。', '',
            '门控使本轮32句的单句平均延迟降低约11%，但510句batch 8总吞吐基本不变。全量有1条重复生成到1024token上限，异常已计入。', '',
            '四个0.5B组合的内容保留得分约96%，差距主要在filter口语清理与rephrase自我修正。数字保护存在103–113条回退，原生数字格式目标与AASR参考也不完全一致；分数衡量当前组合整体，不能将所有差距直接归因于模型规模或训练方式。', '',
            '五组均完成510样本、3416道rubric题，没有失败题。裁判为本机4bit Gemma4，属于机器评分；不是人工准确率，也不是论文分数复现。', '',
        ]
        lines[2:2] = conclusions
    (OUT / 'AASR-Bench五方案速度质量报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    detail = ['# AASR-Bench 中文集合 510 句逐条对照', '',
        '输入为纯文本 oral，不包含音频转录。原始 clean 和规则归一化 clean 分开展示；各输出的逐字一致不等同官方 rubric 得分。', '']
    for scene in sorted(scene_ids):
        detail += [f'## 场景：{scene}', '']
        for index, sample in enumerate((r for r in manifest if r['scene'] == scene), 1):
            ident = sample['id']
            kind = 'passthrough' if sample['oral'] == sample['clean'] else 'oral与clean不同'
            detail += [f"### {index}. {kind} · `{ident}`", '', '**oral 输入**：' + sample['oral'], '',
                '**原始 clean**：' + sample['original_reference'], '',
                '**规则归一化 clean**：' + sample['reference'], '',
                '**规则基线**：' + sample['rule_baseline'], '']
            for name in VARIANTS:
                row = throughput[name][ident]
                flags = []
                if name.endswith('_gate') and not row['route']:
                    flags.append('gate skip')
                if row['numeric_fallback']:
                    flags.append('numeric guard 回退')
                suffix = '（' + '；'.join(flags) + '）' if flags else ''
                detail += [f"**{NAMES[name]}{suffix}**：{row['output']}"]
                if row['numeric_fallback'] or row['numeric_surface_repaired']:
                    detail += ['模型正文：' + row['model_output']]
                detail += ['']
    (OUT / '510句逐条对照.md').write_text('\n'.join(detail) + '\n', encoding='utf-8')
    print(json.dumps({'summary': str(OUT / 'summary.json'), 'report': str(OUT / 'AASR-Bench五方案速度质量报告.md'),
                      'details': str(OUT / '510句逐条对照.md'), 'official_rubric_run': summary['official_rubric_run']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
