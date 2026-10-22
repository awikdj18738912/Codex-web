"""Audit and summarize natural long-audio paced timing runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, fraction):
    """Linear-interpolated percentile, matching the common inclusive definition."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    duration = float(manifest['duration_seconds'])
    expected_packets = math.ceil(duration / 0.160)
    run = directory / 'run'
    rows = read_rows(run / 'results.jsonl')
    events = read_rows(run / 'events.jsonl')
    assert duration > 0
    assert rows and all(isinstance(r.get('round'), int) and r['round'] >= 0 for r in rows)
    round_count = max(r['round'] for r in rows) + 1
    expected_keys = {(i, arm) for i in range(round_count) for arm in ('raw', 'entity')}
    keys = [(r['round'], r['arm']) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) == expected_keys
    keyed = {(r['round'], r['arm']): r for r in rows}
    assert hashlib.sha256(Path(manifest['pcm_path']).read_bytes()).hexdigest() == manifest['pcm_sha256']
    assert hashlib.sha256(Path(manifest['library_path']).read_bytes()).hexdigest() == manifest['library_sha256']
    packet_count = 0
    for row in rows:
        assert row['pcm_sha256'] == manifest['pcm_sha256']
        assert row['duration_seconds'] == duration
        assert len(row['chunk_submissions']) == expected_packets
        previous = 0.
        for packet in row['chunk_submissions']:
            assert packet['audio_seconds'] > previous
            previous = packet['audio_seconds']
            assert packet['submitted_wall_seconds'] + .002 >= (
                row['audio_clock_start_wall_seconds'] + packet['audio_seconds'])
            assert packet['response_wall_seconds'] >= packet['submitted_wall_seconds']
            packet_count += 1
        assert math.isclose(previous, duration, abs_tol=1e-6)
        if row['arm'] == 'raw':
            assert not row['entity_audio_repair'] and not row['ctc']
    timing_fields = ('first_external_text_wall_seconds', 'total_wall_seconds',
        'finish_tail_from_last_packet_submit_seconds', 'finish_tail_from_finish_call_seconds',
        'max_cumulative_schedule_lateness_seconds', 'chunk_request_total_seconds')
    arms = {arm: {name: median([r[name] for r in rows if r['arm'] == arm])
                  for name in timing_fields} for arm in ('raw', 'entity')}
    arms_lateness = {arm: dict(max_seconds=max(r['max_cumulative_schedule_lateness_seconds']
                                  for r in rows if r['arm'] == arm),
                               median_seconds=median(r['max_cumulative_schedule_lateness_seconds']
                                  for r in rows if r['arm'] == arm))
                     for arm in ('raw', 'entity')}
    request_latency_by_minute = {}
    for minute in range(math.ceil(duration / 60)):
        start, end = minute * 60, min((minute + 1) * 60, duration)
        for arm in ('raw', 'entity'):
            latencies = []
            for row in rows:
                if row['arm'] != arm:
                    continue
                for packet in row['chunk_submissions']:
                    audio_time = packet['audio_seconds']
                    if start < audio_time <= end or (minute == 0 and audio_time == 0):
                        latencies.append(packet['response_wall_seconds'] - packet['submitted_wall_seconds'])
            request_latency_by_minute[f'{arm}:minute_{minute + 1}'] = dict(
                audio_start_seconds=start, audio_end_seconds=end, sample_count=len(latencies),
                p50_seconds=percentile(latencies, .50), p95_seconds=percentile(latencies, .95),
                max_seconds=max(latencies) if latencies else None)
    paired = []
    for i in range(round_count):
        raw, entity = keyed[i, 'raw'], keyed[i, 'entity']
        paired.append(dict(round=i, **{name: entity[name] - raw[name] for name in timing_fields},
            actual_primary_text_identical=raw['raw_primary_final_text'] == entity['raw_primary_final_text']))
    targets, changes, spelling_transitions = [], [], []
    for row in rows:
        relevant = [e for e in events if e['id'] == row['id'] and
                    e['round'] == row['round'] and e['arm'] == row['arm']]
        previous_spelling = None
        for event in relevant:
            if event['event'] != 'external_response':
                continue
            text = event.get('text', '')
            spelling = '构皮滩' if '构皮滩' in text else '构壁滩' if '构壁滩' in text else None
            if spelling != previous_spelling:
                if previous_spelling is not None or spelling is not None:
                    spelling_transitions.append(dict(round=row['round'], arm=row['arm'],
                        wall_seconds=event['wall_seconds'], from_spelling=previous_spelling,
                        to_spelling=spelling))
                previous_spelling = spelling
        for term in manifest.get('target_terms', []):
            first = {}
            for field, kinds in [('primary', {'primary_response', 'primary_finish_response'}),
                                 ('published', {'external_response'})]:
                first[field] = next((e['wall_seconds'] for e in relevant
                    if e['event'] in kinds and term in e.get('text', '')), None)
            targets.append(dict(round=row['round'], arm=row['arm'], term=term,
                first_primary_wall_seconds=first['primary'],
                first_published_wall_seconds=first['published'],
                present_in_final=term in row['final_text']))
        if row['arm'] != 'entity':
            continue
        for change in row.get('entity_audio_repair', {}).get('changes', []):
            matching = [e for e in relevant if e['event'] == 'repair_status' and
                any(c.get('canonical') == change['canonical'] and
                    c.get('start') == change['start'] and c.get('end') == change['end']
                    for c in e.get('snapshot', {}).get('changes', []))]
            confirm = matching[0]['wall_seconds'] if matching else None
            observed = next((e['wall_seconds'] for e in relevant
                if e['event'] == 'external_response' and confirm is not None and
                e['wall_seconds'] >= confirm and change['canonical'] in e.get('text', '')), None)
            changes.append(dict(round=row['round'], **change,
                first_confirmation_wall_seconds=confirm,
                first_external_lexical_observation_after_confirmation_wall_seconds=observed,
                lexical_observation_before_finish_call=observed is not None and observed < row['finish_call_wall_seconds']))
    output = dict(validation=dict(complete_round_arm_coverage=True, round_count=round_count,
        pcm_and_library_hash_match=True, expected_packets_per_run=expected_packets,
        pacing_packets_checked=packet_count,
        packets_never_submitted_before_available_within_2ms=True, raw_bypasses_entity_module=True),
        arms_medians=arms, arm_max_and_median_send_lateness=arms_lateness,
        per_minute_packet_request_latency=request_latency_by_minute,
        per_round_deltas_entity_minus_raw=paired,
        paired_delta_medians={name: median([p[name] for p in paired]) for name in timing_fields},
        target_spelling_observations=targets, final_entity_changes=changes,
        goupitan_spelling_transitions=spelling_transitions,
        limitations=['Single manifest input; repetition count and coverage are derived from complete rounds.',
            'No human reference, no CER or entity precision/recall claim.',
            'Target-term publication times are lexical observations, not causal evidence of an entity edit.',
            'Timing excludes model/library preparation and downstream Web refinement.',
            'Returned text/patch observation times are not audio entity-end timestamps.',
            'Real-time pacing measures user-visible streaming latency, not offline file throughput.'])
    (directory / 'timing_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 连续长音频：纯转录与实体修复的真实语速对照', '',
        f"输入时长 {duration:g} 秒，manifest id：{manifest.get('id', '未提供')}。",
        f"共 {round_count} 轮，每臂 {round_count} 次；包间隔 160 毫秒，每次 {expected_packets} 包。",
        'raw 直接请求真实 ASR 服务；entity 使用同一服务及本次配置的实体处理。',
        '包请求耗时为提交至响应的墙钟时间，不等同于 GPU 计算时间。', '',
        '| 指标（各臂中位数） | 纯转录 | 实体处理 |', '|---|---:|---:|']
    labels = {'first_external_text_wall_seconds': '首次非空文本响应（秒）',
        'total_wall_seconds': '开始请求至最终返回（秒）',
        'finish_tail_from_last_packet_submit_seconds': '最后一包提交至最终返回（秒）',
        'finish_tail_from_finish_call_seconds': 'finish 请求至最终返回（秒）',
        'max_cumulative_schedule_lateness_seconds': '每轮最大发送落后（秒）',
        'chunk_request_total_seconds': '每轮音频块请求等待时间合计（秒）'}
    for name, label in labels.items():
        lines.append(f"| {label} | {arms['raw'][name]:.4f} | {arms['entity'][name]:.4f} |")
    lines += ['', '| 臂 | 轮间最大发送落后最大值（秒） | 轮间最大发送落后中位数（秒） |', '|---|---:|---:|']
    for arm in ('raw', 'entity'):
        s = arms_lateness[arm]
        lines.append(f"| {arm} | {s['max_seconds']:.4f} | {s['median_seconds']:.4f} |")
    lines += ['', '## 每分钟 packet 请求耗时', '',
        '按 packet 对应音频时间分段，报告提交至响应耗时；p50/p95 使用线性插值。', '',
        '| 臂 | 音频分钟 | 包数 | p50（秒） | p95（秒） | 最大值（秒） |', '|---|---:|---:|---:|---:|---:|']
    for key, stat in request_latency_by_minute.items():
        arm, minute = key.split(':')
        lines.append(f"| {arm} | {minute.removeprefix('minute_')} | {stat['sample_count']} | "
            f"{stat['p50_seconds']:.4f} | {stat['p95_seconds']:.4f} | {stat['max_seconds']:.4f} |")
    lines += ['', '| 轮次 | 总耗时差（实体−纯转录） | 首次非空响应差 | 收尾差（最后一包起） | 主转写逐字一致 |',
        '|---|---:|---:|---:|---|']
    for pair in paired:
        lines.append(f"| {pair['round']+1} | {pair['total_wall_seconds']:+.4f} 秒 | "
            f"{pair['first_external_text_wall_seconds']:+.4f} 秒 | "
            f"{pair['finish_tail_from_last_packet_submit_seconds']:+.4f} 秒 | {pair['actual_primary_text_identical']} |")
    lines += ['', '## 流式修改记录', '',
        '时间是会话起点后的词面观察；修改和目标词之后出现的关系不能证明修改导致了该输出。', '',
        '| 轮次 | 原词→目标词 | 证据来源 | 确认时间（秒） | 确认后首次词面观察（秒） | 结束前出现 |',
        '|---|---|---|---:|---:|---|']
    for change in changes:
        confirm = change['first_confirmation_wall_seconds']
        observed = change['first_external_lexical_observation_after_confirmation_wall_seconds']
        lines.append(f"| {change['round']+1} | {change['observed']}→{change['canonical']} | "
            f"{change.get('evidence_source')} | {confirm if confirm is not None else '—'} | "
            f"{observed if observed is not None else '—'} | {change['lexical_observation_before_finish_call']} |")
    if not changes:
        lines.append('| — | 本次没有最终保留的修改 | — | — | — | — |')
    lines += ['', '## 构皮滩词面变化', '']
    if spelling_transitions:
        lines += ['以下仅列日志中实际观察到的词面状态变化，不据此推断实体修改造成变化。', '',
            '| 轮次 | 臂 | 时间（秒） | 变化 |', '|---:|---|---:|---|']
        for item in spelling_transitions:
            lines.append(f"| {item['round']+1} | {item['arm']} | {item['wall_seconds']:.3f} | "
                f"{item['from_spelling'] or '无该词'} → {item['to_spelling'] or '无该词'} |")
    else:
        lines.append('日志中没有观察到该拼写的状态变化，因此不作变化推断。')
    lines += ['', '## 运行统计', '']
    for row in rows:
        ctc = row.get('ctc') or {}
        fallback = ctc.get('fallback_reason')
        compute = ('未保留关闭前统计' if fallback else f"{ctc.get('compute_seconds', 0):.3f} 秒")
        lines.append(f"- 轮 {row['round']+1} / {row['arm']}：复核检查 {row.get('audio_review_checks', 0)}，"
            f"CTC 提案 {ctc.get('proposals', 0)}、保留补丁 {ctc.get('committed', 0)}、"
            f"错误计数 {ctc.get('errors', 0)}、计算 {compute}。")
        if fallback:
            lines.append(f"  - 回退状态：`{fallback}`；已提交 {ctc.get('submitted_audio_seconds', 0)} 秒音频。"
                '异常关闭后未保留 acoustic_events/compute 的历史读数，最终零值不能解释为没有运行 CTC。'
                '日志未保存底层异常文本，无法仅凭该状态区分队列超限与 worker 错误。')
    lines += ['', '## 目标词首次对外观察', '',
        '单元格是会话起点后的秒数；“未出现”只指未观察到该精确拼写，不是人工判定的错误率。', '']
    if manifest.get('target_terms'):
        headers = ['目标拼写'] + [f'{arm} 第{i+1}轮' for i in range(round_count) for arm in ('raw', 'entity')] + ['最终含该拼写的会话数']
        lines += ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
        observed = {(t['round'], t['arm'], t['term']): t['first_published_wall_seconds'] for t in targets}
        for term in manifest['target_terms']:
            cells = []
            for i in range(round_count):
                for arm in ('raw', 'entity'):
                    value = observed[i, arm, term]
                    cells.append(f'{value:.3f}' if value is not None else '未出现')
            final_present = sum(t['present_in_final'] for t in targets if t['term'] == term)
            lines.append('| ' + term + ' | ' + ' | '.join(cells) + f' | {final_present}/{len(rows)} |')
    lines += ['', '## 适用范围', ''] + ['- ' + x for x in output['limitations']]
    lines += ['', f'共核验 {packet_count} 包，PCM 和词库 SHA256 匹配。逐包、逐词时序见 `timing_audit.json` 及 `run/events.jsonl`。', '']
    (directory / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(validation=output['validation'], arms_medians=arms,
        arm_max_and_median_send_lateness=arms_lateness,
        per_minute_packet_request_latency=request_latency_by_minute,
        delta_medians=output['paired_delta_medians'], changes=len(changes)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory.resolve())
