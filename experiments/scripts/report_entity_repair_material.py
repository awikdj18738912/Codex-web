"""Report per-mention entity repair outcomes for the frozen philosophy material.

The annotations correct entity spellings only.  This script deliberately does
not compute CER or infer accuracy from a canonical string appearing elsewhere.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/entity_philosophy_repair_20261015'


def normalize_with_positions(text: str):
    """NFKC, casefold, and remove whitespace/punctuation while retaining offsets."""
    chars, positions = [], []
    for i, source_char in enumerate(text):
        for char in unicodedata.normalize('NFKC', source_char).casefold():
            if char.isspace() or unicodedata.category(char).startswith('P'):
                continue
            chars.append(char)
            positions.append(i)
    return ''.join(chars), positions


def project_reference_span(reference: str, hypothesis: str, start: int, end: int):
    """Project one annotated source span through a global sequence alignment.

    Returning None means there was no defensible local projection (for example,
    the annotation coordinates do not identify reference characters).
    """
    ref_norm, ref_positions = normalize_with_positions(reference)
    hyp_norm, _ = normalize_with_positions(hypothesis)
    selected = [i for i, original in enumerate(ref_positions) if start <= original < end]
    if not selected:
        return None
    lo, hi = selected[0], selected[-1] + 1
    # A projection is accepted only when the annotated span is represented by
    # equal blocks and/or a single replacement block. Insertions at its edges
    # do not expand the annotation into unrelated words.
    matcher = SequenceMatcher(a=ref_norm, b=hyp_norm, autojunk=False)
    ops = matcher.get_opcodes()
    overlaps = [op for op in ops if op[1] < hi and op[2] > lo]
    if not overlaps or overlaps[0][1] > lo or overlaps[-1][2] < hi:
        return None
    # A replacement is projectable only when it covers the complete annotated
    # source span. Mixed equal/replace alignments are allowed when contiguous.
    pieces = []
    expected_cursor = lo
    out_cursor = None
    for tag, i1, i2, j1, j2 in overlaps:
        a, b = max(i1, lo), min(i2, hi)
        if a >= b:
            continue
        if a != expected_cursor or tag in ('delete', 'insert'):
            return None
        if tag == 'equal':
            p, q = j1 + (a - i1), j1 + (b - i1)
        elif tag == 'replace':
            if a != i1 or b != i2:
                return None
            p, q = j1, j2
        else:
            return None
        if out_cursor is not None and p != out_cursor:
            return None
        pieces.append(hyp_norm[p:q])
        out_cursor = q
        expected_cursor = b
    if expected_cursor != hi:
        return None
    return ''.join(pieces)


def hit_for(mention, reference_text, hypothesis):
    projected = project_reference_span(reference_text, hypothesis,
                                       mention['reference_start'], mention['reference_end'])
    surface = mention.get('reference_surface') or mention['canonical_entity']
    target, _ = normalize_with_positions(surface)
    return dict(projected_surface=projected, accepted_surface=surface,
                hit=(projected == target) if projected is not None else None)


def load_result(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def arm_mentions(annotations, result):
    if result is None:
        return None
    reference = annotations['reference_text']
    fields = {}
    for field in ('primary_text', 'text'):
        hypothesis = str(result.get(field) or '')
        fields[field] = {m['id']: hit_for(m, reference, hypothesis)
                         for m in annotations['mentions']}
    return fields


def metrics(mentions, hits):
    evaluated = [m for m in mentions if m.get('category') != 'order_or_spoken_variant_uncertain']
    core = [m for m in evaluated if m.get('count_in_core_evaluation')]
    hard = [m for m in core if m.get('category') == 'core_spelling']
    def subset(rows):
        vals = [hits[m['id']]['hit'] for m in rows]
        return dict(mentions=len(rows), hit=sum(v is True for v in vals),
                    miss=sum(v is False for v in vals), unprojectable=sum(v is None for v in vals),
                    accuracy=(sum(v is True for v in vals) / len(rows) if rows else None))
    return dict(core31=subset(core), targeted_hard5=subset(hard),
                uncertain3=subset([m for m in mentions if m.get('category') == 'order_or_spoken_variant_uncertain']))


def transition_counts(mentions, before, after):
    core = [m for m in mentions if m.get('count_in_core_evaluation')]
    bad_before = [m for m in core if before[m['id']]['hit'] is False]
    recovered = sum(after[m['id']]['hit'] is True for m in bad_before)
    lost = sum(before[m['id']]['hit'] is True and after[m['id']]['hit'] is False for m in core)
    return dict(core_errors_before=len(bad_before), errors_recovered=recovered,
                errors_unrecovered=sum(after[m['id']]['hit'] is False for m in bad_before),
                previously_correct_lost=lost,
                before_unprojectable=sum(before[m['id']]['hit'] is None for m in core),
                after_unprojectable=sum(after[m['id']]['hit'] is None for m in core))


def snapshot_details(result):
    snap = result.get('entity_audio_repair') or {}
    inc = snap.get('incremental') or {}
    ctc = inc.get('ctc') or {}
    return dict(snapshot=snap, incremental_errors=inc.get('errors'),
                ctc_fallback_reason=ctc.get('fallback_reason'),
                ctc_errors=ctc.get('errors'), ctc_acoustic_events=ctc.get('acoustic_events'),
                ctc_compute_seconds=ctc.get('compute_seconds'), ctc_submitted_audio_seconds=ctc.get('submitted_audio_seconds'))


def render(directory: Path, annotations, results, arms, output):
    mentions = annotations['mentions']
    on = arms['on']
    lines = ['# 哲学音频实体修复材料报告', '',
             '对照只在同一次 `quality/on.json` 内比较 `primary_text` 与 `text`。',
             '标注 reference 只校正实体拼写，不是全文人工 gold；本报告不计算 CER。',
             '实体命中通过规范化后的 reference 与 hypothesis 全文序列对齐，把每个 reference_start/end 独立投射到对应文本跨度；其他文字仍是原始 Qwen 输出。', '']
    lines += ['## 按提及指标', '',
              '| 臂 / 字段 | core 31 | targeted hard 5 | uncertain 3（单列） |',
              '|---|---:|---:|---:|']
    if on is None:
        lines.append('| quality/on.json | 尚未生成 | 尚未生成 | 尚未生成 |')
    else:
        for arm_name in ('primary_text', 'text'):
            stat = metrics(mentions, on[arm_name])
            def cell(s): return f"{s['hit']}/{s['mentions']} 命中；{s['miss']} miss；{s['unprojectable']} 未投射"
            lines.append(f"| on / `{arm_name}` | {cell(stat['core31'])} | {cell(stat['targeted_hard5'])} | {cell(stat['uncertain3'])} |")
        tr = transition_counts(mentions, on['primary_text'], on['text'])
        lines += ['', f"同一 on run 的 core 转换：初始错误 {tr['core_errors_before']}，修复 {tr['errors_recovered']}，仍错 {tr['errors_unrecovered']}；初始命中后丢失 {tr['previously_correct_lost']}。无法投射：primary {tr['before_unprojectable']}，text {tr['after_unprojectable']}。", '']
    if arms.get('off'):
        lines += ['存在 `quality/off.json`，以下另列，不替代同一 on run 对照。', '']
        for arm_name in ('primary_text', 'text'):
            stat = metrics(mentions, arms['off'][arm_name])
            lines.append(f"- off / `{arm_name}`：core {stat['core31']['hit']}/{stat['core31']['mentions']}，hard {stat['targeted_hard5']['hit']}/{stat['targeted_hard5']['mentions']}，uncertain {stat['uncertain3']['hit']}/{stat['uncertain3']['mentions']}。")
    if arms.get('raw_screen'):
        stat = metrics(mentions, arms['raw_screen']['text'])
        lines += ['', f"筛选基线 `raw_screen.json` final text：core {stat['core31']['hit']}/{stat['core31']['mentions']}，hard {stat['targeted_hard5']['hit']}/{stat['targeted_hard5']['mentions']}，uncertain {stat['uncertain3']['hit']}/{stat['uncertain3']['mentions']}。它用于说明错误富集筛选起点，不纳入 on 同次配对修复统计。", '']
    lines += ['', '## 8 个错例及逐处前后', '',
              '每格为该 mention 对应局部跨度，不是全文替换。uncertain 3 不计入 core denominator。', '',
              '| mention | 类别 | reference 表面 | primary_text | text |', '|---|---|---|---|---|']
    for m in mentions:
        if m.get('category') not in ('core_spelling', 'order_or_spoken_variant_uncertain'):
            continue
        before = on['primary_text'].get(m['id'], {}) if on else {}
        after = on['text'].get(m['id'], {}) if on else {}
        def show(x): return '—' if x.get('projected_surface') is None else x['projected_surface']
        lines.append(f"| {m['id']} | {m['category']} | {m.get('reference_surface') or m['canonical_entity']} | {show(before)} | {show(after)} |")
    lines += ['', '## 26 个正确对照', '', '| mention | reference 表面 | primary_text | text | 是否误改 |', '|---|---|---|---|---|']
    for m in mentions:
        if m.get('category') != 'already_correct_control':
            continue
        before = on['primary_text'].get(m['id'], {}) if on else {}
        after = on['text'].get(m['id'], {}) if on else {}
        b, a = before.get('projected_surface'), after.get('projected_surface')
        change = ('未知' if b is None or a is None else '是' if b != a else '否')
        lines.append(f"| {m['id']} | {m.get('reference_surface') or m['canonical_entity']} | {b if b is not None else '未投射'} | {a if a is not None else '未投射'} | {change} |")
    lines += ['', '## 快照、错误与 CTC fallback', '']
    if results.get('on') is None:
        lines.append('on 结果尚未落盘；快照统计待补。')
    else:
        for arm, res in results.items():
            if res is None: continue
            details = snapshot_details(res)
            ctc = ((details['snapshot'].get('incremental') or {}).get('ctc') or {})
            lines.append(f"### {arm}")
            lines.append(f"incremental.errors={details['incremental_errors']}；CTC errors={details['ctc_errors']}；fallback_reason={details['ctc_fallback_reason']!r}；acoustic_events={details['ctc_acoustic_events']}；compute_seconds={details['ctc_compute_seconds']}；submitted_audio_seconds={details['ctc_submitted_audio_seconds']}。")
            if details['ctc_fallback_reason']:
                lines.append('发生 CTC fallback；fallback 后的 compute/event=0 不能解释为 CTC 未运行。')
            lines.append('完整 entity_audio_repair snapshot：')
            lines.append('```json')
            lines.append(json.dumps(details['snapshot'], ensure_ascii=False, indent=2, sort_keys=True))
            lines.append('```')
    lines += ['', '## 解释范围', '',
              '- 拼写来源核准的是实体规范写法；没有独立人工听审确认每个实际发音。',
              '- source 身份是推断，来源原件未核实；异序或疑似口误的 3 处仍待定，并从 core denominator 排除。',
              '- `reference_text` 是逐处实体修正代理，其他文字保留原始转写；禁止据此计算全文 CER。',
              '- 本段材料按初始错误富集，是开发诊断，不是盲测或总体准确率估计。',
              '- runner 的加速只用于质量跑数；elapsed_seconds 不代表真实用户延迟。', '']
    (directory / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (directory / 'entity_repair_report.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(directory: Path):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    annotations = json.loads((directory / manifest.get('entity_annotation_file', 'entity_annotations.json')).read_text(encoding='utf-8'))
    results = {arm: load_result(directory / 'quality' / f'{arm}.json') for arm in ('on', 'off')}
    screen_path = directory / 'raw_screen.json'
    results['raw_screen'] = None
    if screen_path.exists():
        screen = json.loads(screen_path.read_text(encoding='utf-8'))
        final_payload = screen.get('final_payload') or {}
        screen_text = final_payload.get('text') or ''
        results['raw_screen'] = dict(primary_text=screen_text, text=screen_text,
            timing_scope=screen.get('timing_scope'), elapsed_seconds=screen.get('elapsed_seconds'))
    arms = {arm: arm_mentions(annotations, result) for arm, result in results.items()}
    mentions = annotations['mentions']
    output = dict(manifest_id=manifest.get('id'), annotation_mentions=len(mentions),
                  categories={category: sum(m.get('category') == category for m in mentions)
                              for category in sorted({m.get('category') for m in mentions})},
                  arms={}, transitions={}, run_details={})
    for arm in ('on', 'off', 'raw_screen'):
        if arms[arm] is None:
            output['arms'][arm] = None
            continue
        output['arms'][arm] = {field: dict(metrics(mentions, arms[arm][field]),
            mentions={m['id']: dict(category=m.get('category'), canonical_entity=m['canonical_entity'],
                reference_surface=m.get('reference_surface') or m['canonical_entity'], **arms[arm][field][m['id']])
                for m in mentions}) for field in ('primary_text', 'text')}
        output['run_details'][arm] = snapshot_details(results[arm]) if arm != 'raw_screen' else {
            'timing_scope': results[arm].get('timing_scope'),
            'elapsed_seconds': results[arm].get('elapsed_seconds')}
        if arm == 'on':
            output['transitions']['primary_text_to_text'] = transition_counts(mentions, arms[arm]['primary_text'], arms[arm]['text'])
    render(directory, annotations, results, arms, output)
    print(json.dumps(dict(report=str(directory / 'REPORT.md'), json=str(directory / 'entity_repair_report.json'),
                          available_arms=[a for a in ('on', 'off', 'raw_screen') if results[a] is not None]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIR)
    main(parser.parse_args().directory.resolve())
