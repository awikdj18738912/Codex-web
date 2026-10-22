"""Join paced latency results with frozen per-mention entity diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.report_entity_repair_material import (
    hit_for, metrics, transition_counts, project_reference_span)
from system.entity_audio_repair import normalized


def main(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    annotations = json.loads((directory / 'entity_annotations.json').read_text())
    rows = [json.loads(line) for line in (directory / 'run/results.jsonl').read_text().splitlines() if line]
    assert len(rows) == 2
    by_arm = {r['arm']: r for r in rows}
    assert set(by_arm) == {'raw', 'entity'}
    assert all(r['id'] == manifest['id'] and r['round'] == 0 for r in rows)
    assert hashlib.sha256((directory / 'entity_annotations.json').read_bytes()).hexdigest() == manifest['annotations_sha256']
    for row in rows:
        assert row['pcm_sha256'] == manifest['pcm_sha256'] and row['duration_seconds'] == manifest['duration_seconds']
        assert len(row['chunk_submissions']) == 3750
    raw, entity = by_arm['raw'], by_arm['entity']
    assert entity['ctc'].get('mode') == 'on'
    hypotheses = {'raw_final': raw['final_text'],
                  'entity_primary': entity['raw_primary_final_text'],
                  'entity_final': entity['final_text']}
    per_mention = {name: {m['id']: hit_for(m, annotations['reference_text'], text)
                         for m in annotations['mentions']} for name, text in hypotheses.items()}
    scores = {name: metrics(annotations['mentions'], hits) for name, hits in per_mention.items()}
    transitions = transition_counts(annotations['mentions'], per_mention['entity_primary'], per_mention['entity_final'])
    events = [json.loads(line) for line in (directory / 'run/events.jsonl').read_text().splitlines() if line]
    changes = entity['entity_audio_repair'].get('changes', [])
    publication = []
    latest_primary = ''
    audio_position = 0.
    found = set()
    pairs_seen = set()
    for event in events:
        if event['arm'] != 'entity':
            continue
        if event['event'] == 'packet_submit':
            audio_position = event['audio_seconds']
        if event['event'] in ('primary_response', 'primary_finish_response'):
            latest_primary = event.get('text', '')
        if event['event'] != 'external_response' or not event.get('revision'):
            continue
        visible = event.get('text', '')
        pair = latest_primary, visible
        if visible == latest_primary or pair in pairs_seen:
            continue
        pairs_seen.add(pair)
        for index, change in enumerate(changes):
            if index in found:
                continue
            start, end = change['start'], change['end']
            if latest_primary[start:end] != change['observed']:
                # Earlier partial text can have a slightly different prefix.
                # Locate this same observed span by its recorded left/right
                # anchors instead of applying final offsets to old partials.
                possible = []
                at = latest_primary.find(change['observed'])
                while at >= 0:
                    stop = at + len(change['observed'])
                    left = normalized(latest_primary[:at])
                    right = normalized(latest_primary[stop:])
                    if (left.endswith(change.get('source_left', '')) and
                            right.startswith(change.get('source_right', ''))):
                        possible.append((at, stop))
                    at = latest_primary.find(change['observed'], at + 1)
                if len(possible) != 1:
                    continue
                start, end = possible[0]
            projected = project_reference_span(latest_primary, visible, start, end)
            exact_single_patch = (latest_primary[:start] + change['canonical'] + latest_primary[end:]) == visible
            if projected != change['canonical'] and not exact_single_patch:
                continue
            found.add(index)
            publication.append(dict(change_index=index, observed=change['observed'],
                canonical=change['canonical'], evidence_source=change.get('evidence_source'),
                first_verified_publication_wall_seconds=event['wall_seconds'],
                submitted_audio_seconds=audio_position,
                before_finish_call=event['wall_seconds'] < entity['finish_call_wall_seconds'],
                method='Compare changed source span in immediately preceding primary response with external response; require nonzero repair revision.'))
    output = dict(manifest_id=manifest['id'], scores=scores, per_mention=per_mention,
        entity_same_run_transitions=transitions,
        primary_text_exactly_matches_between_arms=raw['raw_primary_final_text'] == entity['raw_primary_final_text'],
        final_patch_publications=publication, final_changes=changes,
        snapshots={arm: row['entity_audio_repair'] for arm, row in by_arm.items()},
        timing={arm: {k: row[k] for k in ('first_external_text_wall_seconds',
            'total_wall_seconds', 'finish_tail_from_last_packet_submit_seconds',
            'max_cumulative_schedule_lateness_seconds')} for arm, row in by_arm.items()},
        label_scope='Entity spelling diagnostic annotations, pending independent human listening; not full gold or blind evaluation; no CER. Three spoken-order variants separate.')
    (directory / 'entity_paced_report.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 10分钟有错实体材料：真实语速测试', '',
        '相同600秒PCM、53条冻结领域词库；纯转录与当前CTC＋原音复核串行各一次。',
        '每160毫秒提交对应新音频。模型/词库准备、下游Web精修不在计时内。',
        '实体标注是资料核准的开发诊断参考，尚待独立人工听审；3处词序异型单列，不计算全文CER。', '',
        '## 实体结果', '',
        '| 输出 | 核心31处命中 | 5处核心错例命中 | 3处待定异型命中 |',
        '|---|---:|---:|---:|']
    for name, score in scores.items():
        cells = [f"{score[key]['hit']}/{score[key]['mentions']}（未投射{score[key]['unprojectable']}）"
                 for key in ('core31', 'targeted_hard5', 'uncertain3')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', f"实体组同次主转写→最终文本：恢复 {transitions['errors_recovered']} 处核心错误，原正确项丢失 {transitions['previously_correct_lost']} 处。两组主转写逐字相同：{output['primary_text_exactly_matches_between_arms']}。", '',
        '## 五处核心错例', '',
        '| 标注ID | 参考拼写 | 纯转录 | 实体组主转写 | 实体组最终 |',
        '|---|---|---|---|---|']
    for mention in annotations['mentions']:
        if mention['category'] != 'core_spelling':
            continue
        cells = [per_mention[name][mention['id']]['projected_surface'] or '未投射' for name in hypotheses]
        lines.append('| '+mention['id']+' | '+mention['reference_surface']+' | '+' | '.join(cells)+' |')
    lines += ['', '## 延迟', '', '| 指标 | 纯转录 | 实体组 |', '|---|---:|---:|']
    for key, label in [('first_external_text_wall_seconds', '首次非空文本响应'),
                       ('total_wall_seconds', '总耗时（含600秒播放）'),
                       ('finish_tail_from_last_packet_submit_seconds', '末包至最终返回'),
                       ('max_cumulative_schedule_lateness_seconds', '最大发送落后')]:
        lines.append(f"| {label} | {raw[key]:.3f}秒 | {entity[key]:.3f}秒 |")
    lines += ['', '## 最终补丁的流式发布证据', '',
        '逐次对比紧邻主转写和对外响应的对应原始跨度，并要求修订版本非零；不把其他位置出现相同姓名当作本处修复。', '',
        '| 原文→规范名 | 证据来源 | 首次确认对外发布 | 当时已提交音频 | 是否结束前 |',
        '|---|---|---:|---:|---|']
    for patch in publication:
        lines.append(f"| {patch['observed']}→{patch['canonical']} | {patch['evidence_source']} | "
            f"{patch['first_verified_publication_wall_seconds']:.3f}秒 | {patch['submitted_audio_seconds']:.2f}秒 | {patch['before_finish_call']} |")
    if not publication:
        lines.append('| 无可验证的最终补丁发布 | — | — | — | — |')
    ctc = entity['ctc']
    lines += ['', '## 运行状态与范围', '',
        f"- 复核检查 {entity['audio_review_checks']}；CTC errors={ctc.get('errors')}、fallback={ctc.get('fallback_reason')}、finish_timeout={ctc.get('finish_timed_out')}。",
        '- CTC异常关闭后event/compute零值可能是历史统计丢失，不能解释为没有执行。',
        '- 当前是单段、单次配对；无并发性能或总体准确率保证。',
        '- 按原始错误筛选的开发诊断素材；所有结果使用冻结标注，未按修复结果修改词库或阈值。',
        '- 逐包/每分钟延迟见REPORT.md及timing_audit.json，逐处命中及完整状态见entity_paced_report.json。', '']
    (directory / 'ENTITY_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(scores=scores, transitions=transitions, publications=publication), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory.resolve())
