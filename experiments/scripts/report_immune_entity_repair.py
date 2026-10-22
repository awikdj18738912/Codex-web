"""Describe actual patches on the user-approved immune-system glossary."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.report_entity_repair_material import project_reference_span


def main(directory):
    result = json.loads((directory/'quality/off.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    registered = json.loads((directory/'registered_entities.json').read_text())
    assert result['pcm_sha256'] == manifest['pcm_sha256']
    assert result['library_sha256'] == manifest['library_sha256']
    assert hashlib.sha256(Path(manifest['pcm_path']).read_bytes()).hexdigest() == manifest['pcm_sha256']
    assert hashlib.sha256(Path(manifest['library_path']).read_bytes()).hexdigest() == manifest['library_sha256']
    before, after = result['primary_text'], result['text']
    changes = result['entity_audio_repair'].get('changes', [])
    reconstructed = before
    for patch in sorted(changes, key=lambda x: x['start'], reverse=True):
        assert before[patch['start']:patch['end']] == patch['observed']
        reconstructed = reconstructed[:patch['start']]+patch['canonical']+reconstructed[patch['end']:]
    assert reconstructed == after
    (directory/'primary_before.txt').write_text(before+'\n')
    (directory/'entity_after.txt').write_text(after+'\n')
    prior = json.loads((ROOT/'results/immune_system_raw_20261015/raw_transcript.json').read_text())['final_payload']['text']
    drift = [dict(before=prior[i:j], current=before[a:b]) for op,i,j,a,b in
             difflib.SequenceMatcher(None, prior, before, autojunk=False).get_opcodes() if op != 'equal']
    rows = []
    for entry in registered:
        patches = [c for c in changes if c['entity_id'] == entry['entity_id']]
        observed = []
        for surface in entry['raw_surface_examples']:
            examples = []
            for match in re.finditer(re.escape(surface), before):
                following = project_reference_span(before, after, match.start(), match.end())
                examples.append(dict(start=match.start(), end=match.end(), observed=surface,
                                     after=following, context=before[max(0,match.start()-12):match.end()+12]))
            observed.append(dict(surface=surface, occurrences=examples))
        rows.append(dict(**entry, changes=patches, original_surface_occurrences=observed,
            canonical_literal_occurrences_before=before.count(entry['canonical_text']),
            canonical_literal_occurrences_after=after.count(entry['canonical_text'])))
    output = dict(library_entries=len(registered), retained_patches=len(changes),
        changed_entries=sum(bool(x['changes']) for x in rows), rows=rows,
        patches=changes, primary_drift_vs_prior=drift,
        all_output_edits_reconstruct_from_patches=True,
        statistics=result['entity_audio_repair'],
        scope='Candidate glossary is not gold. Counts of retained patches, not verified error recovery or accuracy. No paced timing or CER claim.')
    (directory/'comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# 免疫系统：当前实体策略测试','',
        '完整用户音频，无提示Qwen主识别；确认的31项词库，未添加错写别名。CTC关闭，默认直接拼写与原音复核启用。',
        '加速提交录音，只评估修改效果，运行耗时不代表实时延迟。前后比较使用同次primary_text/text；此前原始转录保持不变。', '',
        f"31个词库条目中，{output['changed_entries']}项产生最终保留的补丁，共{len(changes)}处。此为实际替换数，尚无人工逐处gold，不能等同正确恢复数。",'',
        '[本次原始主转写](primary_before.txt) · [本次修复后全文](entity_after.txt)', '',
        '## 所有最终修改', '', '| 原始跨度 | 替换结果 | 路径 | 规则 |', '|---|---|---|---|']
    for patch in changes:
        lines.append(f"| {patch['observed']} | {patch['canonical']} | {patch.get('evidence_source')} | {patch.get('direct_rule', patch.get('match_type',''))} |")
    if not changes:lines.append('| 无 | — | — | — |')
    lines+=['','## 31项逐项结果','','没有替换不自动等于漏修；其中有本来正确、格式差异、同义名与待听审项。','','| 原编号 | 配置的规范词 | 本轮实际替换 |','|---:|---|---|']
    for row in rows:
        text='；'.join(c['observed']+'→'+c['canonical'] for c in row['changes']) or '无最终替换'
        lines.append(f"| {row['original_id']} | {row['canonical_text']} | {text} |")
    inc=result['entity_audio_repair']['incremental']
    lines+=['','## 状态与限制','',
        f"- 复核检查：{result['entity_audio_repair']['checks']}；原音复核错误计数：{inc.get('errors',0)}。",
        f"- 直接路径最终保留计数：{inc.get('glossary_spelling_committed',0)}；CTC模式：{inc.get('ctc',{}).get('mode')}。",
        f"- 与此前未加词库的转录有{len(drift)}处文本差异；差异另存comparison.json，不当作本次替换收益。",
        '- 已验证所有最终输出差异可由记录补丁重建，没有清单外的隐式文本精修。',
        '- 运行期间代码、阈值及实验词库固定，现有在线词库未修改。未替换项如实保留，不能只据是否改写判断对错。',
        '- 例如HLA与HLDA、ITAM与ITIM必须按具体提及区分；不根据词种在全文出现一次就宣布全部正确。','']
    (directory/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(changed_entries=output['changed_entries'],retained_patches=len(changes),
        patches=[(c['observed'],c['canonical'],c.get('evidence_source')) for c in changes]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    main(parser.parse_args().directory.resolve())
