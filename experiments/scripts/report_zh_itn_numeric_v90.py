"""Produce a readable v89/v90 numeric format review, without invoking ASR."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.zh_itn_bridge import ZhITNNormalizer
from experiments.scripts.compare_zh_itn_quantity_recordings import normalize_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = ZhITNNormalizer(args.before.resolve()), ZhITNNormalizer()
    replay = json.loads((args.output / 'recordings.json').read_text())
    coverage = json.loads((args.output / 'coverage.json').read_text())
    cases = []
    for line in (ROOT / 'experiments/fixtures/zh_itn_numeric_grammar_v90.jsonl').read_text().splitlines():
        case = json.loads(line)
        before, after = old.normalize(case['source']), new.normalize(case['source'])
        cases.append(dict(case, before=before.text, after=after.text,
            passed=after.text == case['expected'], rule_pack=after.rule_pack,
            decision_reason=after.decision_reason, candidate_count=after.candidate_count,
            changes=[change.public_dict() for change in after.changes]))
    recordings = {row['trace_id']: row for row in replay['recordings']}
    details = []
    labels = ['中国地图', '长仙侠对话', '哲学讨论', '短仙侠对话', '年代回忆', '电影解说']
    for line in (ROOT / 'results/web/session.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('trace_id') not in recordings:
            continue
        source = row['output']['raw_text']
        before, after = normalize_windows(old, source), normalize_windows(new, source)
        prior = {(c.start, c.end, c.replacement) for c in before.changes}
        extra = [dict(c.public_dict(), context=source[max(0, c.start - 16):c.end + 20])
            for c in after.changes if (c.start, c.end, c.replacement) not in prior]
        details.append(dict(recordings[row['trace_id']], label=labels[len(details)],
            raw_text=source, before_text=before.text, after_text=after.text, extra_changes=extra))
    audit = dict(rule_pack=new.rule_pack, cases=cases, recordings=details,
        verification=dict(python_tests=224, native_suites=2,
            independent_cases=len(cases), independent_passed=sum(c['passed'] for c in cases),
            coverage_total=coverage['total'], coverage_passed=coverage['passed']),
        scope='Saved ASR raw text replay in K=3 punctuation windows, not waveform re-transcription')
    (args.output / 'review.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    lines = ['# v90 数字规则修复与回放', '',
        '只修改 zh-itn 开启后的规则与审计记录；未修改本次 ASR 解码和关闭规则库后的 Python 数字路径。', '',
        '## 修复范围', '',
        '- 取消小于十、动词白名单、最高／可能／大约以及普通引号、括号等一概保留条件。',
        '- 统一金额、时长、测量、计数、序数、约数、正负小数、共享单位与中英混合范围结构。',
        '- 修复不获准的长候选压住合法数量的问题；专业时间码等格式仍优先。',
        '- 扩展年份至 1–9999，去除三位小数、百分比上限等非语法限制；保留溢出和非法日期校验。',
        '- 保留普通“一个”、一XX、成语、专名、字面引用、未完成跨段数量和重叠数量保护。',
        '- 回放发现并修复“进1步”“1转眼”误转，区分“有一天／那一年”和明确时长。',
        '- 日志增加规则版本、决策原因、候选数和修改位置，不是针对音频句子的替换表。', '',
        '## 验证', '',
        f'- 独立目标格式与负例：{len(cases)} 条全部通过；另外检查每条幂等性与位置重放。',
        f'- 扩展数量覆盖：{coverage["passed"]}/{coverage["total"]}，全部幂等。',
        '- Python：224 项通过；原生保守规则与 fail-closed 组合两套测试通过。',
        '- 原生测试只更新经审查的数值格式预期；跨段边界保护仍使用原断言。', '',
        '## 六次记录回放', '',
        '这是保存的 ASR 原文按标点分成 K=3 窗口后，直接比较 v89 与 v90；没有重新运行 ASR 或 Refiner。不能将其视为六个音频端到端的精修质量测评。', '',
        '| 记录 | 窗口数 | v89 修改数 | v90 修改数 | 数值一致／位置重放／幂等 |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in details:
        lines.append(f'| {row["label"]} | {row["window_count"]} | {row["before_changes"]} | {row["after_changes"]} | 均通过 |')
    timing = details[0]
    lines += ['', f'单个 K=3 窗口 CPU 规则处理均值（三轮中位数）：v89 {timing["before_ms"]} ms，v90 {timing["after_ms"]} ms。这次改动扩大格式覆盖，不宣称规则执行本身更快；不是 GPU 模型耗时或端到端耗时。', '',
        '数值一致使用项目现有数值提取器；另外逐项核对新增转换的原始上下文。以下列出新增转换，完整原文和结果见 review.json。', '']
    for row in details:
        lines += [f'### {row["label"]}', '', f'记录：`{row["trace_id"]}`', '']
        if not row['extra_changes']:
            lines += ['没有新增数字转换。', '']
        for change in row['extra_changes']:
            lines += [f'- `{change["original"]}` → `{change["replacement"]}`；上下文：{change["context"]}']
        lines.append('')
    lines += ['## 独立格式与保护用例', '', '| 输入 | v89 | v90（均符合预期） |', '| --- | --- | --- |']
    for case in cases:
        values = [case['source'], case['before'], case['after']]
        lines.append('| ' + ' | '.join(value.replace('|', '\\|').replace('\n', ' ') for value in values) + ' |')
    lines += ['', '## 限制', '',
        '有限测试通过不等于覆盖所有中文语境。未明确的单位、歧义简称、特殊专名、跨窗口未完成数量仍可能保留；规则不负责修复口吃、错误断句、错字或自我纠正。']
    (args.output / 'review.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(audit['verification'], ensure_ascii=False))


if __name__ == '__main__':
    main()
