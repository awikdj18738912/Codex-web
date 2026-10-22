"""Compare paired, deterministic base/epoch-two Refiner predictions."""
import argparse
import csv
import json
from pathlib import Path
import unicodedata

from evaluate_qwen25_refiner import edit_distance


def normalize(text):
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    data = {}
    metrics = {}
    for name in ['base', 'epoch2']:
        data[name] = [json.loads(x) for x in (root / f'{name}_predictions.jsonl').read_text().splitlines() if x.strip()]
        metrics[name] = json.loads((root / f'{name}_predictions.metrics.json').read_text())
    assert len(data['base']) == len(data['epoch2']) == 256
    result = []
    for base, trained in zip(data['base'], data['epoch2']):
        assert all(base[k] == trained[k] for k in ['id', 'input', 'reference', 'scene'])
        reference = normalize(base['reference'])
        row = dict(id=base['id'], scene=base['scene'], input=base['input'], reference=base['reference'],
            base_output=base['prediction'], epoch2_output=trained['prediction'],
            base_errors=base['metrics']['character_errors'], epoch2_errors=trained['metrics']['character_errors'],
            normalized_reference_characters=len(reference),
            base_content_errors=edit_distance(reference, normalize(base['prediction'])),
            epoch2_content_errors=edit_distance(reference, normalize(trained['prediction'])),
            clean=base['input'] == base['reference'])
        row['comparison'] = ('better' if row['epoch2_errors'] < row['base_errors']
            else 'worse' if row['epoch2_errors'] > row['base_errors'] else 'tie')
        result.append(row)
    for name in ['base', 'epoch2']:
        metrics[name]['content_cer_ignoring_punctuation_whitespace'] = sum(r[f'{name}_content_errors'] for r in result) / sum(r['normalized_reference_characters'] for r in result)
    report = dict(samples=256, models=metrics,
        pairwise_counts={k: sum(r['comparison'] == k for r in result) for k in ['better', 'worse', 'tie']},
        limitations=['Samples come from the same synthetic validation distribution used for checkpoint selection.',
            'Reference text similarity is not a semantic correctness score.',
            'Clean output-change rates have only 24 teacher-labeled examples in their denominator.',
            'Numeric formatting differences are included even though numeric formatting belongs to external rules.',
            'Some teacher references retain explicitly abandoned candidates; removing them can be semantically correct despite increasing character differences.'])
    (root / 'comparison_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (root / 'comparison.jsonl').open('w') as f:
        for row in result:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    columns = ['id', 'scene', 'input', 'reference', 'base_output', 'epoch2_output', 'base_errors', 'epoch2_errors', 'comparison', 'clean']
    headers = ['样本ID', '场景', '原始输入', '老师参考答案', '未微调基座输出', '第二轮输出', '基座字符错误数', '第二轮字符错误数', '相对参考答案变化', '无需修改样本']
    with (root / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in result:
            writer.writerow([row[k] for k in columns])
    text = '# 第二轮精修模型与基座对比\n\n256条固定验证样本，九场景均衡抽样，同样的提示词、贪心解码和输出长度预算。\n\n'
    text += '| 指标 | 基座 | 第二轮 |\n|---|---:|---:|\n'
    for label, key in [('参考答案精确匹配率', 'exact_match_rate'), ('与参考答案的字符差异率（含标点）', 'character_error_rate'), ('老师标为无需修改样本的输出变化率（24条）', 'clean_overedit_rate')]:
        values = [metrics[n]['metrics'][key] for n in ['base', 'epoch2']]
        text += f'| {label} | {values[0]:.2%} | {values[1]:.2%} |\n'
    values = [metrics[n]['content_cer_ignoring_punctuation_whitespace'] for n in ['base', 'epoch2']]
    text += f'| 忽略标点和空白的字符错误率 | {values[0]:.2%} | {values[1]:.2%} |\n\n'
    text += f"逐条比较：第二轮更接近参考答案 {report['pairwise_counts']['better']} 条，更远 {report['pairwise_counts']['worse']} 条，持平 {report['pairwise_counts']['tie']} 条。\n\n"
    text += '这是与合成老师答案的文本相似度比较，不能视为真实口语语义通过率。该验证集也曾用于选择checkpoint，因此结果不是独立测试成绩。数字格式差异会计入字符指标；部分老师答案未删除明确放弃的候选，学生正确清理后也可能在字符指标上变差。\n\n'
    for kind, label in [('better', '字面更接近参考答案的样本'), ('worse', '字面更远离参考答案的样本（需判断语义）')]:
        text += f'## {label}\n\n'
        examples = [r for r in result if r['comparison'] == kind]
        examples.sort(key=lambda r: abs(r['base_errors'] - r['epoch2_errors']), reverse=True)
        for row in examples[:5]:
            text += f"### {row['id']}\n\n输入：{row['input']}\n\n参考：{row['reference']}\n\n基座：{row['base_output']}\n\n第二轮：{row['epoch2_output']}\n\n"
    (root / 'comparison.md').write_text(text)
    print(json.dumps(dict(pairwise_counts=report['pairwise_counts'], metrics={n:metrics[n]['metrics'] for n in metrics}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
