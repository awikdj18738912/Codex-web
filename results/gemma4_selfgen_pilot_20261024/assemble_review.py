"""Package completed pilot rounds for review, never as a training export."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BATCHES = ('batch01', 'batch02_b02', 'batch03_b03_e06', 'batch04_controlled_b03')
NAMES = {'B01': '单次明确改口', 'B02': '多次改口／撤销恢复', 'B03': '口述拼字说明清理',
         'D02': '意外断句修复', 'E03': '否定、条件与立场保持', 'E06': '实体占位符保持'}
FLAGS = {
    'batch01/B01_003': '“应该是延长到一个月”存在纠正事实或建议期限的歧义；候选目标“借期是延长到一个月”不自然，隔离而不自行改答案。',
    'batch01/D02_001': '“那个PPT的第三页数据。有点问题”可能是口语话题—评论结构，句号异常证据不足，隔离。',
    'batch02_b02/B02_009': '前两项“周一交，还是周二交”是疑问备选，未明确构成两次即时改口，B02类别证据不足。',
    'batch02_b02/B02_007': '目标“我想买个破壁机就行”句式不自然；不能仅因能删除得到就作合格答案，隔离待复核。',
}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    if (ROOT / 'pilot_summary.json').exists():
        raise SystemExit('Refusing to overwrite a completed review package')
    summaries, retained, quarantined, global_seen = {}, [], [], {}
    raw_counts, pass_counts = Counter(), Counter()
    total_rejects = 0
    for batch in BATCHES:
        summary = json.loads((ROOT / batch / 'summary.json').read_text())
        assert summary['status'] == 'pilot_finished_pending_human_review'
        summaries[batch] = summary
        raws = read(ROOT / batch / 'raw_candidates.jsonl')
        raw_counts.update(row['category'] for row in raws)
        total_rejects += summary['rejected']
        for row in read(ROOT / batch / 'review_candidates.jsonl'):
            row = dict(row, batch=batch, local_id=row['id'], id=batch + '/' + row['id'],
                       group_id=batch + '/' + row['group_id'], human_review_status='pending')
            pass_counts[row['category']] += 1
            key = hashlib.sha256(''.join(row['input'].split()).encode()).hexdigest()
            if row['id'] in FLAGS:
                row['additional_review'] = {'status': 'quarantined', 'reviewer': 'codex_readthrough', 'reason': FLAGS[row['id']]}
                quarantined.append(row)
            elif key in global_seen:
                row['additional_review'] = {'status': 'quarantined', 'reason': 'cross_round_duplicate_source',
                                            'duplicate_of': global_seen[key]}
                quarantined.append(row)
            else:
                row['additional_review'] = {'status': 'no_flag_in_quick_readthrough',
                                            'note': 'Not human approval or correctness guarantee.'}
                global_seen[key] = row['id']
                retained.append(row)
    for name, rows in [('review_candidates.jsonl', retained), ('quarantined.jsonl', quarantined)]:
        (ROOT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = {'status': 'complete_pending_human_review', 'model': 'gemma4-31b-teacher',
               'model_revision': '52f3f65bc7a02d555763bc923bd1d9094898219d',
               'generated_candidates_including_retries': sum(raw_counts.values()),
               'machine_passed_before_additional_review': sum(pass_counts.values()),
               'machine_rejected': total_rejects, 'additional_quarantined': len(quarantined),
               'pending_human_review': len(retained),
               'by_class': {c: {'generated_including_retries': raw_counts[c], 'machine_passed': pass_counts[c],
                                'additional_quarantined': sum(r['category'] == c for r in quarantined),
                                'pending_human_review': sum(r['category'] == c for r in retained)} for c in NAMES},
               'batch_summaries': summaries, 'teacher_api': 'http://127.0.0.1:8014/v1',
               'training_started': False, 'sft_exported': False,
               'limits': ['All inputs synthetic, not real ASR recordings.',
                          'Same quantized teacher used for generation and a separate blind audit.',
                          'Additional Codex readthrough is not human review.',
                          'No test split assigned; do not treat this as an independent evaluation set.',
                          'B03 controlled round uses a finite verified spelling map; E06 masks are program inserted.']}
    assert summary['generated_candidates_including_retries'] == summary['machine_rejected'] + summary['machine_passed_before_additional_review']
    assert summary['machine_passed_before_additional_review'] == len(retained) + len(quarantined)
    for row in retained:
        assert row['human_review_status'] == 'pending'
        assert row['category'] in NAMES
    (ROOT / 'pilot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    md = ['# 自建精修数据：首批样本阅读与审核', '',
          '> 本批由本地Gemma-4-31B-IT QAT W4A16生成；意外断句、占位符和受控拼字说明由程序构造。全部为候选，不是人工金标准，也未进入训练。', '',
          f'初始计划每类12条、共72条。针对字段错误与拼字错误做了有限重试，总共产生{sum(raw_counts.values())}条候选；'
          f'机器检查与同模型另一次盲审通过{sum(pass_counts.values())}条，其中额外隔离{len(quarantined)}条存疑样本，保留{len(retained)}条供你审核。', '',
          '| 小类 | 含重试生成数 | 机器通过 | 额外隔离 | 本稿待审核 |', '| --- | ---: | ---: | ---: | ---: |']
    for c, name in NAMES.items():
        v = summary['by_class'][c]
        md.append(f'| {c} {name} | {v["generated_including_retries"]} | {v["machine_passed"]} | {v["additional_quarantined"]} | {v["pending_human_review"]} |')
    md += ['', '## 审核方法', '',
           '请在“你的决定”填保留／拒绝／待定，在“修改或理由”填写意见。重点检查输入是否自然、最终意图是否唯一、答案是否少删／多删或改写过度。你在Markdown中修改不会自动回写JSONL。', '',
           'B03首两轮自由生成的拼字说明失败较多；受控重试让模型只生成语境，程序按既定“姓名—字词”映射插入说明。E06同样由程序在输入和目标中同步替换实体。映射与字面原文保存在JSONL的construction字段中。', '',
           '原始请求／响应、各轮脚本快照、拒收原因、逐类汇总均保留；没有从公开数据、文档示例或测试集抽种子。', '']
    for c, name in NAMES.items():
        md += [f'## {c}：{name}', '', '| ID | 输入 | 候选目标 | 你的决定 | 修改或理由 |', '| --- | --- | --- | --- | --- |']
        for row in retained:
            if row['category'] == c:
                esc = lambda x: str(x).replace('|', '／').replace('\n', ' ')
                md.append('| ' + ' | '.join(map(esc, [row['id'], row['input'], row['reference'], '待审核', ''])) + ' |')
        md.append('')
    md += ['## 额外隔离：不要作为合格训练数据使用', '',
           '| ID | 输入 | 候选目标 | 隔离原因 |', '| --- | --- | --- | --- |']
    for row in quarantined:
        esc = lambda x: str(x).replace('|', '／').replace('\n', ' ')
        md.append('| ' + ' | '.join(map(esc, [row['id'], row['input'], row['reference'], row['additional_review']['reason']])) + ' |')
    md += ['', '## 文件与边界', '',
           '- `review_candidates.jsonl`：本稿对应候选，全部human_review_status=pending。',
           '- `quarantined.jsonl`：机器通过后被额外隔离的争议项，原答案未手工修成“正确答案”。',
           '- `pilot_summary.json`：总量、逐类数量、各轮统计及限制。',
           '- 各batch目录：raw_candidates、teacher_calls、rejected、generation_config和generator_snapshot等证据。',
           '- `start_teacher.sh`与`server_retry.log`：本轮成功启动配置和运行日志。', '',
           '小样本通过率只用于检查生成流程，不代表真实语音性能或可扩大生成规模的保证。下一步先根据你的审核意见改提示与准入规则，再确定扩量和训练划分。', '']
    (ROOT / '首批自建数据_阅读审核.md').write_text('\n'.join(md))
    print(json.dumps({k: v for k, v in summary.items() if k != 'batch_summaries'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
