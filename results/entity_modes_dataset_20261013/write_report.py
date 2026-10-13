"""Render the complete paired comparison. No inference or library changes."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
NAMES={'context':'词库辅助＋基础实体替换','audio':'原音复核＋基础实体替换'}


def main():
    score=json.loads((OUT/'summary.json').read_text());assert score['complete']
    groups=score['groups'];all_=groups['all']
    rows=[json.loads(l) for l in (OUT/'comparisons.jsonl').open()]
    assert len(rows)==2506
    def pct(v):return f'{v:.2%}'
    def metric(g,a):
        v=g[a]
        return f"{v['hits']}/{v['expected_mentions']}（{pct(v['entity_recall'])}）" if v['expected_mentions'] else '无目标实体'
    lines=['# 词库辅助与原音复核：同音频配对测试', '',
           '2026-10-13。完整执行 2,506 条既有真实音频、5,012 次新 Web 会话，两种模式各跑一次；没有复用此前 ASR 结果。测试未修改生产功能或正式词库。', '',
           '每条音频使用预先冻结的领域词库，两种模式都保留相同的基础实体替换、精修门控和数字处理。词库辅助只开首遍 ASR context；原音复核只开候选驱动的追加音频确认。二者没有同时开启。', '',
           f"本轮原音复核净多命中 {all_['audio']['hits']-all_['context']['hits']} 次实体，平均增加 {all_['audio']['total_seconds']['mean']-all_['context']['total_seconds']['mean']:.3f} 秒；词库辅助更快。复核有 {all_['audio']['replay_absent_changes']} 次实际改字与参考不符，因此仍需保守使用。速度优先可选词库辅助，准确率优先可按需复核。", '',
           '## 完整结果', '',
           '| 范围 | 模式 | 实体提及正确数／总数 | 规范化 CER | 平均耗时 | 中位耗时 | P90 耗时 | 中位收尾 |',
           '|---|---|---|---|---|---|---|---|']
    for name,key in [('全部','all'),('原大规模测试集（2,410 条）','corpus:large_entity_benchmark_v1_20261010'),
                     ('SpeechIO 汽车／足球（96 条）','corpus:entity_reliable_data_20261012_pilot')]:
        g=groups[key]
        for a in NAMES:
            v=g[a];t=v['total_seconds']
            lines.append(f"|{name}|{NAMES[a]}|{metric(g,a)}|{pct(v['CER'])}|{t['mean']:.3f}s|{t['median']:.3f}s|{t['p90']:.3f}s|{v['tail_seconds']['median']:.3f}s|")
    lines+=['','耗时从本地 WebSocket 连接开始，直到完整 final 返回，包含词库准备、新 ASR、精修及复核。文件读盘、解码、浏览器渲染和用户端代理网络不在计时中。串行运行一个样本，两种模式交错且每对顺序固定随机；上传为每块 1 秒 PCM、最多 4 块待确认，无人工等待。这里的秒数不能直接当作用户浏览器耗时。', '',
            '## 提示容量与实体类型', '',
            '词库辅助沿用当前最多 16 个词、按现有优先级选择的规则，没有利用当前音频的参考答案挑词。原音复核沿用当前候选选择及最多 8 个复核提示词规则。领域词库来自此前允许按测试转录建立的词库，因此本测试衡量的是预知词汇和领域条件下的能力，不代表未知词库泛化。', '',
            '| 目标是否进入前置 16 词提示 | 目标提及总数 | 词库辅助正确数 | 原音复核正确数 |',
            '|---|---|---|---|']
    for label,h,e in [('进入提示','prompt_hits','prompt_expected'),('未进入提示','outside_prompt_hits','outside_prompt_expected')]:
        lines.append(f"|{label}|{all_['context'][e]}|{all_['context'][h]}|{all_['audio'][h]}|")
    lines+=['','| 实体类型 | 提及总数 | 词库辅助正确数／召回率 | 原音复核正确数／召回率 |','|---|---|---|---|']
    for typ,s in score['entity_types'].items():
        lines.append(f"|{typ}|{s['context']['expected']}|{s['context']['hits']}／{pct(s['context']['recall'])}|{s['audio']['hits']}／{pct(s['audio']['recall'])}|")
    lines+=['','## 分领域结果','','| 领域词库 | 音频数 | 词库辅助实体正确 | 原音复核实体正确 | 辅助 CER | 复核 CER | 辅助平均耗时 | 复核平均耗时 |','|---|---|---|---|---|---|---|---|']
    for key,g in sorted(groups.items()):
        if not key.startswith('library:'):continue
        a,b=g['context'],g['audio']
        lines.append(f"|{key[8:]}|{a['samples']}|{metric(g,'context')}|{metric(g,'audio')}|{pct(a['CER'])}|{pct(b['CER'])}|{a['total_seconds']['mean']:.3f}s|{b['total_seconds']['mean']:.3f}s|")
    lines+=['','## 额外调用、对照与误改诊断','','| 指标 | 词库辅助 | 原音复核 |','|---|---|---|']
    for label,key in [('精修会话模型调用总数','refiner_calls'),('其中精修重试','refiner_retries'),
                      ('实际原音复核样本数','replay_sample_count'),('原音复核 checks 总数','replay_checks'),
                      ('原音复核改字样本数','replay_changed_samples'),('原音复核改动总数','replay_applied_changes'),
                      ('原音复核直接改成参考未出现词的次数','replay_absent_changes'),
                      ('提示回显回退样本数','context_guard_fallback_samples'),
                      ('参考中不存在的词库词提及数','dictionary_absent_mentions'),
                      ('超过参考次数的词库词提及数','dictionary_excess_mentions'),
                      ('出现上述词的样本数','dictionary_absent_samples')]:
        lines.append(f"|{label}|{all_['context'][key]}|{all_['audio'][key]}|")
    p=all_['paired']
    lines+=['',f"按音频配对：原音复核实体命中更多 {p['audio_more_entity_hits']} 条，词库辅助更多 {p['context_more_entity_hits']} 条；原音复核 CER 更低 {p['audio_lower_CER_samples']} 条，词库辅助更低 {p['context_lower_CER_samples']} 条，其余 {p['equal_CER_samples']} 条 CER 相同。词库辅助更快 {p['context_faster_samples']} 条，原音复核更快 {p['audio_faster_samples']} 条。", '',
            '精修会话调用总数沿用 session_refiner_call_count 口径，可能包含重复审核；原音复核 ASR checks 单独统计，不计入精修调用数。“参考中不存在的词库词”按完整参考和最长不重叠词库规范名匹配统计，仅作为误改线索，尚未经逐条听音审核；不是 NER 精确率或人工认定的幻觉。超过参考次数的词库词提及数还包括参考中有此词、但输出重复次数更多的情况。负样本／无登记目标对照也可能出现不在登记库中的真实实体。', '',
            '| 对照组 | 音频数 | 辅助 CER | 复核 CER | 辅助无参考词提及 | 复核无参考词提及 |','|---|---|---|---|---|---|']
    for key in ('cohort:negative','cohort:no_registered_target'):
        g=groups[key];a,b=g['context'],g['audio']
        lines.append(f"|{key[7:]}|{a['samples']}|{pct(a['CER'])}|{pct(b['CER'])}|{a['dictionary_absent_mentions']}|{b['dictionary_absent_mentions']}|")
    lines+=['','## 配对不确定性','','按音频配对自举 2,000 次，以下均为“原音复核减词库辅助”的 95% 区间；区间跨零时，本次观察不足以确定差异方向。','','| 差值指标 | 95% 自举区间 |','|---|---|']
    for label,key,mult,unit in [('实体召回率','entity_recall',100,'百分点'),('CER','CER',100,'百分点'),('平均耗时','mean_seconds',1,'秒')]:
        ci=p['audio_minus_context_95pct_bootstrap_CI'][key]
        lines.append(f'|{label}|[{ci[0]*mult:.3f}, {ci[1]*mult:.3f}] {unit}|')
    lines+=['','## 可复查样例','','按配对字符错误差的绝对值排序，双方各展示最多 3 条；全部差异可查 comparisons.jsonl。','','| 更准确的一方 | 音频 ID | 标准参考 | 词库辅助最终文字 | 原音复核最终文字 | 两方字符错误数 |','|---|---|---|---|---|---|']
    def cell(text):return text.replace('|','／').replace('\n',' ')[:240]
    for winner in NAMES:
        other='audio' if winner=='context' else 'context'
        candidates=[d for d in rows if d['values'][winner]['errors']<d['values'][other]['errors']]
        candidates.sort(key=lambda d:(-abs(d['values']['audio']['errors']-d['values']['context']['errors']),d['id']))
        for d in candidates[:3]:
            a,b=d['values']['context'],d['values']['audio']
            lines.append(f"|{NAMES[winner]}|{d['id']}|{cell(d['reference'])}|{cell(a['text'])}|{cell(b['text'])}|{a['errors']}／{b['errors']}|")
    lines+=['','## 复核改成参考未出现词的样例','',
            '以下记录的是复核模块实际应用的改动，不将其他 ASR 错误归因于复核；尚未人工逐条听音确认。', '',
            '| 音频 ID | 标准参考 | 复核观察词 → 修改词 |', '|---|---|---|']
    for d in rows:
        for c in d['values']['audio']['replay_absent_change_details']:
            lines.append(f"|{d['id']}|{cell(d['reference'])}|{cell(c['observed'])} → {cell(c['canonical'])}|")
    lines+=['','## 范围与限制','',
            '- 2,506 条唯一 PCM 全部配对；主集 2,410 条、SpeechIO 补充 96 条，共 2,860 次标注实体提及。主体是 2–15 秒短音频，新闻领域占大多数；必须结合分领域结果，不能将总平均推广到数分钟长音频。',
            '- 实体得分是标准规范名的文本出现次数，按标注次数封顶；不证明人物身份，别名及重叠实体也受此口径限制。CER 对双方和参考应用同一保守数字规范化、繁简及 Unicode 规则。',
            '- 相同的保守数字规则仍不能保证所有语境下格式等价：例如本轮“一千米／1000米”可能未归一为相同文本。因此 CER 含部分数字表示和语气词差异，不能把全部配对 CER 增加都视作实体模块改错。',
            '- 数据来源保留原有标注与参考，未逐条重新人工听音；SpeechIO 补充集为参考中整理的实体标注。没有使用受保护的未见测试集。',
            '- 两种模式都使用目前生产配置中的精修门控等设置；结果是两套组合的最终用户输出，不能归因于只开其中一个模块。精修会话调用数和 ASR 复核 checks 分别列出。',
            '- checks 是算法复核检查次数，提示回显保护可能额外触发无提示 ASR 回退；它不等于全部底层 ASR 请求数，实测总耗时已经包含这些工作。',
            '- 同一实验 Web 进程和同一 ASR／精修服务，生产服务保持运行；时间包含实际处理波动，不代表所有网络、并发或浏览器的表现。', '',
            '证据：protocol.json、manifest.json、worker.json、results.jsonl、summary.json、comparisons.jsonl、verification.json，以及 Luna 独立审计。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
