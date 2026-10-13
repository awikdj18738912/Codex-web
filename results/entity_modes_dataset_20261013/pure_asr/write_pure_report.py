"""Render the pure-ASR add-on report after full collection and offline scoring."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
EXPERIMENT=OUT.parent
PIPELINES={
    'pure_asr_raw':'纯 ASR（raw）',
    'context_clean':'词库辅助＋基础实体替换及精修',
    'audio_clean':'原音复核＋基础实体替换及精修',
}
STAGES={
    'pure_asr_raw':'纯 ASR raw',
    'context_raw':'词库辅助 ASR 输出（后处理前）',
    'context_clean':'词库辅助 clean',
    'audio_raw':'原音复核后 ASR 输出（后处理前）',
    'audio_clean':'原音复核 clean',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def pct(value):
    return f'{value:.2%}' if value is not None else '—'


def metric_text(group, method):
    value=group[method]
    if not value['expected_mentions']:
        return f"无目标实体；CER {pct(value['CER'])}"
    return f"{value['hits']}/{value['expected_mentions']}（{pct(value['entity_recall'])}）；CER {pct(value['CER'])}"


def cell(text):
    return str(text).replace('|','／').replace('\n',' ')[:240]


def render():
    summary=read_json(OUT/'summary.json')
    assert summary.get('complete') and summary['sample_count']==2506 and summary['planned_samples']==2506, 'pure-ASR full scoring is not complete'
    assert summary['pure_asr_request_validation']['validated_sessions']==2506
    assert summary['pure_asr_request_validation']['context_supplied_true']==0
    paired=read_jsonl(OUT/'paired_details.jsonl')
    assert len(paired)==2506 and len({x['id'] for x in paired})==2506
    old_summary=read_json(EXPERIMENT/'summary.json')
    assert old_summary.get('complete') and old_summary['completed_pairs']==2506
    groups=summary['groups']; all_metrics=groups['all']['metrics']; all_timing=groups['all']['run_timing']
    lines=[
        '# 纯 ASR 补充对照：实体辅助、原音复核与无辅助识别', '',
        f"本轮纯 ASR 命中 {all_metrics['pure_asr_raw']['hits']}/{all_metrics['pure_asr_raw']['expected_mentions']} 次实体，词库辅助组合最终命中 {all_metrics['context_clean']['hits']} 次，原音复核组合最终命中 {all_metrics['audio_clean']['hits']} 次。前置词库的 ASR 输出净多命中 {all_metrics['context_raw']['hits']-all_metrics['pure_asr_raw']['hits']} 次；经过完整后处理后，词库辅助组合相对纯 ASR 的净命中差为 {all_metrics['context_clean']['hits']-all_metrics['pure_asr_raw']['hits']} 次。相同总命中数不代表逐条结果完全相同。", '',
        '纯 ASR 补跑覆盖与前两组完全相同的 2,506 条音频，共 2,506 次新 Web 会话。下表的准确率与时间按同一音频 ID 配对汇总。纯 ASR 通过独立私有 WebSocket 透明转发到同一 ASR 服务，只返回原始 ASR 文本。', '',
        '本补跑发生在原词库辅助／原音复核两组之后，不是三组同时交错采集。纯组的 WebSocket 仅走 raw ASR 路径，每条音频均发起新识别，没有复用缓存 ASR 响应；它没有浏览器 UI、实体库、精修、数字规范化或文本清理。仅关闭 UI 中的实体开关不等同于本实验的纯 ASR 旁路。准确率差异也不能全部归因于实体辅助模块，因为三条流程的后处理范围不同。', '',
        '## 三条流程的最终输出', '',
        '| 流程 | 实体命中／提及 | 实体召回 | CER | 平均会话时间 | 中位会话时间 | P90 会话时间 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method,label in PIPELINES.items():
        metric=all_metrics[method]; timing=all_timing['pure_asr' if method=='pure_asr_raw' else ('context' if method=='context_clean' else 'audio')]['total_seconds']
        lines.append(f"|{label}|{metric['hits']}/{metric['expected_mentions']}|{pct(metric['entity_recall'])}|{pct(metric['CER'])}|{timing['mean']:.3f}s|{timing['median']:.3f}s|{timing['p90']:.3f}s|")
    lines += [
        '',
        '纯 ASR 时间来自透明转发 WebSocket 的 connect-to-final；上下文和原音复核时间来自先前完整 Web final 会话，包括其词库准备、ASR 和对应处理。两组均不含浏览器渲染、磁盘读盘／解码或用户代理网络时间。纯组是补跑，服务状态和采集时段也不与前两组完全同步；这些会话时间不能视作同一轮随机交错的直接延迟竞赛。', '',
        '## 五种输出阶段', '',
        '以下 raw/clean 是准确率视图，不是五条独立运行路径。只有上表三条流程的真实会话时间可比较；context/audio 的 raw 与 clean 共用同一次会话，不能为 raw 视图另报一份延迟。', '',
        '| 输出阶段 | 实体命中／提及 | 实体召回 | CER |', '|---|---:|---:|---:|',
    ]
    for method,label in STAGES.items():
        metric=all_metrics[method]
        lines.append(f"|{label}|{metric['hits']}/{metric['expected_mentions']}|{pct(metric['entity_recall'])}|{pct(metric['CER'])}|")

    corpus_names={
        'corpus:large_entity_benchmark_v1_20261010':'原大规模测试集（2,410 条）',
        'corpus:entity_reliable_data_20261012_pilot':'SpeechIO 汽车／足球补充集（96 条）',
    }
    lines += ['', '## 主集与补充集', '',
              '| 数据范围 | 流程 | 音频数 | 实体命中／提及 | 召回 | CER | 平均会话时间 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for key,label in corpus_names.items():
        group=groups[key]
        for method,流程 in PIPELINES.items():
            metric=group['metrics'][method]
            timing_arm='pure_asr' if method=='pure_asr_raw' else ('context' if method=='context_clean' else 'audio')
            timing=group['run_timing'][timing_arm]['total_seconds']
            lines.append(f"|{label}|{流程}|{metric['samples']}|{metric['hits']}/{metric['expected_mentions']}|{pct(metric['entity_recall'])}|{pct(metric['CER'])}|{timing['mean']:.3f}s|")

    lines += ['', '## 实体类型', '',
              '召回只计相应类型的标注实体。CER 列是包含该类型实体的录音上的全文 CER，不是该实体类型单独造成的字符错误。', '',
              '| 类型 | 提及数 | 纯 ASR 命中／召回 | 词库辅助 clean 命中／召回 | 原音复核 clean 命中／召回 | 含该类型样本的全文 CER（纯／词库／复核） |',
              '|---|---:|---:|---:|---:|---|']
    for typ,values in sorted(summary['entity_types'].items()):
        pure=values['pure_asr_raw']; context=values['context_clean']; audio=values['audio_clean']
        errors=' / '.join(pct(values[m]['whole_transcript_CER_on_type_subset']) for m in ('pure_asr_raw','context_clean','audio_clean'))
        lines.append(f"|{typ}|{pure['expected_mentions']}|{pure['hits']}／{pct(pure['entity_recall'])}|{context['hits']}／{pct(context['entity_recall'])}|{audio['hits']}／{pct(audio['entity_recall'])}|{errors}|")

    lines += ['', '## 领域分组', '',
              '| 领域 | 音频数 | 纯 ASR hits/total、CER、平均秒 | 词库 clean hits/total、CER、平均秒 | 复核 clean hits/total、CER、平均秒 |',
              '|---|---:|---|---|---|']
    for key in sorted(k for k in groups if k.startswith('library:')):
        group=groups[key]; n=group['metrics']['pure_asr_raw']['samples']; cols=[]
        for method in ('pure_asr_raw','context_clean','audio_clean'):
            value=group['metrics'][method]
            arm='pure_asr' if method=='pure_asr_raw' else ('context' if method=='context_clean' else 'audio')
            mean=group['run_timing'][arm]['total_seconds']['mean']
            cols.append(f"{value['hits']}/{value['expected_mentions']}，{pct(value['CER'])}，{mean:.3f}s")
        lines.append(f"|{key[len('library:'):]}|{n}|{cols[0]}|{cols[1]}|{cols[2]}|")

    lines += ['', '## 无目标实体对照', '',
              '| 对照组 | 音频数 | 纯 ASR CER | 词库辅助 clean CER | 原音复核 clean CER |',
              '|---|---:|---:|---:|---:|']
    for cohort in ('negative','no_registered_target'):
        key='cohort:'+cohort
        if key not in groups: continue
        group=groups[key]['metrics']; n=group['pure_asr_raw']['samples']
        lines.append(f"|{cohort}|{n}|{pct(group['pure_asr_raw']['CER'])}|{pct(group['context_clean']['CER'])}|{pct(group['audio_clean']['CER'])}|")

    lines += ['', '## 同音频文本差异样例', '',
              '每个比较按全文 CER 的绝对差排序，每一方最多列 3 条更低 CER 的样例；按 ID 配对，字符错误数使用同一保守规范化口径。', '']
    comparisons=(('context_clean','词库辅助 clean'),('audio_clean','原音复核 clean'))
    for baseline,label in comparisons:
        lines += [f'### 纯 ASR 与{label}', '',
                  '| CER 更低的一方 | 音频 ID | 参考 | 纯 ASR raw | 对照最终文字 | 字符错误（纯／对照） | entity hits（纯／对照） |',
                  '|---|---|---|---|---|---:|---:|']
        candidates=[]
        for row in paired:
            pure=row['values']['pure_asr_raw']; other=row['values'][baseline]
            delta=pure['errors']-other['errors']
            if delta:
                winner='纯 ASR raw' if delta<0 else label
                candidates.append((winner,row,abs(delta)))
        for winner_side in ('纯 ASR raw',label):
            side=[item for item in candidates if item[0]==winner_side]
            side.sort(key=lambda item:(-item[2],item[1]['id']))
            for winner,row,_ in side[:3]:
                pure=row['values']['pure_asr_raw']; other=row['values'][baseline]
                lines.append(f"|{winner}|{row['id']}|{cell(row['reference'])}|{cell(pure['text'])}|{cell(other['text'])}|{pure['errors']}／{other['errors']}|{pure['hits']}／{other['hits']}|")

    lines += ['', '## 口径与限制', '',
              '- 全部文本均用相同的保守 zh-ITN、繁转简、NFKC/casefold 和 Unicode 标点／空白／控制符移除规则计 CER 和 canonical 命中。数字规范化不保证所有等价表达收敛，例如“一千米／1000米”可能保留 CER 差异；不要把所有字符错误差都解释为实体改错。',
              '- 实体命中是规范名字面出现次数，并按冻结标注次数封顶；不做身份链接、别名扩展或一般 NER 精确率判断。类型 CER 是包含该类型的录音上的全文 CER。',
              '- 纯 ASR 是新补跑，不与前两组同时三臂交错。它走透明转发的 raw 路径，没有浏览器时间；两组旧数据使用完整 Web 输出/时间。关掉 UI 的实体开关并不能复现纯 ASR 条件，因为这个实验同时旁路实体、精修、数字规则和文本清理。',
              '- ASR 服务与历史实验一致，但补跑时间不同。服务负载、时间段和流程配置可能影响比较；模式间准确率变化不能全部归因于实体辅助或复核模块。',
              '- 主集为 2,410 条短语音，补充 SpeechIO 汽车／足球为 96 条；结果不代表长音频或未知领域的泛化。正例与无目标实体对照按 frozen cohort 单独列出。',
              '',
              f"协议记录：{(OUT/'protocol.json').name}。纯组会话请求验证 {summary['pure_asr_request_validation']['validated_sessions']}/{summary['sample_count']}；ASR start/chunk/finish 请求分别为 {summary['pure_asr_request_validation']['start_requests']}/{summary['pure_asr_request_validation']['chunk_requests']}/{summary['pure_asr_request_validation']['finish_requests']}；context_supplied=true 为 {summary['pure_asr_request_validation']['context_supplied_true']}。", '',
              '评分输入来自父目录冻结 references.json；推理输入未包含参考或 gold。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(OUT/'REPORT.md')


if __name__=='__main__':
    render()
