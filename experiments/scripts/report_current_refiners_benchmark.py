"""Summarize actual text benchmark. No inference and no dataset mutation."""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import statistics
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/refiner_text_benchmark_20261028'
NAMES = {'lora':'0.5B LoRA（已合并）', 'lora_gate':'0.5B LoRA＋RBT3门控',
         'full':'0.5B 全量', 'full_gate':'0.5B 全量＋RBT3门控', 'agentic1b':'原 AgenticASR 1B'}


def read(path):
    return [json.loads(x) for x in Path(path).open()]


def normalize(text):
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def distance(a, b):
    last = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1]+1, last[j]+1, last[j-1]+(x != y)))
        last = row
    return last[-1]


def percentile(values, p):
    a = sorted(values)
    t = (len(a)-1)*p
    lo = int(t)
    return a[lo] + (a[min(lo+1,len(a)-1)]-a[lo])*(t-lo)


def main():
    progress = json.loads((OUT/'progress.json').read_text())
    assert progress['phase'] == 'complete', progress
    allrows = read(OUT/'text_predictions.jsonl')
    first = [r for r in allrows if r['repeat'] == 0]
    second = [r for r in allrows if r['repeat'] == 1]
    manifest = read(OUT/'text_manifest.jsonl')
    assert len(first) == 256*5 and len(second) == 64*5
    by = {name:{r['id']:r for r in first if r['variant']==name} for name in NAMES}
    assert all(set(group)=={r['id'] for r in manifest} for group in by.values())
    summary = {'unique_inputs':len(manifest), 'weak_labels':dict(Counter(r['label'] for r in manifest)),
               'categories':dict(Counter(r['category'] for r in manifest)), 'pipelines':{}, 'gate_comparisons':{}}
    for name, mapping in by.items():
        rows = list(mapping.values())
        times = [r['postprocess_seconds']*1000 for r in rows]
        positive = [r for r in rows if r['label']]
        clean = [r for r in rows if not r['label']]
        repeat = [r for r in second if r['variant']==name]
        m = {'n':len(rows), 'mean_ms':statistics.mean(times), 'median_ms':statistics.median(times),
             'p95_ms':percentile(times,.95), 'mean_gate_ms':statistics.mean(r['gate_seconds']*1000 for r in rows),
             'mean_refiner_ms':statistics.mean(r['refiner_seconds']*1000 for r in rows),
             'refiner_calls':sum(r['route'] for r in rows),
             'exact_matches':sum(r['output']==r['reference'] for r in rows),
             'normalized_matches':sum(normalize(r['output'])==normalize(r['reference']) for r in rows),
             'positive_exact_matches':sum(r['output']==r['reference'] for r in positive),
             'weak_clean_changed':sum(r['output']!=r['text'] for r in clean),
             'normalized_char_error_rate':sum(distance(normalize(r['reference']),normalize(r['output'])) for r in rows)/sum(len(normalize(r['reference'])) for r in rows),
             'generated_tokens_mean':statistics.mean(r['generated_tokens'] for r in rows),
             'non_eos':sum(not r['eos'] for r in rows),
             'repeat_output_changes':sum(r['output']!=mapping[r['id']]['output'] for r in repeat),
             'repeat_first64_mean_ms':statistics.mean(mapping[r['id']]['postprocess_seconds']*1000 for r in repeat),
             'repeat_second64_mean_ms':statistics.mean(r['postprocess_seconds']*1000 for r in repeat)}
        summary['pipelines'][name] = m
    frozen_gate = {r['id']:r for r in read('/home/aim0/data/ASR/training/refiner_noC02_gate_rbt3_20261027/balanced_test_predictions.jsonl')}
    cases = []
    for base in ('lora','full'):
        gated = by[base+'_gate']
        original = by[base]
        losses, gains, gated_differences = [], [], []
        for ident, g in gated.items():
            old = original[ident]
            if g['output'] != old['output']:
                reason = ('lost_exact' if old['output']==old['reference'] else
                          'gained_exact' if g['output']==g['reference'] else 'both_differ_from_reference')
                cases.append({'base':base, 'id':ident, 'input':g['text'], 'reference':g['reference'],
                              'ungated':old['output'], 'gated':g['output'], 'route':g['route'], 'reason':reason})
                gated_differences.append(ident)
                if reason == 'lost_exact': losses.append(ident)
                if reason == 'gained_exact': gains.append(ident)
        gm = summary['pipelines'][base+'_gate']
        bm = summary['pipelines'][base]
        paired = [(original[k]['postprocess_seconds'], gated[k]['postprocess_seconds']) for k in original]
        rng = random.Random(20261028)
        boot = []
        for _ in range(2000):
            sample = rng.choices(paired, k=len(paired))
            boot.append(1-sum(g for b,g in sample)/sum(b for b,g in sample))
        summary['gate_comparisons'][base] = {
            'mean_latency_reduction':1-gm['mean_ms']/bm['mean_ms'],
            'speedup':bm['mean_ms']/gm['mean_ms'],
            'paired_bootstrap_latency_reduction_95ci':[percentile(boot,.025),percentile(boot,.975)],
            'lost_exact_ids':losses, 'gained_exact_ids':gains,
            'changed_output_ids':gated_differences,
            'weak_positive_skipped':sum(not r['route'] and r['label']==1 for r in gated.values()),
            'weak_clean_skipped':sum(not r['route'] and r['label']==0 for r in gated.values()),
            'gate_route_disagrees_with_frozen':sum(r['route']!=frozen_gate[r['id']]['route_to_refiner'] for r in gated.values()),
            'routed_output_differs_from_ungated':sum(r['route'] and r['output']!=original[r['id']]['output'] for r in gated.values())}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'gate_output_differences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    with (OUT/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=['pipeline']+list(next(iter(summary['pipelines'].values())).keys()))
        writer.writeheader()
        for name, m in summary['pipelines'].items(): writer.writerow(dict(pipeline=name,**m))
    lines = ['# GPU0 纯文本精修：速度与候选参考对照','',
        '本轮按要求不使用音频、不运行ASR。下面的总耗时仅指“输入文本→门控（可选）→精修→输出正文”，不能视为转录加精修耗时。','',
        '256条固定测试文本，弱标签需修改128条、不需修改128条。五条链路均逐条实际推理；首轮用于下表，固定前64条再重复一轮检查稳定性。参考答案和类别没有进入模型提示词。', '',
        '| 方案 | 平均总耗时/ms | 中位数/ms | P95/ms | 精修调用 | 匹配候选参考 | 不需修改候选被改动 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,m in summary['pipelines'].items():
        lines.append(f"| {NAMES[name]} | {m['mean_ms']:.1f} | {m['median_ms']:.1f} | {m['p95_ms']:.1f} | {m['refiner_calls']}/256 | {m['exact_matches']}/256 | {m['weak_clean_changed']}/128 |")
    lines += ['', '## 门控带来的变化','',
              '| 精修模型 | 平均耗时减少 | 加速倍数 | 门控平均耗时/ms | 原本匹配但门控后不匹配 | 门控后新增匹配 | 漏掉弱标签需修改 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for base,m in summary['gate_comparisons'].items():
        gm = summary['pipelines'][base+'_gate']
        lines.append(f"| {NAMES[base]} | {m['mean_latency_reduction']:.1%} | {m['speedup']:.2f}× | {gm['mean_gate_ms']:.2f} | {len(m['lost_exact_ids'])} | {len(m['gained_exact_ids'])} | {m['weak_positive_skipped']} |")
    lines += ['', '“漏掉弱标签需修改”按输入与候选参考判断；“原本匹配但门控后不匹配”比较同一句的实际输出。二者含义不同。候选参考仍有噪声，均不能直接称为人工确认的准确率。', '',
        '## 测试条件','',
        '- GPU0：RTX 3090，单卡、逐条串行、batch size 1，模型预热后计时。模型同时驻留但逐次执行，没有模型并行计算。',
        '- 三个精修模型均为BF16、SDPA、贪心生成，max_new_tokens=1024，显式开启KV cache。',
        '- LoRA使用当前去C02的checkpoint-990，在内存合并后推理；全量使用当前去C02的最佳模型。未覆盖模型文件。',
        '- RBT3门控使用原训练权重、float32、验证集选定阈值0.005。超512 token时直接送精修。',
        '- LoRA和全量使用相同训练提示词；AgenticASR 1B使用本机原离线精修提示词。1B生成的KEY也计耗时，正文比较时移除KEY。提示词差异反映本机配置，不是统一提示词的模型架构对照。',
        '- 当前候选参考要求保留输入数字写法；旧1B会自行转换部分数字和时间。这些差异会影响匹配数，需要与实质性漏修、删除信息区分，不能只用逐字匹配宣称某个模型客观更准确。',
        '- 总耗时包含分词、设备传输、门控、完整生成、解码与CUDA同步；不包含启动加载、网络、音频转录、流式分段、实体保护或ITN。',
        '- 模型均使用相同生成上限，不以候选参考长度约束生成；非EOS和重复轮输出差异见下表。', '',
        '## 重复检查','',
        '| 方案 | 相同64句首轮均值/ms | 次轮均值/ms | 两轮输出不同 | 首轮未EOS |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,m in summary['pipelines'].items():
        lines.append(f"| {NAMES[name]} | {m['repeat_first64_mean_ms']:.1f} | {m['repeat_second64_mean_ms']:.1f} | {m['repeat_output_changes']} | {m['non_eos']} |")
    lines += ['', '## 门控造成的逐句差异','']
    if not cases: lines += ['本轮门控与不加门控的正文完全一致。','']
    for i,r in enumerate(cases,1):
        lines += [f"### {i}. {NAMES[r['base']]} · {r['reason']}", '',
            '**输入**：'+r['input'], '', '**候选参考**：'+r['reference'], '',
            '**不加门控**：'+r['ungated'], '', '**加门控**：'+r['gated'], '', '`'+r['id']+'`','']
    lines += ['## 适用范围','',
        '50/50是本次人为固定的测试比例。线上干净文本比例变化时，门控跳过比例和整体加速会变化。测试文本来自之前使用的留出集，部分已被查看；不能由256句宣称真实录音准确率或线上速度获得保证。', '',
        'LoRA合并依据：[Hugging Face PEFT文档](https://huggingface.co/docs/peft/developer_guides/checkpoint)、[官方LoRA实现说明](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md)。原1B模型：[模型卡](https://huggingface.co/Andrew0425/AgenticASR-Refiner)、[官方项目](https://github.com/AnXMuy/AgenticASR)。', '',
        '## 原始结果','',
        f'- [逐条输入和输出]({OUT}/text_predictions.jsonl)',
        f'- [汇总JSON]({OUT}/summary.json)', f'- [CSV表格]({OUT}/summary.csv)',
        f'- [固定测试协议]({OUT}/protocol.json)', f'- [固定输入集]({OUT}/text_manifest.jsonl)',
        f'- [模型加载与运行环境]({OUT}/runtime.json)',
        f'- [执行脚本]({ROOT}/experiments/scripts/benchmark_current_refiners_gpu0.py)', '']
    (OUT/'GPU0纯文本精修速度与质量对照.md').write_text('\n'.join(lines)+'\n')
    detail = ['# 256句五方案逐句对照','',
              '“候选参考”尚未全部人工确认。原1B的数字格式变化可能造成逐字不匹配；所有原始生成含KEY的版本保存在JSONL。','']
    for i, sample in enumerate(manifest,1):
        ident = sample['id']
        detail += [f"## {i:03d} · {sample['category']} · {'弱标签需修改' if sample['label'] else '弱标签不需修改'}",'',
                   '**输入**：'+sample['text'],'','**候选参考**：'+sample['reference'],'']
        for name in NAMES:
            r = by[name][ident]
            suffix = '（门控跳过）' if name.endswith('_gate') and not r['route'] else ''
            detail += ['**'+NAMES[name]+suffix+'**：'+r['output'],'']
        detail += ['样本ID：`'+ident+'`','']
    (OUT/'256句五方案逐句对照.md').write_text('\n'.join(detail)+'\n')
    # Auxiliary quality-only replay; these cached outputs are NOT timed as fresh inference.
    routes = {r['id']:r for r in read('/home/aim0/data/ASR/training/refiner_noC02_gate_rbt3_20261027/test_predictions.jsonl')}
    aux = {}
    for name, filename in [('lora','no_C02_test.jsonl'),('full','no_C02_full_test.jsonl')]:
        rows = read(ROOT/'results/refiner13_no_C02_20261027/evaluation'/filename)
        assert len(rows) == 1757
        assert all(routes[r['id']]['text']==r['input'] for r in rows)
        aux[name] = {'n':len(rows), 'ungated_matches':sum(r['prediction']==r['reference'] for r in rows),
            'gated_matches':sum((r['prediction'] if routes[r['id']]['route_to_refiner'] else r['input'])==r['reference'] for r in rows),
            'lost_matches':sum(not routes[r['id']]['route_to_refiner'] and r['prediction']==r['reference'] and r['input']!=r['reference'] for r in rows),
            'refiner_calls':sum(routes[r['id']]['route_to_refiner'] for r in rows)}
    (OUT/'cached1757_quality_only.json').write_text(json.dumps(aux,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'GPU0纯文本精修速度与质量对照.md').open('a') as f:
        f.write('\n## 附录：已有1,757条输出的门控路由对照\n\n这些精修输出和门控判断来自此前测试，下面只重算质量指标，不计作本轮新推理或时延测量。原集合的需修改比例较高，与本轮50/50不相同。\n\n| 模型 | 不加门控匹配参考 | 加门控匹配参考 | 门控损失匹配 |\n| --- | ---: | ---: | ---: |\n')
        for name,m in aux.items():
            f.write(f"| {NAMES[name]} | {m['ungated_matches']}/1757 | {m['gated_matches']}/1757 | {m['lost_matches']} |\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
