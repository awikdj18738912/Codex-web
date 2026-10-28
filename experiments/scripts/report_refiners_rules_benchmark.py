"""Report text-only native-rules benchmark: distinguish latency from throughput."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from report_current_refiners_benchmark import distance, normalize, percentile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/refiner_rules_text_benchmark_20261028'
NAMES = {'lora':'规则库＋0.5B LoRA', 'lora_gate':'规则库＋门控＋0.5B LoRA',
         'full':'规则库＋0.5B全量', 'full_gate':'规则库＋门控＋0.5B全量',
         'agentic1b':'原AgenticASR 1B'}


def read(path): return [json.loads(s) for s in Path(path).open()]


def main():
    assert json.loads((OUT/'progress.json').read_text())['phase']=='complete'
    inputs = read(OUT/'text_manifest.jsonl')
    predictions = read(OUT/'predictions.jsonl')
    timings = read(OUT/'batch_timings.jsonl')
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert len(predictions)==1440 and len(timings)==320
    by = {v:{r['id']:r for r in predictions if r['mode']=='throughput' and r['variant']==v} for v in NAMES}
    summary = {'unique_inputs':len(inputs),'weak_labels':dict(Counter(r['label'] for r in inputs)),
               'rule_pack':protocol['rule_pack'],'pipelines':{},'gate_comparisons':{}}
    cases = []
    for name,group in by.items():
        rows = list(group.values())
        assert len(rows)==256
        lat = [r for r in timings if r['variant']==name and r['mode']=='latency']
        bulk = [r for r in timings if r['variant']==name and r['mode']=='throughput']
        assert len(lat)==len(bulk)==32
        positive = [r for r in rows if r['label']]
        clean = [r for r in rows if not r['label']]
        lat_outputs = {r['id']:r for r in predictions if r['variant']==name and r['mode']=='latency'}
        elapsed = sum(r['batch_seconds'] for r in bulk)
        summary['pipelines'][name] = {
            'latency_n':32, 'latency_mean_ms':statistics.mean(r['batch_seconds']*1000 for r in lat),
            'latency_median_ms':statistics.median(r['batch_seconds']*1000 for r in lat),
            'latency_p95_ms':percentile([r['batch_seconds']*1000 for r in lat],.95),
            'latency_gate_mean_ms':statistics.mean(r['gate_seconds']*1000 for r in lat),
            'latency_rule_mean_ms':statistics.mean(r['rule_seconds']*1000 for r in lat),
            'latency_numeric_guard_mean_ms':statistics.mean(r['numeric_guard_seconds']*1000 for r in lat),
            'latency_refiner_mean_ms':statistics.mean(r['refiner_seconds']*1000 for r in lat),
            'batch8_seconds_256_inputs':elapsed, 'batch8_inputs_per_second':256/elapsed,
            'batch8_mean_completion_seconds':statistics.mean(r['batch_seconds'] for r in bulk),
            'batch8_refiner_calls':sum(r['route'] for r in rows),
            'batch8_rule_seconds':sum(r['rule_seconds'] for r in bulk),
            'batch8_gate_seconds':sum(r['gate_seconds'] for r in bulk),
            'batch8_refiner_seconds':sum(r['refiner_seconds'] for r in bulk),
            'batch8_numeric_guard_seconds':sum(r['numeric_guard_seconds'] for r in bulk),
            'quality_n':256, 'exact_matches':sum(r['output']==r['reference'] for r in rows),
            'normalized_matches':sum(normalize(r['output'])==normalize(r['reference']) for r in rows),
            'normalized_char_error_rate':sum(distance(normalize(r['reference']),normalize(r['output'])) for r in rows)/sum(len(normalize(r['reference'])) for r in rows),
            'positive_exact_matches':sum(r['output']==r['reference'] for r in positive),
            'before_numeric_guard_exact_matches':sum(r['model_output']==r['reference'] for r in rows),
            'matches_lost_by_numeric_guard':sum(r['model_output']==r['reference'] and r['output']!=r['reference'] for r in rows),
            'weak_clean_changed_vs_rule_baseline':sum(r['output']!=r['rule_baseline'] for r in clean),
            'numeric_fallback_count':sum(r['numeric_fallback'] for r in rows),
            'numeric_surface_repair_count':sum(r['numeric_surface_repaired'] for r in rows),
            'non_eos':sum(not r['eos'] for r in rows),
            'batch1_vs_batch8_output_changes':sum(r['output']!=group[ident]['output'] for ident,r in lat_outputs.items()),
            'batch1_vs_batch8_gate_route_changes':sum(r['route']!=group[ident]['route'] for ident,r in lat_outputs.items()),
            'mean_generated_tokens_per_input':statistics.mean(r['generated_tokens'] for r in rows)}
    for base in ('lora','full'):
        plain,gated = by[base],by[base+'_gate']
        bm,gm = summary['pipelines'][base],summary['pipelines'][base+'_gate']
        lost,gained,differences = [],[],[]
        for ident,g in gated.items():
            old=plain[ident]
            if g['output']!=old['output']:
                reason=('lost_exact' if old['output']==old['reference'] else
                        'gained_exact' if g['output']==g['reference'] else 'both_differ_from_reference')
                cases.append({'base':base,'id':ident,'input':g['text'],'rule_baseline':g['rule_baseline'],
                    'reference':g['reference'],'plain':old['output'],'gated':g['output'],
                    'route':g['route'],'reason':reason})
                differences.append(ident)
                if reason=='lost_exact':lost.append(ident)
                if reason=='gained_exact':gained.append(ident)
        summary['gate_comparisons'][base] = {
            'single_latency_reduction':1-gm['latency_mean_ms']/bm['latency_mean_ms'],
            'batch8_total_time_reduction':1-gm['batch8_seconds_256_inputs']/bm['batch8_seconds_256_inputs'],
            'batch8_throughput_speedup':gm['batch8_inputs_per_second']/bm['batch8_inputs_per_second'],
            'weak_positive_skipped':sum(not r['route'] and r['label'] for r in gated.values()),
            'weak_clean_skipped':sum(not r['route'] and not r['label'] for r in gated.values()),
            'lost_exact_ids':lost,'gained_exact_ids':gained,'different_output_ids':differences,
            'lost_due_to_gate_skipping_ids':[ident for ident in lost if not gated[ident]['route']],
            'lost_while_still_routed_ids':[ident for ident in lost if gated[ident]['route']],
            'routed_outputs_differ':sum(r['route'] and r['output']!=plain[ident]['output'] for ident,r in gated.items())}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'gate_differences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    with (OUT/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['pipeline']+list(summary['pipelines']['lora']))
        writer.writeheader()
        for name,m in summary['pipelines'].items():writer.writerow(dict(pipeline=name,**m))
    lines=['# GPU0：数字规则库＋精修模型的速度与质量对照','',
        '本轮只使用文本，不运行音频转录。LoRA和全量模型的有门控／无门控方案均加入本机原生数字规则库。原AgenticASR 1B沿用本机原提示词，作为对照。','',
        '本轮全量＋规则库＋门控的逐句平均耗时约343ms，相对不加门控减少约30%；batch 8吞吐提高约19%。两者在256句上都匹配236条候选参考。这个小样本支持继续验证该组合，但不能保证真实场景准确率。','',
        '测试同时发现数字保护将8条已匹配候选参考的模型输出回退：全量模型在保护前匹配244条，回退后236条。该问题出在数字完整性保护对自我修正或重复的识别，不应归咎于数字规范化库或微调模型本身。未修改业务保护代码。','',
        '256句固定文本：弱标签需修改128句、不需修改128句。32句（各16句）逐条测试单句延迟；全部256句再以batch size 8测试吞吐与输出质量。每种方案均实际执行。','',
        '## 单句延迟：逐条处理32句','',
        '| 方案 | 平均总耗时/ms | 中位数/ms | P95/ms |',
        '| --- | ---: | ---: | ---: |']
    for name,m in summary['pipelines'].items():
        lines += [f"| {NAMES[name]} | {m['latency_mean_ms']:.1f} | {m['latency_median_ms']:.1f} | {m['latency_p95_ms']:.1f} |"]
    lines += ['', '## 批量处理：batch size 8，256句','',
        '| 方案 | 256句总处理时间/s | 吞吐/句每秒 | 精修调用数 | 匹配规范化候选参考 | 干净候选被额外改动 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,m in summary['pipelines'].items():
        lines += [f"| {NAMES[name]} | {m['batch8_seconds_256_inputs']:.2f} | {m['batch8_inputs_per_second']:.2f} | {m['batch8_refiner_calls']}/256 | {m['exact_matches']}/256 | {m['weak_clean_changed_vs_rule_baseline']}/128 |"]
    lines += ['', '批量吞吐不能当作单句响应时间；批内句子要等待该批生成结束。各方案依次执行，GPU上没有同时跑多个模型任务，也没有请求排队或网络开销。', '',
        '## 门控带来的收益与代价','',
        '| 精修模型 | 单句均值耗时减少 | 批量总时间减少 | 批量吞吐加速 | 因门控跳过而损失匹配 | 门控组合新增匹配 | 弱标签需修改但被跳过 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,m in summary['gate_comparisons'].items():
        lines += [f"| {NAMES[name]} | {m['single_latency_reduction']:.1%} | {m['batch8_total_time_reduction']:.1%} | {m['batch8_throughput_speedup']:.2f}× | {len(m['lost_due_to_gate_skipping_ids'])} | {len(m['gained_exact_ids'])} | {m['weak_positive_skipped']} |"]
    lines += ['', '门控漏掉弱标签需修改，不一定意味着损失一个原本能改对的答案；上表分别列出这两种情况。参考仍是候选答案，匹配率不能直接称为人工确认的准确率。','',
        'LoRA加门控的组合少匹配1条，但该句门控分数0.999901，实际已送精修，不能算门控漏检。门控使精修子批大小变化，同一句出现“还有没有”→“还有”的输出差异。单独复现结果见下方文件；这项批大小稳定性问题使本轮LoRA＋门控不能被描述为完全无损。','',
        '## 数字处理流程','',
        '使用当前已编译的 `zh-itn-strict-conservative-v91`，测试前复制SO并记录SHA256，未重建或修改规则。四个0.5B方案对每条输入执行：','',
        '1. 原生规则库规范数字，形成规则基线。',
        '2. 如启用RBT3门控，它读取原始文本，保持与训练时相同的输入口径。门控跳过时仍返回规则处理结果。',
        '3. 需要精修时，0.5B模型读取规则处理后的文本，只负责口语清理、纠错等精修。',
        '4. 调用现有数字写法恢复与数字完整性检查。仅处理数字相关拒绝原因；数字检查失败时回退规则基线，不加入额外的通用内容过滤或重试。','',
        '规则处理、门控、数字检查与回退耗时都计入结果。参考答案离线经过同一规则库，保留原始参考；模型输出不做事后数字归一化来提高分数。干净句额外改动以规则基线比较，正常数字规范化不算误改。','',
        '原1B按原配置直接读原始输入；生成KEY计耗时，比较正文时移除KEY。它的数字写法、时间格式仍可能与规则目标不同，须结合逐句结果区分格式差异和实质错误。','',
        '## 耗时分解与检查','',
        '| 方案 | 单句规则均值/ms | 门控均值/ms | 精修均值/ms | 数字检查均值/ms | 数字回退/256 | 未正常EOS/256 | batch1与8输出不同/32 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,m in summary['pipelines'].items():
        lines += [f"| {NAMES[name]} | {m['latency_rule_mean_ms']:.3f} | {m['latency_gate_mean_ms']:.3f} | {m['latency_refiner_mean_ms']:.1f} | {m['latency_numeric_guard_mean_ms']:.3f} | {m['numeric_fallback_count']} | {m['non_eos']} | {m['batch1_vs_batch8_output_changes']} |"]
    lines += ['', 'GPU0为RTX 3090。精修模型BF16、SDPA、KV cache开启、贪心生成、最多1024新token；RBT3 float32、阈值0.005。LoRA在内存中合并，原权重文件不变。启动加载和预热不计时，分词、传输、生成、解码、数字规则及检查计时。', '',
        '数字规则可能使输入分布与原训练文本发生变化；这里测的是用户要求的实际组合。32句延迟和256句质量均属小样本，来自已有留出集且部分已被查看，不能承诺真实录音准确率。线上干净句比例变化时，门控收益也会变化。','',
        '## 门控组合与无门控组合的输出差异','']
    if not cases:lines += ['本轮门控与不加门控的最终正文相同。','']
    for i,r in enumerate(cases,1):
        lines += [f"### {i}. {NAMES[r['base']]} · {r['reason']}",'', '**输入**：'+r['input'],'',
            '**规则基线**：'+r['rule_baseline'],'','**候选参考**：'+r['reference'],'',
            '**不加门控**：'+r['plain'],'','**加门控**：'+r['gated'],'']
    lines += ['## 文件与依据','',
        f'- [256句逐句对照]({OUT}/256句五方案逐句对照.md)',
        f'- [数字保护回退的8条]({OUT}/数字保护回退的8条.md)',
        f'- [LoRA批大小差异复现]({OUT}/lora_batch_difference_replay.json)',
        f'- [汇总JSON]({OUT}/summary.json)',f'- [汇总CSV]({OUT}/summary.csv)',
        f'- [原始逐条结果]({OUT}/predictions.jsonl)',f'- [每批实际计时]({OUT}/batch_timings.jsonl)',
        f'- [固定协议]({OUT}/protocol.json)',f'- [固定输入]({OUT}/text_manifest.jsonl)',
        f'- [执行脚本]({ROOT}/experiments/scripts/benchmark_refiners_rules_gpu0.py)',
        f'- [本机原生规则接口]({ROOT}/system/zh_itn_bridge.py)',
        '- [PEFT官方合并说明](https://huggingface.co/docs/peft/developer_guides/checkpoint)、[LoRA官方实现说明](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md)。',
        '- [AgenticASR官方项目](https://github.com/AnXMuy/AgenticASR)、[原1B模型卡](https://huggingface.co/Andrew0425/AgenticASR-Refiner)。','']
    (OUT/'GPU0规则库与精修速度质量对照.md').write_text('\n'.join(lines)+'\n')
    detail=['# 256句：规则库＋精修模型逐句对照','','参考是候选答案。规则基线和规范化参考均由同一冻结SO产生；未修改源数据。','']
    for i,sample in enumerate(inputs,1):
        detail += [f"## {i:03d} · {sample['category']} · {'弱标签需修改' if sample['label'] else '弱标签不需修改'}",'',
            '**原始输入**：'+sample['text'],'','**规则基线**：'+sample['rule_baseline'],'',
            '**原始候选参考**：'+sample['original_reference'],'','**规则处理后参考**：'+sample['reference'],'']
        for name in NAMES:
            r=by[name][sample['id']]
            flags=[]
            if not r['route']:flags.append('门控跳过')
            if r['numeric_fallback']:flags.append('数字检查回退规则基线')
            label=NAMES[name]+('（'+'、'.join(flags)+'）' if flags else '')
            detail += ['**'+label+'**：'+r['output'],'']
            if r['numeric_fallback'] or r['numeric_surface_repaired']:
                detail += ['模型原正文：'+r['model_output'],'']
        detail += ['ID：`'+sample['id']+'`','']
    (OUT/'256句五方案逐句对照.md').write_text('\n'.join(detail)+'\n')
    guard_detail=['# 数字保护退回的8条：模型答案已匹配候选参考','',
        '以下以全量模型的batch 8结果展示。LoRA和全量的有门控／无门控方案均出现相同8条回退。尚未修改保护规则；候选参考仍需人工核对。','']
    selected=[r for r in by['full'].values() if r['numeric_fallback']]
    for i,r in enumerate(selected,1):
        guard_detail += [f"## {i}. {r['category']} · {r['id']}",'', '**原始输入**：'+r['text'],'',
            '**规则处理**：'+r['rule_baseline'],'','**模型原正文**：'+r['model_output'],'',
            '**候选参考**：'+r['reference'],'','**数字保护后的最终结果**：'+r['output'],'']
    (OUT/'数字保护回退的8条.md').write_text('\n'.join(guard_detail)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
