"""Report measured vLLM latency/throughput without reusing scores for changed outputs."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/vllm_aasr_refiners_gpu0_20261028'
PREVIOUS=ROOT/'results/aasr_bench_refiners_gpu0_20261028'
NAMES={'lora':'规则库＋LoRA0.5B','lora_gate':'规则库＋LoRA0.5B＋门控',
       'full':'规则库＋全量0.5B','full_gate':'规则库＋全量0.5B＋门控','agentic1b':'AgenticASR1B'}


def read(path):return [json.loads(s) for s in Path(path).open()]
def quantile(a,p):
    a=sorted(a);x=(len(a)-1)*p;i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)


def main():
    assert json.loads((OUT/'progress.json').read_text())['phase']=='complete'
    sweeps=read(OUT/'sweeps.jsonl')
    assert len(sweeps)==15
    predictions=[r for name in ('lora','full','agentic1b') for r in read(OUT/(name+'.predictions.jsonl'))]
    assert len(predictions)==5260
    assert len({(r['variant'],r['concurrency'],r['id']) for r in predictions})==5260
    oldrows=read(PREVIOUS/'predictions.jsonl')
    old={(r['variant'],r['mode'],r['id']):r for r in oldrows}
    oldsummary=json.loads((PREVIOUS/'summary.json').read_text())
    result={'dataset':'AASR-Bench510 Chinese oral text only','measured_inference_rows':len(predictions),
            'previous_transformers':str(PREVIOUS),'measurements':{},'no_new_rubric_judgement':True}
    differences=[]
    for sweep in sweeps:
        variant,c=sweep['variant'],sweep['concurrency']
        rows=[r for r in predictions if r['variant']==variant and r['concurrency']==c]
        assert len(rows)==(32 if c==1 else 510)
        metric=dict(sweep)
        times=[r['total_seconds']*1000 for r in rows]
        metric.update(mean_ms=statistics.mean(times),median_ms=statistics.median(times),p95_ms=quantile(times,.95),
            mean_rule_ms=statistics.mean(r['rule_seconds']*1000 for r in rows),
            mean_gate_ms=statistics.mean(r['gate_seconds']*1000 for r in rows),
            mean_guard_ms=statistics.mean(r['numeric_guard_seconds']*1000 for r in rows),
            mean_request_ms=statistics.mean(r['request_seconds']*1000 for r in rows),
            refiner_calls=sum(r['route'] for r in rows),numeric_fallbacks=sum(r['numeric_fallback'] for r in rows),
            non_stop=sum(r['route'] and r['finish_reason']!='stop' for r in rows),
            raw_clean_exact=sum(r['output']==r['clean'] for r in rows),
            normalized_clean_exact=sum(r['output']==r['reference'] for r in rows),
            same_output_as_transformers=sum(r['output']==old[variant,'latency' if c==1 else 'throughput',r['id']]['output'] for r in rows),
            mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in rows))
        if c==1:
            baseline=oldsummary['pipelines'][variant]['latency_mean_ms']
            metric.update(previous_mean_ms=baseline,mean_latency_speedup=baseline/metric['mean_ms'])
        else:
            metric['previous_fixed_batch8_seconds_510']=oldsummary['pipelines'][variant]['throughput_batch8_seconds_all_510']
        result['measurements'].setdefault(variant,{})[str(c)]=metric
        for r in rows:
            previous=old[variant,'latency' if c==1 else 'throughput',r['id']]
            if r['output']!=previous['output']:
                differences.append({'variant':variant,'concurrency':c,'id':r['id'],'input':r['text'],
                    'clean':r['clean'],'rule_reference':r['reference'],'transformers_output':previous['output'],
                    'vllm_output':r['output'],'model_output':r['model_output'],'numeric_fallback':r['numeric_fallback']})
    for name,measurements in result['measurements'].items():
        a={r['id']:r for r in predictions if r['variant']==name and r['concurrency']==8}
        b={r['id']:r for r in predictions if r['variant']==name and r['concurrency']==16}
        measurements['8_vs_16_output_changes']=sum(r['output']!=b[k]['output'] for k,r in a.items())
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'output_differences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in differences))
    with (OUT/'summary.csv').open('w') as f:
        keys=['variant','concurrency','n','mean_ms','median_ms','p95_ms','wall_seconds','inputs_per_second',
              'refiner_calls','numeric_fallbacks','non_stop','raw_clean_exact','normalized_clean_exact','same_output_as_transformers']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        for v in NAMES:
            for c in ('1','8','16'):writer.writerow(result['measurements'][v][c])
    lines=['# GPU0 vLLM：AASR中文精修实测','',
        '相同510条AASR-Bench中文oral纯文本，相同模型、任务提示词、BF16和数字规则库。32条相同子集测单请求；510条完整集合测8路和16路并发。未运行音频转录。','',
        '## 单请求：32句预热后逐条处理','',
        '| 方案 | 原Transformers均值/ms | vLLM均值/ms | vLLM中位数/ms | vLLM P95/ms | 均值加速倍数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for v in NAMES:
        m=result['measurements'][v]['1']
        lines += [f"| {NAMES[v]} | {m['previous_mean_ms']:.1f} | {m['mean_ms']:.1f} | {m['median_ms']:.1f} | {m['p95_ms']:.1f} | {m['mean_latency_speedup']:.2f}× |"]
    lines += ['', '## 并发：510条全部输入','',
        '| 方案 | 8路总耗时/s | 8路吞吐/句每秒 | 8路平均响应/ms | 16路总耗时/s | 16路吞吐/句每秒 | 16路平均响应/ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for v in NAMES:
        a,b=result['measurements'][v]['8'],result['measurements'][v]['16']
        lines += [f"| {NAMES[v]} | {a['wall_seconds']:.2f} | {a['inputs_per_second']:.2f} | {a['mean_ms']:.1f} | {b['wall_seconds']:.2f} | {b['inputs_per_second']:.2f} | {b['mean_ms']:.1f} |"]
    lines += ['', '并发是客户端闭环最多同时8或16个文本请求，不是固定batch size。每个请求完成后立即补入下一个；排队和共享GPU带来的等待会进入请求响应时间。吞吐不能当作单句延迟。与此前固定batch8不同，批处理调度方式本身也是部署差异的一部分。','',
        '## 输出与异常：本轮未重新运行rubric裁判','',
        '| 方案 | 并发 | 与Transformers最终输出相同 | 匹配原始clean | 匹配规则归一化clean | 精修调用 | 数字保护回退 | 未stop结束 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for v in NAMES:
        for c in ('8','16'):
            m=result['measurements'][v][c]
            lines += [f"| {NAMES[v]} | {c} | {m['same_output_as_transformers']}/510 | {m['raw_clean_exact']}/510 | {m['normalized_clean_exact']}/510 | {m['refiner_calls']} | {m['numeric_fallbacks']} | {m['non_stop']} |"]
    lines += ['', '输出有变化的句子不能沿用之前rubric分数。逐字匹配只是描述性指标，不是AASR官方rubric得分，也不是人工准确率。较高并发和不同内核可使贪心输出变化，本轮记录所有结果、异常及完整生成耗时，不删除慢样本。','',
        '## 配置与计时边界','',
        '- GPU0 RTX309024GB，vLLM0.26.0，三个精修模型依次单独启动，使用默认编译/CUDA Graph优化，未使用enforce_eager。原服务日志记录实际图捕获情况。',
        '- BF16、greedy temperature0、top_p1、repetition_penalty1、最大1024新token。prefix caching关闭，避免重复输入命中缓存影响速度。',
        '- 最大上下文4096、max_num_seqs16、max_num_batched_tokens4096、显存比例0.50。LoRA从相同checkpoint以BF16合并导出，未改原模型。',
        '- 数字规则在CPU执行；RBT3 float32位于GPU0，通过互斥锁逐请求判断，锁等待计入门控时间。vLLM与门控共享GPU，这是本轮实际组合。',
        '- 总响应包含数字规则、门控、loopback HTTP、服务分词/生成/解码、数字保护；不包含模型启动、图捕获、预热和结果写盘。与此前进程内Transformers相比，多了本机HTTP服务开销。',
        '- 先各方案16路预热，再以固定随机顺序执行各并发设置；每个设置一轮，32条单句延迟是小样本，未声称统计显著。',
        '- 原1B使用允许规范数字的项目默认精修提示词，无native数字规则；四个0.5B方案沿用同一训练提示词、v91规则及数字检查。', '',
        '## 原始文件与依据','',
        f'- [汇总JSON]({OUT}/summary.json)',f'- [CSV]({OUT}/summary.csv)',
        f'- [与Transformers不同的输出]({OUT}/output_differences.jsonl)',
        f'- [输出差异阅读版]({OUT}/vLLM与Transformers输出差异.md)',
        f'- [冻结协议]({OUT}/protocol.json)',f'- [每轮实际墙钟计时]({OUT}/sweeps.jsonl)',
        f'- [执行脚本]({ROOT}/experiments/scripts/benchmark_vllm_aasr_gpu0.py)',
        '- [vLLM官方性能计时口径](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)、[CUDA Graph说明](https://docs.vllm.ai/en/latest/design/cuda_graphs/)。','']
    conclusions=['## 实测结论','',
        '在本轮相同32句单请求测试中，vLLM使0.5B组合的平均延迟从约728–836ms降至95–106ms，约7.6–8.0倍；1B从792ms降至134ms，约5.9倍。0.5B的单句速度优势在此部署配置下体现出来。','',
        '510句在16路并发下约7.81–9.71秒处理完，比8路更快完成全集，但单个请求的平均等待更长。门控在单请求下约降低8%的均值，在8/16路吞吐上的收益较小。','',
        '全量模型在8路和16路测试中都保留了recheck.jsonl:201重复生成至1024token上限的问题，耗时完整计入；因此16路全量的整批完成时间略长于1B。','',
        'vLLM与Transformers、以及8路与16路之间，均有部分正文不同。本轮仅完成速度和输出一致性对照，没有重新打rubric分；不能将上一轮rubric分数原封不动当作vLLM结果。数字保护回退问题也仍然存在。','']
    lines[4:4]=conclusions
    (OUT/'vLLM与Transformers精修速度对照.md').write_text('\n'.join(lines)+'\n')
    detail=['# vLLM与Transformers：实际输出差异','',
            '仅列最终正文不同的样本。候选参考和逐字匹配不等于人工准确率；本轮未重新打rubric分。', '']
    for i,r in enumerate(differences,1):
        detail += [f"## {i}. {NAMES[r['variant']]} · {r['concurrency']}路 · {r['id']}",'',
            '**输入**：'+r['input'],'','**AASR原始参考**：'+r['clean'],'',
            '**规则归一化参考**：'+r['rule_reference'],'',
            '**Transformers最终输出**：'+r['transformers_output'],'',
            '**vLLM最终输出**：'+r['vllm_output'],'']
        if r['numeric_fallback']:
            detail += ['该条vLLM输出经过数字保护回退。模型原正文：'+r['model_output'],'']
    (OUT/'vLLM与Transformers输出差异.md').write_text('\n'.join(detail)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
