"""Balanced synthetic1000 report, including locally judged custom rubric when ready."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/vllm_synthetic_balanced1000_20261028'
NAMES={'lora':'规则库＋LoRA0.5B','lora_gate':'规则库＋LoRA0.5B＋门控',
       'full':'规则库＋全量0.5B','full_gate':'规则库＋全量0.5B＋门控','agentic1b':'AgenticASR1B'}


def read(p):return [json.loads(s) for s in Path(p).open()]


def main():
    assert json.loads((OUT/'progress.json').read_text())['phase']=='complete'
    rows=[r for model in ('lora','full','agentic1b') for r in read(OUT/(model+'.predictions.jsonl'))]
    sweeps=read(OUT/'sweeps.jsonl');manifest=read(OUT/'text_manifest.jsonl')
    assert len(rows)==10500 and len(sweeps)==15 and len(manifest)==1000
    assert len({(r['variant'],r['concurrency'],r['id']) for r in rows})==10500
    summary={'dataset':'Synthetic Gemma4 test1000','labels':{'needs_edit':500,'clean':500},
        'rubric_kind':'custom4dimensionbinary_NOT_AASR_official','rubric_status':'pending','pipelines':{},'gate_effects':{}}
    if (OUT/'custom_rubric_scores.json').exists():
        scores=json.loads((OUT/'custom_rubric_scores.json').read_text())
        if scores['status']=='complete':summary.update(rubric_status='complete',rubric_scores=scores['scores'])
    maps={}
    for v in NAMES:
        summary['pipelines'][v]={}
        for c in (1,8,16):
            a=[r for r in rows if r['variant']==v and r['concurrency']==c]
            assert len(a)==(100 if c==1 else 1000)
            t=next(s for s in sweeps if s['variant']==v and s['concurrency']==c)
            clean=[r for r in a if not r['label']];positive=[r for r in a if r['label']]
            m=dict(t,mean_ms=statistics.mean(r['total_seconds']*1000 for r in a),
                median_ms=statistics.median(r['total_seconds']*1000 for r in a),
                p95_ms=sorted(r['total_seconds']*1000 for r in a)[int((len(a)-1)*.95)],
                refiner_calls=sum(r['route'] for r in a),gate_skip_count=sum(not r['route'] for r in a),
                clean_skip_count=sum(not r['route'] for r in clean),need_edit_skipped=sum(not r['route'] for r in positive),
                exact_reference_matches=sum(r['output']==r['reference'] for r in a),
                positive_exact_matches=sum(r['output']==r['reference'] for r in positive),
                clean_changed=sum(r['output']!=r['rule_baseline'] for r in clean),
                numeric_fallback_count=sum(r['numeric_fallback'] for r in a),
                lost_exact_by_numeric_guard=sum(r['model_output']==r['reference'] and r['output']!=r['reference'] for r in a),
                non_stop=sum(r['route'] and r['finish_reason']!='stop' for r in a),
                mean_gate_ms=statistics.mean(r['gate_seconds']*1000 for r in a))
            summary['pipelines'][v][str(c)]=m;maps[v,c]={r['id']:r for r in a}
    differences=[]
    for v in ('lora','full'):
        summary['gate_effects'][v]={}
        for c in (1,8,16):
            plain,gated=maps[v,c],maps[v+'_gate',c]
            a,b=summary['pipelines'][v][str(c)],summary['pipelines'][v+'_gate'][str(c)]
            lost=[k for k,r in gated.items() if plain[k]['output']==r['reference'] and r['output']!=r['reference']]
            summary['gate_effects'][v][str(c)]={'mean_latency_reduction':1-b['mean_ms']/a['mean_ms'],
                'whole_set_time_reduction':1-b['wall_seconds']/a['wall_seconds'],
                'lost_exact_due_to_skip':[k for k in lost if not gated[k]['route']],
                'lost_exact_while_routed':[k for k in lost if gated[k]['route']],
                'gained_exact':[k for k,r in gated.items() if plain[k]['output']!=r['reference'] and r['output']==r['reference']]}
            if c==16:
                for k,r in gated.items():
                    if r['output']!=plain[k]['output']:
                        differences.append({'model':v,'id':k,'input':r['text'],'reference':r['reference'],
                            'ungated':plain[k]['output'],'gated':r['output'],'route':r['route']})
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'gate_differences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in differences))
    with (OUT/'summary.csv').open('w') as stream:
        keys=['variant','concurrency','n','mean_ms','median_ms','p95_ms','wall_seconds','inputs_per_second',
            'refiner_calls','gate_skip_count','clean_skip_count','need_edit_skipped','exact_reference_matches',
            'positive_exact_matches','clean_changed','numeric_fallback_count','lost_exact_by_numeric_guard','non_stop','mean_gate_ms']
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        for v in NAMES:
            for c in ('1','8','16'):writer.writerow(summary['pipelines'][v][c])
    md=['# 自建1000句：5:5精修／干净文本，vLLM速度与得分','',
        '全部来自Gemma4生成数据的留出test池，需修改500句、不需修改500句；规则处理前后均保持该标签比例。原文及规则基线去重，排除与当前冻结train/dev输入或参考的逐字重叠。没有按门控分数或精修输出挑选样本。','',
        '干净样本包含由留出参考答案构造的文本，候选参考未经全部人工审核，部分留出池曾被查看。本轮反映同来源数据上的表现，不是新的独立真实录音盲测。','',
        '## 单请求：100句，50:50','',
        '| 方案 | 平均/ms | 中位数/ms | P95/ms | 门控跳过 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for v in NAMES:
        m=summary['pipelines'][v]['1'];md += [f"| {NAMES[v]} | {m['mean_ms']:.1f} | {m['median_ms']:.1f} | {m['p95_ms']:.1f} | {m['gate_skip_count']}/100 |"]
    md += ['', '## 全部1000句并发速度','',
        '| 方案 | 8路总耗时/s | 8路吞吐/句每秒 | 16路总耗时/s | 16路吞吐/句每秒 | 16路平均响应/ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for v in NAMES:
        a,b=summary['pipelines'][v]['8'],summary['pipelines'][v]['16']
        md += [f"| {NAMES[v]} | {a['wall_seconds']:.2f} | {a['inputs_per_second']:.2f} | {b['wall_seconds']:.2f} | {b['inputs_per_second']:.2f} | {b['mean_ms']:.1f} |"]
    md += ['', '## 16路输出质量与门控','',
        '| 方案 | 匹配候选参考 | 需修改句匹配 | 干净句被改动 | 跳过干净句 | 跳过需修改句 | 数字检查回退 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for v in NAMES:
        m=summary['pipelines'][v]['16'];md += [f"| {NAMES[v]} | {m['exact_reference_matches']}/1000 | {m['positive_exact_matches']}/500 | {m['clean_changed']}/500 | {m['clean_skip_count']}/500 | {m['need_edit_skipped']}/500 | {m['numeric_fallback_count']} |"]
    md += ['', '精修调用减少不等于同比例加速。门控与精修共享GPU0，逐条门控互斥锁和数字规则开销都计入时间。16路是并发请求，不是固定batch16；全集总耗时与单请求延迟不同。','',
        '## 自建rubric得分','',
        '评分采用自建四维二选一要求：content信息保留、format数字/实体格式、filter口语清理、rephrase最终意图。每句每维1题、每题0或1分；4,000题/方案、满分4,000，四维等权；裁判是本机Gemma4-31B QAT W4A16，temperature0。参考统一按本机数字规则处理，正常强调不作为必须删除项。评分取16路的完整1000条输出，输出与参考相同的样本也送裁判，不自动发满分。','',
        '这是自建rubric，借用了AASR项目的解析与计分代码，不能与AASR官方细粒度题目所得分数直接比较。裁判只对输出评分，参考及rubric从未发送给精修模型。','']
    if summary['rubric_status']=='complete':
        md += ['| 方案 | 总得分 | 需修改500句得分 | 干净500句得分 | Content | Format | Filter | Rephrase |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for v in NAMES:
            s=summary['rubric_scores'][v]
            assert s['run']['completed_sample_count']==1000 and s['overall']['question_count']==4000
            vals=[s['by_category'][k]['normalized_score']*100 for k in ('content','format','filter','rephrase')]
            clean_score=s['by_need_label']['0']['overall']['normalized_score']*100
            edit_score=s['by_need_label']['1']['overall']['normalized_score']*100
            md += [f"| {NAMES[v]} | {s['overall']['normalized_score']*100:.2f}% | {edit_score:.2f}% | {clean_score:.2f}% | "+' | '.join(f'{n:.2f}%' for n in vals)+' |']
    else:md += ['Rubric评分尚未完成；上述参考匹配数不能替代rubric分数。']
    if summary['rubric_status']=='complete':
        md += ['', '干净组rubric得分100%不代表所有输出逐字不变。该rubric允许等义表达，且裁判可能漏判；请同时查看“干净句被改动”数量和逐句对照。0.5B仍有9–11条干净候选被改动。', '',
            f'[裁判评分汇总]({OUT}/custom_rubric_scores.json) · [裁判协议]({OUT}/judge_protocol.json)', '']
    md += ['', '## 配置与文件','',
        'GPU0 RTX3090；三个精修模型依次启动vLLM0.26，BF16、默认CUDA Graph/compile、关闭prefix cache；相同提示词和最大1024新token，规则库v91。门控架构、阈值0.005、逐条GPU互斥执行均与前轮相同，只替换测试数据。启动/预热不计时，规则、门控、HTTP精修、数字检查计时；未用音频。','',
        f'- [抽样与去重检查]({OUT}/selection_audit.json)',f'- [固定1000句]({OUT}/text_manifest.jsonl)',
        f'- [自建rubric]({OUT}/custom_rubric.json)',f'- [汇总JSON]({OUT}/summary.json)',f'- [汇总CSV]({OUT}/summary.csv)',
        f'- [1000句五方案对照]({OUT}/1000句五方案对照.md)',f'- [门控输出差异]({OUT}/gate_differences.jsonl)',
        f'- [协议]({OUT}/protocol.json)',
        '- [vLLM官方计时说明](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)。','']
    (OUT/'1000句平衡测试速度与得分.md').write_text('\n'.join(md)+'\n')
    detail=['# 自建1000句五方案实际输出','','候选参考和需改/干净标签不是完整人工金标准；以下均为16路并发结果。','']
    for i,r in enumerate(manifest,1):
        detail += [f"## {i:04d} · {r['category']} · {'需修改候选' if r['label'] else '干净候选'}",'',
            '**输入**：'+r['text'],'','**规则基线**：'+r['rule_baseline'],'','**候选参考**：'+r['reference'],'']
        for v in NAMES:
            a=maps[v,16][r['id']];flags=('（门控跳过）' if not a['route'] else '')+('（数字检查回退）' if a['numeric_fallback'] else '')
            detail += ['**'+NAMES[v]+flags+'**：'+a['output'],'']
        detail += ['ID：`'+r['id']+'`','']
    (OUT/'1000句五方案对照.md').write_text('\n'.join(detail)+'\n')
    print('Report complete; rubric:',summary['rubric_status'],flush=True)


if __name__=='__main__':main()
