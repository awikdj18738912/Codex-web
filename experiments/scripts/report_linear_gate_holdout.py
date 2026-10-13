"""Separate official-reference speech from synthetic-script diagnostics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d


def percentile(values,fraction):
    values=sorted(values)
    position=(len(values)-1)*fraction;left=int(position);right=min(left+1,len(values)-1)
    return values[left]*(right-position)+values[right]*(position-left) if right!=left else values[left]


def grouped_ci(groups):
    values=[statistics.mean(v) for v in groups.values()]
    rng=random.Random(20261012)
    bootstrap=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(3000)]
    return dict(mean_difference_seconds=statistics.mean(values),
                bootstrap95_seconds=[percentile(bootstrap,.025),percentile(bootstrap,.975)],
                blocks=len(values),unit='template_family_or_human_speaker')


def asr_history(row):
    # Compare hypothesis content, not receipt times, confidence or displayed edits.
    return [e['payload'].get('raw_text') for e in row['events'] if e['payload'].get('event')=='transcript']


def summarize(rows):
    kinds=sorted({r['data_kind'] for r in rows});result={};differences=[]
    for kind in kinds:
        subset=[r for r in rows if r['data_kind']==kind];arms={}
        for policy in ('current','learned'):
            values=[r for r in subset if r['policy']==policy]
            timing={}
            for name in ('completion_lag_from_audio_clock_seconds','finish_tail_seconds','send_overrun_seconds','backpressure_seconds'):
                x=[r['timing'][name] for r in values]
                timing[name]=dict(p50=percentile(x,.5),p95=percentile(x,.95),max=max(x))
            arms[policy]=dict(sessions=len(values),clips=len({r['seed_id'] for r in values}),
                normalized_reference_exact=sum(r['normalized_distance']==0 for r in values),
                normalized_micro_cer=sum(r['normalized_distance'] for r in values)/sum(r['normalized_reference_chars'] for r in values),
                lexical_micro_cer=sum(r['lexical_distance'] for r in values)/sum(r['lexical_reference_chars'] for r in values),
                refiner_calls=sum(len(r['model_calls']) for r in values),plausibility_calls=sum(len(r['plausibility_calls']) for r in values),
                gate_calls=sum(len(r['gate_calls']) for r in values),gate_ms=sum(g['gate_ms'] for r in values for g in r['gate_calls']),
                vetoes=sum(g['action']=='keep' for r in values for g in r['gate_calls']),
                model_errors=sum(bool(m.get('error')) for r in values for m in [*r['model_calls'],*r['plausibility_calls']]),timing=timing)
        pairs=defaultdict(dict)
        for row in subset:pairs[(row['seed_id'],row['repetition'])][row['policy']]=row
        groups=defaultdict(list);degraded=improved=raw_equal=history_equal=final_equal=0
        for identity,pair in pairs.items():
            if set(pair)!={'current','learned'}:raise ValueError('Incomplete paired observation')
            current,learned=pair['current'],pair['learned']
            if current['pcm_sha256']!=learned['pcm_sha256']:raise ValueError('PCM pairing failed')
            equal=current['raw_text']==learned['raw_text'];raw_equal+=equal
            same_history=asr_history(current)==asr_history(learned);history_equal+=same_history
            final_equal+=current['final_text']==learned['final_text']
            degraded+=learned['normalized_distance']>current['normalized_distance']
            improved+=learned['normalized_distance']<current['normalized_distance']
            groups[current['template_group']].append(learned['timing']['completion_lag_from_audio_clock_seconds']-current['timing']['completion_lag_from_audio_clock_seconds'])
            if current['final_text']!=learned['final_text']:
                differences.append(dict(kind=kind,seed_id=identity[0],repetition=identity[1],raw_equal=equal,history_equal=same_history,
                    current=current['final_text'],learned=learned['final_text'],current_distance=current['normalized_distance'],learned_distance=learned['normalized_distance']))
        result[kind]=dict(arms=arms,paired=dict(pairs=len(pairs),raw_asr_exact_equal=raw_equal,asr_hypothesis_history_equal=history_equal,
            final_exact_equal=final_equal,learned_worse=degraded,learned_better=improved),
            paired_audio_clock_lag=grouped_ci(groups))
    return dict(by_source=result,differences=differences)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text());cfg=json.loads((root/'streaming/configuration.json').read_text())
    for key,path in [('model_sha256',root/'frozen_model/model.pkl'),('selection_sha256',root/'selection.json'),
                     ('human_sources_sha256',root/'human_sources.jsonl'),('streaming_adapter_sha256',ROOT/'experiments/scripts/stream_linear_gate_holdout.py')]:
        assert c.file_hash(path)==manifest[key]
    c.verify_frozen(root/'runtime')
    rows=d.read_rows(root/'streaming/sessions.jsonl');failures=d.read_rows(root/'streaming/failures.jsonl')
    assert len(rows)==(manifest['human_audio']+manifest['synthetic_audio'])*cfg['repetitions']*2
    assert len({(r['seed_id'],r['repetition'],r['policy']) for r in rows})==len(rows)
    report=summarize(rows);report.update(sessions=len(rows),clips=manifest['human_audio']+manifest['synthetic_audio'],
        repetitions=cfg['repetitions'],failures=len(failures),complete=True,threshold=manifest['threshold'],
        no_test_tuning=True,production_enabled=False,ASR_pretraining_overlap_unknown=True)
    c.write_json(root/'streaming/summary.json',report)
    c.write_json(root/'test_data/test_consumed.json',dict(status='consumed',subsequent_use='regression_only',
        evaluated_unix=time.time(),selection_sha256=manifest['selection_sha256'],model_sha256=manifest['model_sha256']))
    c.write_json(root/'test_consumed.json',dict(status='consumed',human_and_synthetic=True,subsequent_use='regression_only'))
    lines=['# 冻结轻量门控：新留出真实流式对照，2026-10-12','',
        '保持round4字符TF-IDF线性模型及0.8阈值冻结，用真实WebSocket音频流和HTTP Qwen ASR对照原策略与学习窗口门控。模型输入/原生KEEP-DEFER/局部审核/实体数字处理/终态保护策略未改动。','',
        '## 样本和协议','',
        '24段AISHELL-dev官方朗读音频来自24名随机顺序选取的不同说话人，配官方转写；其录音ID/WAV哈希未出现在已审计的项目ASR/benchmark结果中。文本曾在本地AISHELL-NER注释资源出现，不能称文本从未见过；ASR/Refiner预训练是否见过公开语料未知。',
        '另有24段Qwen TTS控制音频、12个新文本家族：6组三种clean/filler/stutter变体、3组保留重复、3组长句。完整去词槽文本不与旧训练/开发/测试文本重复，但不能证明语义领域或句法模板完全独立。这部分脚本和音频未经人工核验，只是弱代理。',
        '每段两轮，两轮相反策略顺序，每个session独立缓存。0.5秒PCM包按音频捕获时钟发送，最多2块未ACK；记录发送积压以及相对音频结束的完成滞后，避免只报告最后finish之后的等待。',
        'ASR、原float32 Refiner、MLM与字符门控均预热于旧训练素材。共享GPU和同一ASR服务，未停止或重启现有服务；延迟测量仍受外部负载和ASR缓存影响。','',
        '## 分来源结果','',
        '| 来源 | 策略 | 规范化参考完全匹配（会话） | 规范化CER | 原字词CER | Refiner调用 | 音频完成滞后p50/p95/max（秒） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for kind,title in [('human_read_speech','真实朗读/官方转写'),('synthetic_tts','合成/弱脚本')]:
        value=report['by_source'][kind]
        for policy,name in [('current','原策略'),('learned','冻结门控')]:
            s=value['arms'][policy];t=s['timing']['completion_lag_from_audio_clock_seconds']
            lines.append(f"| {title} | {name} | {s['normalized_reference_exact']}/{s['sessions']} | {s['normalized_micro_cer']:.3%} | {s['lexical_micro_cer']:.3%} | {s['refiner_calls']} | {t['p50']:.3f}/{t['p95']:.3f}/{t['max']:.3f} |")
    for kind,title in [('human_read_speech','真实朗读/官方转写'),('synthetic_tts','合成/弱脚本')]:
        value=report['by_source'][kind]
        pair=value['paired'];ci=value['paired_audio_clock_lag']
        lines+=['',f"{title}配对：{pair['pairs']}对，相同最终ASR {pair['raw_asr_exact_equal']} 对、相同逐次假设历史 {pair['asr_hypothesis_history_equal']} 对；门控脚本/参考距离改善 {pair['learned_better']} 对、退化 {pair['learned_worse']} 对。", 
            f"门控−原策略的音频完成滞后按说话人/模板分组均值差 {ci['mean_difference_seconds']:.4f} 秒；分组bootstrap 95%区间 [{ci['bootstrap95_seconds'][0]:.4f}, {ci['bootstrap95_seconds'][1]:.4f}] 秒，{ci['blocks']} 个块。"]
    lines+=['',
        f"总计{len(rows)}个会话，失败尝试{len(failures)}次。Refiner/MLM错误按sessions逐项保留，不把重跑或同一音频两轮当成新的独立样本。两轮只有48段音频，不是192个独立样本。",'',
        '## 口径与发现','',
        '原字词CER只做NFKC/去标点空格；规范化CER额外使用与项目相同的中文数字规则，减少合法数字格式变化的影响。官方语音与合成脚本分别计分，不混合成一项总体金标准确率。',
        '当ASR最终文本或逐步假设不同，两策略差异不能只归因门控。差异列表保留raw/history配对标志；完整时序与发送背压见streaming/sessions.jsonl。',
        '两策略共有的合成长句失败：归档结束样本中“随后检查索引。索引交给值班员。”被局部重复审核合并为一次索引；现场清空样本中“接着核对名单。名单送到调度室。”也删了一次名单，各自弱脚本距离增加2。这是原流程跨句宾语/主语重复保护问题，不能归因模型阈值。未在本轮调整规则或重训，所有冻结结果保留。',
        '门控评分是文本路由分数，未校准，也不是ASR声学置信度。真实朗读流利、口吃少，不能代表自然对话/讲座中的精修需求。样本小，bootstrap区间只描述本批分组波动，不保证部署收益。','',
        '真实朗读子集没有原生窗口REFINE机会，学习窗口门控实际调用0次；两策略各2次Refiner均来自原局部审核。这部分验证了输入配对和基础流程，但没有检验窗口门控触发后的效果或风险，也未证明真实语音加速。',
        '合成子集Refiner调用40→32（减少20%），相同规范化脚本CER、没有额外距离退化。窗口门控共24次调用，其中10次veto，评分总约35毫秒。模板等权平均的完成滞后减少约91毫秒，与按所有会话平均的滞后差不同；前述bootstrap采用模板等权，不能把24个合成变体都当独立模板。',
        '相关检查161项通过。Luna核对了源数据、顺序反转、音频配对、模型冻结、两项共享误删及ASR波动，详细审计见LUNA_AUDIT.md。','',
        '## 下一步','',
        '保留这批为已消费回归数据。是否值得启用需根据独立语音中的调用变化、质量变化和延迟区间决定；没有把本次测试用于调0.8阈值或修改门控模型。',
        '优先独立修复跨句重复的语义保护，再用新的留出样本验证；不能在当前样本上修完后仍称本轮是新鲜测试。还应补充具有实际精修需求的自然口语，当前流利朗读没有触发窗口门控。',
        '生产未启用新门控；详细结果streaming/summary.json，模型frozen_model/model.pkl，冻结选择selection.json，独立审计LUNA_AUDIT.md。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for name in ('run_linear_holdout_audio.py','report_linear_gate_holdout.py'):
        path=ROOT/'experiments/scripts'/name;(root/'executed_code'/name).write_bytes(path.read_bytes())
    hashes={str(path.relative_to(root)):c.file_hash(path) for path in root.rglob('*') if path.is_file()
        and path.name not in {'integrity_check.json','completed_experiment.json'} and str(path.relative_to(root))!='logs/report.log'}
    c.write_json(root/'integrity_check.json',dict(artifacts_sha256=hashes,
        frozen_model_sha256=manifest['model_sha256'],frozen_selection_sha256=manifest['selection_sha256'],
        runtime_verified=True,paired_PCM_verified=True,no_test_tuning=True,production_enabled=False))
    c.write_json(root/'completed_experiment.json',dict(status='complete',sessions=len(rows),new_holdout_consumed=True,
        production_enabled=False,no_test_tuning=True))
    print(json.dumps({k:v for k,v in report.items() if k!='differences'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
