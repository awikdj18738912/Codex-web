"""Report consumed regression and the frozen new bank separately."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c, synthetic_gate_data as d


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    runtime=root/'runtime/pairs_full';c.verify_frozen(runtime)
    manifest=json.loads((root/'test_data/manifest.json').read_text())
    assert manifest['runtime_protocol_sha256']==c.file_hash(runtime/'protocol.json')
    assert manifest['runner_sha256']==c.file_hash(ROOT/'experiments/scripts/evaluate_gate_streaming.py')
    assert manifest['texts_sha256']==c.file_hash(root/'test_data/texts.jsonl')
    old=json.loads((ROOT/'results/gate_round3_20261011/streaming_test/summary.json').read_text())
    reports={}
    for stage,count in [('regression',32),('fresh_streaming',20),('punctuation_regression',20)]:
        summary=json.loads((root/stage/'summary.json').read_text());rows=d.read_rows(root/stage/'sessions.jsonl')
        cfg=json.loads((root/stage/'configuration.json').read_text())
        assert cfg['gate_scope']=='window' and summary['complete'] and len(rows)==count*2
        assert cfg['policy_sha256']==manifest['policy_sha256_before_test_generation']
        stage_runtime=(root/'runtime_punctuation/pairs_full') if stage=='punctuation_regression' else runtime
        c.verify_frozen(stage_runtime)
        assert cfg['teacher_protocol_sha256']==c.file_hash(stage_runtime/'protocol.json')
        pairs={sid:{r['policy']:r for r in rows if r['seed_id']==sid} for sid in {r['seed_id'] for r in rows}}
        assert all(v['current']['audio_sha256']==v['learned']['audio_sha256'] for v in pairs.values())
        assert all(g['input']['task_kind']=='window_refine' for r in rows for g in r['gate_calls'])
        compared=dict(
            raw_exact_equal_pairs=sum(v['current']['raw_text']==v['learned']['raw_text'] for v in pairs.values()),
            final_exact_equal_pairs=sum(v['current']['final_text']==v['learned']['final_text'] for v in pairs.values()),
            learned_better=sum(v['learned']['intended_distance']<v['current']['intended_distance'] for v in pairs.values()),
            learned_worse=sum(v['learned']['intended_distance']>v['current']['intended_distance'] for v in pairs.values()))
        detail={}
        for policy in ('current','learned'):
            arm=[r for r in rows if r['policy']==policy]
            detail[policy]=dict(summary['by_policy'][policy],
                finish_tail_max_seconds=max(r['timing']['finish_tail_seconds'] for r in arm),
                refiner_wall_ms=sum(m['wall_ms'] for r in arm for m in r['model_calls']),
                call_methods=dict(collections.Counter(m['method'] for r in arm for m in r['model_calls'])),
                restorations=sum(len(r['final'].get('repetition_restorations',[])) for r in arm),
                categories={category:dict(recordings=sum(r['category']==category for r in arm),
                    exact=sum(r['category']==category and r['intended_distance']==0 for r in arm))
                    for category in sorted({r['category'] for r in arm})})
        reports[stage]=dict(comparison=compared,by_policy=detail)
    c.write_json(root/'evaluation.json',dict(old_both_task_trial=old['by_policy'],**reports,
        data_boundary='32 consumed recordings are regression; new bank is 20 audio variants over 12 templates',
        reference='unverified_TTS_scripts_not_human_gold',production_student_enabled=False))
    c.write_json(root/'test_data/test_consumed.json',dict(status='consumed',evaluations=['fresh_streaming','punctuation_regression'],
        runtime_protocol_sha256=manifest['runtime_protocol_sha256'],policy_sha256=manifest['policy_sha256_before_test_generation'],
        test_threshold_tuning_prohibited=True,subsequent_use='regression_only',
        engineering_fix_after_consumption='ASR comma broke named reference; generic cue protected unrelated stutter',unix=time.time()))
    lines=['# 窗口门控与终态原文保护：2026-10-11','',
        '学习门控仅调度窗口精修，局部审核恢复原策略。保留现有模型权重及阈值，不重新训练或在测试集上选阈值。',
        '终态原文保护在实体处理、局部审核和确定性清理之后执行：只插入对齐明确的完整重复副本；必须至少保留一个原样副本，替换或数字/实体规范化引起的歧义不回退。',
        '远处保留指令必须复述被重复的完整内容。它仍是保守工程约束，不能据此声称实现了语义意图模型。',
        '恢复发生时最终 source spans 改为一条完整原文→终态文本记录，state=final；分组窗口缓存不倒灌。终态展示可对应权威文本，但失去 K3 分组粒度。','',
        '| 数据 | 策略 | 脚本完全匹配 | 脚本 CER | Refiner 调用 | MLM 调用 | 尾延迟 p95 / 最大值（秒） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for stage,label,count in [('regression','旧32段回归（v1）',32),('fresh_streaming','冻结新20段（v1）',20),('punctuation_regression','标点修后20段回归（v2）',20)]:
        for policy,name in [('current','原策略+终态保护'),('learned','窗口门控+终态保护')]:
            row=reports[stage]['by_policy'][policy]
            lines.append(f"| {label} | {name} | {row['exact_intended']}/{count} | {row['intended_micro_cer']:.3%} | {row['logical_refiner_calls']} | {row['plausibility_calls']} | {row['finish_tail_p95_seconds']:.3f} / {row['finish_tail_max_seconds']:.3f} |")
    lines += ['',
        '旧双任务门控在32段上只有18段脚本完全匹配，原策略22段；本轮两侧均23段。此前四处错跳已由局部审核补救，晚到保留指令的接收完成重复也得到恢复。旧样本已用于分析，不是独立测试。',
        f"新20段：窗口门控相对原策略，改善 {reports['fresh_streaming']['comparison']['learned_better']} 段、退化 {reports['fresh_streaming']['comparison']['learned_worse']} 段；最终原样文本一致 {reports['fresh_streaming']['comparison']['final_exact_equal_pairs']}/20。",'',
        '新20段发现一个工程缺陷：ASR在“前面的已经备案，说了两遍”里插入逗号，v1未识别具名引用，通用指令错保护较近的确认重复，反而丢掉已经备案重复。独立v1结果原样保留，随后将整批标为已使用，再修复标点兼容。',
        'v2允许具名短语后的逗号、冒号、空格及引号。凡是具名引用所在句，都不再把指令回退关联到无关重复；源中没有被引用的重复也不例外。没有更改门控权重或阈值。',
        '019修后恢复了已经备案重复，但确认仍重复出现，脚本编辑距离由6降为2，整句尚未修好。因此完全匹配数量仍是13/20；不能把部分内容修复描述为全句正确。',
        f"修后20段只是回归：两策略脚本完全匹配分别 {reports['punctuation_regression']['by_policy']['current']['exact_intended']}/20 和 {reports['punctuation_regression']['by_policy']['learned']['exact_intended']}/20，不能用它替代未参与修改的新验证集。",'',
        '逐行字符串比较与去标点质量指标须区分：v1及v2两策略的015有标点差异，原样终态一致都是19/20；流式summary的differences为空是按规范化去标点比较，并非所有原样文本逐字一致。',
        '## 数据和延迟限制','',
        '新数据为12个文本模板的20个音频变体（4组clean/filler/stutter，4组保留重复，4组长句）。完整去词槽文本不与已生成的训练/开发/测试文本重复；不据此声称语义领域独立。文本、运行时、策略及执行适配器在生成音频前冻结，Qwen TTS 使用 CUDA0，固定 Vivian 声线。',
        '所有意图脚本和音频未人工核验。CER/完全匹配是诊断代理指标，不能等同于真实语音金标质量。未使用ASR置信度，不保证风险上界。',
        '本轮预加载MLM模型，旧一轮未预加载，因此不能跨轮比较冷启动延迟。旧回归最后一段原策略尾延迟31.924秒，而两次Refiner仅约743毫秒；事件日志显示主要等待来自发送后累计的ASR/WebSocket处理队列。保留最大值，不能把这个异常当成门控加速。',
        '同一机器共享GPU，固定原策略→门控顺序，每条仅一次。调用次数及配对文本可供判断，延迟不能支持可靠加速结论。','',
        f"冻结新20段累计Refiner实测时间减少约 {reports['fresh_streaming']['by_policy']['current']['refiner_wall_ms']-reports['fresh_streaming']['by_policy']['learned']['refiner_wall_ms']:.0f} 毫秒，而门控评分用去 {reports['fresh_streaming']['by_policy']['learned']['gate_total_ms']:.0f} 毫秒，抵消了大部分这一次运行的节省；不能只以少调用两次就认定明显提速。",'',
        '## 后续边界','',
        '没有上线学习门控或重启现有服务。收益首先按真实终态质量和总调用计算；下一轮训练应使用窗口动作对整条转录的边际收益标签，再做新语料验证。v2尚无新独立音频验证；本批20段只能作回归。','',
        '结果：evaluation.json；原始配对：regression/sessions.jsonl、fresh_streaming/sessions.jsonl、punctuation_regression/sessions.jsonl；模型/运行时哈希：integrity_check.json；Luna独立审计：LUNA_AUDIT.md。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    executed=root/'executed_code'
    executed.mkdir(exist_ok=True)
    (executed/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    hashes={str(path.relative_to(root)):c.file_hash(path) for path in root.rglob('*')
            if path.is_file() and path.name not in {'integrity_check.json','completed_experiment.json'}
            and str(path.relative_to(root))!='logs/report.log'}
    c.write_json(root/'integrity_check.json',dict(artifacts_sha256=hashes,
        selected_weights_sha256=c.file_hash(ROOT/'results/gate_round3_20261011/training/encoder_ternary_operation/gate.pt'),
        selection_sha256=c.file_hash(ROOT/'results/gate_round3_20261011/training/selection.json'),
        runtime_verified=True,paired_audio_hashes_verified=True,audio_script_verified=False,human_gold=False,local_learned_calls=0,
        current_code_matches_v2={name:c.file_hash(ROOT/'system'/name)==c.file_hash(root/'runtime_punctuation/pairs_full/frozen/system'/name)
            for name in ('web_app.py','repetition_preservation.py')}))
    c.write_json(root/'completed_experiment.json',dict(status='complete',production_student_enabled=False,
        fresh_test_consumed=True,learning_thresholds_unchanged=True,unix=time.time()))
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
