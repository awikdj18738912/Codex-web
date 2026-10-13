"""Finalize hashes and a bounded report for the terminal-label expansion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d


def report(root):
    dataset=json.loads((root/'dataset/manifest.json').read_text())
    selection=json.loads((root/'linear_training/selection.json').read_text())
    replay=json.loads((root/'linear_replay/summary.json').read_text())
    cfg=json.loads((root/'linear_replay/configuration.json').read_text())
    assert replay['complete'] and replay['all_executed_branches_match']
    assert cfg['selection_sha256']==c.file_hash(root/'linear_training/selection.json')
    for split in ('train','dev'):
        c.verify_frozen(root/f'{split}_collection/runtime')
        for name,expected in dataset['files_sha256'].items():assert c.file_hash(root/'dataset'/name)==expected
    primary=selection['variants'][selection['primary']]
    assert cfg['model_sha256']==primary['model_sha256']==c.file_hash(root/'linear_training'/selection['primary']/'model.pkl')
    assert replay['refiner_calls']==primary['development']['refiner_calls']
    assert replay['degraded_views']==0
    rows=d.read_rows(root/'linear_replay/recordings.jsonl')
    baseline={r['recording_view_id']:r for r in d.read_rows(root/'dev_collection/recordings.jsonl')}
    retained=total=gate_calls=0
    for row in rows:
        base=baseline[row['recording_view_id']];decisions={g['key']:g['action'] for g in row['gate_calls']}
        for effect in base['effects']:
            if effect['label']=='benefit':total+=1;retained+=decisions[effect['window_key']]=='refine'
        gate_calls+=len(row['gate_calls'])
    assert retained==total==14
    native=selection['native'];delta=native['refiner_calls']-replay['refiner_calls']
    facts=dict(train=dataset['summaries']['train'],dev=dataset['summaries']['dev'],
        selected_model=selection['primary'],threshold=primary['threshold'],native_development=native,
        actual_classifier_development_replay=replay,actual_beneficial_windows_retained=retained,
        actual_beneficial_windows=total,actual_gate_mean_ms=replay['gate_total_ms']/max(1,gate_calls),
        refiner_call_reduction=delta,refiner_call_reduction_fraction=delta/native['refiner_calls'],
        independently_tested=False,online_latency_measured=False,production_enabled=False,human_gold=False)
    c.write_json(root/'evaluation.json',facts)
    lines=['# 终态收益标签扩展与轻量门控：2026-10-12','',
        '已采集整个现有训练/开发分区的终态反事实标签，训练两个字符TF-IDF + Logistic Regression原型，并把冻结选中模型接入实际WebSocket终态回放。没有修改生产服务、旧门控权重或原Refiner权重。','',
        '## 数据边界','',
        '训练818个输入视图来自441组原始合成录音（441清晰、377加噪）；开发288个视图来自162组录音（162清晰、126加噪）。不是1106组新录音；全部复用已有ASR响应和音频文件。',
        '全量输入都通过原生终态处理器探测机会，未用旧窗口准备时的gate过滤，以避免漏掉真实分窗后的机会。训练/开发在模板、录音组、PCM及实际窗口输入上无交集，已使用的测试模板不参与本轮。',
        '旧pilot36包含在训练全量中，重新测量一次但没有额外合并计数。开发包括old_validation、新dev与risk_dev，是复用的开发资料；不能称为独立测试。','',
        '| 分区 | 输入视图 | 原始录音组 | 窗口弱标签 | 有益/中性/有害 | 原策略稳定控制 |',
        '|---|---:|---:|---:|---:|---:|']
    for split,title in [('train','训练'),('dev','开发')]:
        s=dataset['summaries'][split];labels=s['labels']
        lines.append(f"| {title} | {s['source_views']} | {s['source_groups']} | {s['window_examples']} | {labels.get('benefit',0)}/{labels.get('neutral',0)}/{labels.get('harm',0)} | {s['native_controls_stable']}/{s['source_views']} |")
    lines+=['',
        '原生窗口从REFINE改为KEEP后，其他窗口、局部审核、MLM与终态原文保护仍按原策略运行。每个签名单独干预，额外原策略控制重复一次。0或1机会的全部跳过复用已执行结果；物理调用数量从arms.jsonl计，不重复计算别名。',
        '标签是整条转录最终脚本编辑距离的变化，不是窗口自身输出是否改善。292条训练输入只有决策时可见的原文、准备后baseline、域和空左上下文；不把参考脚本、标签、成本或终态文本输入模型。',
        '成本另存调用数、实际forward/token位置和模型耗时。质量为neutral却没有保守计算优势的窗口，在保守成本目标中仍可要求执行；不把neutral简单等同廉价skip。','',
        '## 模型与开发选择','',
        '两个原型均为字符1–3gram TF-IDF + 线性逻辑回归：一个学习终态质量收益，一个学习保守成本动作。拟合只用训练分区，录音组及模板加权防止slot值和clean/noise重复占过高权重。',
        f"开发集选择 {selection['primary']}，skip评分阈值 {primary['threshold']}。选择要求逐视图脚本距离不退化、全部有益窗口保留，再比较已执行分支的模型耗时加CPU评分估计。没有使用测试集选阈值，也没有把分数当成校准置信度。",'',
        '| 开发策略 | 脚本完全匹配 | 脚本CER | Refiner调用 | MLM调用 |',
        '|---|---:|---:|---:|---:|',
        f"| 原策略 | {native['exact_script']}/288 | {native['intended_micro_cer']:.3%} | {native['refiner_calls']} | {native['plausibility_calls']} |",
        f"| 冻结轻量门控，实际终态回放 | {replay['exact_script']}/288 | {replay['intended_micro_cer']:.3%} | {replay['refiner_calls']} | {replay['plausibility_calls']} |",'',
        f"实际回放保留 {retained}/{total} 个有益窗口、0个视图脚本距离退化，veto {replay['vetoes']} 个窗口，最终输出逐字匹配事先按模型得分选择的已执行反事实分支。Refiner调用减少 {delta} 次（{delta/native['refiner_calls']:.1%}）。这是模型自身决策的结果，不是按参考答案事后挑选的oracle。",
        f"开发集评分阶段平均每次 {primary['cpu_score_mean_ms']:.3f} 毫秒；实际Web回调总计 {replay['gate_total_ms']:.1f} 毫秒/{gate_calls} 次，均值 {facts['actual_gate_mean_ms']:.3f} 毫秒。记录两种范围，避免把离线估计当作实回调耗时。",'',
        '## 限制与下一步','',
        '这是实际模型callback + 实际WebSocket终态处理器，但ASR采用冻结final响应、没有发送音频块或重新调用ASR。它验证冷终态路由结果和调用数，不能推导在线转录速度或真实流式缓存行为。',
        '所有意图脚本及合成音频都未人工核验，完全匹配/CER是弱脚本诊断指标，不能代表真实语音金标质量。本批没有harm标签，不能宣称风险已经校准或真实语音零退化。',
        '模型输入仍是文本，不含声学特征或ASR置信度。字符线性模型是低开销基线，尚无证据表明具有更强的语义理解或论文创新。',
        '下一个步骤：保持模型和阈值冻结，使用未参与训练/开发的新留出语音做真实带节奏WebSocket+HTTP ASR对照，随机策略顺序并报告最长延迟及失败；通过后再决定是否启用。当前模型只留作实验，生产未启用。','',
        '训练标签：dataset/train.jsonl；开发标签：dataset/dev.jsonl；模型：linear_training/terminal_quality/model.pkl；冻结选择：linear_training/selection.json；实际回放：linear_replay/recordings.jsonl；独立审计：LUNA_AUDIT.md。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for name in ('run_terminal_gain_round4.py','assemble_terminal_gain_round4.py','train_terminal_gain_linear.py','report_terminal_gain_round4.py'):
        path=ROOT/'experiments/scripts'/name;(root/'executed_code'/name).write_bytes(path.read_bytes())
    hashes={str(p.relative_to(root)):c.file_hash(p) for p in root.rglob('*') if p.is_file()
            and p.name not in {'integrity_check.json','completed_experiment.json'} and str(p.relative_to(root))!='logs/report.log'}
    c.write_json(root/'integrity_check.json',dict(artifacts_sha256=hashes,train_dev_disjoint=True,
        old_gate_weights_sha256=c.file_hash(ROOT/'results/gate_round3_20261011/training/encoder_ternary_operation/gate.pt'),
        old_gate_selection_sha256=c.file_hash(ROOT/'results/gate_round3_20261011/training/selection.json'),
        refiner_weights_sha256=cfg['refiner_weights_sha256'],production_enabled=False))
    c.write_json(root/'completed_experiment.json',dict(status='complete',terminal_labels_expanded=True,
        lightweight_prototype_trained=True,development_callback_replayed=True,independent_test_pending=True,
        production_enabled=False))
    print(json.dumps({k:v for k,v in facts.items() if k not in {'train','dev'}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();report(a.root.resolve())
