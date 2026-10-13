"""Export terminal window labels; keep cost, quality and physical counts separate."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.gate_terminal_gain_pilot import paired_effect


def strategy_summary(rows,choose):
    arms=[choose(row) for row in rows]
    fields=('refiner_calls','window_calls','local_calls','plausibility_calls','refiner_token_positions',
            'forward_passes','refiner_wall_ms','plausibility_wall_ms')
    return dict(recordings=len(rows),exact_script=sum(a['intended_distance']==0 for a in arms),
        intended_distance=sum(a['intended_distance'] for a in arms),
        intended_micro_cer=sum(a['intended_distance'] for a in arms)/sum(r['intended_chars'] for r in rows),
        cost={field:sum(a['cost'][field] for a in arms) for field in fields})


def choose_posthoc(row,conservative=True):
    # Use one fully executed whole-recording arm, never sum independent vetoes.
    effect=paired_effect(row['native'],row['skip_all'],row['native_control_stable'])
    eligible=(effect['label'] in {'neutral','harm'}
              and (effect['skip_has_compute_advantage_proxy'] or not conservative))
    return row['skip_all'] if eligible else row['native']


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());cfg=json.loads((root/'configuration.json').read_text())
    c.verify_frozen(root/'runtime')
    assert cfg['manifest_sha256']==c.file_hash(root/'manifest.json')
    assert manifest['source_rows_sha256']==c.file_hash(root/'sources.jsonl')
    assert manifest['adapter_sha256']==c.file_hash(root/'executed_code/gate_terminal_gain_pilot.py')
    rows=d.read_rows(root/'recordings.jsonl');physical=d.read_rows(root/'arms.jsonl')
    assert len(rows)==manifest['recordings'] and len({r['seed_id'] for r in rows})==len(rows)
    assert all(r['synthetic_split']=='train' and r['variant']=='clean' for r in rows)
    assert len({r['pcm_sha256'] for r in rows})==len(rows)
    old_root=ROOT/'results/gate_round3_20261011'
    old_cases=d.read_rows(old_root/'train_data/pairs_full/cases.jsonl')
    clean_ids={r['case_id'] for r in old_cases if r['variant']=='clean'}
    old_labels=defaultdict(list)
    for old in d.read_rows(old_root/'dataset/train.jsonl'):
        if old['case_id'] in clean_ids and old['input']['task_kind']=='window_refine':
            old_labels[(old['recording_group'],old['input']['raw_text'])].append(old)
    labels=[];old_cross=Counter();conflict=defaultdict(set);format_changes=0
    for row in rows:
        native=row['native'];native_routes={r['key']:r['action'] for r in native['routes']}
        for effect in row['effects']:
            skip=row['counterfactuals'][effect['window_key']]
            skip_routes={r['key']:r['action'] for r in skip['routes']}
            assert native_routes.keys()==skip_routes.keys()
            assert [k for k in native_routes if native_routes[k]!=skip_routes[k]]==[effect['window_key']]
            assert skip_routes[effect['window_key']]=='keep'
            assert native['final']['raw_text']==skip['final']['raw_text']==row['raw_text']
            previous=old_labels.get((row['recording_group'],effect['input']['raw_text']),[])
            prior_labels=sorted({r['label'] for r in previous})
            for old in previous:old_cross[(old['label'],effect['label'])]+=1
            format_changes+=effect['label']=='neutral' and native['final_text']!=skip['final_text']
            window_time=sum(native['cost'][k] for k in ('refiner_wall_ms','plausibility_wall_ms'))
            repeat_time=sum(row['native_repeat']['cost'][k] for k in ('refiner_wall_ms','plausibility_wall_ms'))
            veto_time=sum(skip['cost'][k] for k in ('refiner_wall_ms','plausibility_wall_ms'))
            sample=dict(case_id=c.digest([manifest['runtime_protocol_sha256'],row['recording_group'],effect['window_key']])[:24],
                seed_id=row['seed_id'],recording_group=row['recording_group'],split='train',split_group=row['template_group'],
                input=effect['input'],feature_stage='actual_window_pre_action_prepared_source_no_final_context',
                label=effect['label'],supervision_status='terminal_effect_unverified_script_proxy',
                native_action='refine',eligible_for_weak_pretraining=effect['native_control_stable'],
                human_gold=False,audio_script_verified=False,eligible_for_real_risk_calibration=False,
                terminal_effect=effect,previous_operation_labels=prior_labels,
                cost_targets=dict(native=native['cost'],single_window_veto=skip['cost'],
                    measured_model_time_saving_ms=window_time-veto_time,
                    saving_against_faster_native_control_ms=min(window_time,repeat_time)-veto_time,
                    timing_scope='shared_GPU_model_calls_only_with_staging_not_online_latency'))
            labels.append(sample);conflict[effect['window_key']].add(effect['label'])
    file=root/'terminal_window_labels.jsonl'
    file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in labels))
    policies=dict(native=strategy_summary(rows,lambda r:r['native']),
        skip_all=strategy_summary(rows,lambda r:r['skip_all']),
        posthoc_conservative=strategy_summary(rows,choose_posthoc),
        posthoc_quality_only=strategy_summary(rows,lambda r:choose_posthoc(r,False)))
    effects=[e for r in rows for e in r['effects']]
    summary=dict(recordings=len(rows),templates=manifest['templates'],window_signatures=len(effects),
        labels=dict(Counter(e['label'] for e in effects)),stable_native_controls=sum(r['native_control_stable'] for r in rows),
        eligible_lower_compute_windows=sum(e['skip_has_compute_advantage_proxy'] for e in effects),
        neutral_with_format_change=format_changes,input_label_conflicts=sum(len(v)>1 for v in conflict.values()),
        old_operation_to_terminal_labels=[dict(previous=k[0],terminal=k[1],count=v) for k,v in sorted(old_cross.items())],
        old_label_source=str(old_root/'dataset/train.jsonl'),
        old_label_source_sha256=c.file_hash(old_root/'dataset/train.jsonl'),
        old_label_scope='variant=clean exact source window matches; prior teacher/runtime is different, not a pure label-only ablation',
        local_compensation_recordings=sum(r['skip_all']['cost']['local_calls']>r['native']['cost']['local_calls'] for r in rows),
        physical_execution=dict(snapshots=len(physical),refiner_calls=sum(len(r['model_calls']) for r in physical),
            plausibility_calls=sum(len(r['plausibility_calls']) for r in physical)),
        strategies=policies,posthoc_scope='reference_and_counterfactual_aware_selection_of_executed_arms_not_a_trained_or_deployable_policy',
        label_file_sha256=c.file_hash(file),human_gold=False,new_training=False,production_policy_changed=False,
        scope=manifest['scope'])
    c.write_json(root/'summary.json',summary)
    native,skip=policies['native'],policies['skip_all']
    oracle=policies['posthoc_conservative']
    lines=['# 窗口终态收益小规模对照：2026-10-11','',
        '本轮先检查窗口精修的实际终态收益，没有训练新门控或修改生产服务。',
        f"36段训练集无噪声合成音频、{manifest['templates']}个模板，按已有原生REFINE机会以及类别/模板均衡抽取；不根据老师输出或正确率选样。开发、风险开发及已使用的测试模板/音频/文本被排除。",'',
        '## 对照方式','',
        '使用实际WebSocket流式模式的终态处理器，ASR输入为已冻结的原始final响应。没有重新调用ASR或发送音频块，因此测的是冷终态快照，不是实时流式速度。',
        '每条先执行原策略，再逐个精确输入签名把一个原生REFINE窗口改为KEEP。其他窗口、局部审核、MLM补救和终态原文保护均保留原策略。相同输入签名的重复出现视为同一个决策组。',
        '37个单窗口干预均核对只有目标签名变化；原策略另跑一次，36/36终态文本、调用数和token处理量稳定。只有一条有两个签名，并额外执行全部跳过。其他35条的全部跳过复用其单窗口跳过结果，不重复计物理调用。',
        f"实际执行 {summary['physical_execution']['snapshots']} 个终态快照、{summary['physical_execution']['refiner_calls']} 次Refiner调用、{summary['physical_execution']['plausibility_calls']} 次MLM调用。",'',
        '## 结果','',
        '| 策略 | 脚本完全匹配 | 脚本CER | Refiner调用 | 窗口/局部调用 | MLM调用 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,title in [('native','原策略'),('skip_all','所有窗口跳过'),('posthoc_conservative','事后保守选择（仅空间分析）'),('posthoc_quality_only','事后仅按质量选择（仅空间分析）')]:
        arm=policies[name];cost=arm['cost']
        lines.append(f"| {title} | {arm['exact_script']}/36 | {arm['intended_micro_cer']:.3%} | {cost['refiner_calls']} | {cost['window_calls']}/{cost['local_calls']} | {cost['plausibility_calls']} |")
    lines+=['',
        '37个窗口的终态弱标签：4 benefit、33 neutral、0 harm。4个benefit都是填充词嗯的删除：跳过这些窗口会使最终脚本编辑距离增加1；其中一个仍有其他ASR错误，所以完全匹配数只减少3。',
        f"{summary['eligible_lower_compute_windows']}个窗口满足保守的计算代理优势：跳过后脚本距离不增加，Refiner调用、MLM调用和解码输入token位置均不增加，至少一项减少。",
        '6条记录的窗口精修被局部审核补救，Refiner调用保持1→1。窗口生成需要8–17次forward，局部候选打分只需要2次；局部打分的token位置反而略多。这说明次数、token位置和耗时是不同的指标，不能单独作为计算量或延迟真值。',
        '原策略与全部跳过各有1次实际MLM调用，来自同一重重起重复处理。它仍按原策略运行，并没有被窗口veto省掉。',
        '3个neutral干预仍有原样文本变化，只是去标点后脚本距离不变；neutral不表示每个字符都相同。',
        f"历史操作标签一对一比对发现5个benefit现在是终态neutral。它们的窗口可以修复口吃，但跳过后局部审核也能补救。这是把单步输出改善误当成必须调用窗口的典型例子。旧运行时与本次运行时也有保护规则差异，因此不称为纯标签消融。",'',
        '事后策略按已知参考脚本和已执行反事实选择，表明本批训练样本存在节省空间；它不是训练出的门控，也不能部署。保守选择每条只选一条完整执行过的策略，不把多个单窗口收益相加。',
        f"事后保守选择保留 {oracle['exact_script']}/36 完全匹配、脚本CER {oracle['intended_micro_cer']:.3%}，Refiner调用从 {native['cost']['refiner_calls']} 降至 {oracle['cost']['refiner_calls']}。这不是可泛化的安全上界。",'',
        '## 产物与下一步','',
        'terminal_window_labels.jsonl：37条窗口终态收益弱标签；只含train分区，输入为真实决策时可见的原文/准备后baseline/域/空左上下文，没有把参考脚本或终态输出放入模型输入。成本目标分别记录实际Refiner/局部/MLM调用、forward、token位置、两次原策略对照下的模型耗时变化。',
        '下一步值得扩大到整个训练分区收集同类终态标签，并在独立开发分区评估保留有用窗口的召回。当前仅4个benefit、0个harm，且只有21个合成模板，不足以直接重训和风险校准。',
        '相关检查155项通过，Luna独立核对了抽样边界、单窗口干预、重复控制和物理调用去重。',
        '标签比对按整条转录的终态收益进行；纯neutral且会增加后续审核开销的窗口不能直接当作廉价skip。门控推理自身的开销仍需在后续真实流式对照中计入。','',
        '所有脚本和音频未经人工核验，CER/完全匹配仅为弱脚本诊断，不是人工金标质量。只用无噪声train且均衡选择native REFINE机会，不能据此估计真实流量收益。原置信度被禁用，原Refiner权重和float32精度未改变，MLM权重CPU暂存带来额外搬运，模型耗时不代表在线延迟。','',
        '明细：recordings.jsonl；物理执行：arms.jsonl；冻结选择：manifest.json；独立审核：LUNA_AUDIT.md。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    executed=root/'executed_code';(executed/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    hashes={str(path.relative_to(root)):c.file_hash(path) for path in root.rglob('*') if path.is_file()
            and path.name not in {'integrity_check.json','completed_experiment.json'} and str(path.relative_to(root))!='logs/report.log'}
    c.write_json(root/'integrity_check.json',dict(artifacts_sha256=hashes,source_rows_verified=True,runtime_verified=True,
        selected_gate_unchanged_sha256=c.file_hash(ROOT/'results/gate_round3_20261011/training/encoder_ternary_operation/gate.pt'),
        refiner_weights_sha256=cfg['weights_sha256'],human_gold=False,audio_script_verified=False))
    c.write_json(root/'completed_experiment.json',dict(status='complete',pilot_labels_exported=len(labels),
        new_training=False,production_policy_changed=False,next_work='expand_train_terminal_labels_then_development_validation'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.root.resolve()),ensure_ascii=False,indent=2))
