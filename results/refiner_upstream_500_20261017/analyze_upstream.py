"""Summarize original-pipeline acceptance without introducing new filters."""
from collections import Counter
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent


def read(name):
    return [json.loads(s) for s in (OUT/name).read_text().splitlines() if s.strip()]


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    calls=read('api_calls.jsonl')
    qc=read('qc_results.jsonl')
    decisions=Counter()
    for row in calls:
        if row['stage']!='qc':continue
        try:
            value=json.loads(row.get('response','').strip())
        except (ValueError,TypeError):
            decisions['non_json_or_error']+=1;continue
        if not isinstance(value,dict):decisions['non_object_json']+=1
        elif type(value.get('pass')) is bool:decisions['explicit_bool_pass' if value['pass'] else 'explicit_bool_fail']+=1
        elif 'pass' not in value:decisions['missing_pass_upstream_defaults_true']+=1
        else:decisions['non_boolean_pass']+=1
    languages={}
    for language,predicate in [('中文及中英混合',lambda s:not s.startswith('english_')),
                               ('英文',lambda s:s.startswith('english_'))]:
        group=[v for s,v in summary['per_scene'].items() if predicate(s)]
        oral=sum(v['oral'] for v in group);retained=sum(v['final'] for v in group)
        languages[language]={'oral':oral,'retained':retained,'retained_rate':retained/oral if oral else 0}
    statuses=Counter('pass' if r['passed'] else 'fail' for r in qc)
    assert len(qc)==summary['qc_checked']
    assert statuses['pass']==summary['qc_pass']
    assert sum(summary['per_scene'][s]['final'] for s in summary['per_scene'])==summary['final_retained']
    completed_calls=Counter(r['stage'] for r in calls if 'response' in r or 'error' in r)
    cancelled_records=Counter(r['stage'] for r in calls if 'response' not in r and 'error' not in r)
    additional={'language_subsets':languages,'qc_response_schema':dict(decisions),
                'completed_api_records':dict(completed_calls),'cancelled_or_queued_api_records':dict(cancelled_records),
                'generation_attempt_denominator_note':'summary oral_generation_attempts comes from committed upstream batch checkpoints. Calls abandoned during scheduling restart are not in this denominator; raw API records without response/error can include queued calls never submitted. Primary retention denominator is exactly500 valid Oral records.',
                'counts_consistent':True,
                'accuracy_not_measured':'Original acceptance only; no independent semantic relabeling or student/audio evaluation.'}
    (OUT/'analysis.json').write_text(json.dumps(additional,ensure_ascii=False,indent=2)+'\n')
    lines=['# AgenticASR官方原版：500条生成与保留率实验','',
        '## 实验口径','',
        f"- 官方仓库提交：`{summary['upstream_commit']}`，源码与提示词未修改。",
        '- 使用现有Qwen3-30B-A3B-Instruct-2507 AWQ4bit教师；论文使用Gemma-4-31B-IT，故非论文原教师复现。',
        '- 使用官方16个中英文内部场景及原权重、数字处理、口语提示、Clean生成、20% ASR分支和最终语义审查，未使用本地构造标签／数字规则／严格质检改造。',
        '- 每场景用原始提示生成一轮新种子，未扩充至量产3000种子池；500条小规模下种子池差异仍是实验限制。',
        '- 调度脚本配置输出路径、两份相同教师各HTTP并发4、超时600秒、随机数初始化及记录证据。采用原generate_scene补足有效Oral配额；最终检查所有可用记录，未跳过原版关键词QC。途中从单卡切到双卡时保留已完成检查点，未提交批次被中止；这些调用单独统计。',
        '- 接口兼容性调整：英文客服、英文语音指令种子在原json_object模式多次生成不完整JSON，因此仅这两个种子场景改为普通文本JSON响应；原提示、温度、长度上限和解析器未改，Oral／Clean／ASR／QC接口设置未改。英文客服池被上游解析器提取为117项扁平数组，沿用上游SeedSampler支持的数组输入，未手动修补。',
        '', '## 阶段数量','', '| 阶段 | 数量 |', '| --- | ---: |',
        f"| Oral已提交批次的生成尝试（检查点口径） | {summary['oral_generation_attempts']} |",
        f"| 有效Oral | {summary['valid_oral']} |",
        f"| Clean及关键词检查后 | {summary['after_clean_and_keyword_qc']} |",
        f"| ASR模拟后 | {summary['after_asr']} |",
        f"| 组装后 | {summary['after_assembly']} |",
        f"| 最终语义审查通过 | {summary['qc_pass']} |",
        f"| 近似重复移除 | {summary['near_duplicates_removed']} |",
        f"| 最终保留 | {summary['final_retained']} |",'',
        f"**以500条有效Oral为分母：{summary['retained_per_valid_oral']:.1%}。** 切换调度时未提交批次的调用单列，不能把检查点尝试数当作整轮服务请求数。已返回／报错API记录：`{dict(completed_calls)}`；无响应的取消／排队记录：`{dict(cancelled_records)}`。后者可能从未提交HTTP，不用于计算实际模型请求失败率。",'',
        f"最后一次恢复运行耗时{summary['elapsed_seconds']/60:.1f}分钟；包括失败重启及用户间隔的墙钟时间{summary.get('wall_seconds_including_restarts',summary['elapsed_seconds'])/60:.1f}分钟，不应当作纯推理耗时。途中切换为两张RTX3090相同教师并行；上游源码变化{len(summary['source_files_changed'])}个。",'',
        '## 语言子集','', '| 子集 | 有效Oral | 最终保留 | 保留率 |','| --- | ---: | ---: | ---: |']
    for label,value in languages.items():lines.append(f"| {label} | {value['oral']} | {value['retained']} | {value['retained_rate']:.1%} |")
    lines+=['','## 各场景','', '| 场景 | Oral | Clean | ASR | 组装 | QC通过 | 最终 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for scene,v in summary['per_scene'].items():lines.append(f"| {scene} | {v['oral']} | {v['clean']} | {v['asr']} | {v['assembled']} | {v['qc_pass']} | {v['final']} |")
    lines+=['','## 结果边界','',
        '这是原版自动接受率，不是标签正确率或学生效果。原版QC主要判断最终意图是否一致，不包含本地版本的全部严格构造／字面保护／目标残留检查；不能只用保留率高低判断训练数据优劣。',
        f"原始QC响应字段统计：`{dict(decisions)}`。其中缺失pass字段时默认通过属于原版逻辑，若发生会在这里单列，不做静默修正。",
        '不同教师、语言分布、提示与候选文本使本结果不能与此前中文为主的本地小样本当作严格对照。',
        '原始API请求与输出见api_calls.jsonl，逐条QC见qc_results.jsonl，各阶段见raw/，最终对见final/train_with_provenance.jsonl。未手工修标签、未启动学生训练。','',
        '参考：[官方仓库](https://github.com/AnXMuy/AgenticASR)、[原版生成器](https://github.com/AnXMuy/AgenticASR/blob/main/pipeline/src/generators/oral_text_generator.py)、[原版最终筛选](https://github.com/AnXMuy/AgenticASR/blob/main/pipeline/scripts/05_finalize.py)。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'retained':summary['final_retained'],'rate':summary['retained_per_valid_oral'],**additional},ensure_ascii=False))


if __name__=='__main__':main()
