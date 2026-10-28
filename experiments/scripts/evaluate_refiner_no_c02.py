"""Compare the no-C02 LoRA against cached original LoRA on identical inputs."""
import json
from pathlib import Path

from evaluate_refiner13_qwen import BASE, run_model, metric_text, levenshtein

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
MODEL=Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_noC02_lora_20261027')
OUT=ROOT/'results/refiner13_no_C02_20261027/evaluation'


def read(path):
    return [json.loads(s) for s in path.open()]


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def score(rows):
    clean=[r for r in rows if r['input']==r['reference']]
    total=sum(len(metric_text(r['reference'])) for r in rows)
    return {'n':len(rows),'exact_count':sum(r['prediction']==r['reference'] for r in rows),
        'exact_rate':sum(r['prediction']==r['reference'] for r in rows)/len(rows),
        'normalized_char_difference_rate':sum(levenshtein(metric_text(r['prediction']),
            metric_text(r['reference'])) for r in rows)/max(total,1),
        'identity_n':len(clean),'identity_changed_count':sum(r['prediction']!=r['input'] for r in clean)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read(DATA/'split_audit.jsonl') if r['split']=='test']
    assert len(rows)==1757 and all(r['category']!='C02' for r in rows)
    manifest=json.loads((DATA/'manifest.json').read_text())
    old={r['id']:r for r in read(ROOT/'results/refiner13_qwen25_lora_20261026/evaluation/qwen25_lora.jsonl')}
    for r in rows:
        assert old[r['id']]['input']==r['input'] and old[r['id']]['reference']==r['reference']
    run_model('qwen25_05b_no_C02_lora',BASE,MODEL,rows,manifest['system_prompt'],OUT/'no_C02_test.jsonl')
    new=read(OUT/'no_C02_test.jsonl')
    new_by_id={r['id']:r for r in new}
    failures=[r for r in read(ROOT/'results/refiner13_pregate_balanced_20261027/comparison.jsonl')
        if not r['label'] and r['lora_prediction']!=r['text']]
    assert len(failures)==71
    # These are clean-text diagnostics, not training rows or new blind test data.
    missing=[{'id':r['id'],'category':r['category'],'source':r['source'],
        'input':r['text'],'reference':r['text']} for r in failures if r['id'] not in new_by_id]
    if missing:
        run_model('prior_changed_clean_diagnostics',BASE,MODEL,missing,
            manifest['system_prompt'],OUT/'prior_71_extra.jsonl')
        new_by_id.update({r['id']:r for r in read(OUT/'prior_71_extra.jsonl')})
    diagnostics=[{'id':r['id'],'source_category':r['category'],'input':r['text'],
        'previous_prediction':r['lora_prediction'],'new_prediction':new_by_id[r['id']]['prediction'],
        'new_equals_input':new_by_id[r['id']]['prediction']==r['text']} for r in failures]
    (OUT/'prior_71_comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in diagnostics))
    selected_old=[old[r['id']] for r in rows]
    summary={'status':'complete','original_lora_on_same_1757':score(selected_old),
        'no_C02_lora_on_1757':score(new),
        'prior_71_changed_clean_diagnostics':{'n':71,'previous_changed_count':71,
            'new_changed_count':sum(not r['new_equals_input'] for r in diagnostics),
            'now_unchanged_count':sum(r['new_equals_input'] for r in diagnostics)},
        'best_checkpoint':json.loads((MODEL/'trainer_state.json').read_text()).get('best_model_checkpoint'),
        'interpretation':'Candidate references, not human gold. Prior 71 selected on old model failures; diagnostic only, not unbiased new test. One seed and fewer training examples/steps after removing C02.',
        'gate_used':False}
    write('comparison_summary.json',summary)
    a,b=summary['original_lora_on_same_1757'],summary['no_C02_lora_on_1757']
    report=f'''# 移除 C02 后的 LoRA 对照

训练数据仅移除 C02，重新从 Qwen2.5-0.5B-Instruct 基座训练两轮 LoRA。原有其他类别、提示词、学习率及有效批量沿用；新训练集 31,642 条、验证集 1,757 条。数据仍是候选参考。

## 同一 1,757 条非 C02 测试

| 指标 | 原两轮 LoRA | 去 C02 两轮 LoRA |
| --- | ---: | ---: |
| 完全匹配候选参考 | {a['exact_count']}（{a['exact_rate']:.2%}） | {b['exact_count']}（{b['exact_rate']:.2%}） |
| 归一化字符差异率 | {a['normalized_char_difference_rate']:.4%} | {b['normalized_char_difference_rate']:.4%} |
| 不修改候选被改动 | {a['identity_changed_count']}/{a['identity_n']} | {b['identity_changed_count']}/{b['identity_n']} |

## 原 71 条被改动的干净候选

新模型有 {summary['prior_71_changed_clean_diagnostics']['now_unchanged_count']} 条恢复原文，仍有 {summary['prior_71_changed_clean_diagnostics']['new_changed_count']} 条发生文本变化。发生变化不等于有害误改，需结合口癖、重复和信息保留要求逐条看。

这 71 条按旧模型输出筛选，已被查看，只用于错误诊断。即使其中有 C02 来源参考派生文本，也仅作干净文本推理诊断，从未加入新训练数据。训练规模与步数随删类减少，单一随机种子的对照不能证明每个具体删除的来源。

逐句结果见 `prior_71_comparison.jsonl`；正式子集结果见 `no_C02_test.jsonl`；汇总见 `comparison_summary.json`。
'''
    for r in diagnostics:
        if r['id'] in {'FCGEC:3428dafbea193cc4b05c49432451b6da','FCGEC:543719ff90326a1579a60640912dd0ad','FCGEC:31cb65d9d19c0b81ea5cd92ac55e2443'}:
            report+='\n### '+r['id']+'\n\n输入：'+r['input']+'\n\n原模型：'+r['previous_prediction']+'\n\n去 C02 模型：'+r['new_prediction']+'\n'
    (OUT/'移除C02微调对照报告.md').write_text(report)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
