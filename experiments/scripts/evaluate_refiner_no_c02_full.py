"""Compare full SFT to LoRA on the same frozen no-C02 evaluation items."""
import json
from pathlib import Path

from evaluate_refiner13_qwen import BASE, run_model, metric_text, levenshtein

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
FULL=Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_noC02_full_20261027')
OUT=ROOT/'results/refiner13_no_C02_20261027/evaluation'


def read(path): return [json.loads(s) for s in path.open()]
def write(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


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
    lora={r['id']:r for r in read(OUT/'no_C02_test.jsonl')}
    for r in rows:
        assert old[r['id']]['input']==r['input'] and old[r['id']]['reference']==r['reference']
        assert lora[r['id']]['input']==r['input'] and lora[r['id']]['reference']==r['reference']
    full_path=OUT/'no_C02_full_test.jsonl'
    run_model('qwen25_05b_no_C02_full',FULL,None,rows,manifest['system_prompt'],full_path)
    full={r['id']:r for r in read(full_path)}
    # Reuse the clean-text diagnostic challenge created for the LoRA comparison.
    original=[r for r in read(ROOT/'results/refiner13_pregate_balanced_20261027/comparison.jsonl')
        if not r['label'] and r['lora_prediction']!=r['text']]
    missing=[r for r in original if r['id'] not in full]
    if missing:
        challenge=[{'id':r['id'],'category':r['category'],'source':r['source'],
            'input':r['text'],'reference':r['text']} for r in missing]
        run_model('no_C02_full_clean_diagnostics',FULL,None,challenge,
            manifest['system_prompt'],OUT/'no_C02_full_71_extra.jsonl')
        full.update({r['id']:r for r in read(OUT/'no_C02_full_71_extra.jsonl')})
    lora_diag={r['id']:r for r in read(OUT/'prior_71_comparison.jsonl')}
    diags=[]
    for r in original:
        row=full[r['id']]
        diags.append({'id':r['id'],'category':r['category'],'source':r['source'],
            'input':r['text'],'reference':r['text'],'old_lora_prediction':r['lora_prediction'],
            'no_c02_lora_prediction':lora_diag[r['id']]['new_prediction'],
            'no_c02_full_prediction':row['prediction'],
            'no_c02_full_changed':row['prediction']!=r['text']})
    (OUT/'prior_71_full_comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in diags))
    original_test=[old[r['id']] for r in rows]
    lora_test=[lora[r['id']] for r in rows]
    full_test=[full[r['id']] for r in rows]
    summary={'status':'complete','test_split_n':len(rows),
        'original_full_C02_lora':score(original_test),
        'no_C02_lora':score(lora_test),'no_C02_full':score(full_test),
        'prior_71_clean_diagnostics':{'n':len(diags),
            'old_lora_changed':sum(x['old_lora_prediction']!=x['input'] for x in diags),
            'no_C02_lora_changed':sum(x['no_c02_lora_prediction']!=x['input'] for x in diags),
            'no_C02_full_changed':sum(x['no_c02_full_changed'] for x in diags)},
        'full_best_checkpoint':json.loads((FULL/'trainer_state.json').read_text()).get('best_model_checkpoint'),
        'qualification':'Candidate references, not human gold. Main test split excludes C02. The selected 71 diagnostic items were seen before and are not an unbiased test.',
        'gate_used':False}
    write('full_vs_lora_summary.json',summary)
    a,b,c=summary['original_full_C02_lora'],summary['no_C02_lora'],summary['no_C02_full']
    report=f'''# 去除 C02 后的 LoRA 与全量微调对照

全量微调从 Qwen2.5-0.5B-Instruct 基座开始，使用与去 C02 LoRA 相同的 12 类训练／验证／测试划分、系统提示词、两轮训练和有效批量。C02 完全没有进入这轮全量微调。

## 同一 1,757 条非 C02 测试集

| 指标 | 原版 LoRA（含 C02） | 去 C02 LoRA | 去 C02 全量微调 |
| --- | ---: | ---: | ---: |
| 完全匹配候选参考 | {a['exact_count']}（{a['exact_rate']:.2%}） | {b['exact_count']}（{b['exact_rate']:.2%}） | {c['exact_count']}（{c['exact_rate']:.2%}） |
| 归一化字符差异率 | {a['normalized_char_difference_rate']:.4%} | {b['normalized_char_difference_rate']:.4%} | {c['normalized_char_difference_rate']:.4%} |
| E07 等身份句被改动 | {a['identity_changed_count']}/{a['identity_n']} | {b['identity_changed_count']}/{b['identity_n']} | {c['identity_changed_count']}/{c['identity_n']} |

原版 LoRA 作为背景对照时，其输出只在同一 1,757 条、非 C02 测试上重新计分；不含 C02 的结果仍是候选参考匹配率，不是人工正确率。

## 原 71 条干净候选误改诊断集

旧 LoRA 改动 {summary['prior_71_clean_diagnostics']['old_lora_changed']} 条；去 C02 LoRA 改动 {summary['prior_71_clean_diagnostics']['no_C02_lora_changed']} 条；去 C02 全量微调改动 {summary['prior_71_clean_diagnostics']['no_C02_full_changed']} 条。71 条之前已查看，且其中有弱标签，故这里只用于定位删除行为。

明细：`prior_71_full_comparison.jsonl`；主测试结果：`no_C02_full_test.jsonl`；汇总：`full_vs_lora_summary.json`。
'''
    (OUT/'移除C02后LoRA与全量微调对照.md').write_text(report)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
