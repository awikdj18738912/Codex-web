"""Compare a new gate to the frozen tiny gate using identical LoRA outputs."""
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
MODEL=Path('/home/aim0/data/ASR/training/refiner_noC02_gate_rbt3_20261027')
OLD=Path('/home/aim0/data/ASR/training/refiner_noC02_pregate_gpu0_20261027')
ACTUAL=ROOT/'results/refiner13_no_C02_gate_20261027/all3056_refiner'
OUT=ROOT/'results/refiner_noC02_gate_rbt3_20261027'


def read(p):return [json.loads(s) for s in p.open()]


def main():
    original=read(ACTUAL/'predictions.jsonl')
    new=read(MODEL/'balanced_test_predictions.jsonl')
    assert len(original)==len(new)==3056
    routes={r['id']:r for r in new}
    assert set(routes)=={r['id'] for r in original}
    for r in original:
        assert routes[r['id']]['text']==r['text'] and routes[r['id']]['label']==r['label']
    def result(use_new):
        rows=[]
        for r in original:
            route=routes[r['id']]['route_to_refiner'] if use_new else r['route_to_refiner']
            rows.append(dict(r,route=route,final=r['prediction'] if route else r['text']))
        clean=[r for r in rows if not r['label']]
        return {'n':len(rows),'refiner_calls':sum(r['route'] for r in rows),
            'gate_send_lora_edit':sum(r['route'] and r['changed_after_outer_whitespace'] for r in rows),
            'gate_send_lora_keep':sum(r['route'] and not r['changed_after_outer_whitespace'] for r in rows),
            'gate_keep_lora_edit':sum(not r['route'] and r['changed_after_outer_whitespace'] for r in rows),
            'gate_keep_lora_keep':sum(not r['route'] and not r['changed_after_outer_whitespace'] for r in rows),
            'weak_positive_missed':sum(not r['route'] and r['label']==1 for r in rows),
            'missed_by_category':dict(Counter(r['category'] for r in rows if not r['route'] and r['label']==1)),
            'weak_clean_skip_rate':sum(not r['route'] for r in clean)/len(clean),
            'final_exact_count':sum(r['final']==r['reference'] for r in rows),
            'lost_exact_vs_all_lora':sum(r['prediction']==r['reference'] and r['final']!=r['reference'] for r in rows),
            'gained_exact_vs_all_lora':sum(r['prediction']!=r['reference'] and r['final']==r['reference'] for r in rows)}
    old_result=result(False);new_result=result(True)
    nine=[r for r in original if not r['route_to_refiner'] and r['prediction']==r['reference'] and r['text']!=r['reference']]
    assert len(nine)==9
    diag=[{'id':r['id'],'input':r['text'],'lora_output':r['prediction'],
        'tiny_score':r['probability'],'rbt3_score':routes[r['id']]['probability'],
        'rbt3_sends_to_refiner':routes[r['id']]['route_to_refiner']} for r in nine]
    (OUT/'prior_nine_comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in diag))
    summary={'status':'complete','tiny_gate':old_result,'rbt3_gate':new_result,
        'prior_nine_now_sent':sum(r['rbt3_sends_to_refiner'] for r in diag),'prior_nine_n':9,
        'rbt3_model_metrics':json.loads((MODEL/'summary.json').read_text()),
        'loRA_outputs_reused':str(ACTUAL/'predictions.jsonl'),
        'qualification':'Same paired text splits, weak candidate labels, two training epochs and dev threshold rule. Prior nine already inspected and used only for diagnostic reporting, not threshold tuning. Larger backbone does not remedy weak label inconsistency. No latency benchmark.'}
    (OUT/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# RBT3 全量微调门控对照','',
        '此前870万参数门控也使用全量微调。本次将预训练编码器换为hfl/rbt3，编码器和分类头全部训练。原去C02训练／验证／测试、两轮训练、学习率和验证集阈值规则沿用，不因9条诊断样本调整标签或阈值。',
        '', '## 相同3,056条测试、相同LoRA实际输出','',
        '| 指标 | 870万参数门控 | RBT3门控 |','| --- | ---: | ---: |']
    for label,k in [('精修调用数','refiner_calls'),('送去但LoRA不改','gate_send_lora_keep'),
        ('放行但LoRA会改','gate_keep_lora_edit'),('原弱标签需修改漏检','weak_positive_missed'),
        ('最终匹配候选参考数','final_exact_count'),('门控损失原本可匹配参考的答案','lost_exact_vs_all_lora')]:
        lines.append(f'| {label} | {old_result[k]} | {new_result[k]} |')
    lines += ['',f'先前漏掉的9条中，本轮门控送精修 {summary["prior_nine_now_sent"]} 条。模型实际改写不等于客观需要精修，候选参考仍待人工审核；这些9条已被查看，不是独立盲测。',
        '', '## 9条诊断逐句对照','']
    for i,r in enumerate(diag,1):
        lines += [f'### {i}. '+('本轮送精修' if r['rbt3_sends_to_refiner'] else '本轮仍放行'),'',
            '原文：','',r['input'],'','LoRA可给出的输出：','',r['lora_output'],'',
            f'旧分数 {r["tiny_score"]:.6f}；新分数 {r["rbt3_score"]:.6f}。ID：`{r["id"]}`。','']
    lines += ['', '## 限制','',
        '这里只比较更换编码器后的效果。分数未经校准，阈值仍在验证集选取；换模型和全量训练不能消除标签噪声。测试来自同一生成方式，不能据此断言真实ASR泛化。推理时延和整体吞吐未测。',
        '', '模型来源：[hfl/rbt3](https://huggingface.co/hfl/rbt3)，官方中文预训练项目：[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。']
    (OUT/'RBT3门控全量微调对照.md').write_text('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
