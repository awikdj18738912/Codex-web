"""Compare frozen v1/v2 entity results with identical gold and primary text."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'results/entity_challenge_current_20261014_v1'
DATA = ROOT / 'results/entity_challenge_20261014/dataset'

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]

original_inputs = rows(DATA / 'inference_inputs.jsonl')
mixed = {r['id'] for r in original_inputs if r['domain'] == '人名机构地名'}
old = {r['id']:r for r in rows(OLD / 'comparisons.jsonl')}
new = {r['id']:r for r in rows(OUT / 'comparisons.jsonl')}
assert len(old) == len(new) == 2000 and set(old) == set(new)
for rid in old:
    assert old[rid]['reference'] == new[rid]['reference'], rid
    assert old[rid]['arms']['paired_primary']['text'] == new[rid]['arms']['paired_primary']['text'], rid

def aggregate(ids, source, arm):
    values = [source[rid]['arms'][arm] for rid in ids]
    hits = sum(x['entity']['hits'] for x in values)
    mentions = sum(x['entity']['expected'] for x in values)
    errors = sum(x['cer_errors'] for x in values)
    chars = sum(x['cer_reference_chars'] for x in values)
    return dict(samples=len(ids), hits=hits, mentions=mentions, entity_recall=hits/mentions,
                cer_errors=errors, reference_chars=chars, CER=errors/chars)

comparison = dict(conditions='Same frozen entity source, primary transcript and gold. 727 changed library inputs rerun against actual ASR; 1273 unchanged inputs reused.',
    total={}, affected_727={},
    original_capacity_error_samples=json.loads((OLD/'summary.json').read_text())['module_error_count'],
    corrected_capacity_error_samples=json.loads((OUT/'summary.json').read_text())['module_error_count'])
for name, ids in [('total',set(old)),('affected_727',mixed)]:
    comparison[name] = dict(baseline=aggregate(ids,old,'paired_primary'),
        original_mixed_library=aggregate(ids,old,'entity_audio_raw'),
        scene_libraries=aggregate(ids,new,'entity_audio_raw'),
        scene_libraries_and_text_rules=aggregate(ids,new,'entity_rules_auto'))
    values=comparison[name]
    values['audio_recall_change_pp'] = (values['scene_libraries']['entity_recall']-values['original_mixed_library']['entity_recall'])*100
    values['new_audio_more_hits_samples'] = sum(new[r]['arms']['entity_audio_raw']['entity']['hits']>old[r]['arms']['entity_audio_raw']['entity']['hits'] for r in ids)
    values['new_audio_fewer_hits_samples'] = sum(new[r]['arms']['entity_audio_raw']['entity']['hits']<old[r]['arms']['entity_audio_raw']['entity']['hits'] for r in ids)
summary=json.loads((OUT/'summary.json').read_text())
comparison['new_false_edit_audit']=summary['false_edit_audit']
comparison['new_audio_review']=summary['audio_review']
comparison['new_paired_changes']=summary['paired_changes']
(OUT/'library_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
report=['# 场景分库修正前后对照','',
    '原有 2,000 条样本、音频、标准标注和首轮 ASR 文本全部相同；保持上一轮冻结的实体模块源码，仅改变原混合专名组的 727 条词库配置。其余 1,273 条在源码、PCM 和完整词库字节一致时复用历史真实输出。',
    '', '|范围|原始 ASR|原混合词库|场景分库|相对旧词库变化|', '|---|---:|---:|---:|---:|']
for name,label in [('total','全部 2,000 条'),('affected_727','受影响 727 条')]:
    v=comparison[name]
    report.append(f"|{label}|{v['baseline']['hits']}/{v['baseline']['mentions']}（{v['baseline']['entity_recall']:.2%}）|{v['original_mixed_library']['hits']}/{v['original_mixed_library']['mentions']}（{v['original_mixed_library']['entity_recall']:.2%}）|{v['scene_libraries']['hits']}/{v['scene_libraries']['mentions']}（{v['scene_libraries']['entity_recall']:.2%}）|{v['audio_recall_change_pp']:+.2f} 个百分点|")
report += ['',f"默认文字匹配器容量错误：{comparison['original_capacity_error_samples']} → {comparison['corrected_capacity_error_samples']} 条。新版文字规则诊断覆盖 {summary['arms']['entity_rules_auto']['samples']}/2000；未运行神经 Refiner 或完整 Web。",
    '',f"新版原音复核新增恢复 {summary['audio_review']['recovered_mentions']} 次 gold 提及，丢失 {summary['audio_review']['lost_mentions']} 次。自动误改候选 {summary['false_edit_audit']['candidate_sample_count']} 个样本，不能仅凭净命中增益宣称无误改。",
    '', '37 个词库均通过默认匹配器加载，最大 4,177 词；原混合库 6,808 个词的并集全部保留，目标实体覆盖无缺失。',
    '', '原混合组中 433/727 条（59.56%）仍缺乏场景线索，保留为“综合新闻／场景未定”。场景标签由目标实体遮蔽后的参考文本离线生成，属于已知场景条件评测，不代表自动场景识别或独立盲测。',
    '', '数字明细见 `library_comparison.json`；新版与原始 ASR 的配对评分见 `REPORT.md`；数据集构建与分组证据见 `../entity_challenge_20261014_scene_v2/`。','']
(OUT/'LIBRARY_COMPARISON.md').write_text('\n'.join(report))
print(json.dumps(comparison,ensure_ascii=False,indent=2))
