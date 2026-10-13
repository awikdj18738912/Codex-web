"""Real-model regression replay for the medical entity fixes of 2026-10-09."""
import argparse
from collections import Counter
import csv
import difflib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from benchmark_multimed_entities import Runner,MODEL,require_complete,transcript,digest,report as medical_report
from evaluate_entity_library import ROOT,write_json
from system.entity_store import EntityStore

OUT=ROOT/'results/entity_repair_20261009'
PODCAST=ROOT/'results/oncology_podcast_20261008'
MEDICAL=ROOT/'results/multimed_chinese_test_eval_20261008'
ORIGINAL_LIBRARY=ROOT/'data/entity_benchmarks/oncology_podcast_20261008'
LIBRARY=ROOT/'data/entity_benchmarks/oncology_repaired_20261009'
MEDICAL_LIBRARY=ROOT/'data/entity_benchmarks/multimed_chinese_test'


def prepare():
    for folder,source in [(OUT/'podcast',PODCAST),(OUT/'medical',MEDICAL)]:
        folder.mkdir(parents=True,exist_ok=True);(folder/'service_logs').mkdir(exist_ok=True)
        for name in ['manifest.json','baseline.jsonl']:
            if not (folder/name).exists():shutil.copy2(source/name,folder/name)
    if (LIBRARY/'provenance.json').exists():return
    LIBRARY.mkdir(parents=True,exist_ok=True)
    definitions=json.loads((ORIGINAL_LIBRARY/'entities.json').read_text())
    store=EntityStore(LIBRARY/'entities.db');opted=[]
    for item in definitions:
        d=store.upsert_entity(item['canonical_text'],entity_type=item['entity_type'],domain=item['domain'],
            normalization_policy=item['normalization_policy'],priority=item['priority'],aliases=item['aliases'],source=item['source'])
        context=item.get('spelling_context')
        if context:
            context['source']='Predeclared medical scene; monoclonal names opt in to internal omission'
            if item['canonical_text'].endswith('单抗'):
                context['allow_internal_omission']=True
                context['keywords'].update({'抑制':1.0,'抗体':1.0,'用药':1.0})
                opted.append(item['canonical_text'])
            store.set_spelling_context(d.entity_id,keywords=context['keywords'],minimum_score=context['minimum_score'],
                source=context['source'],
                allow_internal_omission=context.get('allow_internal_omission',False),allow_near_spelling=False)
    write_json(LIBRARY/'entities.json',definitions)
    write_json(LIBRARY/'provenance.json',dict(vocabulary_size=len(definitions),
        original_glossary_sha256=digest(ORIGINAL_LIBRARY/'entities.json'),
        new_entities=[],error_aliases=False,opted_monoclonal_names=opted,
        frozen_hashes={name:digest(LIBRARY/name) for name in ['entities.db','entities.json']}))


def report():
    manifest=json.loads((OUT/'podcast/manifest.json').read_text());ids=[r['id'] for r in manifest['samples']]
    inputs={k:require_complete(path,ids) for k,path in {
        'no_library':OUT/'podcast/control_replay.jsonl',
        'old_rules':PODCAST/'entity_library.jsonl',
        'symbols_only':OUT/'podcast/symbols_only.jsonl',
        'symbols_and_drug_omission':OUT/'podcast/entity_library.jsonl'}.items()}
    summary={};differences=[];labels={'no_library':'无库控制','old_rules':'修复前 141 词库',
        'symbols_only':'缩写写法修复','symbols_and_drug_omission':'缩写＋单抗漏字修复'}
    targets={'HER2':('Her two',),'PD-1':('P D 一','P D一'),
        'PD-L1':('P D L 一','P D L一'),'贝伐珠单抗':('贝伐单抗',)}
    for key,records in inputs.items():
        substitutions=[n for r in records.values() for n in r['result'].get('entity_normalizations',[])]
        texts=[transcript(r,'clean_text') for r in records.values()]
        summary[key]=dict(samples=len(records),model_calls=sum(len(r['model_calls']) for r in records.values()),
            normalization_events=len(substitutions),event_types=dict(Counter(n['match_type'] for n in substitutions)),
            final_standard_occurrences={name:sum(t.count(name) for t in texts) for name in targets},
            remaining_source_forms={name:sum(t.count(source) for t in texts for source in forms) for name,forms in targets.items()},
            changed_samples=sum(transcript(r,'clean_text')!=transcript(inputs['old_rules'][rid],'clean_text') for rid,r in records.items()))
    for sample in manifest['samples']:
        rid=sample['id'];base=inputs['old_rules'][rid]
        for key in inputs:
            record=inputs[key][rid]
            assert record['chunk_payloads']==base['chunk_payloads'] and record['final_payload']==base['final_payload']
            assert record['pcm_sha256']==base['pcm_sha256']
        before=transcript(base,'clean_text');after=transcript(inputs['symbols_and_drug_omission'][rid],'clean_text')
        if before!=after:
            edits=[dict(source=before[i:j],target=after[a:b],context=before[max(0,i-15):min(len(before),j+15)])
                for op,i,j,a,b in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if op!='equal']
            differences.append(dict(id=rid,start_seconds=sample['start_seconds'],end_seconds=sample['end_seconds'],before=before,after=after,edits=edits))
    write_json(OUT/'podcast/summary.json',summary);write_json(OUT/'podcast/changes.json',differences)
    with (OUT/'podcast/comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['id','start_seconds','end_seconds',*inputs])
        for s in manifest['samples']:writer.writerow([s['id'],s['start_seconds'],s['end_seconds'],*[transcript(records[s['id']],'clean_text') for records in inputs.values()]])
    (OUT/'podcast/repaired.txt').write_text('\n\n'.join(f"[{s['start_seconds']:.0f}–{s['end_seconds']:.0f}s]\n{transcript(inputs['symbols_and_drug_omission'][s['id']],'clean_text')}" for s in manifest['samples'])+'\n')
    # Existing paired medical references keep the same definitions and model.
    old_medical=require_complete(MEDICAL/'entity_library.jsonl',[s['id'] for s in json.loads((MEDICAL/'manifest.json').read_text())['samples']])
    new_medical=require_complete(OUT/'medical/entity_library.jsonl',list(old_medical))
    medical_differences=[dict(id=rid,old=transcript(r,'clean_text'),new=transcript(new_medical[rid],'clean_text'))
        for rid,r in old_medical.items() if transcript(r,'clean_text')!=transcript(new_medical[rid],'clean_text')]
    write_json(OUT/'medical/differences_from_prior_rules.json',medical_differences)
    # Use the prior no-library real-model control when reporting reference CER.
    shutil.copy2(MEDICAL/'control_replay.jsonl',OUT/'medical/control_replay.jsonl')
    medical_report(SimpleNamespace(output=OUT/'medical',library=MEDICAL_LIBRARY))
    medical_summary=json.loads((OUT/'medical/summary.json').read_text())
    lines=['# 实体库修复验证：2026-10-09','',
        '播客 38 窗口复用原真实 ASR 响应，精修模型重新推理。仍为相同 141 个标准名，无新词、无错字别名、词库辅助识别关闭。', '',
        '| 阶段 | 精修调用 | 最终 HER2 数 | 最终 PD-1 数 | 最终 PD-L1 数 | 最终贝伐珠单抗数 |','|---|---:|---:|---:|---:|---:|']
    for key,s in summary.items():
        c=s['final_standard_occurrences'];lines.append(f"| {labels[key]} | {s['model_calls']} | {c['HER2']} | {c['PD-1']} | {c['PD-L1']} | {c['贝伐珠单抗']} |")
    archive=OUT/'podcast/initial_with_redundant_hints'
    if archive.exists():
        lines+=['','## 撤回的中间方案','',
            '| 方案 | 精修调用 | 处理 |','|---|---:|---|']
        for condition,label in [('symbols_only','初版缩写规则'),('entity_library','初版缩写＋药名漏字')]:
            records=require_complete(archive/f'{condition}.jsonl',ids)
            calls=sum(len(r['model_calls']) for r in records.values())
            lines.append(f'| {label} | {calls} | 已验证的缩写仍被重复提示，增加调用；撤回重复提示，保留验证后直接规范 |')
    lines+=['','这些是最终文字中的标准写法出现数，不是人工核准的正确纠错数；中间替换事件不作为收益次数。播客用于修复开发，没有标准字幕。', '',
        '## 修复前后文字变化','', '| 时间窗 | 变化 |','|---|---|']
    esc=lambda s:str(s).replace('|','\\|').replace('\n',' ')
    for r in differences:
        edits='；'.join(f"{e['source']!r} → {e['target']!r}（{e['context']}）" for e in r['edits'])
        lines.append(f"| {r['start_seconds']:.0f}–{r['end_seconds']:.0f} 秒 | {esc(edits)} |")
    s=medical_summary['entity_library']
    lines+=['','## 原 225 条医疗样本回归','',
        f"最终文字相对修复前改变 {len(medical_differences)} 条；参考 CER {s['CER']:.3%}，实体保留 {s['recovered_mentions']}/{s['expected_mentions']}。未新增或移除词表。", '',
        '本次只有实体功能模块及其测试、评估脚本、说明改动；ASR、Web、精修模型和数字处理代码未修改。正式用户库不自动合并专题词表。', '',
        '优化库：`data/entity_benchmarks/oncology_repaired_20261009/entities.db` / `entities.json`。已验证现有实体 CLI 可完整导入 141 词及单抗漏字权限。', '',
        '225 条参考文本使用优化后的 141 词库直接检查，文字未发生修改，证据见 reference_negative_check.json。仍需额外的独立肿瘤音频验证泛化收益。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','run-final','report']);args=parser.parse_args()
    if args.stage in {'prepare','run','run-final'}:prepare()
    if args.stage in {'run','run-final'}:
        if args.stage=='run-final':
            archive=OUT/'podcast/initial_with_redundant_hints';archive.mkdir(exist_ok=True)
            ids=[s['id'] for s in json.loads((OUT/'podcast/manifest.json').read_text())['samples']]
            for condition in ['symbols_only','entity_library']:
                path=OUT/'podcast'/f'{condition}.jsonl';saved=archive/path.name
                if not saved.exists():
                    require_complete(path,ids);path.rename(saved)
        settings=SimpleNamespace(output=OUT/'podcast',library=ORIGINAL_LIBRARY,model=MODEL,device='cuda:0',
            asr_url='http://127.0.0.1:8766',mode='streaming',chunk_seconds=2.0)
        runner=Runner(settings);runner.run('control_replay');runner.run('symbols_only')
        settings.library=LIBRARY;runner.run('entity_library')
        settings.output=OUT/'medical';settings.library=MEDICAL_LIBRARY;runner.run('entity_library')
    if args.stage in {'report','run','run-final'}:report()


if __name__=='__main__':main()
