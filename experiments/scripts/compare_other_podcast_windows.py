"""Repeat the four-way evaluation on podcast windows omitted last round."""
import argparse
from collections import Counter
import csv
import json
from types import SimpleNamespace

import compare_entity_methods as framework
from benchmark_multimed_entities import MODEL
from evaluate_entity_library import ROOT,write_json

PREVIOUS=ROOT/'results/entity_methods_comparison_20261009/podcast/protocol.json'
framework.OUT=ROOT/'results/entity_methods_other_podcast_20261009'
framework.SOURCES={'podcast':framework.SOURCES['podcast']}


def configure():
    prior=json.loads(PREVIOUS.read_text())
    excluded={s['row'] for s in prior['samples']}
    manifest=json.loads((framework.SOURCES['podcast'][0]/'manifest.json').read_text())
    framework.PODCAST_ROWS={s['row'] for s in manifest['samples']}-excluded
    return prior


def prepare():
    prior=configure();framework.prepare()
    path=framework.OUT/'podcast/protocol.json';protocol=json.loads(path.read_text())
    assert {s['id'] for s in protocol['samples']}.isdisjoint({s['id'] for s in prior['samples']})
    assert protocol['context_text']==prior['context_text']
    assert protocol['library_hashes']==prior['library_hashes']
    protocol['cohort']='All 27 remaining podcast windows, disjoint from the prior 11-window factorial comparison; earlier full-podcast experiments exist; no reference captions'
    protocol['excluded_previous_ids']=[s['id'] for s in prior['samples']]
    protocol['comparison_to_previous']='Same 141-term library and exact 16-term prompt; no production rule changes'
    write_json(path,protocol)


def report():
    framework.report()
    folder=framework.OUT/'podcast';protocol=framework.validate(folder)
    details=json.loads((folder/'comparison.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    targets=('HER2','PD-L1','CAR-T','PD-1','贝伐珠单抗')
    counts={arm:{name:sum(d['results'][arm]['text'].count(name) for d in details) for name in targets} for arm in framework.ARMS}
    write_json(folder/'standard_form_counts.json',counts)
    with (folder/'comparison_with_times.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['id','start_seconds','end_seconds',*framework.ARMS])
        for d in details:writer.writerow([d['id'],d['start_seconds'],d['end_seconds'],*[d['results'][arm]['text'] for arm in framework.ARMS]])
    raw_on=framework.require_complete(folder/'asr_on.jsonl',[d['id'] for d in details])
    echoes=[rid for rid,r in raw_on.items() if '参考术语仅用于辅助拼写' in r['final_payload'].get('text','')]
    norms={arm:Counter(n['match_type'] for d in details for n in d['results'][arm]['normalizations']) for arm in framework.ARMS}
    prior=PREVIOUS.parent
    old_details=json.loads((prior/'comparison.json').read_text())
    old_protocol=json.loads(PREVIOUS.read_text())
    assert protocol['context_text']==old_protocol['context_text'] and protocol['library_hashes']==old_protocol['library_hashes']
    assert protocol['source_hashes']==old_protocol['source_hashes']
    assert {d['id'] for d in details}.isdisjoint({d['id'] for d in old_details})
    all_details=sorted([*old_details,*details],key=lambda d:d['id'])
    assert len(all_details)==38
    old_summary=json.loads((prior/'summary.json').read_text())
    whole={arm:dict(samples=38,refiner_calls=old_summary[arm]['refiner_calls']+summary[arm]['refiner_calls'],
        standard_forms={name:sum(d['results'][arm]['text'].count(name) for d in all_details) for name in targets}) for arm in framework.ARMS}
    write_json(framework.OUT/'whole_podcast_summary.json',whole)
    lines=['# 其他 27 段癌症播客：四组结果解读','',
        '使用上一轮四组对照未选的全部 27 个窗口，约 26 分 29 秒。保持相同 141 词库、16 个提示词和生产规则，完成 54 次新的真实 ASR、108 次实际 Web／精修流程回放。', '',
        '| 条件 | HER2 标准形式 | PD-L1 标准形式 | CAR-T 标准形式 | 精修调用 |','|---|---:|---:|---:|---:|']
    for arm in framework.ARMS:
        c=counts[arm];lines.append(f"| {framework.LABELS[arm]} | {c['HER2']} | {c['PD-L1']} | {c['CAR-T']} | {summary[arm]['refiner_calls']} |")
    lines+=['','本轮后处理的 4 条规范事件全部为 `Her2 → HER2` 大小写统一；组合组为 3 条，其余一处由带提示 ASR 直接输出 HER2。最后两组都保留 8 处 HER2 标准形式。', '',
        '24–25 分钟窗口中，无提示结果为 `CAR T`、`PDL一CAR T`；带提示结果为 `CAR-T`、`PD-L1 CAR-T`。辅助识别在本批增加 4 处 CAR-T、1 处 PD-L1 的标准字面写法，组合同时保留后处理的大小写规范。', '',
        '这体现了局部写法规范上的互补，不是新修复 5 个中文医学实体错误。本批没有新增中文药名／病名自动纠正记录。该窗口的 `PDE` 写法仍然保留，缺少标准转录，不能认定所有缩写都已正确。', '',
        f'本批真实 ASR 中提示说明复制次数为 {len(echoes)}，不代表上一轮医疗样本中的问题已经修复；生产提示格式没有改动。', '',
        '其他变化主要包括标点、空格、口头词，也出现它／他、在／再及节目作者名称写法差异。完整原音和标准字幕未经人工逐句校对，无法宣布组合整体准确率更高或没有误改。', '',
        '## 两轮合并：完整 38 个窗口','',
        '| 条件 | HER2 | PD-L1 | CAR-T | PD-1 | 贝伐珠单抗 | 精修调用 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in whole.items():
        c=s['standard_forms'];lines.append(f"| {framework.LABELS[arm]} | {c['HER2']} | {c['PD-L1']} | {c['CAR-T']} | {c['PD-1']} | {c['贝伐珠单抗']} | {s['refiner_calls']} |")
    lines+=['','两轮参数和代码相同，合并后覆盖完整 37 分 29 秒播客；每个 60 秒窗口独立会话，不等同于一次连续转录。之前也有该播客开发实验，不是未见独立语料。', '',
        '表中是标准字面形式出现数，不是准确率或人工确认的纠错数；大小写、连字符和数字读法差异不能混计成真实识别错误。没有标准字幕，不计算 CER。', '',
        '本轮没有修改生产代码、正式词库或默认开关；字幕对照 CSV 包含时间窗，见 `podcast/comparison_with_times.csv`。']
    (framework.OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','decode','replay','report','run']);args=p.parse_args()
    configure()
    if args.stage in {'prepare','run'}:prepare()
    if args.stage in {'decode','run'}:framework.decode()
    if args.stage in {'replay','run'}:
        settings=SimpleNamespace(output=framework.OUT/'podcast',library=framework.OUT/'podcast/library',
            model=MODEL,device='cuda:0',asr_url=framework.ASR,mode='streaming',chunk_seconds=2.0)
        runner=framework.FactorialRunner(settings)
        for arm in framework.ARMS:runner.run_cell('podcast',arm)
    if args.stage in {'report','run'}:report()


if __name__=='__main__':main()
