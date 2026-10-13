"""Independent report-only audit of the frozen large-entity experiment.

Reads references only after inference. Adds intervention, negative-control,
dictionary-surface insertion, and per-topic metrics without changing any rules.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/large_entity_benchmark_v1_20261010'
ARMS = ('baseline', 'entity_rules', 'entity_audio_repair')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(out):
    sys.path.insert(0, str(out / 'source_snapshot'))
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.entity_audio_repair import normalized
    manifest = json.loads((out / 'manifest.json').read_text())
    samples = {s['id']: s for s in manifest['samples']}
    gold_rows = json.loads((BASE / 'dataset/sample_entities.json').read_text())
    gold = {r['sample_id']: r for r in gold_rows}
    references = {r['sample_id']: r['reference'] for r in json.loads((BASE / 'dataset/references.json').read_text())}
    audio_rows = read_rows(out / 'audio.jsonl')
    web_rows = read_rows(out / 'web.jsonl')
    audio = {r['id']: r for r in audio_rows}
    web = {(r['id'], r['condition']): r for r in web_rows}
    assert len(audio_rows) == len(audio) == len(samples)
    assert len(web_rows) == len(web) == len(samples) * 3
    assert set(audio) == set(samples)
    assert set(web) == {(sid, arm) for sid in samples for arm in ARMS}
    converter = opencc.OpenCC('t2s')
    itn = ZhITNNormalizer(ROOT / 'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    cache = {}

    def norm(text):
        if text not in cache:
            cache[text] = normalized(converter.convert(itn.normalize(text).text))
        return cache[text]

    names = {norm(e['canonical_text']) for e in json.loads((BASE / 'dataset/entities.json').read_text())}
    assert '' not in names
    expression = re.compile('|'.join(re.escape(n) for n in sorted(names, key=lambda n: (-len(n), n))))

    def surfaces(text):
        # Leftmost, longest non-overlapping dictionary surfaces, not full NER.
        return Counter(m.group() for m in expression.finditer(text))

    def difference(previous, current, ref):
        before = Levenshtein.distance(ref, previous)
        after = Levenshtein.distance(ref, current)
        added = surfaces(current) - surfaces(previous)
        unsupported = Counter({word: count for word, count in added.items() if word not in ref})
        return dict(changed=previous != current, improved=after < before, regressed=after > before,
                    equal_error_changed=previous != current and after == before,
                    error_delta=after - before, new_surfaces_absent_from_reference=dict(unsupported))

    pairs = [('baseline', 'entity_rules'), ('entity_rules', 'entity_audio_repair')]
    summaries = {f'{a}_to_{b}': Counter() for a, b in pairs}
    groups = {}
    entity_types = {}
    target_transitions = {f'{a}_to_{b}': Counter() for a, b in pairs}
    negatives = {arm: Counter() for arm in ARMS}
    repair = Counter()
    raw_summary = {stage: Counter() for stage in ('initial', 'production_final')}
    downstream_targets = {arm: Counter() for arm in ARMS}
    details = []
    calls = {arm: Counter() for arm in ARMS}
    call_times = {arm: [] for arm in ARMS}
    pipelines = {arm: [] for arm in ARMS}
    for sid, sample in samples.items():
        ref = norm(references[sid])
        values = {arm: norm(web[sid, arm]['result'].get('clean_text', '')) for arm in ARMS}
        expected = Counter((e['entity_type'], norm(e['canonical'])) for e in gold[sid]['gold_entities'])
        target_hits = {arm: {} for arm in ARMS}
        for (entity_type, name), count in expected.items():
            group = entity_types.setdefault(entity_type, {arm: Counter() for arm in ARMS})
            for arm in ARMS:
                hits = min(count, values[arm].count(name))
                target_hits[arm][entity_type, name] = hits
                group[arm].update(expected_mentions=count, hits=hits)
        for a, b in pairs:
            gains = sum(max(0, target_hits[b][key] - target_hits[a][key]) for key in expected)
            losses = sum(max(0, target_hits[a][key] - target_hits[b][key]) for key in expected)
            target_transitions[f'{a}_to_{b}'].update(gained_mentions=gains, lost_mentions=losses,
                gained_samples=int(gains > 0), lost_samples=int(losses > 0))
        detail = dict(id=sid, topics=sample['topics'], cohort=gold[sid]['cohort'], reference=references[sid], transitions={})
        for arm in ARMS:
            row = web[sid, arm]
            declared = row['result'].get('session_refiner_call_count')
            calls[arm].update(audited_calls=len(row['model_calls']), sessions=len(row['model_calls']) > 0,
                             declared_call_count=declared or 0,
                             declared_count_mismatch=declared is not None and declared != len(row['model_calls']))
            call_times[arm].extend(c['wall_seconds'] for c in row['model_calls'])
            pipelines[arm].append(row['elapsed_seconds'])
            if gold[sid]['cohort'] == 'negative':
                negatives[arm].update(samples=1, reference_chars=len(ref), errors=Levenshtein.distance(ref, values[arm]))
        for a, b in pairs:
            key = f'{a}_to_{b}'
            metric = difference(values[a], values[b], ref)
            unsupported = metric['new_surfaces_absent_from_reference']
            counts = dict(samples=1, changed_samples=int(metric['changed']), improved_samples=int(metric['improved']),
                          regressed_samples=int(metric['regressed']), equal_error_changed_samples=int(metric['equal_error_changed']),
                          error_delta=metric['error_delta'], new_absent_surface_samples=int(bool(unsupported)),
                          new_absent_surface_mentions=sum(unsupported.values()))
            summaries[key].update(counts)
            for topic in sample['topics']:
                groups.setdefault(topic, {}).setdefault(key, Counter()).update(counts)
            if gold[sid]['cohort'] == 'negative':
                summaries[key].update(negative_changed_samples=int(metric['changed']), negative_improved_samples=int(metric['improved']),
                                      negative_regressed_samples=int(metric['regressed']),
                                      negative_new_absent_surface_mentions=sum(unsupported.values()))
            if metric['changed'] or unsupported:
                detail['transitions'][key] = dict(metric, before=web[sid, a]['result'].get('clean_text', ''),
                                                 after=web[sid, b]['result'].get('clean_text', ''))
        a = audio[sid]
        expected_names = Counter(norm(e['canonical']) for e in gold[sid]['gold_entities'])
        for arm in ARMS:
            source_stage = 'production_final' if arm == 'entity_audio_repair' else 'initial'
            raw = norm(a[source_stage].get('text', ''))
            gains = losses = 0
            for name, count in expected_names.items():
                before, after = min(count, raw.count(name)), min(count, values[arm].count(name))
                gains += max(0, after - before)
                losses += max(0, before - after)
            downstream_targets[arm].update(gained_mentions=gains, lost_mentions=losses,
                                          gained_samples=int(gains > 0), lost_samples=int(losses > 0))
        for stage, value in raw_summary.items():
            raw = norm(a[stage].get('text', ''))
            value.update(samples=1, expected_mentions=sum(expected_names.values()),
                         hits=sum(min(count, raw.count(name)) for name, count in expected_names.items()),
                         errors=Levenshtein.distance(ref, raw), reference_chars=len(ref))
        status = a['production_final']['entity_audio_repair']
        changes = status.get('changes', [])
        raw_metric = difference(norm(a['initial'].get('text', '')), norm(a['production_final'].get('text', '')), ref)
        absent = [c for c in changes if norm(c['canonical']) not in ref]
        repair.update(samples=1, checks=status.get('checks', 0), applied_samples=bool(changes), changes=len(changes),
                      canonical_absent_changes=len(absent), canonical_absent_samples=bool(absent),
                      raw_improved_samples=raw_metric['improved'], raw_regressed_samples=raw_metric['regressed'],
                      raw_equal_error_changed_samples=raw_metric['equal_error_changed'])
        if detail['transitions'] or changes:
            detail['raw_repair'] = dict(raw_metric, changes=changes, canonical_absent_changes=absent)
            details.append(detail)
    for value in negatives.values():
        value['CER'] = value['errors'] / value['reference_chars'] if value['reference_chars'] else None
    for group in entity_types.values():
        for value in group.values():
            value['entity_recall'] = value['hits'] / value['expected_mentions']
    for value in raw_summary.values():
        value['CER'] = value['errors'] / value['reference_chars']
        value['entity_recall'] = value['hits'] / value['expected_mentions']

    def timing(values):
        if not values:
            return dict(count=0)
        ordered = sorted(values)
        return dict(count=len(values), total_seconds=sum(values), mean_seconds=statistics.mean(values),
                    p50_seconds=statistics.median(values), p95_seconds=ordered[min(len(ordered)-1, int(.95*len(ordered)))])

    health = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in (8051, 8081, 8766):
        try:
            health[str(port)] = json.load(opener.open(f'http://127.0.0.1:{port}/health', timeout=10))
        except Exception as error:
            health[str(port)] = dict(error=repr(error))
    integrity = dict(audio_rows=len(audio_rows), unique_audio_rows=len(audio), web_rows=len(web_rows), unique_web_rows=len(web),
                     complete_pairs=True, formal_database_sha256=digest(manifest['formal_database']),
                     formal_database_expected_sha256=manifest['formal_database_sha256'],
                     formal_database_unchanged=digest(manifest['formal_database']) == manifest['formal_database_sha256'],
                     dataset_manifest_unchanged=digest(BASE / 'dataset/manifest.json') == manifest['dataset_manifest_sha256'],
                     library_unchanged=digest(BASE / 'dataset/library/entities.db') == manifest['library_sha256'],
                     snapshot_unchanged=all(digest(out / name) == h for name, h in manifest['snapshot_sha256'].items()),
                     services_health=health, all_services_healthy=all(v.get('ok') is True for v in health.values()),
                     audit_script_sha256=digest(__file__),
                     evaluation_script_sha256=digest(ROOT / 'experiments/scripts/evaluate_large_entity_benchmark.py'))
    resume_file = out / 'resume_processes.json'
    if resume_file.exists():
        integrity['collection_resume'] = json.loads(resume_file.read_text())
    topic_summary = json.loads((out / 'topic_summary.json').read_text())
    macro = {arm: dict(topic_mean_CER=statistics.mean(t[arm]['CER'] for t in topic_summary.values()),
                       topic_mean_entity_recall=statistics.mean(t[arm]['entity_recall'] for t in topic_summary.values() if t[arm]['entity_recall'] is not None)) for arm in ARMS}
    audit = dict(topic_macro_means=macro, transitions=summaries, target_transitions=target_transitions,
                 per_entity_type=entity_types, negative_controls=negatives, raw_audio_repair=repair,
                 raw_asr_summary=raw_summary, downstream_target_transitions=downstream_targets,
                 per_topic_transitions=groups,
                 refiner_calls=calls, actual_model_call_timing={a: timing(v) for a, v in call_times.items()},
                 serial_web_replay_timing={a: timing(v) for a, v in pipelines.items()}, integrity=integrity,
                 limitations=[
                     'Audited model_calls capture refine* text methods only. Web session counters also include repetition review; a count mismatch is a scope difference. Final-event repetition counts are not session totals.',
                     'New absent dictionary surfaces are a literal insertion proxy, not semantic false-positive NER; source annotations may be incomplete.',
                     'Canonical absent changes compare against supplied normalized references; no per-item human listening or identity linking is claimed.',
                     'CER regressions are reference agreement regressions and are not adjudicated acoustic errors.',
                     'ASR collection used four concurrent workers; its request elapsed time is not a single-client latency benchmark.',
                     'Serial Web timings replay fixed actual ASR responses and exclude network ASR inference; model-call timings describe actual GPU calls.',
                     'Topic sizes overlap and are highly imbalanced; the news cohort must be assessed separately.',
                 ])
    write(out / 'extended_audit.json', audit)
    write(out / 'intervention_details.json', details)
    report = out / 'REPORT.md'
    text = report.read_text()
    marker = '\n## 完整性、误改与负例补充\n'
    text = text.split(marker)[0]
    text += marker + '\n|比较|改动样本|CER改善|CER退化|等错误量改动|新增库表面词不在参考中|\n|---|---:|---:|---:|---:|---:|\n'
    for key, value in summaries.items():
        text += f"|{key}|{value['changed_samples']}|{value['improved_samples']}|{value['regressed_samples']}|{value['equal_error_changed_samples']}|{value['new_absent_surface_mentions']}处/{value['new_absent_surface_samples']}条|\n"
    text += '\n|目标实体变化|新增命中|丢失命中|新增命中的样本|丢失命中的样本|\n|---|---:|---:|---:|---:|\n'
    for key, value in target_transitions.items():
        text += f"|{key}|{value['gained_mentions']}|{value['lost_mentions']}|{value['gained_samples']}|{value['lost_samples']}|\n"
    text += '\n|实体类型|无库字面命中|仅规则字面命中|加原音复核字面命中|\n|---|---:|---:|---:|\n'
    for entity_type, group in sorted(entity_types.items()):
        text += '|' + entity_type + '|' + '|'.join(f"{group[arm]['hits']}/{group[arm]['expected_mentions']}（{group[arm]['entity_recall']:.2%}）" for arm in ARMS) + '|\n'
    text += '\n|进入Web精修流程之前|目标字面命中|CER|\n|---|---:|---:|\n'
    for stage, value in raw_summary.items():
        text += f"|{stage}|{value['hits']}/{value['expected_mentions']}（{value['entity_recall']:.2%}）|{value['CER']:.2%}|\n"
    text += '\n上述原始ASR表用于与完整Web结果对照。Web还包含精修门控、数字规范等现有处理，其整体差异不能全部归因于精修模型。\n'
    text += '\n|ASR结果经过Web后续处理|新增目标命中|丢失目标命中|\n|---|---:|---:|\n'
    for arm, value in downstream_targets.items():
        text += f"|{arm}|{value['gained_mentions']}|{value['lost_mentions']}|\n"
    text += '\n话题等权平均（话题可交叉，非自然语料总体估计）：' + '；'.join(f"{arm} CER {value['topic_mean_CER']:.2%}、目标字面召回 {value['topic_mean_entity_recall']:.2%}" for arm, value in macro.items()) + '。\n'
    text += '\n新增库表面词统计为最长非重叠字面代理指标，不等同语义误插入；来源参考/实体标注可能不完整。完整变更详情见 `intervention_details.json`。\n'
    text += '\n|负例条件|样本|字符错误|参考字符|CER|\n|---|---:|---:|---:|---:|\n'
    for arm, value in negatives.items():
        text += f"|{arm}|{value['samples']}|{value['errors']}|{value['reference_chars']}|{value['CER']:.2%}|\n"
    text += '\n原音复核实际应用 ' + str(repair['changes']) + ' 处/' + str(repair['applied_samples']) + '条；其中规范名不在归一参考中 ' + str(repair['canonical_absent_changes']) + ' 处。原始ASR参考字符错误改善 ' + str(repair['raw_improved_samples']) + ' 条、退化 ' + str(repair['raw_regressed_samples']) + ' 条。\n'
    text += '\n四并发ASR耗时不能解释为单并发延迟。三组Web回放串行执行，耗时不含ASR网络推理。实际模型调用时间与分话题干预计数见 `extended_audit.json`。\n'
    text += '\n计时日志 model_calls 仅独立捕获 refine* 文本精修方法，未捕获 review_repetition_candidate 等重复审核方法；actual_model_call_timing 对应前者。Web session_refiner_call_count 是含重复审核的会话累计逻辑调用数，declared_count_mismatch 记录的是这两种口径之差。\n'
    if resume_file.exists():
        text += '\n采集曾因ASR服务不可达中断，从2,074条检查点续跑；恢复后的服务健康配置字段与烟雾评估时一致。已完成的记录原样保留，每条音频的三组仍共享同一实际初稿。该运行包含服务重启，不能作为连续稳定性或统一热机延迟结论。\n'
    text += '\n输出唯一性检查：' + str(len(audio)) + '条独立ASR记录、' + str(len(web)) + '条唯一Web会话。正式实体库哈希一致：' + str(integrity['formal_database_unchanged']) + '。\n'
    report.write_text(text)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    assert all(integrity[k] for k in ('formal_database_unchanged', 'dataset_manifest_unchanged', 'library_unchanged', 'snapshot_unchanged', 'all_services_healthy')), 'Final integrity or service health check failed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=BASE / 'full_evaluation')
    run(parser.parse_args().out)
