"""Deterministic, baseline-only selection and auditable benchmark export."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from build_entity_challenge import OUT, ROOT, OLD, SEED, scored_pool, scorable_label, read, rank, norm, score, sha, write, lines
from system.entity_store import EntityStore

ALIASES = {'农业生态': '农业种植', '医疗健康': '临床医疗', '法律政策': '法律司法',
           '科技计算机通信': '数码电子', '自然科学化学': '科学教育'}


def choose(target=2000, minimum=30, person_cap=900, tolerance=.005, dry_run=False):
    all_rows = scored_pool()
    for r in all_rows:
        r['original_domain'] = r['domain']; r['domain'] = ALIASES.get(r['domain'], r['domain'])
    counts = Counter(r['domain'] for r in all_rows)
    domains = sorted(d for d, n in counts.items() if n >= minimum)
    if len(domains) < 15:
        write(OUT / 'selection_feasibility.json', dict(feasible=False, reason='fewer_than_15_populated_domains',
              domains=dict(counts), minimum=minimum)); print('NEED_DOMAINS', len(domains)); return False
    # A variable counts recordings with equal domain and baseline mention/hit
    # counts; the solver cannot see entity-module outputs or acoustic alterations.
    groups = defaultdict(list)
    for r in all_rows:
        if r['domain'] in domains:
            s = r['baseline_score']
            groups[r['domain'], s['hits'], s['expected_mentions']].append(r)
    keys = sorted(groups); sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    costs = np.array([(.8 if k[0] == '人名机构地名' else 0) +
                      (.01 * k[1]/k[2]) + int(rank(repr(k))[:8], 16)/2**32 * .00001 for k in keys])
    vectors = [np.ones(len(keys))]; lower = [target]; upper = [target]
    for d in domains:
        vectors.append(np.array([k[0] == d for k in keys], dtype=float))
        lower.append(minimum)
        upper.append(person_cap if d == '人名机构地名' else min(250, counts[d]))
    vectors += [np.array([h-(.5-tolerance)*n for d,h,n in keys]),
                np.array([h-(.5+tolerance)*n for d,h,n in keys])]
    lower += [0, -np.inf]; upper += [np.inf, 0]
    constraints = LinearConstraint(np.array(vectors), np.array(lower), np.array(upper))
    result = milp(costs, integrality=np.ones(len(keys)), bounds=Bounds(np.zeros(len(keys)), sizes),
                  constraints=constraints, options={'time_limit':120, 'mip_rel_gap':.002})
    feasibility = dict(feasible=result.x is not None, status=int(result.status), message=result.message,
        pool_recordings=len(all_rows), eligible_domains={d:counts[d] for d in domains}, target_recordings=target,
        per_domain_minimum=minimum, person_recording_cap=person_cap, target_recall=.5, tolerance=tolerance,
        selector_source_sha256=sha(__file__), seed=SEED)
    if result.x is None:
        base_constraints = LinearConstraint(np.array(vectors[:-2]), np.array(lower[:-2]), np.array(upper[:-2]))
        ratio = .5
        for _ in range(8):
            objective = np.array([h-ratio*n for d,h,n in keys])
            floor = milp(objective, integrality=np.ones(len(keys)), bounds=Bounds(np.zeros(len(keys)),sizes),
                         constraints=base_constraints, options={'time_limit':30,'mip_rel_gap':.0001})
            if floor.x is None: break
            counts_at_floor = np.rint(floor.x).astype(int)
            floor_hits = sum(int(c)*k[1] for c,k in zip(counts_at_floor,keys))
            floor_mentions = sum(int(c)*k[2] for c,k in zip(counts_at_floor,keys))
            next_ratio = floor_hits/floor_mentions
            if abs(next_ratio-ratio) < 1e-8: break
            ratio = next_ratio
        if floor.x is not None:
            feasibility.update(best_current_recall=floor_hits/floor_mentions,
                               best_current_hits=floor_hits,best_current_mentions=floor_mentions)
        write(OUT / 'selection_feasibility.json', feasibility); print('INFEASIBLE', result.message); return False
    picked = []
    for k, n in zip(keys, np.rint(result.x).astype(int)):
        picked += sorted(groups[k], key=lambda r: rank(r['id']))[:n]
    picked.sort(key=lambda r: (r['domain'], rank(r['id'])))
    hits = sum(r['baseline_score']['hits'] for r in picked)
    expected = sum(r['baseline_score']['expected_mentions'] for r in picked)
    actual_counts = Counter(r['domain'] for r in picked)
    assert len(picked) == target and len(actual_counts) >= 15
    assert min(actual_counts.values()) >= minimum and len({r['pcm_sha256'] for r in picked}) == target
    assert len({norm(r['reference']) for r in picked}) == target
    assert abs(hits/expected - .5) <= tolerance + 1e-10
    feasibility.update(selected_counts=dict(actual_counts), hits=hits, expected_mentions=expected,
                       baseline_entity_recall=hits/expected)
    write(OUT / 'selection_feasibility.json', feasibility)
    print('SELECTED', target, len(actual_counts), 'DOMAINS', hits, '/', expected, hits/expected)
    if dry_run: return True
    destination = OUT / 'dataset'
    if (destination / 'manifest.json').exists(): raise FileExistsError('Dataset already frozen')
    destination.mkdir(parents=True, exist_ok=True)
    # Build each glossary from the complete admitted candidate pool for that
    # domain. Correct/easy and unselected distractor names remain in the library.
    vocab = defaultdict(dict)
    active = json.loads((OUT/'active_pools.json').read_text())['pools']
    vocabulary_sources = {}
    protected_file = ROOT/'results/entity_context_confirmation_20261012/unseen/summary.json'
    protected = set(json.loads(protected_file.read_text())['pcm_sha256'].values())
    for pool in active:
        folder = OUT/pool
        inputs_by_id = {s['id']:s for s in read(folder/'inference_inputs.jsonl')}
        frozen_labels = json.loads((folder/'references.json').read_text())['samples']
        vocabulary_sources[str(folder/'references.json')] = sha(folder/'references.json')
        for sid, original in frozen_labels.items():
            sample = inputs_by_id[sid]
            if sample['pcm_sha256'] in protected: continue
            label = scorable_label(sample, original)
            d = ALIASES.get(sample['domain'], sample['domain'])
            for e in label['gold_entities']:
                if norm(e['canonical']) and norm(e['canonical']) in norm(label['reference']):
                    vocab[d].setdefault(norm(e['canonical']), e)
    libraries = {}
    for index, d in enumerate(domains):
        path = destination / 'libraries' / (str(index+1).zfill(2) + '_' + d + '.db')
        store = EntityStore(path)
        words = sorted(vocab[d].values(), key=lambda e:e['canonical'])
        assert all(isinstance(e['canonical'],str) and e['canonical'].strip() and
                   e['entity_type'] in {'PERSON','ORG','LOC','TERM'} for e in words)
        stamp = datetime.now(timezone.utc).isoformat()
        # Initialize the actual EntityStore schema, then import all fresh rows in
        # one transaction. Per-word CRUD calls otherwise re-read the whole table.
        with store._connect() as connection:
            connection.executemany('''INSERT INTO protected_entities
                (canonical_text,entity_type,domain,normalization_policy,priority,enabled,source,created_at,updated_at)
                VALUES (?,?,'general','normalize',0,1,?,?,?)''',
                [(e['canonical'].strip(),e['entity_type'],
                  'Real-audio challenge: full reference-grounded pool vocabulary; evaluation only',stamp,stamp)
                 for e in words])
        imported = store.list_entities(domain='general')
        assert len(imported)==len(words) and all(e.normalization_policy=='normalize' and not e.aliases for e in imported)
        print('LIBRARY',d,len(vocab[d]),flush=True)
        libraries[d] = dict(path=str(path), sha256=sha(path), entity_count=len(vocab[d]))
    inputs, labels, baseline = [], {}, []
    from scipy.io.wavfile import write as write_wave
    (destination/'audio').mkdir(exist_ok=True)
    for r in picked:
        # Audio paths are preserved; source files are never rewritten or noised.
        input_keys = {'id','origin','source','source_recording_id','speaker_id','split','domain','original_domain',
                      'pcm_path','pcm_sha256','duration_seconds','source_audio_sha256','audio_path',
                      'source_parquet','source_row_index','subset','topics','corpus','cohort'}
        s = {k:v for k,v in r.items() if k in input_keys}
        pcm = Path(r['pcm_path']).read_bytes()
        assert hashlib.sha256(pcm).hexdigest() == r['pcm_sha256']
        wav_path = destination/'audio'/(r['id']+'.wav')
        write_wave(wav_path,16000,np.frombuffer(pcm,dtype='<f4'))
        s['evaluation_wav_path'] = str(wav_path)
        s['evaluation_wav_sha256'] = sha(wav_path)
        s['library_path'] = libraries[r['domain']]['path']
        s['source_group'] = r.get('speaker_id') if r['source'].startswith('AISHELL') else \
            r['source_recording_id'].rsplit('_',1)[0] if r.get('source_recording_id') else r['id']
        inputs.append(s)
        labels[r['id']] = dict(reference=r['reference'], gold_entities=r['gold_entities'],
                               annotation_method=r['annotation_method'])
        if 'reference_original' in r:
            labels[r['id']]['reference_original'] = r['reference_original']
        baseline.append(dict(id=r['id'], pcm_sha256=r['pcm_sha256'], raw_text=r['baseline_raw_text'],
            score=r['baseline_score'], source_result=r['baseline_source'], reused=r['baseline_reused']))
    lines(destination / 'inference_inputs.jsonl', inputs)
    write(destination / 'references.json', dict(schema_version=1, samples=labels))
    lines(destination / 'baseline_raw.jsonl', baseline)
    lines(destination / 'samples.jsonl', picked)
    summary = dict(recordings=target, domains={}, baseline_hits=hits, expected_mentions=expected,
        baseline_entity_recall=hits/expected, duration_seconds=sum(r['duration_seconds'] for r in picked),
        origins=dict(Counter(r['origin'] for r in picked)),
        types=dict(Counter(e['entity_type'] for r in picked for e in r['gold_entities'])),
        unique_entities=len({norm(e['canonical']) for r in picked for e in r['gold_entities']}),
        unique_source_groups=len({s['source_group'] for s in inputs}),
        error_recordings=sum(r['baseline_score']['hits'] < r['baseline_score']['expected_mentions'] for r in picked),
        normalization='Same conservative zh-ITN/OpenCC/NFKC canonical mention scoring as original pure-ASR report',
        scope='baseline-selected known-vocabulary challenge; not representative natural-traffic accuracy',
        audio_quality='Unmodified real source audio, no noise synthesis, no cropping to force errors; references not all listened to')
    for d in domains:
        rows = [r for r in picked if r['domain'] == d]
        h = sum(r['baseline_score']['hits'] for r in rows); n = sum(r['baseline_score']['expected_mentions'] for r in rows)
        summary['domains'][d] = dict(recordings=len(rows), hits=h, expected_mentions=n, entity_recall=h/n,
            unique_entities=len({norm(e['canonical']) for r in rows for e in r['gold_entities']}),
            source_groups=len({s['source_group'] for s in inputs if s['domain']==d}),
            library_size=libraries[d]['entity_count'])
    write(destination / 'summary.json', summary)
    files = {str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    write(destination / 'manifest.json', dict(schema_version=1, selector=feasibility,
        files_sha256=files, libraries=libraries,
        pool_manifests={str(p):sha(p) for p in OUT.glob('pool_*/manifest.json')},
        glossary_source_reference_sha256=vocabulary_sources,
        selected_pcm={r['id']:dict(path=r['pcm_path'], sha256=r['pcm_sha256']) for r in picked},
        selection_uses_module_outputs=False, source_registry=str(OUT/'source_registry.json')))
    report = ['# 实体难例评测集（2026-10-14）', '',
        f'真实录音片段 {target} 条，{len(domains)} 个领域；纯 ASR 实体命中 {hits}/{expected}，{hits/expected:.2%}。', '',
        '候选参考与领域词库先固定，再依据无词库、无提示、无精修 ASR 的命中情况筛选。'
        '未使用实体模块的纠错结果挑选样本。录音按 PCM 哈希和规范化参考文本去重。', '',
        '|领域|录音|实体命中/提及|原始 ASR 命中率|词库规模|', '|---|---:|---:|---:|---:|']
    for d, s in summary['domains'].items():
        report.append(f"|{d}|{s['recordings']}|{s['hits']}/{s['expected_mentions']}|{s['entity_recall']:.2%}|{s['library_size']}|")
    report += ['', '## 使用', '',
        '`inference_inputs.jsonl` 只含音频和独立词库路径；`references.json` 仅用于评分。'
        '`baseline_raw.jsonl` 保存逐条原始 ASR 输出及采集来源。实体模块评估应固定这份名单，配对比较原始输出和模块输出。', '',
        '## 范围', '',
        '50% 是整套数据的实体提及加权命中率，不代表每个领域均为 50%。无目标实体录音未参与主集配额。'
        '这套数据依据当前 ASR 的失败筛选，属于难例压力评测；不能用来推断自然流量准确率。', '',
        'AISHELL 项采用官方 NER；SpeechIO 项为参考支持的专名和术语词库匹配；CMeEE 项是医学词表到音频参考的匹配。'
        '所有来源均保留来源记录和标注方法，未声称逐条人工听审。部分音频片段来自同一原始节目，后续训练与评估应按 source_group 隔离。', '',
        '独立评估词库从全部候选的参考词条构建，含未选录音支持的干扰词；推理不传参考文本。'
        '音频与词库仅用于本地评测，SpeechIO 镜像许可不代表底层节目的统一再分发许可。', '',
        '[SpeechIO 官方](https://github.com/SpeechColab/Leaderboard) · '
        '[AISHELL-NER 官方](https://github.com/Alibaba-NLP/AISHELL-NER)', '']
    (destination / 'REPORT.md').write_text('\n'.join(report))
    return True


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--target',type=int,default=2000)
    p.add_argument('--minimum',type=int,default=30); p.add_argument('--person-cap',type=int,default=900)
    p.add_argument('--tolerance',type=float,default=.005); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); raise SystemExit(0 if choose(a.target,a.minimum,a.person_cap,a.tolerance,a.dry_run) else 2)
