"""Frozen, source-held-out entity evaluation; references never enter inference.

prepare uses pyarrow in the base environment; subsequent stages use agentic-asr.
The reference-derived glossary tests advance knowledge of vocabulary, not unseen
vocabulary generalization. One fresh unprompted ASR stream is shared by all arms;
the production repair wrapper additionally replays the actual original audio.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/generic_entity_blind_20261010'
ASR = 'http://127.0.0.1:8766'
MODEL = Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
ARMS = ('no_dictionary', 'dictionary_rules', 'audio_repair')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def verify():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['frozen_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen input or implementation changed: ' + name)
    return manifest


def prepare():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('The evaluation has already been frozen')
    import pyarrow as pa
    from system.entity_store import EntityStore
    metadata = json.loads((OUT / 'metadata.json').read_text())
    glossaries = json.loads((OUT / 'glossaries.json').read_text())
    # Input schema is explicit so dataset selection cannot accidentally leak
    # reference fields into the later inference manifest.
    samples, references, tables = [], {}, {}
    for source, definitions in glossaries.items():
        store = EntityStore(OUT / 'libraries' / source / 'entities.db')
        for definition in definitions:
            store.upsert_entity(definition['term'], entity_type='TERM',
                domain='general', normalization_policy='normalize', source='Reference-derived vocabulary frozen before any ASR request')
    for sample in metadata['samples']:
        identifier = sample['sample_id']
        if identifier in references:
            raise ValueError('Duplicate selected sample')
        if sample['source'].startswith('fleurs'):
            shard = str(Path('/home/aim0/data/datasets/ASR/fleurs/zh-CN_test') / sample['location']['shard'])
            if shard not in tables:
                with pa.memory_map(shard, 'r') as stream:
                    tables[shard] = pa.ipc.open_stream(stream).read_all()
            record = tables[shard].slice(sample['location']['row_index'], 1).to_pylist()[0]
            if record['raw_transcription'] != sample['reference']:
                raise ValueError('Reference changed in FLEURS')
            source_bytes = record['audio']['bytes']
            decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'],
                                    input=source_bytes, capture_output=True, check=True)
        else:
            path = Path('/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN') / sample['location']['clip_path']
            source_bytes = path.read_bytes()
            decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'],
                                    capture_output=True, check=True)
        pcm = decode.stdout
        if not pcm or len(pcm) % 4:
            raise ValueError('Invalid decoded PCM')
        audio_path = OUT / 'audio' / (identifier + '.pcm')
        audio_path.parent.mkdir(exist_ok=True)
        audio_path.write_bytes(pcm)
        samples.append(dict(id=identifier, source=sample['source'], cohort=sample['label'],
            audio_path=str(audio_path), pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            source_audio_sha256=hashlib.sha256(source_bytes).hexdigest(), duration_seconds=len(pcm)/64000,
            speaker_hash=sample.get('speaker_hash'), row=sample['location'].get('row_index', sample['location'].get('line'))))
        references[identifier] = dict(reference=sample['reference'], terms=sample['terms'])
    if len({s['source_audio_sha256'] for s in samples}) != len(samples):
        raise ValueError('Duplicate audio in selected cohort')
    write(OUT / 'references.json', references)
    paths = list((ROOT / 'system').rglob('*.py')) + [ROOT / 'system/web/index.html', Path(__file__).resolve()]
    paths += [OUT / 'metadata.json', OUT / 'glossaries.json', OUT / 'references.json']
    paths += list((OUT / 'libraries').rglob('entities.db'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    formal = Path('/home/aim0/桌面/tan/ASR/AgenticASR_entity_e5bedb3/data/entities.db')
    write(OUT / 'manifest.json', dict(samples=samples, frozen_hashes=hashes, arms=list(ARMS),
        formal_entity_database=str(formal), formal_entity_database_sha256=digest(formal),
        inference_reference_used=False, sampling='Reference-only preselection before ASR; deterministic hash order',
        vocabulary_scope='Known-vocabulary evaluation from references, new audio sources; not unseen-vocabulary evaluation',
        refiner_model=str(MODEL), asr_url=ASR, chunk_seconds=2.0))
    print('FROZEN', len(samples), 'samples', flush=True)


async def collect():
    import httpx
    from system.entity_store import EntityStore
    from system.entity_audio_repair import EntityAudioRepair
    manifest = verify()
    done = {r['id'] for r in rows(OUT / 'audio_results.jsonl')}
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        health = (await client.get(ASR + '/health')).raise_for_status().json()
        if not health.get('context_supported'):
            raise ValueError('ASR service has no context support')
        for index, sample in enumerate(manifest['samples']):
            if sample['id'] in done:
                continue
            pcm = Path(sample['audio_path']).read_bytes()
            if hashlib.sha256(pcm).hexdigest() != sample['pcm_sha256']:
                raise ValueError('Audio changed')
            sessions, recordings, request_times, trace = {}, [], Counter(), []
            async def request(endpoint, session_id=None, data=b'', params=None):
                query = dict(params or {})
                if session_id:
                    query['session_id'] = session_id
                started = time.perf_counter()
                result = (await client.post(ASR + endpoint, params=query, content=data)).raise_for_status().json()
                trace.append(dict(endpoint=endpoint, session_id=session_id, context=query.get('context', ''),
                                  data_sha256=hashlib.sha256(data).hexdigest(), result=deepcopy(result)))
                if endpoint == '/stream/start':
                    state = dict(ready=deepcopy(result), chunk_payloads=[], pcm=bytearray(),
                                 prompted=bool(query.get('context')), context=query.get('context', ''))
                    recordings.append(state)
                    sessions[result['session_id']] = state
                    role = 'prompted' if state['prompted'] else 'initial'
                else:
                    state = sessions.get(session_id)
                    role = 'prompted' if state and state['prompted'] else 'initial'
                    if endpoint == '/stream/chunk':
                        state['pcm'].extend(data)
                        state['chunk_payloads'].append(deepcopy(result))
                    elif endpoint == '/stream/finish':
                        if hashlib.sha256(state['pcm']).hexdigest() != sample['pcm_sha256']:
                            raise ValueError('Replayed PCM differs from original')
                        state['final_payload'] = deepcopy(result)
                        state['pcm_sha256'] = sample['pcm_sha256']
                        sessions.pop(session_id)
                    elif endpoint == '/stream/cancel':
                        sessions.pop(session_id, None)
                request_times[role] += time.perf_counter() - started
                return result
            definitions = EntityStore(OUT / 'libraries' / sample['source'] / 'entities.db').list_entities(domain='general')
            repair = EntityAudioRepair(request, definitions)
            started = time.perf_counter()
            try:
                ready = await repair.request('/stream/start', params=dict(language='Chinese', chunk_seconds=2.0))
                if ready.get('context_applied'):
                    raise ValueError('The initial ASR stream was prompted')
                for offset in range(0, len(pcm), 64000):
                    await repair.request('/stream/chunk', ready['session_id'], pcm[offset:offset+64000])
                final = await repair.request('/stream/finish', ready['session_id'])
            finally:
                repair.close()
                for sid in list(sessions):
                    await request('/stream/cancel', sid)
            for record in recordings:
                record.pop('pcm')
            if 'final_payload' not in recordings[0]:
                raise ValueError('Initial real ASR recording is incomplete')
            initial = recordings[0]
            append(OUT / 'audio_results.jsonl', dict(id=sample['id'], initial=initial,
                production_final=final, recordings=recordings, request_trace=trace, timings=dict(request_times),
                elapsed_seconds=time.perf_counter()-started))
            print('REAL_AUDIO', index+1, '/', len(manifest['samples']), sample['id'],
                  final.get('entity_audio_repair', {}).get('change_count', 0), flush=True)


def pipeline(device):
    from fastapi.testclient import TestClient
    from benchmark_multimed_entities import Runner
    from system.zh_itn_bridge import DEFAULT_LIBRARY
    manifest = verify()
    actual = {r['id']: r for r in rows(OUT / 'audio_results.jsonl')}
    if set(actual) != {s['id'] for s in manifest['samples']}:
        raise ValueError('Real audio collection is incomplete')
    runner = Runner(SimpleNamespace(output=OUT, library=OUT/'libraries', model=MODEL, device=device,
                                   asr_url=ASR, mode='streaming', chunk_seconds=2.0))
    active = {}
    async def request(url, endpoint, session_id=None, data=b'', params=None, **kwargs):
        sample, record = active['sample'], active['record']
        trace = record['request_trace']
        cursor = active['cursor']
        if endpoint == '/stream/cancel' and (cursor == len(trace) or trace[cursor]['endpoint'] != endpoint):
            return dict(cancelled=True)
        if cursor == len(trace):
            raise ValueError('Web made an unexpected ASR request')
        expected = trace[cursor]
        if (endpoint != expected['endpoint'] or session_id != expected['session_id']
                or (params or {}).get('context', '') != expected['context']
                or hashlib.sha256(data).hexdigest() != expected['data_sha256']):
            raise ValueError('Web ASR request differs from frozen real-audio request: ' + sample['id'])
        active['cursor'] += 1
        return deepcopy(expected['result'])
    runner.web._stream_request_async = request
    completed = {(r['id'], r['condition']) for r in rows(OUT / 'web_results.jsonl')}
    for source in dict.fromkeys(s['source'] for s in manifest['samples']):
        with ExitStack() as stack:
            clients = {}
            for arm in ARMS:
                log = OUT / 'service_logs' / (source + '-' + arm + '.jsonl')
                db = None if arm == 'no_dictionary' else OUT / 'libraries' / source / 'entities.db'
                app = runner.web.create_app(MODEL, device, ASR, 'Chinese', 256, log,
                    entity_db=db, entity_fuzzy_mode='off' if db is None else 'auto',
                    refinement_gate_mode='tri_state', final_refinement_mode='off', zh_itn_library=DEFAULT_LIBRARY,
                    entity_asr_context=False, entity_audio_repair=arm == 'audio_repair')
                clients[arm] = stack.enter_context(TestClient(app))
            for index, sample in enumerate(s for s in manifest['samples'] if s['source'] == source):
                active.update(sample=sample, record=actual[sample['id']])
                pcm = Path(sample['audio_path']).read_bytes()
                for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS))):
                    if (sample['id'], arm) in completed:
                        continue
                    runner.calls.clear(); events = Counter(); started = time.perf_counter()
                    active['cursor'] = 0
                    query = urlencode(dict(domain='general', language='Chinese', mode='streaming',
                        asr_chunk_seconds=2.0, entity_asr_context=0, entity_audio_repair=int(arm == 'audio_repair')))
                    with clients[arm].websocket_connect('/ws/stream?' + query) as socket:
                        ready = socket.receive_json()
                        if ready.get('event') != 'ready':
                            raise ValueError(ready)
                        for offset in range(0, len(pcm), 64000):
                            socket.send_bytes(pcm[offset:offset+64000])
                            while True:
                                event = socket.receive_json(); events[event.get('event')] += 1
                                if event.get('event') == 'error': raise RuntimeError(event)
                                if event.get('event') == 'chunk_ack': break
                        socket.send_json(dict(event='finish'))
                        while True:
                            event = socket.receive_json(); events[event.get('event')] += 1
                            if event.get('event') == 'error': raise RuntimeError(event)
                            if event.get('event') == 'final': break
                    expected = actual[sample['id']]['production_final'] if arm == 'audio_repair' else actual[sample['id']]['initial']['final_payload']
                    if event.get('raw_text', '').strip() != expected.get('text', '').strip():
                        raise ValueError('Web changed the expected ASR source')
                    if arm == 'audio_repair' and active['cursor'] != len(active['record']['request_trace']):
                        raise ValueError('Web did not replay the complete real ASR trace')
                    append(OUT / 'web_results.jsonl', dict(id=sample['id'], condition=arm, result=event,
                        model_calls=list(runner.calls), elapsed_seconds=time.perf_counter()-started, events=dict(events)))
                    print('WEB', sample['id'], arm, event.get('clean_text', ''), flush=True)
    verify()


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from evaluate_entity_library import normalized
    from evaluate_medical_entities import lexicon_occurrences
    manifest = verify()
    references = json.loads((OUT / 'references.json').read_text())
    glossaries = json.loads((OUT / 'glossaries.json').read_text())
    actual = {r['id']: r for r in rows(OUT / 'audio_results.jsonl')}
    web = {(r['id'], r['condition']): r for r in rows(OUT / 'web_results.jsonl')}
    converter, itn = opencc.OpenCC('t2s'), ZhITNNormalizer()
    cache = {}
    def norm(text):
        if text not in cache:
            cache[text] = normalized(converter.convert(itn.normalize(text).text))
        return cache[text]
    def final_text(record):
        result = record['result']
        if 'clean_text' in result:
            return result['clean_text']
        # The Web endpoint omits clean_text when the ASR output is empty.
        # Keep this failure in the cohort and score the empty hypothesis.
        if not result.get('raw_text', '').strip():
            return ''
        raise ValueError('Nonempty Web output has no final transcript')
    summaries, comparisons, repair_decisions = {}, [], []
    for sample in manifest['samples']:
        rid = sample['id']; ref = norm(references[rid]['reference'])
        names = [d['term'] for d in glossaries[sample['source']]]
        expected = Counter(e['text'] for e in lexicon_occurrences(references[rid]['reference'], names))
        texts = dict(raw_initial=actual[rid]['initial']['final_payload']['text'], raw_repair=actual[rid]['production_final']['text'])
        texts.update({arm: final_text(web[rid, arm]) for arm in ARMS})
        values = {}
        for arm, text in texts.items():
            have = Counter(e['text'] for e in lexicon_occurrences(text, names))
            value = dict(text=text, errors=Levenshtein.distance(ref, norm(text)),
                hits=sum(min(n, have[w]) for w,n in expected.items()),
                extras=sum(max(0, n-expected[w]) for w,n in have.items()))
            values[arm] = value
            for group in ('all', sample['source'], sample['source'] + ':' + sample['cohort']):
                counter = summaries.setdefault(group, {}).setdefault(arm, Counter())
                counter.update(samples=1, reference_chars=len(ref), errors=value['errors'], hits=value['hits'],
                               expected_mentions=sum(expected.values()), unreferenced_mentions=value['extras'])
                baseline = texts.get('dictionary_rules', texts['raw_initial']) if arm == 'audio_repair' else texts['raw_initial']
                baseline_errors = Levenshtein.distance(ref, norm(baseline))
                counter.update(improved=int(value['errors'] < baseline_errors), regressed=int(value['errors'] > baseline_errors))
        comparison = dict(id=rid, source=sample['source'], cohort=sample['cohort'],
            reference=references[rid]['reference'], expected=dict(expected), values=values,
            repair=actual[rid]['production_final'].get('entity_audio_repair', {}), timings=actual[rid]['timings'])
        comparisons.append(comparison)
        for change in comparison['repair'].get('changes', []):
            before = texts['raw_initial']
            single = before[:change['start']] + change['canonical'] + before[change['end']:]
            delta = Levenshtein.distance(ref, norm(single)) - values['raw_initial']['errors']
            repair_decisions.append(dict(id=rid, reference=references[rid]['reference'], change=change,
                                        single_edit_error_delta=delta))
    for groups in summaries.values():
        for arm, counter in groups.items():
            counter['CER'] = counter['errors']/counter['reference_chars'] if counter['reference_chars'] else 0
            counter['entity_recall'] = counter['hits']/counter['expected_mentions'] if counter['expected_mentions'] else None
    write(OUT / 'summary.json', summaries)
    write(OUT / 'comparisons.json', comparisons)
    write(OUT / 'repair_decisions.json', repair_decisions)
    write(OUT / 'changed_samples.json', [r for r in comparisons if r['values']['dictionary_rules']['text'] != r['values']['audio_repair']['text']])
    timings = [r['timings'].get('prompted', 0) for r in comparisons]
    changed = sum(bool(r['repair'].get('changes')) for r in comparisons)
    formal_unchanged = digest(manifest['formal_entity_database']) == manifest['formal_entity_database_sha256']
    write(OUT / 'scope_verification.json', dict(production_sources_unchanged=True, formal_database_unchanged=formal_unchanged,
        actual_audio_requests=len(actual), repaired_samples=changed, independent_reference_not_used_in_inference=True))
    lines = ['# 新来源预建词库验证', '',
        '规则和词库在任何 ASR 请求前冻结；标准转录只用于提前建词库和最后评分。两来源此前未发现用于实体规则开发的记录，但不能证明模型预训练从未见过。FLEURS 未提供说话人 ID，因此不宣称说话人独立。', '',
        '全部样本新调用无词库 ASR；正式原音复核对原始 PCM 实际重识别。三个 Web 条件共享相同初稿，使用实际 GPU Refiner；Web 阶段回放本轮真实 ASR 响应以控制解码波动。', '',
        '|来源/分组|条件|样本|实体命中|CER|额外未参考实体次数|改善/退化|', '|---|---|---:|---:|---:|---:|---:|']
    for group, values in summaries.items():
        for arm, value in values.items():
            lines.append(f"|{group}|{arm}|{value['samples']}|{value['hits']}/{value['expected_mentions']}|{value['CER']:.2%}|{value['unreferenced_mentions']}|{value['improved']}/{value['regressed']}|")
    lines += ['', 'audio_repair 的改善/退化相对 dictionary_rules；其他行相对原始 ASR，不能将精修导致的变化归因于实体功能。', '',
        f'原音修正 {changed} 个样本、{len(repair_decisions)} 次替换。逐次独立替换的字符距离改善/持平/变差：'
        f"{sum(r['single_edit_error_delta']<0 for r in repair_decisions)}/{sum(r['single_edit_error_delta']==0 for r in repair_decisions)}/{sum(r['single_edit_error_delta']>0 for r in repair_decisions)}；此指标不是语义人工正确率。",
        f'总额外带词库 ASR 请求耗时 {sum(timings):.2f} 秒，均摊每样本 {sum(timings)/len(timings):.2f} 秒；仅发生复核的样本耗时见 comparisons.json。', '',
        '负例仅表示所建词库实体不在参考中，仍使用对应完整词库；不会逐句预告实体。所有完整参考参与 CER，括号英文是否口述等数据标注风险需单独人工检查；不得按结果删样。',
        f'生产代码冻结哈希通过，正式数据库保持原样：{formal_unchanged}。本轮不修改生产规则。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries['all'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'collect', 'pipeline', 'report'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'collect': asyncio.run(collect())
    elif args.stage == 'pipeline': pipeline(args.device)
    else: report()
