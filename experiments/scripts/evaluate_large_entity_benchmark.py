"""Paired large-glossary evaluation through actual ASR and actual Web Refiner.

Default cohort is the predeclared 24-clip smoke set. --all prepares all 2,410
clips in a separate output folder. References are opened only by report().
"""
import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/large_entity_benchmark_v1_20261010'
DATA=BASE/'dataset'
ASR='http://127.0.0.1:8766'
MODEL=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
ARMS=('baseline','entity_rules','entity_audio_repair')
OUT=BASE/'smoke_evaluation'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    # Bind this experiment to its own source copy rather than concurrently
    # changing production files. All operational resources use absolute paths.
    path=OUT/'source_snapshot'
    if str(path) not in sys.path:sys.path.insert(0,str(path))


def verify():
    m=json.loads((OUT/'manifest.json').read_text())
    assert digest(DATA/'manifest.json')==m['dataset_manifest_sha256']
    assert digest(DATA/'library/entities.db')==m['library_sha256']
    for name,h in m['snapshot_sha256'].items():assert digest(OUT/name)==h,name
    return m


def prepare(all_samples):
    if (OUT/'manifest.json').exists():raise FileExistsError('Cohort already frozen')
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((BASE/'evaluation_inputs/input_manifest.json').read_text())['samples']
    selected=None if all_samples else set(json.loads((BASE/'evaluation_inputs/smoke_ids.json').read_text())['sample_ids'])
    samples=[s for s in inputs if selected is None or s['id'] in selected]
    assert samples and all('reference' not in s and 'gold_entities' not in s for s in samples)
    shutil.copytree(ROOT/'system',OUT/'source_snapshot/system',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(OUT)):digest(p) for p in (OUT/'source_snapshot').rglob('*') if p.is_file()}
    formal=Path('/home/aim0/桌面/tan/ASR/AgenticASR_entity_e5bedb3/data/entities.db')
    write(OUT/'manifest.json',dict(samples=samples,sample_count=len(samples),scope='full' if all_samples else 'smoke',
        library=str(DATA/'library/entities.db'),library_sha256=digest(DATA/'library/entities.db'),
        dictionary_size=1346,dataset_manifest_sha256=digest(DATA/'manifest.json'),snapshot_sha256=hashes,
        formal_database=str(formal),formal_database_sha256=digest(formal),
        conditions=list(ARMS),refiner_model=str(MODEL),refinement_gate='tri_state',final_refinement='off',
        source_reference_at_inference=False,asr='Actual service once for initial stream, plus actual prompted replay; shared first pass'))
    print('PREPARED',len(samples),flush=True)


async def collect(workers=1):
    import httpx
    snapshot()
    from system.entity_audio_repair import EntityAudioRepair
    from system.entity_store import EntityStore
    m=verify();definitions=EntityStore(DATA/'library/entities.db').list_entities(domain='general')
    assert len(definitions)==m['dictionary_size']
    done={r['id'] for r in rows(OUT/'audio.jsonl')}
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        semaphore=asyncio.Semaphore(workers)
        async def process(index,sample):
            if sample['id'] in done:return
            async with semaphore:
                pcm=Path(sample['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
                trace=[];initial=None;roles={};times=Counter()
                async def request(endpoint,session_id=None,data=b'',params=None):
                    nonlocal initial
                    query=dict(params or {})
                    if session_id:query['session_id']=session_id
                    started=time.perf_counter()
                    result=(await client.post(ASR+endpoint,params=query,content=data)).raise_for_status().json()
                    trace.append(dict(endpoint=endpoint,session_id=session_id,context=query.get('context',''),
                        data_sha256=hashlib.sha256(data).hexdigest(),result=deepcopy(result)))
                    if endpoint=='/stream/start':roles[result['session_id']]=bool(query.get('context'))
                    prompted=roles.get(session_id,bool(query.get('context')))
                    times['prompted' if prompted else 'initial']+=time.perf_counter()-started
                    if endpoint=='/stream/finish' and not prompted:initial=deepcopy(result)
                    if endpoint in ('/stream/finish','/stream/cancel'):roles.pop(session_id,None)
                    return result
                guard=EntityAudioRepair(request,definitions);started=time.perf_counter()
                try:
                    ready=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0))
                    assert not ready.get('context_applied')
                    for offset in range(0,len(pcm),64000):await guard.request('/stream/chunk',ready['session_id'],pcm[offset:offset+64000])
                    final=await guard.request('/stream/finish',ready['session_id'])
                finally:
                    guard.close()
                    for sid in list(roles):await request('/stream/cancel',sid)
                assert initial is not None
                append(OUT/'audio.jsonl',dict(id=sample['id'],initial=initial,production_final=final,trace=trace,
                    actual_asr_seconds=dict(times),elapsed_seconds=time.perf_counter()-started))
                print('ASR',index+1,'/',len(m['samples']),sample['id'],final['entity_audio_repair']['change_count'],flush=True)
        await asyncio.gather(*(process(i,s) for i,s in enumerate(m['samples'])))
    verify()


def pipeline(device):
    snapshot()
    from fastapi.testclient import TestClient
    from benchmark_multimed_entities import Runner
    m=verify();actual={r['id']:r for r in rows(OUT/'audio.jsonl')}
    assert set(actual)=={s['id'] for s in m['samples']}
    runner=Runner(SimpleNamespace(output=OUT,library=DATA/'library',model=MODEL,device=device,
        asr_url=ASR,mode='streaming',chunk_seconds=2.0))
    active={}
    async def request(url,endpoint,session_id=None,data=b'',params=None,**kwargs):
        trace=active['record']['trace'];cursor=active['cursor']
        if endpoint=='/stream/cancel' and (cursor==len(trace) or trace[cursor]['endpoint']!=endpoint):return dict(cancelled=True)
        if cursor==len(trace):raise ValueError('Unexpected ASR request')
        expected=trace[cursor]
        assert endpoint==expected['endpoint'] and session_id==expected['session_id']
        assert (params or {}).get('context','')==expected['context']
        assert hashlib.sha256(data).hexdigest()==expected['data_sha256']
        active['cursor']+=1
        return deepcopy(expected['result'])
    runner.web._stream_request_async=request
    done={(r['id'],r['condition']) for r in rows(OUT/'web.jsonl')}
    with ExitStack() as stack:
        clients={}
        for arm in ARMS:
            db=None if arm=='baseline' else DATA/'library/entities.db'
            app=runner.web.create_app(MODEL,device,ASR,'Chinese',256,OUT/'service_logs'/(arm+'.jsonl'),
                entity_db=db,entity_fuzzy_mode='off' if db is None else 'auto',refinement_gate_mode='tri_state',
                final_refinement_mode='off',zh_itn_library=ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so',
                entity_asr_context=False,entity_audio_repair=arm=='entity_audio_repair')
            clients[arm]=stack.enter_context(TestClient(app))
        for index,sample in enumerate(m['samples']):
            active.update(record=actual[sample['id']]);pcm=Path(sample['pcm_path']).read_bytes()
            assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
            for arm in (ARMS if index%2==0 else tuple(reversed(ARMS))):
                if (sample['id'],arm) in done:continue
                active['cursor']=0;runner.calls.clear();started=time.perf_counter()
                query=urlencode(dict(mode='streaming',domain='general',language='Chinese',asr_chunk_seconds=2.0,
                    entity_asr_context=0,entity_audio_repair=int(arm=='entity_audio_repair')))
                with clients[arm].websocket_connect('/ws/stream?'+query) as socket:
                    ready=socket.receive_json();assert ready['event']=='ready'
                    for offset in range(0,len(pcm),64000):
                        socket.send_bytes(pcm[offset:offset+64000])
                        while True:
                            event=socket.receive_json()
                            if event.get('event')=='error':raise RuntimeError(event)
                            if event.get('event')=='chunk_ack':break
                    socket.send_json(dict(event='finish'))
                    while True:
                        event=socket.receive_json()
                        if event.get('event')=='error':raise RuntimeError(event)
                        if event.get('event')=='final':break
                expected=actual[sample['id']]['production_final'] if arm=='entity_audio_repair' else actual[sample['id']]['initial']
                assert event.get('raw_text','').strip()==expected.get('text','').strip()
                if arm=='entity_audio_repair':assert active['cursor']==len(active['record']['trace'])
                append(OUT/'web.jsonl',dict(id=sample['id'],condition=arm,result=event,model_calls=list(runner.calls),
                    elapsed_seconds=time.perf_counter()-started))
                print('WEB',sample['id'],arm,event.get('clean_text',''),flush=True)
    verify()


def report():
    snapshot()
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.entity_audio_repair import normalized
    m=verify();samples={s['id']:s for s in m['samples']}
    references={r['sample_id']:r['reference'] for r in json.loads((DATA/'references.json').read_text())}
    gold={r['sample_id']:r['gold_entities'] for r in json.loads((DATA/'sample_entities.json').read_text())}
    data={(r['id'],r['condition']):r for r in rows(OUT/'web.jsonl')}
    assert len(data)==len(samples)*3
    converter,itn=opencc.OpenCC('t2s'),ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(text):return normalized(converter.convert(itn.normalize(text).text))
    summary={arm:Counter() for arm in ARMS};details=[]
    for rid,s in samples.items():
        ref=norm(references[rid]);expected=Counter(e['canonical'] for e in gold[rid]);values={}
        for arm in ARMS:
            row=data[rid,arm];result=row['result'];text=result.get('clean_text','')
            if 'clean_text' not in result and result.get('raw_text','').strip():raise ValueError('Missing nonempty final transcript')
            compact=norm(text);hits=sum(min(n,compact.count(norm(word))) for word,n in expected.items())
            errors=Levenshtein.distance(ref,compact);summary[arm].update(samples=1,hits=hits,expected_mentions=sum(expected.values()),
                errors=errors,reference_chars=len(ref),refiner_calls=len(row['model_calls']),
                text_refinement_calls=len(row['model_calls']),
                session_refiner_calls=result['session_refiner_call_count'])
            values[arm]=dict(text=text,hits=hits,errors=errors)
        summary['entity_audio_repair'].update(improved=int(values['entity_audio_repair']['errors']<values['entity_rules']['errors']),
            regressed=int(values['entity_audio_repair']['errors']>values['entity_rules']['errors']))
        details.append(dict(id=rid,topics=s['topics'],reference=references[rid],expected=dict(expected),values=values))
    for v in summary.values():v['CER']=v['errors']/v['reference_chars']
    per_topic={}
    for d in details:
        expected_mentions=sum(d['expected'].values());reference_chars=len(norm(d['reference']))
        for topic in d['topics']:
            group=per_topic.setdefault(topic,{arm:Counter() for arm in ARMS})
            for arm in ARMS:
                value=d['values'][arm]
                group[arm].update(samples=1,hits=value['hits'],expected_mentions=expected_mentions,
                                  errors=value['errors'],reference_chars=reference_chars)
    for group in per_topic.values():
        for value in group.values():
            value['CER']=value['errors']/value['reference_chars']
            value['entity_recall']=value['hits']/value['expected_mentions'] if value['expected_mentions'] else None
    write(OUT/'topic_summary.json',per_topic)
    audio_records=rows(OUT/'audio.jsonl')
    write(OUT/'asr_performance.json',dict(actual_initial_streams=len(audio_records),
        actual_prompted_streams=sum(r['production_final']['entity_audio_repair']['checks'] for r in audio_records),
        actual_request_seconds={key:sum(r['actual_asr_seconds'].get(key,0) for r in audio_records) for key in ['initial','prompted']},
        applied_samples=sum(bool(r['production_final']['entity_audio_repair']['changes']) for r in audio_records),
        applied_changes=sum(len(r['production_final']['entity_audio_repair']['changes']) for r in audio_records)))
    write(OUT/'summary.json',summary);write(OUT/'comparisons.json',details)
    write(OUT/'verification.json',dict(actual_audio_samples=len(samples),web_sessions=len(data),
        real_refiner_calls=sum(v['session_refiner_calls'] for v in summary.values()),
        audited_text_refinement_calls=sum(v['text_refinement_calls'] for v in summary.values()),
        refiner_call_accounting='Session counters include repetition review; model_calls traces capture refine* methods only.',
        formal_database_unchanged=digest(Path(m['formal_database']))==m['formal_database_sha256'],
        exact_original_pcm_and_context_trace_verified=True,full_dataset_evaluated=m['scope']=='full'))
    lines=['# 千实体库完整精修流程检查','',f"本轮为 {m['scope']}，{len(samples)} 条音频，加载全库 1,346 个实体。初稿和原音复核实际调用 ASR；三组共享响应，经过实际 Web/精修门控，Refiner 按门控实际推理。",'',
        '|条件|实体字面命中|CER|文本精修调用（独立捕获）|Refiner累计调用（含重复审核）|','|---|---:|---:|---:|---:|']
    for arm,v in summary.items():lines.append(f"|{arm}|{v['hits']}/{v['expected_mentions']}|{v['CER']:.2%}|{v['text_refinement_calls']}|{v['session_refiner_calls']}|")
    lines+=['','文本精修日志只捕获 refine* 方法，共576次；会话累计统计还包含重复审核，共1,035次逻辑调用，不是GPU forward次数。两者口径不同。每组数量见上表；末次事件的重复审核计数不是全会话累计，不能用于简单相减对齐。'] if m['scope']=='full' else ['','文本精修日志仅捕获 refine* 方法；会话累计还包含重复审核，二者口径不同。']
    lines+=['',f"原音方案相对仅实体规则：改善 {summary['entity_audio_repair']['improved']} 条，退化 {summary['entity_audio_repair']['regressed']} 条。",'',
        '字面命中基于来源标注的目标名，不代表完整实体语义识别正确率。全量固定集含2,110条正例、300条负例；未按结果删除或改选音频。' if m['scope']=='full' else '字面命中基于来源标注的目标名，不代表完整实体语义识别正确率。烟雾检查不作为全量2,410条结果，也不用于删除或改选测试音频。',
        '', '此为已知词汇库评估：标准名来自来源标注及来源参考匹配，参考只在汇总阶段打开。新闻话题占比很高，各话题可能交叉；应同时查看分话题结果。音频已逐条解码校验，未声称逐条人工听审。正例原计划2,500条，最终合格唯一正例2,110条，缺口390条未用重复数据填充。']
    lines+=['','|话题|样本|无实体库命中|仅实体规则命中|加原音复核命中|原音方案 CER|',
            '|---|---:|---:|---:|---:|---:|']
    for topic,group in sorted(per_topic.items()):
        a,b,c=(group[arm] for arm in ARMS)
        lines.append(f"|{topic}|{a['samples']}|{a['hits']}/{a['expected_mentions']}|{b['hits']}/{b['expected_mentions']}|{c['hits']}/{c['expected_mentions']}|{c['CER']:.2%}|")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','pipeline','report']);p.add_argument('--all',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.all:OUT=BASE/'full_evaluation'
    if a.stage=='prepare':prepare(a.all)
    elif a.stage=='collect':
        if not 1<=a.workers<=4:p.error('--workers must be between 1 and 4')
        asyncio.run(collect(a.workers))
    elif a.stage=='pipeline':pipeline(a.device)
    else:report()
