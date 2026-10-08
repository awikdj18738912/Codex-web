"""Reproducible paired evaluation of the current entity pipeline on MultiMed.

Stages: prepare audio -> real no-library baseline -> reference-only glossary
-> identical-ASR Web/Refiner replay with glossary -> report. Test references
supply vocabulary by explicit user request; this is a known-vocabulary test.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlencode

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,normalized
from evaluate_medical_entities import lexicon_occurrences

DATASET=Path('/home/aim0/data/datasets/ASR/MultiMed/Chinese/test-00000-of-00001.parquet')
DEFAULT_OUT=ROOT/'results/multimed_chinese_test_eval_20261008'
DEFAULT_LIBRARY=ROOT/'data/entity_benchmarks/multimed_chinese_test'
MODEL=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
# Independently identified complete medical phrases, never ASR-error aliases.
REVIEWED=('白内障手术','前列腺癌','骨质疏松症','颈椎手术','韧带重建手术',
    '肿瘤切除手术','大肠癌','大肠癌症筛检','结肠癌','全身麻醉','局部麻醉',
    '人造肛门','巡回护士','器械护士','防辐射服','无菌手套','手术衣','无影灯',
    'MRSA感染','胰脏癌','胰脏发炎','高压氧治疗','心肺复苏','人工呼吸')
GENERIC={'手术','检查','治疗','患者','病人','医生','医院','疾病','症状','细胞','器官','组织',
    '疼痛','切除','变性','增生','意外损伤','切除肿瘤组织','肠切除','改善症状','缓解症状','VIP','有阴影'}
CONTEXT_WORDS=('患者','病人','医生','医院','手术','治疗','检查','诊断','症状','疾病',
    '接受','进行','患有','患上','需要','开刀')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_configuration(args):
    path=args.output/'manifest.json'
    if not path.exists():return
    config=json.loads(path.read_text())['configuration']
    for key,value in [('mode',args.mode),('asr_chunk_seconds',args.chunk_seconds),('refiner_model',str(args.model)),('asr_url',args.asr_url)]:
        if config[key]!=value:raise ValueError(f'{key} differs from frozen run; use a new output folder')


def require_complete(path,ids):
    rows=read_jsonl(path);by_id={r['id']:r for r in rows}
    if len(rows)!=len(by_id):raise ValueError(f'Duplicate records in {path}')
    missing=set(ids)-set(by_id)
    if missing:raise ValueError(f'{path}: {len(missing)} unfinished samples')
    return by_id


def transcript(record,key):
    result=record['result']
    if key in result:return result[key]
    if record.get('final_payload',{}).get('text','')=='':return ''
    raise ValueError(f"{record['id']}: missing {key} for nonempty ASR")


def prepare(args):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    if (args.output/'manifest.json').exists():
        if json.loads((args.output/'manifest.json').read_text())['dataset_sha256']!=digest(args.dataset):
            raise ValueError('Dataset changed; use a new output folder')
        return
    args.output.mkdir(parents=True,exist_ok=True)
    rows=pq.read_table(args.dataset).to_pylist();samples=[]
    for index,row in enumerate(rows):
        rid=f'MultiMed_Chinese_test_{index:04d}'
        audio_bytes=row['audio']['bytes']
        if not audio_bytes:raise ValueError(f'{rid}: missing embedded audio')
        wave,rate=sf.read(io.BytesIO(audio_bytes),dtype='float32')
        if wave.ndim>1:wave=wave.mean(axis=1)
        if rate!=16000:
            factor=gcd(rate,16000);wave=resample_poly(wave,16000//factor,rate//factor).astype('float32')
        if len(wave)==0 or not np.isfinite(wave).all():raise ValueError(f'{rid}: invalid audio')
        path=args.output/'audio'/(rid+'.wav');path.parent.mkdir(exist_ok=True)
        sf.write(path,wave,16000,subtype='FLOAT')
        samples.append(dict(id=rid,row=index,reference=row['text'],audio_path=str(path.resolve()),
            duration_seconds=len(wave)/16000,annotated_duration_seconds=row['duration'],
            source_audio_path=row['audio']['path'],original_audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            pcm_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()))
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'system').rglob('*')
        if p.is_file() and p.suffix in {'.py','.json','.html','.md'}}
    write_json(args.output/'manifest.json',dict(dataset=str(args.dataset),dataset_sha256=digest(args.dataset),
        sample_count=len(samples),duration_seconds=sum(r['duration_seconds'] for r in samples),samples=samples,
        configuration=dict(asr_url=args.asr_url,language='Chinese',mode=args.mode,asr_chunk_seconds=args.chunk_seconds,
            transport_chunk_samples=16000,refiner_model=str(args.model),max_new_tokens=256,
            gate='tri_state',final_refinement='off',numeric_backend='zh-itn-v90',entity_asr_context=False),
        source_hashes=source_hashes,comparison='Reference-derived known vocabulary; same ASR responses in both arms',
        stages=['baseline_without_entity_db','build_reference_entities','entity_library_replay'],
        caveats=['Supplied captions may omit speech; CER is caption agreement, not human-verified ASR accuracy',
            'This corpus has appeared in earlier project experiments; not untouched independent generalization',
            'Glossary occurrences are longest non-overlapping strings, not manually annotated complete NER']))
    print('PREPARED',len(samples),'audio records',flush=True)


def extract_entities(reference,labels):
    candidates=[]
    for name,types in labels.items():
        if name in GENERIC or not 3<=len(name)<=48:continue
        if name not in reference:continue
        for match in re.finditer(re.escape(name),reference):
            if name.isascii() and ((match.start() and reference[match.start()-1].isascii() and reference[match.start()-1].isalnum())
                or (match.end()<len(reference) and reference[match.end()].isascii() and reference[match.end()].isalnum())):continue
            candidates.append(dict(text=name,start=match.start(),end=match.end(),annotation_types=sorted(types)))
    for name in REVIEWED:
        for match in re.finditer(re.escape(name),reference):
            candidates.append(dict(text=name,start=match.start(),end=match.end(),annotation_types=['reviewed_complete_phrase']))
    chosen=[]
    for entry in sorted(candidates,key=lambda r:(-(r['end']-r['start']),r['start'],r['text'])):
        if not any(entry['start']<q['end'] and q['start']<entry['end'] for q in chosen):chosen.append(entry)
    return sorted(chosen,key=lambda r:r['start'])


def build_entities(args):
    from system.entity_store import EntityStore
    manifest=json.loads((args.output/'manifest.json').read_text())
    require_complete(args.output/'baseline.jsonl',[r['id'] for r in manifest['samples']])
    args.library.mkdir(parents=True,exist_ok=True)
    if (args.library/'provenance.json').exists():
        p=json.loads((args.library/'provenance.json').read_text())
        if p['dataset_sha256']!=manifest['dataset_sha256']:raise ValueError('Library belongs to another dataset')
        return
    labels=defaultdict(set)
    for row in json.loads(args.medical_lexicon.read_text()):
        for entry in row['entities']:
            name=entry['entity']
            if re.fullmatch(r'[\u3400-\u9fffA-Za-z0-9-]+',name):labels[name].add(entry['type'])
    records={};sample_entities={}
    for row in manifest['samples']:
        entities=extract_entities(row['reference'],labels);sample_entities[row['id']]=entities
        for entry in entities:
            record=records.setdefault(entry['text'],dict(canonical_text=entry['text'],annotation_types=set(),evidence=[]))
            record['annotation_types'].update(entry['annotation_types'])
            record['evidence'].append(dict(id=row['id'],reference=row['reference'],start=entry['start'],end=entry['end']))
    definitions=[]
    for name,record in sorted(records.items()):
        keywords={w:1.0 for w in CONTEXT_WORDS if w not in name and name not in w}
        context=None
        if re.fullmatch(r'[\u3400-\u9fff]{3,16}',name):
            context=dict(keywords=keywords,minimum_score=1.0,source='Fixed medical scene keywords; no per-sample reference at inference',
                allow_internal_omission=False,allow_near_spelling=False)
        definitions.append(dict(canonical_text=name,entity_type='TERM',domain='medical',normalization_policy='normalize',
            enabled=True,priority=0,aliases=[],source='MultiMed Chinese test reference plus independent CMeEE medical spelling annotation',spelling_context=context))
    vocabulary=[dict(canonical_text=n,annotation_types=sorted(r['annotation_types']),evidence=r['evidence']) for n,r in sorted(records.items())]
    db=args.library/'entities.db'
    if db.exists():raise FileExistsError('Refuse to overwrite an unfinished existing library')
    store=EntityStore(db)
    for item in definitions:
        d=store.upsert_entity(item['canonical_text'],entity_type='TERM',domain='medical',normalization_policy='normalize',source=item['source'])
        if item['spelling_context']:
            p=item['spelling_context'];store.set_spelling_context(d.entity_id,keywords=p['keywords'],minimum_score=p['minimum_score'],source=p['source'])
    write_json(args.library/'entities.json',definitions);write_json(args.library/'vocabulary_evidence.json',vocabulary)
    write_json(args.library/'sample_entities.json',sample_entities)
    write_json(args.library/'provenance.json',dict(dataset_sha256=manifest['dataset_sha256'],lexicon_sha256=digest(args.medical_lexicon),
        vocabulary_size=len(definitions),reference_mentions=sum(len(v) for v in sample_entities.values()),
        extraction='Independent CMeEE medical names matched in references, with reviewed whole phrases; longest spans',
        reference_at_inference=False,error_aliases=False,asr_context=False,near_permission=False,
        frozen_hashes={p.name:digest(p) for p in [db,args.library/'entities.json',args.library/'sample_entities.json',args.library/'vocabulary_evidence.json']}))
    print('BUILT',len(definitions),'standard terms',flush=True)


class Runner:
    def __init__(self,args):
        import system.web_app as web
        from system.live_qwen_refiner import TransformersRefiner
        from system.repetition_plausibility import RepetitionPlausibility
        from system.zh_itn_bridge import DEFAULT_LIBRARY
        self.args=args;self.web=web;self.calls=[];self.active=None;self.transport=[];self.final_payload=None
        (args.output/'service_logs').mkdir(parents=True,exist_ok=True)
        self.original_http=web._stream_request_async
        self.model=TransformersRefiner(args.model,args.device,256)
        owner=self
        class Audited:
            def __getattr__(self,name):
                value=getattr(owner.model,name)
                if not callable(value) or not name.startswith('refine'):return value
                def invoke(*a,**kw):
                    start=time.perf_counter();result=value(*a,**kw)
                    owner.calls.append(dict(method=name,input=list(a),kwargs=kw,output=result,wall_seconds=time.perf_counter()-start))
                    return result
                return invoke
        model=Audited();web.TransformersRefiner=lambda *a,**k:model
        plausibility=RepetitionPlausibility(args.device);web.RepetitionPlausibility=lambda *a,**k:plausibility
        async def request(url,endpoint,session_id=None,data=b'',params=None,**kw):
            if owner.active is None:
                result=await owner.original_http(url,endpoint,session_id=session_id,data=data,params=params,**kw)
                if endpoint=='/stream/chunk':owner.transport.append(result)
                elif endpoint=='/stream/finish':owner.final_payload=result
                return result
            if endpoint=='/stream/start':owner.replay_index=0;return dict(session_id='paired-real-asr-replay',chunk_seconds=args.chunk_seconds,context_applied=False)
            if endpoint=='/stream/chunk':
                result=owner.active['chunk_payloads'][owner.replay_index];owner.replay_index+=1;return result
            if endpoint=='/stream/finish':return owner.active['final_payload']
            if endpoint=='/stream/cancel':return dict(cancelled=True)
            raise ValueError(endpoint)
        web._stream_request_async=request
        self.app_factory=lambda db,condition:web.create_app(args.model,args.device,args.asr_url,'Chinese',256,
            args.output/'service_logs'/f'{condition}.jsonl',entity_db=db,entity_fuzzy_mode='auto' if db else 'off',
            refinement_gate_mode='tri_state',final_refinement_mode='off',zh_itn_library=DEFAULT_LIBRARY,entity_asr_context=False)

    def run(self,condition):
        import soundfile as sf
        from fastapi.testclient import TestClient
        args=self.args;manifest=json.loads((args.output/'manifest.json').read_text());samples=manifest['samples']
        db=None if condition in {'baseline','control_replay'} else args.library/'entities.db'
        baseline=require_complete(args.output/'baseline.jsonl',[r['id'] for r in samples]) if condition!='baseline' else {}
        if db:
            p=json.loads((args.library/'provenance.json').read_text())
            for name,value in p['frozen_hashes'].items():
                if digest(args.library/name)!=value:raise ValueError(f'Frozen library changed: {name}')
        dest=args.output/f'{condition}.jsonl';done={r['id'] for r in read_jsonl(dest)}
        app=self.app_factory(db,condition);query=urlencode(dict(mode=args.mode,domain='medical',language='Chinese',asr_chunk_seconds=args.chunk_seconds,entity_asr_context='0'))
        with TestClient(app) as client:
            if db:
                loaded=client.get('/api/entities').raise_for_status().json()['entities']
                if len(loaded)!=p['vocabulary_size']:raise ValueError('Loaded vocabulary mismatch')
            for index,row in enumerate(samples):
                if row['id'] in done:continue
                self.active=None if condition=='baseline' else baseline[row['id']]
                self.transport=[];self.final_payload=None;self.calls.clear();events=[]
                wave,rate=sf.read(row['audio_path'],dtype='float32');assert rate==16000
                pcm=wave.astype('<f4').tobytes()
                if hashlib.sha256(pcm).hexdigest()!=row['pcm_sha256']:raise ValueError('PCM changed')
                start=time.perf_counter()
                with client.websocket_connect('/ws/stream?'+query) as socket:
                    ready=socket.receive_json()
                    if ready.get('event')!='ready' or ready.get('entity_asr_context_applied'):raise ValueError(ready)
                    for offset in range(0,len(pcm),64000):
                        socket.send_bytes(pcm[offset:offset+64000] if self.active is None else b'\0\0\0\0')
                        while True:
                            e=socket.receive_json();events.append(e)
                            if e.get('event')=='error':raise RuntimeError(e)
                            if e.get('event')=='chunk_ack':break
                    socket.send_json(dict(event='finish'))
                    while True:
                        e=socket.receive_json();events.append(e)
                        if e.get('event')=='error':raise RuntimeError(e)
                        if e.get('event')=='final':break
                if self.active is not None and e.get('raw_text','')!=transcript(self.active,'raw_text'):raise ValueError('Paired ASR text differs')
                append_jsonl(dest,dict(id=row['id'],row=row['row'],condition=condition,reference=row['reference'],
                    pcm_sha256=row['pcm_sha256'],ready=ready,result=e,model_calls=list(self.calls),
                    chunk_payloads=self.transport if self.active is None else self.active['chunk_payloads'],
                    final_payload=self.final_payload if self.active is None else self.active['final_payload'],
                    elapsed_seconds=time.perf_counter()-start,event_counts=dict(Counter(v.get('event') for v in events))))
                print(condition,index+1,'/',len(samples),row['id'],e.get('clean_text','<empty ASR>'),flush=True)


def report(args):
    from system.zh_itn_bridge import ZhITNNormalizer
    manifest=json.loads((args.output/'manifest.json').read_text());ids=[r['id'] for r in manifest['samples']]
    live_baseline=require_complete(args.output/'baseline.jsonl',ids)
    baseline=require_complete(args.output/'control_replay.jsonl',ids) if (args.output/'control_replay.jsonl').exists() else live_baseline
    paired=require_complete(args.output/'entity_library.jsonl',ids)
    entities=json.loads((args.library/'sample_entities.json').read_text());names=[r['canonical_text'] for r in json.loads((args.library/'entities.json').read_text())]
    itn=ZhITNNormalizer();summaries={k:Counter() for k in ['raw_asr','live_baseline','baseline','entity_library']};details=[];changes=[];control_differences=[]
    numeric_cache={}
    def norm(text):
        if text not in numeric_cache:numeric_cache[text]=normalized(itn.normalize(text).text)
        return numeric_cache[text]
    for row in manifest['samples']:
        rid=row['id'];a=baseline[rid];b=paired[rid]
        if a['pcm_sha256']!=b['pcm_sha256'] or transcript(a,'raw_text')!=transcript(b,'raw_text'):raise ValueError('Not a causal pair')
        ref=norm(row['reference']);expected=Counter(e['text'] for e in entities[rid]);results={}
        live=live_baseline[rid]
        if transcript(live,'clean_text')!=transcript(a,'clean_text'):
            control_differences.append(dict(id=rid,live=transcript(live,'clean_text'),replay=transcript(a,'clean_text')))
        for label,text in [('raw_asr',transcript(a,'raw_text')),('live_baseline',transcript(live,'clean_text')),('baseline',transcript(a,'clean_text')),('entity_library',transcript(b,'clean_text'))]:
            value=norm(text);errors=Levenshtein.distance(ref,value)
            have=Counter(e['text'] for e in lexicon_occurrences(text,names))
            recovered=sum(min(n,have[w]) for w,n in expected.items());spurious=sum(max(0,n-expected[w]) for w,n in have.items())
            summaries[label].update(samples=1,reference_chars=len(ref),character_errors=errors,expected_mentions=sum(expected.values()),recovered_mentions=recovered,unreferenced_mentions=spurious)
            results[label]=dict(text=text,character_errors=errors,recovered_mentions=recovered,unreferenced_mentions=spurious)
        ea,eb=results['baseline']['character_errors'],results['entity_library']['character_errors']
        s=summaries['entity_library'];s.update(changed_samples=int(transcript(a,'clean_text')!=transcript(b,'clean_text')),improved_samples=int(eb<ea),regressed_samples=int(eb>ea),
            entity_gains=max(0,results['entity_library']['recovered_mentions']-results['baseline']['recovered_mentions']),
            entity_losses=max(0,results['baseline']['recovered_mentions']-results['entity_library']['recovered_mentions']))
        for label,record in [('live_baseline',live),('baseline',a),('entity_library',b)]:
            summaries[label]['refiner_calls']+=len(record['model_calls'])
            summaries[label]['entity_normalizations']+=len(record['result'].get('entity_normalizations',[]))
        detail=dict(id=rid,row=row['row'],reference=row['reference'],entities=entities[rid],results=results,
            error_delta=eb-ea,normalizations=b['result'].get('entity_normalizations',[]),candidates=b['result'].get('entity_candidates',[]),
            hints=b['result'].get('entity_refinement_hints',[]),reject_reasons=b['result'].get('refiner_reject_reasons',[]))
        details.append(detail)
        if transcript(a,'clean_text')!=transcript(b,'clean_text'):changes.append(detail)
    summary={}
    for k,s in summaries.items():
        summary[k]=dict(s,CER=s['character_errors']/s['reference_chars'] if s['reference_chars'] else 0,
            entity_retention=s['recovered_mentions']/s['expected_mentions'] if s['expected_mentions'] else None)
    write_json(args.output/'summary.json',summary);write_json(args.output/'changes.json',changes)
    write_json(args.output/'control_replay_differences.json',control_differences)
    missing_entities=[]
    for d in details:
        expected=Counter(e['text'] for e in d['entities'])
        have=Counter(e['text'] for e in lexicon_occurrences(d['results']['entity_library']['text'],names))
        missing=expected-have
        if missing:
            missing_entities.append(dict(id=d['id'],missing=dict(missing),reference=d['reference'],
                raw=d['results']['raw_asr']['text'],final=d['results']['entity_library']['text'],
                candidates=d['candidates'],hints=d['hints']))
    write_json(args.output/'missing_entities.json',missing_entities)
    (args.output/'comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
    with (args.output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file);writer.writerow(['id','reference','baseline','entity_library','error_delta','entity_gains','normalizations'])
        for d in details:writer.writerow([d['id'],d['reference'],d['results']['baseline']['text'],d['results']['entity_library']['text'],d['error_delta'],d['results']['entity_library']['recovered_mentions']-d['results']['baseline']['recovered_mentions'],json.dumps(d['normalizations'],ensure_ascii=False)])
    lines=['# MultiMed 中文测试集：当前实体库对照评估','',
        f"数据：{manifest['sample_count']} 条原音，{manifest['duration_seconds']/60:.2f} 分钟。实体库来自这些参考文本，属于用户指定的已知词表覆盖测试。",'',
        '基线为真实 PCM→当前 ASR→真实精修／数字流程，实体库关闭。第二遍复用逐块及最终真实 ASR 响应，使用相同模型与设置，仅加载预建实体库，词表辅助识别仍关闭。', '',
        '| 条件 | 参考文字 CER | 名称保留／参考出现 | 精修调用 | 实体规范次数 |','|---|---:|---:|---:|---:|']
    for label,k in [('原始 ASR','raw_asr'),('无库：实际首次转录','live_baseline'),('无库：相同 ASR 质量回放','baseline'),('预建实体库：相同 ASR 质量回放','entity_library')]:
        s=summary[k];lines.append(f"| {label} | {s['CER']:.3%} | {s['recovered_mentions']}/{s['expected_mentions']} | {s.get('refiner_calls','—')} | {s.get('entity_normalizations','—')} |")
    s=summary['entity_library'];lines+=['',f"相对无库回放：改动 {s['changed_samples']} 条，文字改善 {s['improved_samples']} 条、退化 {s['regressed_samples']} 条；名称新增保留 {s['entity_gains']} 次、丢失 {s['entity_losses']} 次。{len(ids)} 对输入保持相同 ASR 响应。实际首次转录与无库回放有 {len(control_differences)} 条文字差异，单列于 control_replay_differences.json，不归为实体收益。",'',
        'CER 对参考与输出应用相同现有数字规范、NFKC及标点／空白处理。字幕可能省略口头语、英语或背景语音；该指标是字幕文字一致度，不是经逐字听辨核准的转录准确率。实体计数是提取词表内最长非重叠名称出现，参考本身没有完整人工 NER 标签。', '',
        '词库仅从参考及独立 CMeEE 标准词提取，附完整短语审核及参考位置。没有错误别名、逐句参考提示、近音／漏字权限自动开启。所有中文条目附相同固定医疗语境；名单中可能含器官、设备、操作等通用术语。', '',
        f"词库：[{args.library/'entities.json'}]({args.library/'entities.json'})，数据库：[{args.library/'entities.db'}]({args.library/'entities.db'})。",'',
        '## 逐条有变化的结果','', '| 样本 | 参考 | 无库 | 有库 | 文字错误增减 |','|---|---|---|---|---:|']
    esc=lambda t:t.replace('|','\\|').replace('\n',' ')
    for d in changes:lines.append(f"| {d['id']} | {esc(d['reference'])} | {esc(d['results']['baseline']['text'])} | {esc(d['results']['entity_library']['text'])} | {d['error_delta']:+d} |")
    if not changes:lines.append('| — | 全部样本无最终文字变化 | — | — | 0 |')
    lines+=['','详细证据：`comparison.csv`、`comparison.jsonl`、`changes.json`，以及两组包含原始响应、候选、替换记录和真实模型调用的 `*.jsonl`。第二遍耗时仅为固定识别结果的质量回放，不作为完整转录速度指标。']
    lines+=['','## 未恢复的名称','',
        '| 样本 | 未恢复名称及次数 | 参考 | 有库输出 |','|---|---|---|---|']
    for d in missing_entities:
        missing='、'.join(f'{name} × {count}' for name,count in d['missing'].items())
        lines.append(f"| {d['id']} | {esc(missing)} | {esc(d['reference'])} | {esc(d['final'])} |")
    if not missing_entities:lines.append('| — | 全部参考名称均保留 | — | — |')
    lines+=['','召回候选、纠正提示及完整文字见 `missing_entities.json`。']
    if s['entity_gains']==0:
        lines+=['','本轮没有新增实体纠正；其他文字变化不能算实体纠错收益。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','baseline','build-entities','control','paired','report','run'])
    parser.add_argument('--dataset',type=Path,default=DATASET);parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--library',type=Path,default=DEFAULT_LIBRARY)
    parser.add_argument('--medical-lexicon',type=Path,default=ROOT/'results/medical_entity_20261003/source/CMeEE_train.json')
    parser.add_argument('--model',type=Path,default=MODEL);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--asr-url',default='http://127.0.0.1:8766');parser.add_argument('--mode',choices=['streaming','offline'],default='streaming')
    parser.add_argument('--chunk-seconds',type=float,default=2.0)
    args=parser.parse_args();args.output=args.output.resolve();args.library=args.library.resolve()
    validate_configuration(args)
    if args.stage in {'prepare','run'}:prepare(args)
    if args.stage=='run':
        runner=Runner(args);runner.run('baseline');build_entities(args);runner.run('control_replay');runner.run('entity_library');report(args)
    elif args.stage=='baseline':Runner(args).run('baseline')
    elif args.stage=='build-entities':build_entities(args)
    elif args.stage=='control':Runner(args).run('control_replay')
    elif args.stage=='paired':Runner(args).run('entity_library')
    elif args.stage=='report':report(args)


if __name__=='__main__':main()
