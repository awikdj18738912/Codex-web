"""Fresh real-ASR factorial evaluation: glossary context x entity processing.

Four cells share two real ASR recordings per audio. Refiner inference is real
in every cell; an experiment-only frozen context builder isolates prompting
from dictionary protection when testing the context-only cell.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from urllib.parse import urlencode

from rapidfuzz.distance import Levenshtein
from benchmark_multimed_entities import Runner,MODEL,digest,require_complete,transcript
from evaluate_entity_library import ROOT,read_jsonl,append_jsonl,write_json,normalized
from evaluate_medical_entities import lexicon_occurrences
from system.entity_store import EntityStore
from system.entity_asr_context import build_entity_asr_context,EntityASRContext,context_digest

OUT=ROOT/'results/entity_methods_comparison_20261009'
ASR='http://127.0.0.1:8766'
ARMS=('baseline','post_only','context_only','combined')
LABELS=dict(baseline='无库',post_only='仅实体后处理',context_only='仅词库辅助识别',combined='辅助识别＋实体后处理')
SOURCES={
    'medical':(ROOT/'results/multimed_chinese_test_eval_20261008',ROOT/'data/entity_benchmarks/multimed_chinese_test'),
    'podcast':(ROOT/'results/oncology_podcast_20261008',ROOT/'data/entity_benchmarks/oncology_repaired_20261009')}
PODCAST_ROWS={0,13,15,19,21,22,23,26,27,28,35}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for dataset,(source,library) in SOURCES.items():
        folder=OUT/dataset;folder.mkdir(exist_ok=True);(folder/'service_logs').mkdir(exist_ok=True)
        path=folder/'protocol.json'
        if path.exists():continue
        manifest=json.loads((source/'manifest.json').read_text())
        samples=manifest['samples'] if dataset=='medical' else [s for s in manifest['samples'] if s['row'] in PODCAST_ROWS]
        target=folder/'library';target.mkdir(exist_ok=True)
        for name in ['entities.db','entities.json','sample_entities.json']:
            if (library/name).exists():shutil.copy2(library/name,target/name)
        definitions=EntityStore(target/'entities.db').list_entities(domain='medical')
        context=build_entity_asr_context(definitions,'medical')
        write_json(path,dict(dataset=dataset,samples=samples,sample_count=len(samples),
            duration_seconds=sum(s['duration_seconds'] for s in samples),source_manifest_sha256=digest(source/'manifest.json'),
            vocabulary_size=len(definitions),context_text=context.text,context_terms=context.terms,
            context_digest=context_digest(context.text),max_terms=16,max_chars=1024,
            mode='streaming',chunk_seconds=2.0,transport_chunk_samples=16000,
            asr_url=ASR,refiner_model=str(MODEL),entity_asr_context_frozen=True,
            library_hashes={name:digest(target/name) for name in ['entities.db','entities.json']},
            source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'system').rglob('*.py')},
            refiner='Real model in all cells, same tri_state / final off / zh-itn v90 settings',
            cohort=('All 225 paired reference samples; vocabulary previously extracted from these references by user request'
                if dataset=='medical' else '11 preselected development windows: six previously corrected windows plus five controls; no reference captions')))
        print('PREPARED',dataset,len(samples),'samples','prompt',context.terms,flush=True)


def validate(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    for name,expected in protocol['library_hashes'].items():
        if digest(folder/'library'/name)!=expected:raise ValueError(f'{folder}: library changed')
    for name,expected in protocol['source_hashes'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'{name}: code changed during experiment; use a fresh run')
    return protocol


def decode():
    import httpx
    import soundfile as sf
    with httpx.Client(timeout=180) as client:
        health=client.get(ASR+'/health').raise_for_status().json()
        if health.get('context_supported') is not True:raise ValueError('ASR context unavailable')
        write_json(OUT/'asr_health.json',health)
        for dataset in SOURCES:
            folder=OUT/dataset;protocol=validate(folder)
            completed={arm:{r['id'] for r in read_jsonl(folder/f'asr_{arm}.jsonl')} for arm in ['off','on']}
            for index,sample in enumerate(protocol['samples']):
                wave,rate=sf.read(sample['audio_path'],dtype='float32');assert rate==16000
                pcm=wave.astype('<f4').tobytes()
                if hashlib.sha256(pcm).hexdigest()!=sample['pcm_sha256']:raise ValueError('PCM changed')
                for arm in (['off','on'] if index%2==0 else ['on','off']):
                    if sample['id'] in completed[arm]:continue
                    context=protocol['context_text'] if arm=='on' else ''
                    started=time.perf_counter();params=dict(language='Chinese',chunk_seconds=2.0)
                    if context:params['context']=context
                    ready=client.post(ASR+'/stream/start',params=params).raise_for_status().json()
                    if ready.get('context_applied')!=bool(context) or ready.get('context_digest')!=context_digest(context):
                        raise ValueError('Actual ASR context acknowledgement mismatch')
                    sid=ready['session_id'];payloads=[]
                    try:
                        for offset in range(0,len(pcm),64000):
                            payloads.append(client.post(ASR+'/stream/chunk',params=dict(session_id=sid),
                                content=pcm[offset:offset+64000]).raise_for_status().json())
                        final=client.post(ASR+'/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
                    except BaseException:
                        client.post(ASR+'/stream/cancel',params=dict(session_id=sid))
                        raise
                    append_jsonl(folder/f'asr_{arm}.jsonl',dict(id=sample['id'],row=sample['row'],condition=arm,
                        reference=sample['reference'],pcm_sha256=sample['pcm_sha256'],context_text=context,
                        context_terms=protocol['context_terms'] if context else [],ready=ready,chunk_payloads=payloads,
                        final_payload=final,elapsed_seconds=time.perf_counter()-started))
                    print('ASR',dataset,index+1,'/',len(protocol['samples']),arm,sample['id'],final.get('text',''),flush=True)


class FactorialRunner(Runner):
    def run_cell(self,dataset,arm):
        from fastapi.testclient import TestClient
        from system.zh_itn_bridge import DEFAULT_LIBRARY
        folder=OUT/dataset;protocol=validate(folder);samples=protocol['samples'];ids=[s['id'] for s in samples]
        context_on=arm in {'context_only','combined'};post_on=arm in {'post_only','combined'}
        sources=require_complete(folder/f"asr_{'on' if context_on else 'off'}.jsonl",ids)
        destination=folder/f'{arm}.jsonl';done={r['id'] for r in read_jsonl(destination)}
        owner=self
        async def replay(url,endpoint,session_id=None,data=b'',params=None,**kwargs):
            if endpoint=='/stream/start':
                if (params or {}).get('context','')!=owner.active['context_text']:raise ValueError('Replay prompt mismatch')
                owner.replay_index=0;return owner.active['ready']
            if endpoint=='/stream/chunk':
                result=owner.active['chunk_payloads'][owner.replay_index];owner.replay_index+=1;return result
            if endpoint=='/stream/finish':return owner.active['final_payload']
            if endpoint=='/stream/cancel':return dict(cancelled=True)
            raise ValueError(endpoint)
        self.web._stream_request_async=replay
        original_builder=self.web.build_entity_asr_context
        # The context-only cell receives the exact frozen real prompt while
        # EntityProtector/Matcher see no database. This is experimental wiring,
        # not a fake recognizer or a fake refiner response.
        self.web.build_entity_asr_context=lambda *a,**k:EntityASRContext(protocol['context_text'],tuple(protocol['context_terms']))
        try:
            app=self.web.create_app(MODEL,'cuda:0',ASR,'Chinese',256,folder/'service_logs'/f'{arm}.jsonl',
                entity_db=folder/'library/entities.db' if post_on else None,entity_fuzzy_mode='auto' if post_on else 'off',
                refinement_gate_mode='tri_state',final_refinement_mode='off',zh_itn_library=DEFAULT_LIBRARY,
                entity_asr_context=context_on)
            query=urlencode(dict(mode='streaming',domain='medical',language='Chinese',asr_chunk_seconds=2.0,
                entity_asr_context=int(context_on)))
            with TestClient(app) as client:
                for index,sample in enumerate(samples):
                    if sample['id'] in done:continue
                    self.active=sources[sample['id']];self.calls.clear();events=[];started=time.perf_counter()
                    with client.websocket_connect('/ws/stream?'+query) as socket:
                        ready=socket.receive_json()
                        if ready.get('event')!='ready' or ready.get('entity_asr_context_applied')!=context_on:raise ValueError(ready)
                        if ready.get('entity_asr_context_term_count')!=(len(protocol['context_terms']) if context_on else 0):raise ValueError('Term count mismatch')
                        for _ in self.active['chunk_payloads']:
                            socket.send_bytes(b'\0\0\0\0')
                            while True:
                                event=socket.receive_json();events.append(event)
                                if event.get('event')=='error':raise RuntimeError(event)
                                if event.get('event')=='chunk_ack':break
                        socket.send_json(dict(event='finish'))
                        while True:
                            event=socket.receive_json();events.append(event)
                            if event.get('event')=='error':raise RuntimeError(event)
                            if event.get('event')=='final':break
                    if self.replay_index!=len(self.active['chunk_payloads']):raise ValueError('Incomplete replay')
                    if not post_on and event.get('entity_normalizations'):raise ValueError('Entity treatment leaked into context-only/control cell')
                    append_jsonl(destination,dict(id=sample['id'],row=sample['row'],condition=arm,reference=sample['reference'],
                        pcm_sha256=sample['pcm_sha256'],asr_source='on' if context_on else 'off',ready=ready,result=event,
                        model_calls=list(self.calls),final_payload=self.active['final_payload'],
                        elapsed_seconds=time.perf_counter()-started,event_counts=dict(Counter(e.get('event') for e in events))))
                    print('PIPELINE',dataset,arm,index+1,'/',len(samples),sample['id'],event.get('clean_text','<empty ASR>'),flush=True)
        finally:self.web.build_entity_asr_context=original_builder


def outside_entity_errors(reference,hypothesis,names):
    spans=lexicon_occurrences(reference,names);mask=[False]*len(reference)
    for span in spans:
        for i in range(span['start'],span['end']):mask[i]=True
    errors=0
    for op in Levenshtein.editops(reference,hypothesis):
        i=op.src_pos
        inside=(0<i<len(mask) and mask[i-1] and mask[i]) if op.tag=='insert' else (i<len(mask) and mask[i])
        errors+=not inside
    return errors,len(reference)-sum(mask)


def report():
    from system.zh_itn_bridge import ZhITNNormalizer
    itn=ZhITNNormalizer();cache={}
    def norm(text):
        if text not in cache:cache[text]=normalized(itn.normalize(text).text)
        return cache[text]
    all_summary={};lines=['# 实体后处理与词库辅助识别：四组对照','',
        '同一音频分别进行两次真实 ASR（提示关闭／开启），交替运行顺序；每份真实响应分别通过有库／无库的真实精修流程。生产功能代码和正式库未修改。', '',
        '辅助识别使用当前默认：16 个标准名、1024 字符，名单在识别前冻结。只有后处理启用时，精修才能访问完整实体库。']
    for dataset in SOURCES:
        folder=OUT/dataset;protocol=validate(folder);ids=[s['id'] for s in protocol['samples']]
        rows={arm:require_complete(folder/f'{arm}.jsonl',ids) for arm in ARMS}
        asr={arm:require_complete(folder/f'asr_{arm}.jsonl',ids) for arm in ['off','on']}
        names=[d['canonical_text'] for d in json.loads((folder/'library/entities.json').read_text())]
        gt=json.loads((folder/'library/sample_entities.json').read_text()) if dataset=='medical' else None
        totals={arm:Counter() for arm in ARMS};details=[]
        raw_totals={arm:Counter() for arm in ['off','on']}
        prompt_names=set(protocol['context_terms'])
        for sample in protocol['samples']:
            rid=sample['id'];texts={arm:transcript(rows[arm][rid],'clean_text') for arm in ARMS}
            for record in [asr['off'][rid],asr['on'][rid],*[rows[a][rid] for a in ARMS]]:
                if record['pcm_sha256']!=sample['pcm_sha256']:raise ValueError('Audio pair differs')
            for arm in ARMS:
                expected_asr=asr['on' if arm in {'context_only','combined'} else 'off'][rid]['final_payload']
                if rows[arm][rid]['final_payload']!=expected_asr:raise ValueError('ASR cell input differs')
            if transcript(rows['baseline'][rid],'raw_text')!=transcript(rows['post_only'][rid],'raw_text'):
                raise ValueError('Off-context raw outputs differ between post-processing cells')
            if transcript(rows['context_only'][rid],'raw_text')!=transcript(rows['combined'][rid],'raw_text'):
                raise ValueError('On-context raw outputs differ between post-processing cells')
            values={};reference=norm(sample['reference']) if gt is not None else ''
            expected=Counter(e['text'] for e in gt[rid]) if gt is not None else Counter()
            for arm in ['off','on']:
                text=asr[arm][rid]['final_payload'].get('text','')
                raw_totals[arm].update(samples=1)
                if gt is not None:
                    have=Counter(e['text'] for e in lexicon_occurrences(text,names))
                    raw_totals[arm].update(character_errors=Levenshtein.distance(reference,norm(text)),reference_chars=len(reference),
                        expected_mentions=sum(expected.values()),recovered_mentions=sum(min(c,have[n]) for n,c in expected.items()))
            for arm in ARMS:
                record=rows[arm][rid];text=texts[arm];have=Counter(e['text'] for e in lexicon_occurrences(text,names))
                recovered={name:min(count,have[name]) for name,count in expected.items()}
                value=dict(text=text,model_calls=len(record['model_calls']),normalizations=record['result'].get('entity_normalizations',[]),
                    glossary_mentions=dict(have),recovered_by_name=recovered)
                totals[arm].update(samples=1,refiner_calls=len(record['model_calls']),normalization_events=len(value['normalizations']),
                    changed_samples=int(text!=texts['baseline']))
                if gt is not None:
                    error=Levenshtein.distance(reference,norm(text));outside,chars=outside_entity_errors(reference,norm(text),names)
                    value.update(character_errors=error,outside_entity_errors=outside,recovered_mentions=sum(recovered.values()))
                    totals[arm].update(reference_chars=len(reference),character_errors=error,expected_mentions=sum(expected.values()),
                        recovered_mentions=sum(recovered.values()),unreferenced_mentions=sum(max(0,count-expected[name]) for name,count in have.items()),
                        outside_entity_chars=chars,outside_entity_errors=outside,
                        prompted_expected=sum(c for n,c in expected.items() if n in prompt_names),
                        prompted_recovered=sum(c for n,c in recovered.items() if n in prompt_names))
                values[arm]=value
            for arm in ARMS:
                if gt is None:continue
                gains,losses=mention_changes(values['baseline']['recovered_by_name'],values[arm]['recovered_by_name'])
                delta=values[arm]['character_errors']-values['baseline']['character_errors']
                totals[arm].update(entity_gains=gains,entity_losses=losses,improved_samples=int(delta<0),regressed_samples=int(delta>0))
            details.append(dict(id=rid,reference=sample['reference'],start_seconds=sample.get('start_seconds'),
                end_seconds=sample.get('end_seconds'),results=values,
                raw_asr={a:asr[a][rid]['final_payload'].get('text','') for a in ['off','on']}))
        summary={}
        for arm,count in totals.items():
            summary[arm]=dict(count)
            if gt is not None:
                summary[arm].update(CER=count['character_errors']/count['reference_chars'],
                    outside_entity_CER=count['outside_entity_errors']/count['outside_entity_chars'])
        raw_summary={a:dict(c,CER=c['character_errors']/c['reference_chars']) if gt is not None else dict(c) for a,c in raw_totals.items()}
        write_json(folder/'raw_summary.json',raw_summary)
        write_json(folder/'summary.json',summary);write_json(folder/'comparison.json',details)
        changes=[d for d in details if any(d['results'][a]['text']!=d['results']['baseline']['text'] for a in ARMS)]
        write_json(folder/'changes.json',changes)
        with (folder/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            metrics=('character_errors','recovered_mentions','outside_entity_errors')
            writer=csv.writer(f);writer.writerow(['id','reference','raw_without_context','raw_with_context',*ARMS,
                *[f'{a}_{metric}' for a in ARMS for metric in metrics]])
            for d in details:writer.writerow([d['id'],d['reference'],d['raw_asr']['off'],d['raw_asr']['on'],
                *[d['results'][a]['text'] for a in ARMS],*[d['results'][a].get(metric,'') for a in ARMS for metric in metrics]])
        for arm in ARMS:
            (folder/f'{arm}.txt').write_text('\n\n'.join(f"[{d['id']}]\n{d['results'][arm]['text']}" for d in details)+'\n')
        all_summary[dataset]=summary
        lines+=['',f"## {dataset}：{len(ids)} 条，{protocol['duration_seconds']/60:.2f} 分钟",'',
            '实际提示词：'+ '、'.join(protocol['context_terms'])+'。','']
        if gt is not None:
            lines+=['| 条件 | CER | 实体保留 | 实体增益／丢失 | 提示词目标保留 | 非实体区 CER | 改善／退化样本 | 精修调用 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
            for arm,s in summary.items():lines.append(f"| {LABELS[arm]} | {s['CER']:.3%} | {s['recovered_mentions']}/{s['expected_mentions']} | {s['entity_gains']}/{s['entity_losses']} | {s['prompted_recovered']}/{s['prompted_expected']} | {s['outside_entity_CER']:.3%} | {s['improved_samples']}/{s['regressed_samples']} | {s['refiner_calls']} |")
            entity_changes=[]
            for d in details:
                base=d['results']['baseline']['recovered_by_name'];both=d['results']['combined']['recovered_by_name']
                changed={name:dict(baseline=base.get(name,0),combined=both.get(name,0)) for name in set(base)|set(both) if base.get(name,0)!=both.get(name,0)}
                if changed:entity_changes.append(dict(id=d['id'],changes=changed,reference=d['reference'],baseline=d['results']['baseline']['text'],combined=d['results']['combined']['text']))
            write_json(folder/'entity_gain_loss_cases.json',entity_changes)
            lines+=['','### 实体增益与丢失案例','', '| 样本 | 标准名 | 无库保留 | 组合保留 |','|---|---|---:|---:|']
            for d in entity_changes:
                for name,c in sorted(d['changes'].items()):lines.append(f"| {d['id']} | {name} | {c['baseline']} | {c['combined']} |")
            prefix=protocol['context_text'].split('术语列表：',1)[0]
            leaked=[d for d in details if prefix and prefix in d['raw_asr']['on'] and prefix not in d['reference']]
            write_json(folder/'prompt_echo_cases.json',leaked)
            if leaked:
                lines+=['','### ASR 提示文字进入转录','',
                    '这些输出含实际发送的提示说明原文，参考没有这些说明。不能通过删掉坏样本提高汇总成绩；它们仍计入全部指标。', '',
                    '| 样本 | 组合相对无库字符错误增量 | 提示说明进入最终文字 |','|---|---:|---|']
                for d in leaked:
                    delta=d['results']['combined']['character_errors']-d['results']['baseline']['character_errors']
                    lines.append(f"| {d['id']} | {delta:+d} | {'是' if prefix in d['results']['combined']['text'] else '否'} |")
                lines+=['','实际被复制的提示说明：'+prefix+'。完整原始及最终文字见 `prompt_echo_cases.json`。']
        else:
            lines+=['| 条件 | 文字变化窗口 | 实体规范事件 | 精修调用 |','|---|---:|---:|---:|']
            for arm,s in summary.items():lines.append(f"| {LABELS[arm]} | {s['changed_samples']} | {s['normalization_events']} | {s['refiner_calls']} |")
            lines+=['','| 条件 | HER2 标准字面形式 | PD-1 标准字面形式 | PD-L1 标准字面形式 | 贝伐珠单抗标准字面形式 |',
                '|---|---:|---:|---:|---:|']
            targets=('HER2','PD-1','PD-L1','贝伐珠单抗')
            for arm in ARMS:
                counts=[sum(d['results'][arm]['text'].count(name) for d in details) for name in targets]
                lines.append(f"| {LABELS[arm]} | "+' | '.join(map(str,counts))+' |')
            lines+=['',f"播客无标准字幕，{len(ids)} 个窗口的选样说明：{protocol['cohort']}。文字变化／标准名称出现不等于准确率。"]
    write_json(OUT/'summary.json',all_summary)
    lines+=['','CER 为参考字幕文字一致度，使用相同 ITN、NFKC 和标点／空白处理。字幕含省略与英语缺标，不作为人工核准的听写准确率。', '',
        '实体计数为词表内最长非重叠标准名称；非实体区错误按与参考的字符对齐排除词表实体区域，并非人工语义误改标签。', '',
        '各条件完整结果、原始 ASR、候选、替换和实际模型调用见各子目录 JSONL、comparison.csv、comparison.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(all_summary,ensure_ascii=False,indent=2),flush=True)


def mention_changes(baseline,current):
    keys=set(baseline)|set(current)
    return (sum(max(0,current.get(k,0)-baseline.get(k,0)) for k in keys),
        sum(max(0,baseline.get(k,0)-current.get(k,0)) for k in keys))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','decode','replay','report','run']);args=parser.parse_args()
    if args.stage in {'prepare','run'}:prepare()
    if args.stage in {'decode','run'}:decode()
    if args.stage in {'replay','run'}:
        settings=SimpleNamespace(output=OUT/'medical',library=OUT/'medical/library',model=MODEL,device='cuda:0',
            asr_url=ASR,mode='streaming',chunk_seconds=2.0)
        runner=FactorialRunner(settings)
        for dataset in SOURCES:
            for arm in ARMS:runner.run_cell(dataset,arm)
    if args.stage in {'report','run'}:report()


if __name__=='__main__':main()
