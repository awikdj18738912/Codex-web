"""New professionally transcribed SpeechIO source pilot, entity stage only.

Glossaries and audio are frozen before inference. No current reference enters
the ASR request. NER spans are reference-curated, not official NER annotations.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/entity_reliable_data_20261012'
PILOT = OUT / 'pilot'
sys.path.insert(0, str(ROOT))
SEED = 'reliable-speechio-entity-pilot-v1'
DOMAINS = {'football': 'SPEECHIO_ASR_ZH00003', 'automotive': 'SPEECHIO_ASR_ZH00021'}
SEEDS = {
    'football': {
        'PERSON': '巴乔 罗纳尔迪尼奥 贝克汉姆 巴蒂斯图塔 马拉多纳 克林斯曼 克洛泽 齐达内 罗纳尔多 卡纳瓦罗 科里纳 布冯 马尔蒂尼 皮尔洛 内德维德 西蒙尼 菲戈 里瓦尔多 德尚 霍德尔 斯科拉里 弗格森 巴尔德拉马 巴尔达诺 普拉蒂尼 穆里尼奥 德罗巴 鲁尼 杰拉德 兰帕德 梅西 勒夫 范加尔 图拉姆',
        'ORG': '巴黎圣日耳曼 国际米兰 曼联 皇家马德里 巴塞罗那 尤文图斯 佛罗伦萨 博卡青年 拜仁慕尼黑 切尔西 多特蒙德 阿森纳 利物浦 圣埃蒂安 那不勒斯',
        'LOC': '斯坦福桥 伯纳乌 圣西罗 马拉卡纳 阿根廷 英格兰 尼日利亚 南斯拉夫 意大利 荷兰 巴西 乌拉圭 哥伦比亚 联邦德国',
        'TERM': '欧洲冠军联赛 欧洲联盟杯 意大利杯 世界杯 金球奖 欧洲杯 任意球 点球大战 欧冠 意甲 英超 西甲 联赛冠军 世界足球先生',
    },
    'automotive': {
        'PERSON': '陈乐儿 陈震 闫闯 韩路',
        'ORG': '大众 宝马 奔驰 奥迪 沃尔沃 保时捷 斯巴鲁 特斯拉 雷克萨斯 本田 丰田 比亚迪 马自达 雪佛兰 阿尔法罗密欧 阿斯顿马丁 玛莎拉蒂 长城汽车 米其林 长安汽车 福特 哈弗 林肯 路虎',
        'TERM': '卡罗拉 雷凌 思域 探岳 探险者 浩悦 新能源汽车 双离合变速箱 涡轮增压 发动机 变速箱 分时四驱 全时四驱 四驱系统 日间行车灯 车道保持 自动驾驶 自适应巡航 主动刹车 胎压监测 电动助力 转弯半径 独立悬架 多连杆 扭力梁 麦弗逊 差速器 可变气门 悬架系统 后轮转向 无线充电 轴距 制动系统',
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def rank(value):
    return hashlib.sha256((SEED+value).encode()).hexdigest()


def entities(text, vocabulary):
    found = []
    for word in sorted(vocabulary, key=lambda w:(-len(w),w)):
        for match in re.finditer(re.escape(word), text):
            if any(match.start()<e['end'] and e['start']<match.end() for e in found):
                continue
            found.append(dict(canonical=word, start=match.start(), end=match.end(),
                              entity_type=vocabulary[word]))
    return sorted(found, key=lambda e:e['start'])


def prepare():
    import numpy as np
    import pyarrow.parquet as pq
    from system.entity_store import EntityStore
    if (PILOT/'manifest.json').exists():
        raise FileExistsError('Pilot already frozen')
    PILOT.mkdir(parents=True, exist_ok=True)
    old_raw, old_pcm = set(), set()
    for f in [ROOT/'results/large_entity_benchmark_v1_20261010/dataset/samples.jsonl'] + [
        ROOT/'results/entity_confirmation_20261011'/folder/'samples.jsonl'
        for folder in ('holdout','holdout_news','holdout_v2')]:
        for r in read(f):
            old_raw.add(r.get('audio_sha256',r.get('source_audio_sha256')))
            old_pcm.add(r['pcm_sha256'])
    # Hash-only metadata, never read the new181's reference sidecar.
    old_pcm.update(json.loads((ROOT/'results/entity_context_confirmation_20261012/unseen/summary.json').read_text())['pcm_sha256'].values())
    selected, references, glossaries, sources = [], {}, {}, {}
    known_raw, known_pcm = set(), set()
    for domain, subset in DOMAINS.items():
        source = OUT/'source/speechio'/f'{subset}.parquet'
        table = pq.read_table(source)
        rows = table.to_pylist()
        words = {w:kind for kind, value in SEEDS[domain].items() for w in value.split()
                 if any(w in r['text'] for r in rows)}
        evidence = {w:[] for w in words}
        for row in rows:
            for e in entities(row['text'],words):
                if len(evidence[e['canonical']])<3:
                    evidence[e['canonical']].append(dict(source_id=row['segment_id'], text=row['text'],
                                                        start=e['start'],end=e['end']))
        words = {w:k for w,k in words.items() if evidence[w]}
        glossaries[domain] = [dict(canonical=w, entity_type=kind, evidence=evidence[w],
                                 annotation_method='agent_reference_curated') for w,kind in sorted(words.items())]
        store = EntityStore(PILOT/'libraries'/domain/'entities.db')
        for w,kind in sorted(words.items()):
            store.upsert_entity(w, entity_type=kind, domain='general', normalization_policy='normalize',
                                source='SpeechIO original reference; vocabulary frozen before ASR; evaluation only')
        candidates = []
        for i,row in enumerate(rows):
            text = row['text']
            if not 2<=row['duration']<=20 or len(text)<6:
                continue
            # Exclude a visibly suspicious surface before any inference, without
            # rewriting reference. Neither this flag nor its token enters ASR.
            if any(flag in text for flag in ('巴蒂斯图塔斯','卡拉雷凌')):
                continue
            gold = entities(text, words)
            candidates.append(dict(row=i, raw=row, gold=gold, key=rank(subset+row['segment_id'])))
        positives = [r for r in candidates if r['gold']]
        negatives = sorted([r for r in candidates if not r['gold']], key=lambda r:r['key'])
        picked, covered = [], set()
        for _ in range(32):
            if not positives:
                break
            item = min(positives, key=lambda r:(-len({e['canonical'] for e in r['gold']}-covered),r['key']))
            positives.remove(item); covered.update(e['canonical'] for e in item['gold']); picked.append(item)
        picked += negatives[:16]
        assert len(picked)==48, (domain,len(picked))
        for item in picked:
            r = item['raw']; source_bytes=r['audio']['bytes']
            raw_hash=hashlib.sha256(source_bytes).hexdigest()
            with wave.open(io.BytesIO(source_bytes),'rb') as stream:
                assert stream.getframerate()==16000 and stream.getnchannels()==1 and stream.getsampwidth()==2
                frames=stream.readframes(stream.getnframes())
            pcm=(np.frombuffer(frames,dtype='<i2').astype('<f4')/32768.0).tobytes()
            pcm_hash=hashlib.sha256(pcm).hexdigest()
            assert abs(len(pcm)/64000-r['duration'])<0.03
            assert raw_hash not in old_raw|known_raw and pcm_hash not in old_pcm|known_pcm
            known_raw.add(raw_hash); known_pcm.add(pcm_hash)
            sid='rel_'+rank(subset+r['segment_id'])[:20]
            audio=PILOT/'audio'/f'{sid}.wav'; audio.parent.mkdir(exist_ok=True); audio.write_bytes(source_bytes)
            path=PILOT/'pcm'/f'{sid}.f32le'; path.parent.mkdir(exist_ok=True);path.write_bytes(pcm)
            selected.append(dict(id=sid, source=domain, subset=subset, source_id=r['segment_id'],
                                 row_index=item['row'], pcm_path=str(path), audio_path=str(audio),
                                 raw_audio_sha256=raw_hash, pcm_sha256=pcm_hash,
                                 duration_seconds=len(pcm)/64000, cohort='positive' if item['gold'] else 'no_registered_target'))
            references[sid]=dict(reference=r['text'], gold_entities=item['gold'],
                                 annotation_method='agent_reference_curated_not_official_NER')
        sources[domain]=dict(path=str(source),sha256=digest(source), rows=len(rows),
                             library_sha256=digest(store.path))
    write(PILOT/'references.json', references)
    write(PILOT/'glossaries.json', glossaries)
    frozen = {str(PILOT/'references.json'):digest(PILOT/'references.json'),
              str(PILOT/'glossaries.json'):digest(PILOT/'glossaries.json')}
    for name in ('entity_audio_repair.py','entity_asr_echo_guard.py','entity_retrieval.py','entity_matcher.py','entity_boundaries.py'):
        frozen[str(ROOT/'system'/name)] = digest(ROOT/'system'/name)
    write(PILOT/'manifest.json',dict(samples=selected, sources=sources,frozen_hashes=frozen,
          target_reference_at_inference=False, prior_audio_hash_overlap=0,
          sampling='Reference-only,32 positives maximizing distinct-name coverage +16 hash negatives per source. Before ASR.',
          suspicious_reference_filters=['巴蒂斯图塔斯','卡拉雷凌'],
          scope='Known vocabulary, actual entity ASR stage. No full Web/Refiner evaluation.',
          asr_url='http://127.0.0.1:8766', library_size_by_domain={d:len(g) for d,g in glossaries.items()}))
    print('FROZEN',len(selected),'libraries',{d:len(g) for d,g in glossaries.items()},flush=True)


def verify():
    plan=json.loads((PILOT/'manifest.json').read_text())
    for path,expected in plan['frozen_hashes'].items():
        assert digest(path)==expected
    for domain,s in plan['sources'].items():
        assert digest(PILOT/'libraries'/domain/'entities.db')==s['library_sha256']
    return plan


async def collect():
    import httpx
    from system.entity_store import EntityStore
    from system.entity_audio_repair import EntityAudioRepair
    from validate_entity_confirmation import exercise
    plan=verify()
    destination=PILOT/'audio_results.jsonl'
    done={r['id'] for r in read(destination)} if destination.exists() else set()
    libraries={d:EntityStore(PILOT/'libraries'/d/'entities.db').list_entities(domain='general') for d in DOMAINS}
    semaphore=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async def process(sample):
            if sample['id'] in done:
                return
            async with semaphore:
                pcm=Path(sample['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
                trace=[];initial=None;sessions=set();started=time.perf_counter()
                async def actual(endpoint,session_id=None,data=b'',params=None):
                    nonlocal initial
                    query=dict(params or {})
                    if session_id:
                        query['session_id']=session_id
                    result=(await client.post(plan['asr_url']+endpoint,params=query,content=data)).raise_for_status().json()
                    trace.append(dict(endpoint=endpoint,session_id=session_id,
                                      context=query.get('context',''),data_sha256=hashlib.sha256(data).hexdigest(),result=deepcopy(result)))
                    if endpoint=='/stream/start':sessions.add(result['session_id'])
                    if endpoint in ('/stream/finish','/stream/cancel'):sessions.discard(session_id)
                    if endpoint=='/stream/finish' and initial is None:initial=deepcopy(result)
                    return result
                try:
                    final=await exercise(EntityAudioRepair,actual,libraries[sample['source']],pcm)
                finally:
                    for sid in list(sessions):
                        await actual('/stream/cancel',sid)
                assert initial is not None and not initial.get('context_applied')
                record=dict(id=sample['id'],initial=initial,production_final=final,trace=trace,
                            elapsed_seconds=time.perf_counter()-started)
                with destination.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                print('COLLECTED',sample['id'],final['entity_audio_repair']['change_count'],flush=True)
        await asyncio.gather(*(process(s) for s in plan['samples']))
    verify()


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    cc=opencc.OpenCC('t2s');itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(text):return normalized(cc.convert(itn.normalize(text).text))
    plan=verify();samples={s['id']:s for s in plan['samples']}
    refs=json.loads((PILOT/'references.json').read_text());results=read(PILOT/'audio_results.jsonl')
    assert len(results)==len({r['id'] for r in results})==len(samples) and {r['id'] for r in results}==set(samples)
    counts={};changes=[]
    for row in results:
        sample=samples[row['id']];label=refs[row['id']];ref=norm(label['reference'])
        gold=Counter(norm(e['canonical']) for e in label['gold_entities'])
        for arm in ('initial','production_final'):
            value=row[arm];text=norm(value['text']);status=value.get('entity_audio_repair',{})
            for group in ('all',sample['source'],sample['cohort']):
                c=counts.setdefault(group,{}).setdefault(arm,Counter())
                c.update(samples=1,hits=sum(min(n,text.count(w)) for w,n in gold.items()),
                         expected_mentions=sum(gold.values()),errors=Levenshtein.distance(ref,text),
                         reference_chars=len(ref),replays=status.get('checks',0),
                         absent_changes=sum(norm(e['canonical']) not in ref for e in status.get('changes',[])))
        if row['initial']['text']!=row['production_final']['text']:
            changes.append(dict(id=row['id'],source=sample['source'],reference=label['reference'],
                                before=row['initial']['text'],after=row['production_final']['text'],
                                error_delta=Levenshtein.distance(ref,norm(row['production_final']['text']))-Levenshtein.distance(ref,norm(row['initial']['text'])),
                                changes=row['production_final']['entity_audio_repair']['changes']))
    for group in counts.values():
        for c in group.values():c['CER']=c['errors']/c['reference_chars']
    write(PILOT/'summary.json',counts);write(PILOT/'changes.json',changes)
    print(json.dumps(counts,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','report']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='collect':asyncio.run(collect())
    else:report()
