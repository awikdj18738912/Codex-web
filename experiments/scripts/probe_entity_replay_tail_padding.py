"""Development-only: add one second of zeros to entity replay, no other edits."""
import asyncio
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_reliable_data_20261012/pilot'
sys.path.insert(0,str(ROOT))


async def main():
    import httpx
    from system.entity_audio_repair import EntityAudioRepair
    from system.entity_store import EntityStore
    from validate_entity_confirmation import exercise,replay_request
    def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    original=json.loads((OUT/'manifest.json').read_text())
    plan_path=OUT/'tail_padding_plan.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(dict(development_only=True,padding_bytes=64000,
            source_manifest_sha256=sha(OUT/'manifest.json'),baseline_sha256=sha(OUT/'audio_results.jsonl'),
            production_entity_module_sha256=sha(ROOT/'system/entity_audio_repair.py'),
            proposal_after_known_failures=True,reference_at_inference=False,
            change='Only prompted entity replay gains1 second of float32 silence at finish. Original recording stays exact prefix.',
            acceptance='No lost target hits, no CER increase, no changed negative controls; at least1 added hit or lower CER.'),ensure_ascii=False,indent=2)+'\n')
    plan=json.loads(plan_path.read_text())
    assert sha(OUT/'manifest.json')==plan['source_manifest_sha256']
    assert sha(OUT/'audio_results.jsonl')==plan['baseline_sha256']
    assert sha(ROOT/'system/entity_audio_repair.py')==plan['production_entity_module_sha256']
    samples={r['id']:r for r in original['samples']}
    defs={d:EntityStore(OUT/'libraries'/d/'entities.db').list_entities(domain='general') for d in original['sources']}
    output=OUT/'tail_padding_results.jsonl';done={r['id'] for r in read(output)} if output.exists() else set()
    semaphore=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async def process(row):
            if row['id'] in done:return
            async with semaphore:
                sample=samples[row['id']];pcm=Path(sample['pcm_path']).read_bytes()
                assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
                first=row['trace'][0];sid=first['result']['session_id']
                initial_trace=[first]+[r for r in row['trace'][1:] if r['session_id']==sid]
                trace=[];prompted=set();audio={};pads=0
                async def actual(endpoint,session_id=None,data=b'',params=None):
                    nonlocal pads
                    query=dict(params or {})
                    if session_id:query['session_id']=session_id
                    if endpoint=='/stream/finish' and session_id in prompted:
                        assert bytes(audio[session_id])==pcm
                        zeros=b'\x00'*plan['padding_bytes']
                        extra=(await client.post(original['asr_url']+'/stream/chunk',params={'session_id':session_id},content=zeros)).raise_for_status().json()
                        trace.append(dict(endpoint='/stream/chunk',session_id=session_id,data_sha256=hashlib.sha256(zeros).hexdigest(),
                                          padding_only=True,result=extra))
                        pads+=1
                    result=(await client.post(original['asr_url']+endpoint,params=query,content=data)).raise_for_status().json()
                    trace.append(dict(endpoint=endpoint,session_id=session_id,context=query.get('context',''),
                                      data_sha256=hashlib.sha256(data).hexdigest(),result=deepcopy(result)))
                    if endpoint=='/stream/start' and query.get('context'):
                        new_sid=result['session_id'];prompted.add(new_sid);audio[new_sid]=bytearray()
                    if endpoint=='/stream/chunk' and session_id in prompted:audio[session_id].extend(data)
                    if endpoint in ('/stream/finish','/stream/cancel'):prompted.discard(session_id)
                    return result
                request,remaining,_=replay_request(initial_trace,real_request=actual)
                proposed=await exercise(EntityAudioRepair,request,defs[sample['source']],pcm)
                assert not remaining
                with output.open('a') as f:f.write(json.dumps(dict(id=row['id'],initial=row['initial'],production_final=proposed,
                    padding_replays=pads,trace=trace),ensure_ascii=False)+'\n')
                print('PADDED',row['id'],'replays',pads,'changes',proposed['entity_audio_repair']['change_count'],flush=True)
        await asyncio.gather(*(process(r) for r in read(OUT/'audio_results.jsonl')))


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    original=json.loads((OUT/'manifest.json').read_text())
    samples={s['id']:s for s in original['samples']}
    refs=json.loads((OUT/'references.json').read_text())
    def read(name):return [json.loads(s) for s in (OUT/name).read_text().splitlines() if s]
    baseline={r['id']:r for r in read('audio_results.jsonl')}
    results=read('tail_padding_results.jsonl')
    assert len(results)==len({r['id'] for r in results})==len(samples)
    assert {r['id'] for r in results}==set(baseline)==set(samples)
    cc=opencc.OpenCC('t2s');itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(text):return normalized(cc.convert(itn.normalize(text).text))
    counts={};changes=[]
    for row in results:
        sample=samples[row['id']];label=refs[row['id']]
        ref=norm(label['reference']);gold=Counter(norm(e['canonical']) for e in label['gold_entities'])
        for arm,payload in [('baseline',baseline[row['id']]['production_final']),('padded',row['production_final'])]:
            text=norm(payload['text']);status=payload['entity_audio_repair']
            for group in ('all',sample['source'],sample['cohort']):
                c=counts.setdefault(group,{}).setdefault(arm,Counter())
                c.update(samples=1,hits=sum(min(n,text.count(w)) for w,n in gold.items()),expected_mentions=sum(gold.values()),
                    errors=Levenshtein.distance(ref,text),reference_chars=len(ref),replays=status['checks'],
                    absent_changes=sum(norm(e['canonical']) not in ref for e in status['changes']))
        if baseline[row['id']]['production_final']['text']!=row['production_final']['text']:
            changes.append(dict(id=row['id'],reference=label['reference'],before=baseline[row['id']]['production_final']['text'],after=row['production_final']['text']))
    for group in counts.values():
        for c in group.values():c['CER']=c['errors']/c['reference_chars']
    for name,value in [('tail_padding_summary.json',counts),('tail_padding_changes.json',changes)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(counts['all'],ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['collect','report'],default='collect',nargs='?');args=parser.parse_args()
    if args.stage=='collect':asyncio.run(main())
    else:report()
