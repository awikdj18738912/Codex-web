"""Real-audio evaluation of verification-only entity ASR echo protection."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,append_jsonl,write_json,normalized
from evaluate_medical_entities import lexicon_occurrences
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard

OUT=ROOT/'results/entity_echo_guard_20261010'
PREVIOUS=ROOT/'results/entity_methods_comparison_20261009/medical'
PILOT=ROOT/'results/entity_prompt_format_20261009/protocol.json'


async def decode(stage,paired=False):
    import httpx
    import soundfile as sf
    p=json.loads(PILOT.read_text());samples=p['pilot_samples'] if stage=='pilot' else p['all_samples']
    folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True);dest=folder/'guarded.jsonl'
    done={r['id'] for r in read_jsonl(dest)}
    async with httpx.AsyncClient(timeout=180) as client:
        async def network_request(endpoint,session_id=None,data=b'',params=None):
            query=dict(params or {})
            if session_id:query['session_id']=session_id
            return (await client.post(p['asr_url']+endpoint,params=query,content=data)).raise_for_status().json()
        originals={r['id']:r for r in read_jsonl(PREVIOUS/'asr_on.jsonl')} if paired else {}
        for index,s in enumerate(samples):
            if s['id'] in done:continue
            wave,rate=sf.read(s['audio_path'],dtype='float32');assert rate==16000
            pcm=wave.astype('<f4').tobytes();assert hashlib.sha256(pcm).hexdigest()==s['pcm_sha256']
            context=EntityASRContext(p['variants']['legacy'],tuple(p['terms']))
            original=originals.get(s['id']);replay_index=0
            async def request(endpoint,session_id=None,data=b'',params=None):
                nonlocal replay_index
                if original is None:return await network_request(endpoint,session_id,data,params)
                recorded_sid=original['ready']['session_id']
                if endpoint=='/stream/start' and (params or {}).get('context'):
                    assert params['context']==original['context_text'];replay_index=0
                    return original['ready']
                if session_id==recorded_sid:
                    if endpoint=='/stream/chunk':
                        assert data==pcm[replay_index*64000:(replay_index+1)*64000]
                        response=original['chunk_payloads'][replay_index];replay_index+=1;return response
                    if endpoint=='/stream/finish':return original['final_payload']
                    if endpoint=='/stream/cancel':return {'cancelled':True}
                return await network_request(endpoint,session_id,data,params)
            guard=EntityASREchoGuard(request,context);started=time.perf_counter();chunks=[]
            ready=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0,context=context.text))
            try:
                for offset in range(0,len(pcm),64000):chunks.append(await guard.request('/stream/chunk',ready['session_id'],pcm[offset:offset+64000]))
                final=await guard.request('/stream/finish',ready['session_id'])
            except BaseException:
                try:await guard.request('/stream/cancel',ready['session_id'])
                finally:guard.close()
                raise
            finally:guard.close()
            append_jsonl(dest,dict(id=s['id'],row=s['row'],reference=s['reference'],pcm_sha256=s['pcm_sha256'],
                ready=ready,context_text=context.text,context_terms=list(context.terms),chunk_payloads=chunks,
                final_payload=final,guard=guard.snapshot(),elapsed_seconds=time.perf_counter()-started,
                recorded_on_live_off=paired))
            print(stage,index+1,'/',len(samples),s['id'],guard.snapshot(),final.get('text',''),flush=True)


def report(stage):
    from system.zh_itn_bridge import ZhITNNormalizer
    itn=ZhITNNormalizer();cache={}
    def norm(t):
        if t not in cache:cache[t]=normalized(itn.normalize(t).text)
        return cache[t]
    p=json.loads(PILOT.read_text());gt=json.loads((Path(p['library'])/'sample_entities.json').read_text())
    names=[r['canonical_text'] for r in json.loads((Path(p['library'])/'entities.json').read_text())]
    guarded=read_jsonl(OUT/stage/'guarded.jsonl');ids={r['id'] for r in guarded}
    old_folder=ROOT/'results/entity_prompt_format_20261009/pilot' if stage=='pilot' else PREVIOUS
    rows={'off':[r for r in read_jsonl(old_folder/('off.jsonl' if stage=='pilot' else 'asr_off.jsonl')) if r['id'] in ids],
          'legacy':[r for r in read_jsonl(old_folder/('legacy.jsonl' if stage=='pilot' else 'asr_on.jsonl')) if r['id'] in ids],
          'guarded':guarded}
    summary={};details=[]
    for arm,records in rows.items():
        stats=Counter()
        for r in records:
            text=r['final_payload'].get('text','');ref=norm(r['reference']);expected=Counter(e['text'] for e in gt[r['id']]);have=Counter(e['text'] for e in lexicon_occurrences(text,names))
            recovered={n:min(c,have[n]) for n,c in expected.items()}
            stats.update(samples=1,reference_chars=len(ref),character_errors=Levenshtein.distance(ref,norm(text)),
                expected_mentions=sum(expected.values()),recovered_mentions=sum(recovered.values()),
                unreferenced_mentions=sum(max(0,c-expected[n]) for n,c in have.items()),
                instruction_echo_samples=int('参考术语仅用于辅助拼写' in text),
                fallback_samples=int(r.get('guard',{}).get('fallback_active',False)))
            details.append(dict(id=r['id'],arm=arm,reference=r['reference'],text=text,recovered_by_name=recovered))
        summary[arm]=dict(stats,CER=stats['character_errors']/stats['reference_chars'])
    write_json(OUT/stage/'summary.json',summary);write_json(OUT/stage/'comparison.json',details)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','full','paired','pilot-report','full-report','paired-report']);a=p.parse_args()
    stage=a.stage.split('-')[0]
    if not a.stage.endswith('-report'):asyncio.run(decode(stage,paired=stage=='paired'))
    report(stage)


if __name__=='__main__':main()
