"""Development probe: real same-PCM ASR with reference-free glossary controls."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/large_entity_benchmark_v1_20261010'
OUT = ROOT / 'results/entity_confirmation_20261011'
sys.path.insert(0, str(BASE / 'full_evaluation/source_snapshot'))


def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare():
    path = OUT / 'stability_plan.json'
    if path.exists(): raise FileExistsError('Probe plan already frozen')
    samples = {r['id']:r for r in read(BASE/'dataset/samples.jsonl')}
    rows = []
    from system.entity_audio_repair import verify_entity_audio_changes
    for row in read(OUT/'development_replays.jsonl'):
        _, changes = verify_entity_audio_changes(row['source'], row['replay'], row['candidates'])
        short = [c for c in changes if max(len(c['observed']),len(c['canonical'])) <= 2 and any(
            (x['start'],x['end'])==(c['start'],c['end']) and x['canonical']!=c['canonical'] for x in row['candidates'])]
        if not short: continue
        local_terms = list(dict.fromkeys(x['canonical'] for x in row['candidates'] if any(
            (x['start'],x['end'])==(c['start'],c['end']) for c in short)))[:8]
        sample=samples[row['id']]
        rows.append(dict(id=row['id'], pcm_path=sample['pcm_path'], pcm_sha256=sample['pcm_sha256'],
                         source=row['source'], candidates=row['candidates'], baseline_changes=changes,
                         short_changes=short, scenarios=dict(reverse_terms=list(reversed(row['prompt_terms'])),
                                                            local_competitors=local_terms)))
    write(path,dict(development_only=True, selection='All applied ambiguous short changes in the prior full cohort',
                    reference_at_inference=False, samples=rows))
    print('PREPARED',len(rows),sum(len(r['short_changes']) for r in rows),flush=True)


async def collect():
    import httpx
    from system.entity_asr_context import EntityASRContext
    from system.entity_asr_echo_guard import EntityASREchoGuard
    from system.entity_audio_repair import REPLAY_PREFIX
    plan=json.loads((OUT/'stability_plan.json').read_text())
    dest=OUT/'stability_audio.jsonl';done={(r['id'],r['scenario']) for r in read(dest)} if dest.exists() else set()
    sem=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async def process(row,scenario,terms):
            if (row['id'],scenario) in done:return
            async with sem:
                pcm=Path(row['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
                trace=[]
                async def request(endpoint,session_id=None,data=b'',params=None):
                    query=dict(params or {})
                    if session_id:query['session_id']=session_id
                    result=(await client.post('http://127.0.0.1:8766'+endpoint,params=query,content=data)).raise_for_status().json()
                    trace.append(dict(endpoint=endpoint,session_id=session_id,context=query.get('context',''),
                                      data_sha256=hashlib.sha256(data).hexdigest(),result=result))
                    return result
                context=EntityASRContext(REPLAY_PREFIX+json.dumps(terms,ensure_ascii=False),tuple(terms))
                guard=EntityASREchoGuard(request,context,replay_chunk_bytes=64000)
                sid=None
                try:
                    ready=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0,context=context.text))
                    assert ready.get('context_applied') is True
                    sid=ready['session_id']
                    for offset in range(0,len(pcm),64000):await guard.request('/stream/chunk',sid,pcm[offset:offset+64000])
                    final=await guard.request('/stream/finish',sid)
                    with dest.open('a') as f:f.write(json.dumps(dict(id=row['id'],scenario=scenario,terms=terms,
                        final=final,fallback_active=guard.fallback_active,pcm_sha256=row['pcm_sha256'],trace=trace),ensure_ascii=False)+'\n')
                    print('ASR',row['id'],scenario,final.get('text',''),flush=True)
                finally:
                    if sid is not None:
                        try:await guard.request('/stream/cancel',sid)
                        except Exception:pass
                    guard.close()
        await asyncio.gather(*(process(row,scenario,terms) for row in plan['samples'] for scenario,terms in row['scenarios'].items()))


def score():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized,verify_entity_audio_changes
    from system.zh_itn_bridge import ZhITNNormalizer
    itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so');cc=opencc.OpenCC('t2s')
    def norm(s):return normalized(cc.convert(itn.normalize(s).text))
    plan=json.loads((OUT/'stability_plan.json').read_text());selected={r['id']:r for r in plan['samples']}
    results={(r['id'],r['scenario']):r for r in read(OUT/'stability_audio.jsonl')}
    assert len(results)==2*len(selected)
    refs={r['sample_id']:r['reference'] for r in json.loads((BASE/'dataset/references.json').read_text())}
    gold={r['sample_id']:r['gold_entities'] for r in json.loads((BASE/'dataset/sample_entities.json').read_text())}
    variants=['baseline','reverse_terms','local_competitors','both']
    summary={v:Counter() for v in variants};details={v:[] for v in variants}
    for old in read(BASE/'full_evaluation/audio.jsonl'):
        sid=old['id'];ref=norm(refs[sid]);expected=Counter(norm(e['canonical']) for e in gold[sid])
        prior=old['production_final']['text'];prior_hits=sum(min(n,norm(prior).count(w)) for w,n in expected.items())
        prior_errors=Levenshtein.distance(ref,norm(prior))
        supports={}
        if sid in selected:
            row=selected[sid]
            for scenario in row['scenarios']:
                result=results[sid,scenario]
                _,confirmed=verify_entity_audio_changes(row['source'],result['final'].get('text',''),row['candidates'])
                supports[scenario]={(c['start'],c['end'],c['canonical']) for c in confirmed} if not result['fallback_active'] else set()
        for variant in variants:
            value=prior;changes=old['production_final']['entity_audio_repair']['changes']
            if variant!='baseline' and sid in selected:
                row=selected[sid];short={(c['start'],c['end'],c['canonical']) for c in row['short_changes']}
                required=list(supports) if variant=='both' else [variant]
                changes=[c for c in row['baseline_changes'] if (c['start'],c['end'],c['canonical']) not in short or all(
                    (c['start'],c['end'],c['canonical']) in supports[s] for s in required)]
                value=row['source']
                for c in sorted(changes,key=lambda c:c['start'],reverse=True):value=value[:c['start']]+c['canonical']+value[c['end']:]
            compact=norm(value);hits=sum(min(n,compact.count(w)) for w,n in expected.items());errors=Levenshtein.distance(ref,compact)
            summary[variant].update(samples=1,hits=hits,errors=errors,reference_chars=len(ref),
                absent_changes=sum(norm(c['canonical']) not in ref for c in changes),
                lost_mentions=max(0,prior_hits-hits),improved_samples=errors<prior_errors,regressed_samples=errors>prior_errors)
            if value!=prior:details[variant].append(dict(id=sid,reference=refs[sid],before=prior,after=value,hits_delta=hits-prior_hits,errors_delta=errors-prior_errors))
    for v in summary.values():v['CER']=v['errors']/v['reference_chars']
    write(OUT/'stability_summary.json',summary);write(OUT/'stability_changes.json',details)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','score']);a=parser.parse_args()
    if a.stage=='collect':asyncio.run(collect())
    elif a.stage=='prepare':prepare()
    else:score()
