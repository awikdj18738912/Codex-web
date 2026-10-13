"""Development-only prompt-format probe; never modifies production rules."""
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import httpx
from evaluate_fresh_entity_blind import ROOT, rows, append, write
from system.entity_audio_repair import GenericEntitySelector, REPLAY_PREFIX, verify_entity_audio_changes
from system.entity_store import EntityStore
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard

SOURCE = ROOT/'results/generic_entity_blind_20261010'
OUT = ROOT/'results/generic_entity_optimization_20261010'


async def run():
    samples = {s['id']:s for s in json.loads((SOURCE/'manifest.json').read_text())['samples']}
    records = rows(SOURCE/'audio_results.jsonl')
    chosen = [r for r in records if r['production_final']['entity_audio_repair']['changes']]
    chosen += [r for r in records if r['id']=='fleurs:data-00001-of-00002.arrow:242:1758']
    negative = [r for r in records if samples[r['id']]['cohort']=='lexicon_absent' and r['production_final']['entity_audio_repair']['checks']]
    negative.sort(key=lambda r:hashlib.sha256(('compact-probe-v1:'+r['id']).encode()).hexdigest())
    chosen += negative[:2]
    done = {(r['id'],r['variant']) for r in rows(OUT/'context_probe.jsonl')}
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        async def request(endpoint,session_id=None,data=b'',params=None):
            query = dict(params or {})
            if session_id:query['session_id']=session_id
            return (await client.post('http://127.0.0.1:8766'+endpoint,params=query,content=data)).raise_for_status().json()
        for record in chosen:
            sample=samples[record['id']]
            definitions=EntityStore(SOURCE/'libraries'/sample['source']/'entities.db').list_entities()
            original=record['initial']['final_payload']['text']
            candidates,terms=GenericEntitySelector(definitions).select(original)
            pcm=Path(sample['audio_path']).read_bytes()
            assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
            variants=dict(current=REPLAY_PREFIX+json.dumps(terms,ensure_ascii=False),
                          json_only=json.dumps(terms,ensure_ascii=False),comma_only='、'.join(terms))
            for variant,prompt in variants.items():
                if (sample['id'],variant) in done:continue
                guard=EntityASREchoGuard(request,EntityASRContext(prompt,tuple(terms)))
                started=time.perf_counter();ready=None
                try:
                    ready=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0,context=prompt))
                    assert ready['context_applied']
                    for offset in range(0,len(pcm),64000):
                        await guard.request('/stream/chunk',ready['session_id'],pcm[offset:offset+64000])
                    final=await guard.request('/stream/finish',ready['session_id'])
                    corrected,changes=verify_entity_audio_changes(original,final['text'],candidates)
                    if guard.fallback_active:corrected,changes=original,[]
                    append(OUT/'context_probe.jsonl',dict(id=sample['id'],variant=variant,initial=original,
                        prompt_terms=terms,replay=final['text'],corrected=corrected,changes=changes,
                        guard=guard.snapshot(),elapsed_seconds=time.perf_counter()-started))
                    print('CONTEXT_PROBE',variant,sample['id'],[(c['observed'],c['canonical']) for c in changes],flush=True)
                finally:
                    if ready is not None:
                        try:await guard.request('/stream/cancel',ready['session_id'])
                        except Exception:pass
                    guard.close()


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.entity_audio_repair import normalized
    references=json.loads((SOURCE/'references.json').read_text())
    converter,itn=opencc.OpenCC('t2s'),ZhITNNormalizer()
    def norm(text):return normalized(converter.convert(itn.normalize(text).text))
    scores={}
    for r in rows(OUT/'context_probe.jsonl'):
        c=scores.setdefault(r['variant'],Counter())
        ref=norm(references[r['id']]['reference'])
        c.update(samples=1,changes=len(r['changes']),errors=Levenshtein.distance(ref,norm(r['corrected'])),
                 reference_chars=len(ref),echo_fallbacks=int(r['guard']['fallback_active']))
    write(OUT/'context_probe_summary.json',dict(results=scores,
        scope='Targeted development probe on known examples; cannot claim independent accuracy gains'))
    print(json.dumps(scores,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    asyncio.run(run());report()
