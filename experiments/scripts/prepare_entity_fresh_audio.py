"""New, reference-auditable audio holdout; dictionary remains train-only."""
from collections import Counter
import hashlib
import json
import random
import time
from types import SimpleNamespace
import httpx
from evaluate_entity_library import ROOT, read_jsonl, write_json, append_jsonl, audio_bytes
from evaluate_medical_entities import lexicon_occurrences

OUT=ROOT/'results/entity_context_20261006'
CORPUS=ROOT.parent.parent.parent/'data'  # Actual corpus path is explicit below.
CORPUS=__import__('pathlib').Path('/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell')


def main():
    refs={k:''.join(t.split()) for k,t in [l.split(maxsplit=1) for l in (CORPUS/'transcript/aishell_transcript_v0.8.txt').read_text().splitlines()]}
    words=[v['canonical_text'] for v in json.loads((ROOT/'results/medical_entity_20261003/vocabulary.json').read_text())]
    excluded={r['id'] for r in read_jsonl(ROOT/'results/entity_library_20261003/asr.jsonl')}
    groups={'medical':[],'ordinary':[]}
    for path in sorted((CORPUS/'wav/test').rglob('*.wav')):
        if path.stem in excluded:continue
        entities=lexicon_occurrences(refs[path.stem],words)
        row=dict(id=path.stem,speaker=path.parent.name,audio_path=str(path),reference=refs[path.stem],entities=entities)
        groups['medical' if entities else 'ordinary'].append(row)
    random.Random(20261006).shuffle(groups['ordinary'])
    selected=groups['medical']+[dict(r) for r in groups['ordinary'][:40]]
    for r in selected:r['group']='medical' if r['entities'] else 'ordinary'
    manifest=dict(seed=20261006,method='All 19 unused test utterances covered by the frozen medical glossary + 40 seeded ordinary negatives',
                  excluded_ids=sorted(excluded),dictionary_source='CMeEE + MultiMed training glossary; no new aliases',samples=selected)
    p=OUT/'fresh_selection.json'
    if p.exists():
        if json.loads(p.read_text())!=manifest:raise ValueError('Frozen selection changed')
    else:write_json(p,manifest)
    dest=OUT/'fresh_asr.jsonl';complete={(r['id'],r['variant']) for r in read_jsonl(dest)}
    with httpx.Client(base_url='http://127.0.0.1:8766',timeout=180) as client:
        write_json(OUT/'fresh_asr_health.json',client.get('/health').raise_for_status().json())
        for i,row in enumerate(selected):
            variants=[('clean',None),('noise10',10.0)]+([('noise5',5.0)] if row['group']=='medical' else [])
            for variant,snr in variants:
                if (row['id'],variant) in complete:continue
                pcm,duration=audio_bytes(row,'clean' if snr is None else 'noise',SimpleNamespace(seed=20261006,noise_snr=snr))
                start=time.perf_counter()
                sid=client.post('/stream/start',params=dict(language='Chinese',chunk_seconds=0.5)).raise_for_status().json()['session_id']
                payloads=[]
                try:
                    for offset in range(0,len(pcm),64000):
                        payloads.append(client.post('/stream/chunk',params=dict(session_id=sid),content=pcm[offset:offset+64000]).raise_for_status().json())
                    final=client.post('/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
                except BaseException:
                    client.post('/stream/cancel',params=dict(session_id=sid));raise
                append_jsonl(dest,dict(**row,variant=variant,noise_snr_db=snr,duration_seconds=duration,
                                      asr_elapsed_seconds=time.perf_counter()-start,pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                                      raw_text=final['text'],chunk_payloads=payloads,final_payload=final))
                print(i+1,'/',len(selected),variant,row['id'],final['text'],flush=True)
    print('Complete',len(read_jsonl(dest)),flush=True)


if __name__=='__main__':main()
