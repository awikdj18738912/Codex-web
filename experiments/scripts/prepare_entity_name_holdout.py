"""Fresh speaker-balanced name/entity audio, excluded from all earlier trials."""
import json,random,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import httpx
import soundfile as sf
from evaluate_entity_library import ROOT,parse_ner,normalized,balanced_sample,audio_bytes,read_jsonl,write_json,append_jsonl

OUT=ROOT/'results/entity_context_20261006'
BASE=ROOT/'results/entity_library_20261003'
CORPUS=Path('/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell')


def main():
    vocabulary=json.loads((BASE/'vocabulary.json').read_text());known={v['canonical_text'] for v in vocabulary}
    annotations={}
    for line in (BASE/'source/aishell_ner_transcript.txt').read_text().splitlines():
        key,text=line.split(maxsplit=1);annotations[key]=parse_ner(text)
    refs={k:''.join(t.split()) for k,t in [l.split(maxsplit=1) for l in (CORPUS/'transcript/aishell_transcript_v0.8.txt').read_text().splitlines()]}
    exclude={r['id'] for p in [BASE/'asr.jsonl',OUT/'fresh_asr.jsonl'] for r in read_jsonl(p)}
    pools={g:[] for g in ['covered_names','uncovered','ordinary']}
    for path in sorted((CORPUS/'wav/test').rglob('*.wav')):
        if path.stem in exclude or path.stem not in annotations:continue
        text,entities=annotations[path.stem]
        if normalized(text)!=normalized(refs[path.stem]):continue
        has=any(e['entity_type']=='PERSON' and e['text'] in known for e in entities)
        group='covered_names' if has else 'uncovered' if entities else 'ordinary'
        for e in entities:e['covered']=e['text'] in known
        pools[group].append(dict(id=path.stem,speaker=path.parent.name,audio_path=str(path),reference=refs[path.stem],entities=entities,group=group))
    rng=random.Random(20261006)
    selected=[]
    for group,count in [('covered_names',120),('uncovered',40),('ordinary',40)]:
        selected.extend(balanced_sample(pools[group],min(count,len(pools[group])),rng))
    manifest=dict(seed=20261006,noise_snr_db=[10.0,5.0],samples=selected,excluded_ids=sorted(exclude),
                  dictionary_source='Same frozen AISHELL train/dev 500-entry glossary',
                  profiles_source='Only train/dev contexts; no test-derived aliases',
                  method='New speaker-balanced test utterances. Covered names: clean/10dB/5dB; controls: clean/10dB')
    p=OUT/'name_holdout_selection.json'
    if p.exists() and json.loads(p.read_text())!=manifest:raise ValueError('Frozen selection changed')
    write_json(p,manifest)
    dest=OUT/'name_holdout_asr.jsonl';done={(r['id'],r['variant']) for r in read_jsonl(dest)}
    with httpx.Client(base_url='http://127.0.0.1:8766',timeout=180) as client:
        for i,row in enumerate(selected):
            variants=[('clean',None),('noise10',10.0)]+([('noise5',5.0)] if row['group']=='covered_names' else [])
            for variant,snr in variants:
                if (row['id'],variant) in done:continue
                pcm,duration=audio_bytes(row,'clean' if snr is None else 'noise',SimpleNamespace(seed=20261006,noise_snr=snr))
                started=time.perf_counter();sid=client.post('/stream/start',params=dict(language='Chinese',chunk_seconds=0.5)).raise_for_status().json()['session_id'];chunks=[]
                try:
                    for offset in range(0,len(pcm),64000):
                        chunks.append(client.post('/stream/chunk',params=dict(session_id=sid),content=pcm[offset:offset+64000]).raise_for_status().json())
                    final=client.post('/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
                except BaseException:
                    client.post('/stream/cancel',params=dict(session_id=sid));raise
                append_jsonl(dest,dict(**row,variant=variant,noise_snr_db=snr,duration_seconds=duration,
                                      asr_elapsed_seconds=time.perf_counter()-started,pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                                      raw_text=final['text'],chunk_payloads=chunks,final_payload=final))
                print(i+1,'/',len(selected),variant,row['id'],final['text'],flush=True)
    print('Complete',len(read_jsonl(dest)),flush=True)


if __name__=='__main__':main()
