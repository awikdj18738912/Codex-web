"""Sparse-entity development stress streams made from real referenced clips."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/entity_local_replay_20261012/sparse_development'
PILOT=ROOT/'results/entity_reliable_data_20261012/pilot'


async def main():
    import httpx
    from system.entity_store import EntityStore
    from system.refinement_guard import join_refined_segments
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'inputs.jsonl').exists():raise FileExistsError('Sparse input is already frozen')
    original=json.loads((PILOT/'manifest.json').read_text());refs=json.loads((PILOT/'references.json').read_text())
    samples={s['id']:s for s in original['samples']};changes=json.loads((PILOT/'changes.json').read_text())
    cases=[];labels={}
    for index,change in enumerate(changes):
        sample=samples[change['id']]
        negatives=sorted([s for s in samples.values() if s['source']==sample['source'] and s['cohort']=='no_registered_target'],key=lambda s:s['id'])
        pieces=[];seconds=0
        while seconds<115:
            s=negatives[len(pieces)%len(negatives)];pieces.append(s);seconds+=s['duration_seconds']+.8
        insertion=(0,len(pieces)//2,len(pieces))[index%3];pieces.insert(insertion,sample)
        pcm=b''.join(Path(s['pcm_path']).read_bytes()+b'\x00'*51200 for s in pieces)
        sid=f'sparse_{index+1}_{sample["source"]}';path=OUT/(sid+'.f32le');path.write_bytes(pcm)
        gold=change['changes'][0];store=EntityStore(OUT/'libraries'/sid/'entities.db')
        store.upsert_entity(gold['canonical'],entity_type=gold['entity_type'],normalization_policy='normalize')
        cases.append(dict(id=sid,pcm_path=str(path),pcm_sha256=hashlib.sha256(pcm).hexdigest(),library=str(store.path),
                          library_sha256=hashlib.sha256(store.path.read_bytes()).hexdigest(),kind='known_real_clip_sparse_concatenation',
                          duration_seconds=len(pcm)/64000,source_ids=[s['source_id'] for s in pieces],insertion=insertion))
        labels[sid]=dict(reference=join_refined_segments(refs[s['id']]['reference'] for s in pieces),canonical=gold['canonical'],
                         caveat='Repeated real negative clips and800ms gaps. Development stress, not natural continuous speech or independent accuracy evidence.')
    (OUT/'references.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        for case in cases:
            pcm=Path(case['pcm_path']).read_bytes()
            ready=(await client.post('http://127.0.0.1:8766/stream/start',params=dict(language='Chinese',chunk_seconds=2))).raise_for_status().json()
            sid=ready['session_id'];payloads=[]
            try:
                for offset in range(0,len(pcm),64000):
                    payloads.append((await client.post('http://127.0.0.1:8766/stream/chunk',params={'session_id':sid},content=pcm[offset:offset+64000])).raise_for_status().json())
                final=(await client.post('http://127.0.0.1:8766/stream/finish',params={'session_id':sid})).raise_for_status().json()
            finally:
                await client.post('http://127.0.0.1:8766/stream/cancel',params={'session_id':sid})
            case['initial']=dict(ready=ready,chunk_payloads=payloads,final_payload=final)
            with (OUT/'inputs.jsonl').open('a') as f:f.write(json.dumps(case,ensure_ascii=False)+'\n')
            print('INITIAL',case['id'],case['duration_seconds'],flush=True)
    plan=json.loads((OUT.parent/'plan.json').read_text())
    plan.update(cases=len(cases),input_sha256=hashlib.sha256((OUT/'inputs.jsonl').read_bytes()).hexdigest(),
                scope='Sparse known-development real-clip concatenations, shared new actual unprompted first pass.')
    (OUT/'original_entity_audio_repair.py').symlink_to(OUT.parent/'original_entity_audio_repair.py')
    (OUT/'candidate_entity_audio_repair.py').symlink_to(OUT.parent/'candidate_entity_audio_repair.py')
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':asyncio.run(main())
