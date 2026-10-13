"""Replay recorded requests for prior changed clips and the new96 short set."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_local_replay_20261012'
sys.path.insert(0,str(ROOT))


async def main():
    from system.entity_store import EntityStore
    from validate_entity_confirmation import exercise,replay_request
    spec=importlib.util.spec_from_file_location('system._local_short_candidate',OUT/'candidate_v2_entity_audio_repair.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s]
    large=ROOT/'results/large_entity_benchmark_v1_20261010';fresh=ROOT/'results/entity_reliable_data_20261012/pilot'
    samples={s['id']:s for s in read(large/'dataset/samples.jsonl')}
    small=json.loads((fresh/'manifest.json').read_text());samples.update({s['id']:s for s in small['samples']})
    records=[]
    for r in read(large/'full_evaluation/audio.jsonl'):
        if r['production_final']['entity_audio_repair']['changes']:records.append((r,large/'dataset/library/entities.db'))
    for r in read(fresh/'audio_results.jsonl'):records.append((r,fresh/'libraries'/samples[r['id']]['source']/'entities.db'))
    libraries={};results=[]
    for i,(record,library) in enumerate(records):
        if library not in libraries:libraries[library]=EntityStore(library).list_entities()
        sample=samples[record['id']];pcm=Path(sample['pcm_path']).read_bytes()
        assert len(pcm)<45*64000
        assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
        request,remaining,_=replay_request(record['trace'])
        final=await exercise(module.EntityAudioRepair,request,libraries[library],pcm)
        assert not remaining
        old={k:v for k,v in record['production_final'].items() if k!='entity_audio_repair'}
        new={k:v for k,v in final.items() if k!='entity_audio_repair'}
        assert old==new,record['id']
        assert record['production_final']['entity_audio_repair']['changes']==final['entity_audio_repair']['changes']
        results.append(dict(id=record['id'],business_payload_identical=True,all_recorded_requests_consumed=True,
                            changes=final['entity_audio_repair']['change_count']))
        if (i+1)%40==0:print('REGRESSED',i+1,flush=True)
    (OUT/'short_regression.json').write_text(json.dumps(dict(samples=len(results),passed=len(results),
        old_library_changed_recordings=sum(library==large/'dataset/library/entities.db' for _,library in records),
        fresh_short_recordings=96,source='Saved actual traces, no new ASR requests',results=results),ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(results),flush=True)


if __name__=='__main__':asyncio.run(main())
