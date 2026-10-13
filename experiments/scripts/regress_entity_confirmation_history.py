"""Read-only replay of 627 historical entity scenarios against the candidate."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime/entity_guard_test_deps'))
import soundfile as sf
import optimize_entity_replay_gate as legacy
from validate_entity_confirmation import classes,OUT,write
from system.entity_store import EntityStore


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]


async def main():
    original,new=classes();old=ROOT/'results/generic_entity_validation_20261010'
    previous=ROOT/'results/generic_entity_blind_20261010';recent=ROOT/'results/generic_entity_optimization_20261010/holdout'
    raw={r['id']:r for r in read(old/'asr_baseline.jsonl')}
    prompted={r['id']:r for r in read(ROOT/'results/generic_entity_audio_20261010/outputs_neighbor.jsonl') if r['replayed']}
    prompted.update({r['id']:r['replay'] for r in read(old/'new_audio_requests.jsonl') if r['replay']})
    fixtures=[(s,old/'libraries'/s['library_key']/'entities.db',None) for s in json.loads((old/'manifest.json').read_text())['samples']]
    for folder in (previous,recent):
        results={r['id']:r for r in read(folder/'audio_results.jsonl')}
        fixtures.extend((s,folder/'libraries'/s['source']/'entities.db',results[s['id']])
                        for s in json.loads((folder/'manifest.json').read_text())['samples'])
    assert len(fixtures)==627
    report_dir=(OUT/'short_expansion_v3' if '--short-expansion-v3' in sys.argv
                else OUT/'short_expansion' if '--short-expansion' in sys.argv else OUT)
    report_dir.mkdir(parents=True,exist_ok=True)
    destination=report_dir/'historical_regression.jsonl';done={r['id'] for r in read(destination)} if destination.exists() else set()
    libraries={}
    for index,(sample,db,record) in enumerate(fixtures):
        if sample['id'] in done:continue
        if str(db) not in libraries:libraries[str(db)]=EntityStore(db).list_entities()
        if record:
            initial, replay, trace=record['initial'],None,record['request_trace']
            pcm=Path(sample['audio_path']).read_bytes()
        else:
            initial,replay,trace=raw[sample['id']],prompted.get(sample['id']),None
            wave,rate=sf.read(sample['audio_path'],dtype='float32');assert rate==16000
            pcm=wave.astype('<f4').tobytes()
        assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
        legacy.EntityAudioRepair=original
        baseline=await legacy.exercise(sample,initial,replay,libraries[str(db)],pcm,.63,trace)
        legacy.EntityAudioRepair=new
        proposed=await legacy.exercise(sample,initial,replay,libraries[str(db)],pcm,.63,trace)
        assert baseline['authoritative_sha256']==proposed['authoritative_sha256'],sample['id']
        assert proposed['final_payload']['entity_audio_repair'].get('short_confirmation_checks',0)==0,sample['id']
        with destination.open('a') as f:f.write(json.dumps(dict(id=sample['id'],authoritative_payload_identical=True,
            starts_before=baseline['start_count'],starts_after=proposed['start_count']),ensure_ascii=False)+'\n')
        print('HISTORY',index+1,'/627',sample['id'],flush=True)
    rows=read(destination);assert len(rows)==627
    write(report_dir/'historical_summary.json',dict(scenarios=627,identical_authoritative_payloads=627,
        new_confirmation_invocations=0,request_counts_unchanged=all(r['starts_before']==r['starts_after'] for r in rows)))


if __name__=='__main__':asyncio.run(main())
