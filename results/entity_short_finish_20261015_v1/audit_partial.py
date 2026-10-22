"""Audit completed samples without claiming the interrupted full run passed."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

OUT = Path(__file__).resolve().parent / 'evaluation'
def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

protocol = json.loads((OUT/'protocol.json').read_text())
for name,digest in protocol['frozen_sha256'].items():
    assert sha(OUT/name)==digest,name
dataset = Path(protocol['dataset'])
assert sha(dataset/'manifest.json')==protocol['dataset_manifest_sha256']
assert sha(dataset/'inference_inputs.jsonl')==protocol['dataset_inputs_sha256']
inputs = {r['id']:r for r in rows(OUT/'inference_inputs.jsonl')}
outputs = rows(OUT/'results.jsonl')
assert len({r['id'] for r in outputs})==len(outputs)<2000
trace = defaultdict(list)
for r in rows(OUT/'asr_trace.jsonl'):
    trace[r['id']].append(r)
assert set(trace)=={r['id'] for r in outputs}
frames=changes=finish_checks=0
libraries={}
for row in outputs:
    sample=inputs[row['id']];events=trace[row['id']]
    pcm=Path(sample['pcm_path']).read_bytes()
    assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']==row['pcm_sha256']
    starts=[e for e in events if e['primary'] and e['endpoint']=='/stream/start']
    finals=[e for e in events if e['primary'] and e['endpoint']=='/stream/finish']
    chunks=[e for e in events if e['primary'] and e['endpoint']=='/stream/chunk']
    assert len(starts)==len(finals)==1
    assert not starts[0]['context'] and not starts[0]['result']['context_applied']
    assert len(chunks)==(len(pcm)+63999)//64000
    for i,e in enumerate(chunks):
        part=pcm[i*64000:(i+1)*64000]
        assert e['data_bytes']==len(part) and e['data_sha256']==hashlib.sha256(part).hexdigest()
    frames+=len(chunks)
    repair=row['result']['entity_audio_repair'];raw=repair['evaluation_backend']['primary_finish_text']
    assert raw==finals[0]['result']['text']
    library=sample['library_path']
    if library not in libraries:
        with sqlite3.connect('file:'+library+'?mode=ro',uri=True) as db:
            libraries[library]={r[0] for r in db.execute('SELECT canonical_text FROM protected_entities WHERE enabled=1')}
    spans=[];corrected=raw
    for c in sorted(repair['changes'],key=lambda c:c['start'],reverse=True):
        a,b=c['start'],c['end']
        assert raw[a:b]==c['observed'] and c['canonical'] in libraries[library]
        assert not any(a<y and x<b for x,y in spans)
        spans.append((a,b));corrected=corrected[:a]+c['canonical']+corrected[b:];changes+=1
    assert corrected==row['result']['raw_text']
    if repair['incremental']['short_finish_checks']:
        assert repair['incremental']['short_finish_checks']==1 and len(pcm)<=16*64000
        # The new short review is the last replay session and sends every PCM
        # byte before asking ASR for its final result.
        replay_starts=[e for e in events if not e['primary'] and e['endpoint']=='/stream/start']
        sid=replay_starts[-1]['result']['session_id']
        replay_chunks=[e for e in events if e['session_id']==sid and e['endpoint']=='/stream/chunk']
        assert len(replay_chunks)==len(chunks)
        for a,b in zip(replay_chunks,chunks):
            assert (a['data_sha256'],a['data_bytes'])==(b['data_sha256'],b['data_bytes'])
        assert any(e['session_id']==sid and e['endpoint']=='/stream/finish' for e in events)
        finish_checks+=1
result=dict(status='PASS_COMPLETED_SUBSET_ONLY',completed_samples=len(outputs),expected_samples=2000,
    full_run_complete=False,primary_pcm_frames_verified=frames,primary_finish_texts_verified=len(outputs),
    bounded_entity_changes_verified=changes,complete_short_finish_pcm_passes_verified=finish_checks,
    frozen_sources_inputs_libraries_verified=True,original_dataset_manifest_unchanged=True,
    failed_attempts_preserved_separately='failed_asr_trace_20261015.jsonl')
(OUT/'partial_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
