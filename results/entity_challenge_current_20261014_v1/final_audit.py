"""Independent all-sample transport, pairing and bounded-edit audit."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

OUT = Path(__file__).resolve().parent

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

protocol = json.loads((OUT / 'protocol.json').read_text())
for name, expected in protocol['frozen_sha256'].items():
    assert sha(OUT / name) == expected, name
dataset = Path(protocol['dataset'])
assert sha(dataset / 'manifest.json') == protocol['dataset_manifest_sha256']
assert sha(dataset / 'inference_inputs.jsonl') == protocol['dataset_inputs_sha256']
inputs = rows(OUT / 'inference_inputs.jsonl')
outputs = rows(OUT / 'results.jsonl')
by_id = {s['id']: s for s in outputs}
assert len(inputs) == len(outputs) == len(by_id) == 2000
assert set(by_id) == {s['id'] for s in inputs}
trace = defaultdict(list)
for event in rows(OUT / 'asr_trace.jsonl'):
    if event['primary']:
        trace[event['id']].append(event)
assert set(trace) == set(by_id)
vocabulary = {}
frames = edits = changed_samples = 0
for sample in inputs:
    sid = sample['id']
    result = by_id[sid]['result']
    pcm = Path(sample['pcm_path']).read_bytes()
    assert hashlib.sha256(pcm).hexdigest() == sample['pcm_sha256'] == by_id[sid]['pcm_sha256']
    starts = [e for e in trace[sid] if e['endpoint'] == '/stream/start']
    finals = [e for e in trace[sid] if e['endpoint'] == '/stream/finish']
    chunks = [e for e in trace[sid] if e['endpoint'] == '/stream/chunk']
    assert len(starts) == len(finals) == 1, sid
    assert not starts[0]['context'] and not starts[0]['result'].get('context_applied'), sid
    assert len(chunks) == (len(pcm) + 63999) // 64000, sid
    for index, event in enumerate(chunks):
        expected = pcm[index*64000:(index+1)*64000]
        assert event['data_bytes'] == len(expected), sid
        assert event['data_sha256'] == hashlib.sha256(expected).hexdigest(), sid
    frames += len(chunks)
    repair = result['entity_audio_repair']
    original = repair['evaluation_backend']['primary_finish_text']
    assert original == finals[0]['result']['text'], sid
    path = sample['library_path']
    if path not in vocabulary:
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            vocabulary[path] = {r[0] for r in db.execute('SELECT canonical_text FROM protected_entities WHERE enabled=1')}
    corrected = original
    spans = []
    for change in sorted(repair['changes'], key=lambda c: c['start'], reverse=True):
        a, b = change['start'], change['end']
        assert original[a:b] == change['observed'], sid
        assert change['canonical'] in vocabulary[path], sid
        assert not any(a < old_b and old_a < b for old_a, old_b in spans), sid
        spans.append((a,b))
        corrected = corrected[:a] + change['canonical'] + corrected[b:]
        edits += 1
    assert corrected == result['raw_text'], sid
    assert not repair['incremental']['full_audio_fallbacks'], sid
    changed_samples += bool(spans)
report = dict(status='PASS', samples=2000, primary_frames_checked=frames,
    actual_primary_finish_texts_checked=2000, exact_input_pcm_and_frame_hashes_verified=True,
    original_dataset_manifest_and_inputs_unchanged=True, frozen_sources_and_libraries_unchanged=True,
    initial_context_absent=True, bounded_output_reconstruction_verified=True,
    changed_samples=changed_samples, changes=edits, whole_audio_fallbacks=0)
(OUT / 'final_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
