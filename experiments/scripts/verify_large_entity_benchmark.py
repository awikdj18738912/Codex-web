"""Independent verification of the frozen large entity benchmark.

Writes an audit outside the immutable dataset; no production database changes.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sqlite3
import unicodedata

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/large_entity_benchmark_v1_20261010'
DATA = BASE/'dataset'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    stats=json.loads((DATA/'stats.json').read_text())
    entities=json.loads((DATA/'entities.json').read_text())
    samples=[json.loads(s) for s in (DATA/'samples.jsonl').read_text().splitlines() if s]
    converter=OpenCC('t2s')
    def norm(text):
        value=unicodedata.normalize('NFKC',converter.convert(text)).casefold()
        return ''.join(c for c in value if unicodedata.category(c)[0] not in {'P','Z','C'})
    for name,expected in manifest['staging_input_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        assert digest(path)==expected,('input hash',name)
    for name,expected in manifest['output_sha256'].items():
        assert digest(DATA/name)==expected,('output hash',name)
    names={norm(e['canonical_text']):e for e in entities}
    assert len(names)==len(entities)>=1000
    assert all(re.fullmatch(r'[\u3400-\u9fff]{2,24}',e['canonical_text']) for e in entities)
    assert sum(e['entity_type']=='TERM' for e in entities)>=200
    assert len({t for e in entities for t in e['topics']})>=8
    assert len({s['id'] for s in samples})==len(samples)==manifest['sample_count']
    assert len({s['audio_sha256'] for s in samples})==len(samples)
    assert len({s['pcm_sha256'] for s in samples})==len(samples)
    policy=json.loads((BASE/'quality_policy.json').read_text())
    banned={norm(s) for s in [*policy['reject_ner'],*policy['reject_terms']]}
    assert not banned & names.keys()
    support=Counter();mention_count=0
    for sample in samples:
        audio,pcm=Path(sample['audio_path']),Path(sample['pcm_path'])
        assert audio.is_file() and pcm.is_file()
        assert digest(audio)==sample['audio_sha256']==manifest['sample_audio_sha256'][sample['id']]
        assert digest(pcm)==sample['pcm_sha256']==manifest['sample_pcm_sha256'][sample['id']]
        assert pcm.stat().st_size>0 and pcm.stat().st_size%4==0
        assert sample['sample_rate']==16000 and sample['channels']==1 and sample['sample_format']=='f32le'
        decoded_seconds=pcm.stat().st_size/64000
        assert abs(decoded_seconds-sample['duration_seconds'])<=max(.15,sample['duration_seconds']*.02)
        reference=sample['reference']
        for hit in sample['gold_entities']:
            assert 0<=hit['start']<hit['end']<=len(reference)
            key=norm(hit['canonical'])
            assert key in names and norm(reference[hit['start']:hit['end']])==key
            assert reference[hit['start']:hit['end']]==hit['surface']
            support[key]+=1;mention_count+=1
        if sample['cohort']=='negative':
            assert not sample['gold_entities']
            assert not any(key in norm(reference) for key in names)
        else:assert sample['gold_entities']
    assert set(support)==set(names)
    with sqlite3.connect(f'file:{DATA}/library/entities.db?mode=ro',uri=True) as db:
        stored=db.execute('SELECT canonical_text,entity_type,domain,normalization_policy,enabled FROM protected_entities').fetchall()
        assert len(stored)==len(entities)
        assert {norm(r[0]) for r in stored}==set(names)
        assert all(r[2]=='general' and r[3]=='normalize' and r[4]==1 for r in stored)
        assert db.execute('SELECT COUNT(*) FROM entity_aliases').fetchone()[0]==0
    for index,topic in enumerate(stats['topics'],1):
        path=DATA/'library'/f'topic_{index:02d}_{topic}.db'
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            found={norm(r[0]) for r in db.execute('SELECT canonical_text FROM protected_entities')}
        expected={norm(e['canonical_text']) for e in entities if topic in e['topics']}
        assert found==expected,(topic,len(found),len(expected))
    references={r['sample_id']:r for r in json.loads((DATA/'references.json').read_text())}
    gold={r['sample_id']:r for r in json.loads((DATA/'sample_entities.json').read_text())}
    assert set(references)==set(gold)=={s['id'] for s in samples}
    assert all(references[s['id']]['reference']==s['reference'] and gold[s['id']]['gold_entities']==s['gold_entities'] for s in samples)
    cohorts=Counter(s['cohort'] for s in samples);types=Counter(e['entity_type'] for e in entities)
    assert dict(cohorts)==stats['sample_cohorts'] and dict(types)==stats['distinct_entities_by_type']
    report=dict(status='PASS',unique_canonical_names=len(names),audio_recordings=len(samples),cohorts=dict(cohorts),
        entity_types=dict(types),supported_gold_mentions=mention_count,all_entities_have_audio_support=True,
        all_audio_and_float32_pcm_hashes_match=True,all_frozen_inputs_and_outputs_match=True,
        global_and_topic_databases_match=True,all_gold_spans_match=True,negative_samples_have_no_registered_surface=True,
        no_source_or_decoded_audio_duplicates=True,reference_and_gold_sidecars_match=True,
        sample_and_entity_counts_match=True,
        limitations='Reference/annotation/codec verification, not per-item acoustic or authoritative real-world entity identity review')
    (BASE/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
