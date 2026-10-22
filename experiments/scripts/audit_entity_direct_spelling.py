"""Regression audit of standalone glossary patches on saved development text."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.evaluate_entity_ctc_fusion import definitions
from experiments.scripts.report_entity_backend_comparison import score
from system.entity_direct_spelling import DirectEntitySpeller


def main(output):
    base = ROOT / 'results/entity_backend_comparison_20261015'
    inputs = [json.loads(s) for s in (base/'inputs.jsonl').read_text().splitlines()]
    raw = {r['id']: r for r in map(json.loads, (base/'qwen_primary_streams.jsonl').read_text().splitlines())}
    refs = json.loads((base/'references.json').read_text())['samples']
    cache, records = {}, []
    for row in inputs:
        if row['split'] not in ('dev', 'control'):
            continue
        library = row['library_path']
        if library not in cache:
            cache[library] = DirectEntitySpeller(definitions(library))
        text = raw[row['id']]['final_payload']['text']
        patches = cache[library].select(text, True)
        revised = text
        for p in reversed(patches):
            revised = revised[:p['start']]+p['canonical']+revised[p['end']:]
        ref = refs[row['id']]
        records.append(dict(id=row['id'], split=row['split'], patches=patches,
            before=score(ref['reference'], text, ref.get('gold_entities', [])),
            after=score(ref['reference'], revised, ref.get('gold_entities', [])),
            text=text if patches else None, revised=revised if patches else None))
    summary = {}
    for split in ('dev', 'control'):
        selected = [r for r in records if r['split'] == split]
        summary[split] = dict(samples=len(selected), changed_samples=sum(bool(r['patches']) for r in selected),
            entity_hits_before=sum(r['before']['entity_hits'] for r in selected),
            entity_hits_after=sum(r['after']['entity_hits'] for r in selected),
            cer_errors_before=sum(r['before']['cer_errors'] for r in selected),
            cer_errors_after=sum(r['after']['cer_errors'] for r in selected))
    output.write_text(json.dumps(dict(summary=summary, records=records,
        scope='Saved raw development/control text, direct route only; controls inspected during rule development, so this is regression evidence, not untouched held-out precision.'), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
