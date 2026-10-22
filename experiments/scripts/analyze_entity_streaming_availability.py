"""Audit when held-out final entity gains were already present in partials.

This is an offline quality/availability analysis. Audio-availability positions
are not wall-clock delays or gold acoustic word-end timestamps.
"""
from collections import Counter
import argparse
import json
from pathlib import Path

from report_entity_backend_comparison import norm, read_jsonl


def counts(text, gold):
    text = norm(text)
    return Counter({word: min(n, text.count(word)) for word, n in gold.items()})


def main(out):
    inputs = {r['id']: r for r in read_jsonl(out/'inputs.jsonl') if r['split']=='test'}
    refs = json.loads((out/'references.json').read_text())['samples']
    selection = json.loads((out/'selection.json').read_text())
    qwen = {r['id']: r for r in read_jsonl(out/'qwen_primary_streams.jsonl')}
    hybrid = {r['id']: r for r in read_jsonl(out/'hybrid_results.jsonl')
              if r['minimum_gap']==selection['hybrid_minimum_gap'] and
              r['minimum_margin']==selection['hybrid_minimum_margin']}
    native = {r['id']: r for r in read_jsonl(out/'native_results.jsonl')
              if r['condition']=='hotword' and r['score']==selection['native_hotword_score']}
    assert inputs.keys()<=hybrid.keys() and inputs.keys()<=native.keys()
    aggregates = {arm: Counter() for arm in ['qwen_ctc_sidecar','native_hotword']}
    samples = []
    for sid, sample in inputs.items():
        gold = Counter(norm(str(e.get('canonical') or e.get('surface') or ''))
                       for e in refs[sid]['gold_entities'])
        gold.pop('',None)
        baseline = counts(qwen[sid]['final_payload'].get('text',''), gold)
        for arm, record in [('qwen_ctc_sidecar',hybrid[sid]),('native_hotword',native[sid])]:
            final = record['text'] if arm=='qwen_ctc_sidecar' else record['final']['text']
            final_counts = counts(final,gold)
            gained = Counter({w:max(0,final_counts[w]-baseline[w]) for w in gold})
            events = record['events'] if arm=='qwen_ctc_sidecar' else [
                e for e in record['partials'] if not e['is_flush_padding']]
            sequence = [counts(e.get('text',''),gold) for e in events]
            seen=Counter(); twice=Counter()
            for c in sequence:
                for w in gold: seen[w]=max(seen[w],c[w])
            for a,b in zip(sequence,sequence[1:]):
                for w in gold: twice[w]=max(twice[w],min(a[w],b[w]))
            item = dict(id=sid,arm=arm,final_gain_mentions=sum(gained.values()),
                gain_already_in_nonfinal_partial=sum(min(gained[w],max(0,seen[w]-baseline[w])) for w in gold),
                gain_in_two_consecutive_nonfinal_partials=sum(min(gained[w],max(0,twice[w]-baseline[w])) for w in gold),
                final_correct_mentions=sum(final_counts.values()),
                final_correct_seen_nonfinal=sum(min(final_counts[w],seen[w]) for w in gold),
                final_correct_in_two_consecutive_nonfinal=sum(min(final_counts[w],twice[w]) for w in gold),
                nonfinal_updates=len(events))
            samples.append(item)
            aggregates[arm].update({k:v for k,v in item.items() if isinstance(v,int)})
            aggregates[arm]['samples']+=1
    result=dict(test_samples=len(inputs),arms={k:dict(v) for k,v in aggregates.items()},
        definitions={
          'final_gain_mentions':'Gold canonical mention counts gained over Qwen final, clipped per reference.',
          'gain_already_in_nonfinal_partial':'Final gains whose canonical already occurred in an ordinary partial before end-of-input flush. For hybrid this is audio-availability replay, not a measured concurrent schedule.',
          'gain_in_two_consecutive_nonfinal_partials':'Same condition in two adjacent ordinary update events. This is an observation, not a guarantee the text never later changes.',
          'native_vs_qwen':'Native gains include a recognizer/model change, not only the hotword effect.',
          'latency_limit':'No wall-clock or gold spoken entity-end timestamps are inferred. A saved Qwen partial may already have paid network/model latency.'})
    (out/'streaming_availability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'streaming_availability_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir.resolve())
