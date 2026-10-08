"""Audit numeric values, source edits, and runtime on recorded transcripts."""

import argparse
from collections import Counter
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.zh_itn_bridge import ZhITNNormalizer
from system.refinement_guard import _numeric_values
from system.chunking import _STREAM_WINDOW_PUNCTUATION
from system.numeric_normalizer import NumericNormalization, NumericNormalizationResult


def punctuation_windows(text):
    edges = [m.end() for m in _STREAM_WINDOW_PUNCTUATION.finditer(text)]
    start, windows = 0, []
    for end in edges[2::3]:
        windows.append(text[start:end]); start = end
    if start < len(text): windows.append(text[start:])
    assert ''.join(windows) == text
    return windows


def normalize_windows(engine, text):
    offset, parts, changes = 0, [], []
    for window in punctuation_windows(text):
        result = engine.normalize(window)
        parts.append(result.text)
        changes.extend(NumericNormalization(c.original, c.replacement, c.kind, offset + c.start, offset + c.end)
                       for c in result.changes)
        offset += len(window)
    return NumericNormalizationResult(''.join(parts), tuple(changes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = ZhITNNormalizer(args.before.resolve())
    new = ZhITNNormalizer(args.after.resolve())
    selected = {'3c69275ebfbc40c68209afea6f6ffea9', 'b6aa6dcd86184da8b86ff80b714c3f01',
                '89fcbb5eb76541ff913daa21138b8264', '0f4b41e520b1448eb8ab5e98e3f109b5',
                '12afb2c24bdc4437a0be1eed7bac67f0', '2e872f4fcaa24fae9b7dc6359ad7c8c1'}
    recordings = []
    for line in (ROOT / 'results/web/session.jsonl').open():
        row = json.loads(line)
        if row.get('trace_id') not in selected:
            continue
        text = row['output']['raw_text']
        a, b = normalize_windows(old, text), normalize_windows(new, text)
        edits = sorted(b.changes, key=lambda c: c.start)
        position, parts = 0, []
        for change in edits:
            assert text[change.start:change.end] == change.original
            assert change.start >= position
            parts.extend([text[position:change.start], change.replacement])
            position = change.end
        parts.append(text[position:])
        old_edits = Counter((c.original, c.replacement) for c in a.changes)
        new_edits = Counter((c.original, c.replacement) for c in b.changes)
        samples = {(c.original, c.replacement): c for c in b.changes}
        record = dict(trace_id=row['trace_id'], captured_at=row['captured_at'],
            chars=len(text), before_changes=len(a.changes), after_changes=len(b.changes),
            numeric_values_equal=_numeric_values(text) == _numeric_values(b.text),
            unmapped_text_preserved=''.join(parts) == b.text,
            idempotent=normalize_windows(new, b.text).text == b.text,
            window_size=3, window_count=len(punctuation_windows(text)),
            added=[dict(original=source, replacement=target, rule=samples[(source, target)].kind)
                   for (source, target), count in new_edits.items()
                   for _ in range(max(0, count - old_edits[(source, target)]))])
        print('recording', row['trace_id'], 'values_equal', record['numeric_values_equal'], flush=True)
        if not recordings:
            # Benchmark the bounded punctuation window used by streaming,
            # not repeated whole-document normalization (final refinement is off).
            windows = punctuation_windows(text)
            record['timing_window_count'] = len(windows)
            record['timing_mean_window_chars'] = round(len(text) / len(windows), 1)
            for engine, key in ((old, 'before_ms'), (new, 'after_ms')):
                elapsed = []
                for _ in range(3):
                    start = time.perf_counter()
                    for window in windows: engine.normalize(window)
                    elapsed.append((time.perf_counter() - start) * 1000 / len(windows))
                record[key] = round(statistics.median(elapsed), 3)
        recordings.append(record)
    gold = []
    for line in (ROOT / 'experiments/fixtures/numeric_refinement_gold.jsonl').open():
        case = json.loads(line)
        a, b = old.normalize(case['input']).text, new.normalize(case['input']).text
        gold.append(dict(case, before=a, after=b,
            before_exact=a == case['expected'], after_exact=b == case['expected'],
            numeric_values_equal_to_before=a == b or _numeric_values(a) == _numeric_values(b)))
    result = dict(before_rule_pack=old.rule_pack, after_rule_pack=new.rule_pack,
                  recordings=recordings, gold=gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(recordings=recordings, gold_total=len(gold),
        gold_before_exact=sum(c['before_exact'] for c in gold),
        gold_after_exact=sum(c['after_exact'] for c in gold),
        gold_value_changes=[c for c in gold if not c['numeric_values_equal_to_before']]), ensure_ascii=False))
    assert len(recordings) == 6
    assert all(r['numeric_values_equal'] and r['unmapped_text_preserved'] and r['idempotent'] for r in recordings)
    assert all(c['numeric_values_equal_to_before'] for c in gold)


if __name__ == '__main__':
    main()
