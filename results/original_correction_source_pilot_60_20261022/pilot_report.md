# Existing correction-pair audit, round 2

60 new source-target pairs were sampled from the reviewed train split, with no target or source-ID overlap with the first 60 or the 300-record target-first audit. Review was category-blind and performed by ChatGPT, not a human gold annotator.

- Source quality: 20 pass, 14 borderline, 26 fail.
- Pair fidelity: 49 pass, 11 fail.
- Correction signal: 25 clear, 15 partial/overlong, 12 absent, 8 unresolved.
- Critical issues: 28; strict high-confidence correction candidates: 17.

The old reviewed corpus contains some usable correction pairs, but topic mixing and ambiguous correction histories remain common. These source pairs were already part of the old synthetic corpus and should not be added twice without deduplication.

Files: `high_confidence_correction_candidates.jsonl`, `rejected_or_hold.jsonl`, `reviewed_pairs_joined.jsonl`, `pilot_report.json`.
