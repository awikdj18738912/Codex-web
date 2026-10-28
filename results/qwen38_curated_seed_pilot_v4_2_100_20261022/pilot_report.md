# Curated seed pilot v4.2 result

This was a 100-request controlled pilot using seeds selected by blinded review. The requested category was bound to the seed: all 15 mixed-code seeds contained existing ASCII/code text, and passthrough controls had no common filler/correction markers. The local Qwen3.8 service has been stopped; GPU0 is free. No training file or production prompt was changed.

## Generation and blinded review

- Requests: 100; successful outputs: 83; teacher refusals: 17.
- Source quality: PASS 70, borderline 9, FAIL 4.
- Pair fidelity: PASS 79, FAIL 4.
- Critical issue flags: 6; posthoc number/unit or protected-surface failures: 0.
- Strict pilot candidates: 54 after blind review and surface checks.

## Category results

| Category | Outputs | Expected signal observed | Signal rate | Strict candidates |
|---|---:|---:|---:|---:|
| boundary_punctuation | 10 | 9/10 | 90% | 4 |
| clean_passthrough | 12 | 12/12 | 100% | 12 |
| fillers | 20 | 15/20 | 75% | 15 |
| natural_mixed_code_switch | 15 | 15/15 | 100% | 15 |
| repetition_false_start | 17 | 14/17 | 82% | 8 |
| self_correction | 9 | 1/9 | 11% | 0 |

Self-correction remains weak: 1 of 9 successful outputs showed a usable correction signal, and zero passed the strict candidate gate. Filler generation was stronger; entity-safe boundary punctuation still needs better control. Mixed-code selection is now correctly paired, but most resulting pairs are no-op preservation examples rather than refiner transformations.

## Pilot pair split

- Non-identical transformation pairs: 28.
- Exact no-op preservation pairs: 26.
- These are research candidates, not approved training data. The independent reviewer was ChatGPT; this is not a human gold audit.

## Files

- `strict_pilot_candidates.jsonl`: candidates passing the current strict screen.
- `refiner_transform_pilot.jsonl`: non-identical source/target candidates.
- `gate_preservation_pilot.jsonl`: exact source/target no-op examples.
- `reviewed_candidates.jsonl`: category, provenance, blind judgments, and retention decisions.
- `posthoc_surface_validation.json`: exact surface validation result.
- `pilot_report.json`: full machine-readable metrics.

This result still does not justify expanding to 100k. The next data step is to source self-correction pairs from naturally occurring correction histories and to constrain boundary punctuation to clause boundaries, then run another small audit.
