# Curated seed pilot v4.1 final audit

The local teacher service is stopped and GPU0 is free. These records are experimental only; no production prompt or training file was changed.

## Generation

- 100 requests: 84 outputs, 16 teacher refusals. The refusals stayed rejected.
- The original hard validator passed the 84 outputs, but independent review found one numeric-unit change: `60帧每秒` became `60fps`. A posthoc unit validator now catches it.

## Blind review

- Source quality: 69 pass, 15 borderline, 0 fail.
- Source/target fidelity: 82 pass, 2 fail.
- Critical issues: 5, including the numeric-unit rewrite, one unsupported added claim, and three punctuation splits inside entity/address names.

## Category results

| Requested category | Successful outputs | Review signal | Strict candidates |
|---|---:|---:|---:|
| boundary_punctuation | 14 | 10/14 | 6 |
| clean_passthrough | 12 | 12/12 | 10 |
| fillers | 20 | 18/20 | 18 |
| natural_mixed_code_switch | 11 | invalid sample | held |
| repetition_false_start | 17 | 11/17 | 3 |
| self_correction | 10 | 1/10 | 0 |

The mixed-code category is **invalid as an evaluation**: only 4 of its 15 assigned seed targets had an ASCII/code span. The sampler selected code-bearing targets first, then shuffled category labels independently. All 11 successful outputs in this tag are held from the strict candidate set; do not interpret 5/11 as a valid trigger rate.

After rejecting pair-fidelity failures, critical issues, borderline sources, no-op outputs for transform categories, and the misassigned mixed-code class, **37 outputs remain as pilot candidates**. They are not yet training data. The self-correction category had a 10% blind trigger rate (1/10); it is not ready for scaling. Repetition yielded 11/17 signals, fillers 18/20, and boundary punctuation 10/14, with entity-boundary risks among the latter.

The independent reviewer was ChatGPT, not a human gold annotator. Use these results to fix the sampler and validator before another controlled pilot; do not scale to 100k yet.

Files: `strict_pilot_candidates.jsonl` (37 experimental candidates), `reviewed_candidates.jsonl`, `posthoc_surface_validation.json`, `mixed_category_sampling_audit.json`, `blind_review_84_results.jsonl`, `pilot_report.json`.
