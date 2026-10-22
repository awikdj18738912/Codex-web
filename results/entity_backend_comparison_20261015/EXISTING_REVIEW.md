# Existing entity review comparison

All text arms were rescored with the backend report's shared normalizer/scorer. The cached Qwen text was required to match the old evaluation's `paired_primary` and `saved_baseline` exactly for every comparison ID.

## Existing scene review versus backend test

The existing scene review has 2000 IDs; this comparison uses its exact intersection with the full backend test set (1829 IDs). `existing_review` is the historical `entity_audio_raw` output.

| Arm | Samples | Entity hits / mentions | Recall | CER |
|---|---:|---:|---:|---:|
| qwen | 1829 | 1224 / 2467 | 49.61% | 6.76% |
| existing_review | 1829 | 1314 / 2467 | 53.26% | 6.50% |
| nativehotword | 1829 | 1570 / 2467 | 63.64% | 8.54% |
| hybrid | 1829 | 1675 / 2467 | 67.90% | 5.43% |

Paired differences against `existing_review` (candidate minus review):

| Candidate | Recovered | Lost | Entity recall difference | CER difference | CER improved / degraded samples |
|---|---:|---:|---:|---:|---:|
| qwen | 0 | 90 | -3.65% | 0.25% | 0 / 86 |
| nativehotword | 455 | 199 | 10.38% | 2.04% | 514 / 641 |
| hybrid | 416 | 55 | 14.63% | -1.08% | 403 / 59 |

## Short-finish shared test subset

The short-finish module has 315 recorded comparisons overall, of which 230 overlap the backend test set (85 are dev). The table below is limited to those 230 test IDs and must not be generalized to all 1829 test examples.

| Arm | Samples | Entity hits / mentions | Recall | CER |
|---|---:|---:|---:|---:|---:|
| qwen | 230 | 100 / 280 | 35.71% | 8.53% |
| current_short_finish | 230 | 228 / 280 | 81.43% | 5.81% |
| nativehotword | 230 | 120 / 280 | 42.86% | 11.19% |
| hybrid | 230 | 145 / 280 | 51.79% | 7.56% |
| existing_review | 230 | 125 / 280 | 44.64% | 8.04% |

## Recorded entity-review timings

Timing fields from historical `results.jsonl` are summarized in the JSON under `observed_review_timings`. Module review elapsed/wall fields and whole Web request totals retain their recorded scope. Replayed-audio seconds are audio duration, not elapsed time. These values are not native CPU timing or concurrent backend latency.

The short-finish measurements represent only the 230 common test samples. The older scene timing sample may include rows marked `reused_from`; the JSON reports their count.
