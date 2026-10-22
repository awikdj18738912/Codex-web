# Entity backend comparison analysis

Status: **complete** (1829/1829 paired test rows). Test settings come from the dev selection recorded in `summary.json`.

Pair differences are candidate minus baseline. Recall differences use summed gold mentions; CER differences use summed normalized reference characters. Negative CER is an improvement. Confidence intervals use 1,000 source-group cluster bootstrap replicates with seed 20261015.

| Paired comparison | N | Recovered hits | Lost hits | CER improved samples | CER degraded samples | Recall difference | CER net difference | Recall 95% CI | CER 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nativehotword_vs_qwen | 1829 | 488 | 142 | 534 | 616 | +14.0251% | +1.7810% | [+10.9596%, +16.9459%] | [+1.1212%, +2.4326%] |
| hybrid_vs_qwen | 1829 | 451 | 0 | 433 | 7 | +18.2813% | -1.3323% | [+16.0206%, +20.4459%] | [-1.5636%, -1.1012%] |
| nativehotword_vs_nativebeam | 1829 | 182 | 18 | 190 | 44 | +6.6478% | -0.4488% | [+5.3813%, +7.8908%] | [-0.5662%, -0.3278%] |
| hybrid_vs_nativehotword | 1829 | 342 | 237 | 772 | 284 | +4.2562% | -3.1133% | [+2.0620%, +6.5997%] | [-3.6587%, -2.5816%] |

## Native bare-character token coverage

Supported means the original gold canonical (or surface fallback) is present verbatim in that sample's enabled library and every original non-whitespace character is a standalone token in native tokens.txt. `Unsupported` means present in the library but not encodable; `not_in_library` is reported separately. Hits remain normalized and mention-clipped. Rates are mention-weighted.

| Coverage class | Gold mentions | Qwen hits / recall | Native hotword hits / recall | Hybrid hits / recall |
|---|---:|---:|---:|---:|
| supported | 1295 | 817/1295 (63.09%) | 1052/1295 (81.24%) | 980/1295 (75.68%) |
| unsupported | 1172 | 407/1172 (34.73%) | 518/1172 (44.20%) | 695/1172 (59.30%) |
| not_in_library | 0 | 0/0 (NA) | 0/0 (NA) | 0/0 (NA) |

## Control CER and lexical audit candidates

For native hotword, the lexical baseline is native beam; other arms are compared with Qwen raw. Candidates are library canonicals absent from the reference whose occurrence count increased. These are automatic audit candidates, not human-reviewed precision.

| Arm | Baseline | N | CER | New candidate entities | Added candidate mentions |
|---|---|---:|---:|---:|---:|
| qwen_raw | qwen_raw | 100 | 1.07% | 0 | 0 |
| ctc_greedy | qwen_raw | 100 | 2.22% | 0 | 0 |
| native_greedy | qwen_raw | 100 | 2.58% | 0 | 0 |
| native_beam | qwen_raw | 100 | 2.08% | 0 | 0 |
| native_hotword | native_beam | 100 | 2.08% | 0 | 0 |
| hybrid | qwen_raw | 100 | 1.07% | 0 | 0 |
