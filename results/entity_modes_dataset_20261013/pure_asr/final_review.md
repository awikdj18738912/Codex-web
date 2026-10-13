# Pure-ASR full-output read-only review

Reviewed `summary.json`, all 2,506 rows of `paired_details.jsonl`, and the current `write_pure_report.py`. This is an output/schema consistency review only; I did not run `score_pure_asr.py`, inference, or the report writer. The report writer has not generated `REPORT.md` yet.

## Full-run and request checks recorded by the scorer

- Full scope is 2,506/2,506 recordings. The paired ID-set hash is `0bbe893273a66a027938f7202334903c345cefa1ab691a3f7c708686d0ce717a`, matching the old experiment's ID set. The 2,506 pure rows have SHA-256 `24886e2d83044fa939ef952b77093366dcbc025cc660c5ff5d77b01b0edbf12a`.
- All 2,506 pure rows passed the scorer's assertions: `raw_text == clean_text == asr_payload.text`; refiner calls and retries are zero; ready context is unapplied with zero terms; context and audio review are unrequested; refiner/postprocessing, numeric normalization, and text cleanup are disabled.
- The request trace has 2,506 starts, 15,246 chunks, and 2,506 finishes; all 15,246 request records have `context_supplied=false`. The per-row assertions require chunk requests to equal transport frames, acknowledgements to equal frames, max outstanding frames at most four, and no endpoint except start/chunk/finish. No incomplete final row was ignored in this full run.
- Both old-arm summary cross-checks pass for `context` and `audio`: independently rebuilt raw/clean hits and edit errors match the parent summary exactly.

## Accuracy totals and aggregation

| Transcript view | Hits / expected | CER | Errors / reference characters |
|---|---:|---:|---:|
| Pure ASR raw | 2,543 / 2,860 | 2.6244% | 1,118 / 42,601 |
| Context raw | 2,545 / 2,860 | 2.6220% | 1,117 / 42,601 |
| Context clean | 2,543 / 2,860 | 2.7182% | 1,158 / 42,601 |
| Audio-review raw | 2,716 / 2,860 | 2.0845% | 888 / 42,601 |
| Audio-review clean | 2,714 / 2,860 | 2.1830% | 930 / 42,601 |

The group partitions reconcile to all totals: 2,410 main recordings plus 96 supplemental recordings; 2,506 recordings across the 12 library/domain groups; and 300 negative, 2,174 positive, plus 32 no-registered-target controls. For each of the five transcript views, type-level hit totals sum to the overall hits and the four type denominators sum to 2,860. LOC/ORG/PERSON/TERM hit totals are 799/928/486/330 for pure raw; 800/926/487/332 for context raw; 800/924/487/332 for context clean; 821/954/606/335 for audio raw; and 821/951/607/335 for audio clean.

Main-subset and supplement values present in the summary are consistent with their frozen sizes. Main pure/context-clean/audio-clean hits are 2,434/2,434/2,602 out of 2,745, with CER 2.6202%/2.6905%/2.1479%. The SpeechIO supplement is 109/109/112 out of 115, with CER 2.6834%/3.1127%/2.6834%.

## Interpretation points for the report

- Pure ASR and context-clean have the same aggregate 2,543 hits, but they are not identical transcripts: the paired rows differ in text on 762 IDs. There are hit-count differences in 49 IDs (24 context-clean has more hits; 25 pure has more), which net to zero aggregate hits. State this as equal aggregate hit totals, not equivalent outputs.
- Context-raw is a net +2 hits over pure ASR (2,545 vs 2,543), while context-clean returns to 2,543. This reflects differences across individual samples; the raw count gain does not carry through as an aggregate clean-output gain.
- Context-clean CER is 2.7182% versus pure raw 2.6244%, but the score uses conservative numeric ITN. It is a character agreement difference, not sufficient by itself to claim that overall perceived or semantic quality became worse. The writer already includes the “一千米／1000米” caveat; keep that qualification near CER comparisons.
- The report writer maps the final workflows to pure raw, context clean, and audio clean for its main table, and keeps all five raw/clean transcript views in a separate accuracy table without assigning separate time to raw views. Timing mappings are pure session timing versus old context/audio full-session timing. This matches the scorer schema and the stated difference in measurement paths.
- It also has corpus, entity-type, library/domain, and negative-control sections; per-type CER is explicitly whole-transcript CER on recordings containing that type. Example tables compare same-ID outputs and cap each side at three examples. Report generation is guarded by a complete 2,506-row summary/details check.
- The writer labels now distinguish “纯 ASR（raw）” from context/audio final workflows including basic entity replacement/refinement. Two generated Chinese phrases currently contain stray spaces (“三条流程 的…” and “五种 输出阶段”); consider removing those before rendering. This is a wording-only suggestion; I did not edit the writer.

Pure ASR was collected as a supplementary run after the original two workflows, through a transparent raw-path WebSocket relay with no browser time. It is not a third simultaneously interleaved arm, and its accuracy/time differences cannot be attributed solely to the entity modules because it also bypasses refinement, numeric normalization, and cleanup. The five stage accuracy views must not be reported as five independent latency measurements.
