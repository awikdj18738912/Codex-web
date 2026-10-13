# Full scoring and report audit

I independently audited the completed `score_modes.py` output with `audit_score_modes.py --full`. The audit script does not import the root scorer. It uses the completed comparison IDs, frozen references and read-only SQLite glossary snapshots, plus final `results.jsonl` rows. No inference was run, no production code or glossary was changed, and protected unseen data was not read.

## Full-run verification

The saved score has exactly 2,506 paired IDs and 5,012 final Web rows. Every ID has one `context` and one `audio` row; each pair has the same PCM hash. Independently recomputed per-ID metrics and aggregate summary values match with no mismatches.

| Metric | Context | Audio review |
|---|---:|---:|
| Gold hits / expected mentions | 2,543 / 2,860 | 2,714 / 2,860 |
| CER | 1,158 / 42,601 = 2.7182% | 930 / 42,601 = 2.1830% |
| Session refiner calls | 305 | 271 |
| Session refiner retry calls | 29 | 29 |
| Mean `timing.total_seconds` | 0.8068 s | 1.1967 s |
| Median `timing.total_seconds` | 0.6721 s | 1.1794 s |
| Dictionary strings absent from the whole reference | 16 | 20 |
| Dictionary excess mentions vs reference dictionary matches | 35 | 39 |
| Replay changes whose canonical is absent from the whole reference | 0 | 4 |

The independent full-run record is [audit_score_modes_report_full_2506_0bbe893273a6.json](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_modes_dataset_20261013/audit_score_modes_report_full_2506_0bbe893273a6.json). Snapshot identifiers:

- 2,506-ID set SHA-256 (lexicographically sorted IDs joined with LF, no terminal LF): `0bbe893273a66a027938f7202334903c345cefa1ab691a3f7c708686d0ce717a`
- `comparisons.jsonl`: `9440b05c46469ad6ae2ceb42ee1d7965d5188f24d7e76fa3003b26fa96b8c690`
- `summary.json`: `275c45074766a5c1cfb517bb31929176df790f1164822f960cbc4c42fb9e6a1f`
- Selected 5,012 result rows: `2b3354e417e15776f8ed779286874995912357ed302aad4ad470d0b1487cf0ee`
- `REPORT.md` checked in this audit: `1067d13f536400d0fea7a098437d7c320b06f594c3258f619452e171887c4019`

## Replay changes absent from the reference

I independently counted final `result.entity_audio_repair.changes` whose normalized `canonical` string does not occur in the complete normalized reference. This yields 4, exactly matching `summary.json`, `comparisons.jsonl`, and the four report examples:

| ID | Observed → canonical | Reference context |
|---|---|---|
| leb1_38459bb96fdd7d36ecc1 | 董剑 → 董健 | Reference contains 佟健 |
| leb1_6c86c4a79561e0611e5e | 惊爆 → 金报 | Reference contains 惊爆 |
| leb1_8c2510722ec5a19f9656 | 终战 → 中传 | Reference contains 终站 |
| leb1_d39d7292a9660cc1eef6 | 冈商 → 冈山 | Reference contains 港商 |

These are reference-string mismatches, not independently adjudicated acoustic errors. The report correctly describes them as applied review changes and states they were not individually confirmed by listening.

## Report table cross-check

The overall and corpus table rows match `summary.json`, including hits, recall percentages, CER, mean/median/P90 total seconds and median tail time. The main-set and 96-sample SpeechIO subgroup figures also match.

The prompt partition matches: the frozen context16 contains 193 gold mentions, of which context gets 190 and audio 188; the 2,667 mentions outside it get 2,353 and 2,526, respectively. Entity-type rows match exactly:

| Type | Expected | Context hits / recall | Audio hits / recall |
|---|---:|---:|---:|
| LOC | 836 | 800 / 95.69% | 821 / 98.21% |
| ORG | 999 | 924 / 92.49% | 951 / 95.20% |
| PERSON | 683 | 487 / 71.30% | 607 / 88.87% |
| TERM | 342 | 332 / 97.08% | 335 / 97.95% |

All four per-domain tables and the paired totals shown in `REPORT.md` use the corresponding summary fields. Percentage formatting and seconds rounding are consistent. The CER and entity-hit percentages use annotated mention counts; they should not be read as identity resolution or general NER precision.

## CER limitation in the highlighted sample

For `leb1_6580ec42d85c81ab27be`, the reference says “杨扬拿到了五百米和一千米两项短道速的金牌”. After the benchmark’s conservative normalization, the context output retains “一千米”, while the audio-review output contains “1000米”; the normalized strings still differ. The scored character errors are 1 for context and 5 for audio, so this formatting difference contributes to the CER gap even though those number forms can be semantically equivalent.

This does not change the main entity-hit metric for this sample: its only gold entity is PERSON “杨扬”, while both outputs say “杨洋”, so both score 0/1 gold hits. The report now explicitly notes that conservative number normalization does not make every numeric form equivalent; that caveat is appropriate alongside the CER results.

## Scoring interpretation and report review

- Gold mentions are accumulated with `Counter(norm(canonical))`; repeated occurrences remain in the expected count and hit credit is capped at that count. Distinct overlapping canonical strings can still both receive mention credit; this is literal mention scoring rather than span-aligned detection.
- Both modes score the same delivered `result.clean_text`; `raw_text` is separate diagnostic scoring. Reference and both outputs use the same zh-ITN, OpenCC `t2s`, NFKC/casefold, and P/Z/C-removal pipeline.
- `dictionary_absent_mentions` counts output glossary matches only if that exact string appears nowhere in the reference. `dictionary_excess_mentions` separately catches excess matched counts for strings that may appear in the reference. Neither is a manually labeled hallucination rate.
- `refiner_calls` uses session-level logical call counts, which can include repetition review. Retry-call counts are separate from ASR replay `checks`; the report distinguishes these.
- Timing is serial WebSocket connect-to-final, including dictionary setup and fresh ASR/refinement. File read/decode, browser rendering and proxy time are excluded; no cached ASR response was used. The report gives mean, median and P90 seconds and describes the serial setting. `tail_seconds` is finish-to-final after audio acknowledgements; the total end-to-end metric is the relevant user-facing comparison.
- The report’s known-vocabulary, short-clip, source-reference, and no-protected-holdout caveats match the frozen evaluation setup. Its overall conclusion (audio gains 171 exact entity mentions, at +0.390 s average) agrees with independent totals.
