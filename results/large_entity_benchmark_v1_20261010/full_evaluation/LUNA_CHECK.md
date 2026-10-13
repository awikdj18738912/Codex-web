# Full evaluation verification

Read-only cross-check of `summary.json`, `extended_audit.json`, `REPORT.md`, `manifest.json`, and service recovery records.

## Completeness and uniqueness

- `audio.jsonl`: 2,410 rows and 2,410 unique IDs.
- `web.jsonl`: 7,230 rows and 7,230 unique `(id, condition)` pairs, with 2,410 rows each for `baseline`, `entity_rules`, and `entity_audio_repair`.
- All Web IDs correspond to an audio ID; all three conditions are present for each audio ID.
- Collection resumed at the recorded checkpoint of 2,074 audio rows. The restored ASR health configuration is recorded as matching the smoke evaluation.

## Summary table cross-check

| Condition | Hits | CER | Text refinement calls (`refine*`) | Session calls (includes repetition review) |
|---|---:|---:|---:|---:|
| baseline | 2431/2745 | 2.71% | 111 | 265 |
| entity_rules | 2433/2745 | 2.72% | 278 | 430 |
| entity_audio_repair | 2603/2745 | 2.16% | 187 | 340 |

All three rows match both `summary.json` and the aggregate table in `REPORT.md` (CER rounded to two decimal places). `entity_rules` to `entity_audio_repair` reports 167 improved and 3 regressed samples in the summary, extended audit, and report. The audit also records 2 changed samples with equal error counts and 4 new absent-surface matches; these details are consistent with the report's change table.

Text refinement counts sum to 576; session-wide Refiner calls sum to 1,035 and match `verification.json`. These are different scopes, not a cache/reuse discrepancy: `_RefinerSessionStats.begin_call` counts both text refinement and repetition-review invocations, while the evaluation harness wraps only callable method names beginning with `refine`, and the audit counts that narrower `model_calls` list. The `review_repetition*` methods are actual Refiner calls but are not captured in that list. For example, Web session `leb1_04662ea0d92b98d5702d` (`entity_audio_repair`) has `session_refiner_call_count=2`, `repetition_review_model_calls=2`, and an empty `model_calls` list. The final event's repetition-review count covers that event; the session counter also includes calls from earlier streaming windows.

## Database and service recovery

- Formal entity database SHA-256, recomputed from the manifest path: `a45a6473d81608b63f30f7270c137c027814ebc179d1858e01d7e2572caf8a1d`. It matches the manifest and audit hashes; the audit records the formal database, dataset manifest, local library, and source snapshot as unchanged.
- `service_restoration.json` records port 8081, PID 1141118, and a successful health response. The recorded PID is alive and the recorded health response matches a fresh `/health` response.
- Fresh health checks for ports 8051, 8081, and 8766 all returned `ok: true`, consistent with the extended audit's service health record.

## Outstanding items

No missing evaluation or audit artifacts were found. The report separates text refinement counts from session-wide Refiner calls to make the accounting scopes explicit.
