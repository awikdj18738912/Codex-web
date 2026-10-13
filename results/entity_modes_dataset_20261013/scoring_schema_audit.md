# Scoring schema and payload audit

Read-only audit of the frozen scoring sidecar, prior scoring conventions, the existing 24-sample large-benchmark smoke artifacts, and a snapshot of the current paired collection. No inference was run. The protected unseen set was not accessed. The formal collection is live, so the partial counts below are only the state observed during this audit.

## Frozen scoring inputs

`references.json` is an object with top-level `schema_version` and `samples`; `samples` maps each recording ID directly to its scoring data. It contains 2,506 IDs and 2,860 entity mentions. Every entity has `canonical`, zero-based half-open `start`/`end`, and `entity_type`; all 2,860 slices satisfy `reference[start:end] == canonical` in the stored scoring reference.

The 2,410 large-benchmark records have `reference`, `reference_original`, `gold_entities`, and a top-level `annotation_method` label (`mixed_source_annotations`). The gold entries retain their more informative per-entity fields, including `annotation_method`, `annotation_layer`, `topic`, `surface`, and `canonical_key`. The 96 SpeechIO pilot records have `reference`, `gold_entities`, and `annotation_method`; each gold entry has the four common fields. Pilot annotations are reference-curated, not official NER. Combined gold type counts are ORG 999, LOC 836, PERSON 683, TERM 342.

The predecessor source formats differ. Large-benchmark sources are two aligned arrays: `dataset/references.json[]` (`sample_id`, `reference`, `reference_original`) and `dataset/sample_entities.json[]` (`sample_id`, `cohort`, `gold_entities`). The reliable pilot source is a dictionary keyed by ID in `pilot/references.json`, with the same `reference` plus `gold_entities` and `annotation_method` fields as the combined sidecar. Scorers should use the combined `references.json` keyed by ID and `reference`; do not assume the original array or source-pilot shape.

## Existing scoring convention

The prior large-benchmark report code in `experiments/scripts/audit_large_entity_full_evaluation.py` normalizes both reference and final `clean_text` using zh-ITN normalization, OpenCC traditional-to-simplified conversion, then `system.entity_audio_repair.normalized` (NFKC, casefold, remove Unicode punctuation/separators/control characters). It computes character edit distance/CER after that normalization. Gold recall is exact canonical-string occurrence matching, capped by each gold occurrence count; its more detailed audit groups by `(entity_type, canonical)`. Main references—not `reference_original`—are the target text.

The reliable pilot report code in `experiments/scripts/evaluate_reliable_entity_audio.py` applies the same normalizer and compares each response’s `.text` field for `initial` versus `production_final`. It counts exact normalized canonical substring occurrences, capped by gold multiplicity, and uses normalized CER. For this combined run, preserve a common scoring rule across both corpora: use stored `reference`, normalize it and the selected final hypothesis identically, then compute CER and per-entity-type exact mention hits. Keep raw, unnormalized scores available only as a separately labeled diagnostic.

These are reference-agreement and literal-mention metrics. The large benchmark’s source audit explicitly does not establish per-item acoustic correctness or real-world identity. Exact string counting can miss valid aliases/variants and can miscount nested or overlapping gold terms; spans are useful for checking the reference annotation, but recognition scoring must still define a consistent surface/alias policy. Do not treat `reference_original` as an additional transcript target or double-count the same mention across fields.

## Final payload paths

For the current `results.jsonl`, each row is one completed arm/session. The useful paths are:

| Purpose | Path |
|---|---|
| Pair key and mode | `row.id`, `row.arm` (`context` or `audio`) |
| Corpus/cohort/library strata | `row.corpus`, `row.cohort`, `row.library_key`, `row.topics` |
| Input identity | `row.pcm_sha256`, `row.audio_seconds` |
| Primary delivered transcript | `row.result.clean_text` |
| Pre-clean final ASR text diagnostic | `row.result.raw_text` |
| Outer elapsed time | `row.timing.total_seconds` |
| Timing breakdown | `row.timing.session_setup_seconds`, `.audio_stream_seconds`, `.tail_seconds` |
| Context arm was requested/applied | `row.ready.entity_asr_context_requested`, `.entity_asr_context_applied`; final confirmation at `row.result.entity_asr_context_requested`, `.entity_asr_context_applied` |
| Context size/digest | `row.ready.entity_asr_context_term_count`, `.entity_asr_context_digest`; final count at `row.result.entity_asr_context_term_count` |
| Audio-review mode was requested | `row.ready.entity_audio_repair_requested`, and `row.result.entity_audio_repair_requested` |
| Final audio-review outcome | `row.result.entity_audio_repair.phase`, `.reason`, `.candidate_count`, `.checks`, `.replay_context_applied`, `.applied`, `.change_count`, `.changes`, `.elapsed_seconds`, `.replay_audio_seconds`, `.replay_windows`, `.replay_mode`, `.local_replay_reason`, `.streaming_replay.*` |
| Refiner retry diagnostics | `row.result.refiner_executed`, `.refiner_retry_count`, `.placeholder_retry_count`, `.refiner_retry_reasons` |
| Session call accounting | `row.result.refiner_session_stats.call_count`, `.initial_call_count`, `.retry_call_count`, `.completed_call_count`, `.failed_call_count`, `.completed_wall_latency_ms`; also `row.result.session_refiner_call_count`, `.session_refiner_initial_call_count`, `.session_refiner_retry_call_count`, `.session_refiner_intermediate_call_count`, `.session_refiner_final_call_count`, `.repetition_review_model_calls` |

The older large-benchmark smoke is at `results/large_entity_benchmark_v1_20261010/smoke_evaluation/`. Its `web.jsonl` final response is under `row.result.raw_text` and `row.result.clean_text`, with retries and request flags under the same `result.*` paths. Its separate `audio.jsonl` ASR responses use `row.initial.text` and `row.production_final.text`; these are not the Web final payload schema. The smoke conditions set entity ASR context off for all 24 samples, so it checks the payload shape but does not validate the new context arm. `web.jsonl` includes `row.model_calls[]` and `row.elapsed_seconds`; `audio.jsonl` includes `row.actual_asr_seconds`, and repair diagnostics at `row.production_final.entity_audio_repair.*`.

## Interpretation and pitfalls

- Use `clean_text` as the primary final-output score. `raw_text` is useful for identifying downstream cleanup/refinement changes; mixing `raw_text` for one arm with `clean_text` for another changes the treatment being scored.
- Requested is not applied. In the current snapshot, every context row had context requested and applied; every audio row requested repair. Yet the audio repair can finish with `reason=no_candidates` or `weak_candidates`, `checks=0`, and no replay. Score the assigned arm by intention-to-treat, and separately report how many sessions actually replayed/changed text. Do not condition the primary arm comparison on successful replay.
- `ready.entity_audio_repair` is an early session snapshot (often `phase=capturing`); use `result.entity_audio_repair` for final outcome. `checks`, `replay_context_applied`, `applied`, `change_count`, and `changes` are distinct: a replay/check may occur without a text change.
- Retries are not total model calls. `refiner_retry_count` and `placeholder_retry_count` are retry diagnostics. `session_refiner_call_count` is broader; repetition-review calls have their own `repetition_review_model_calls`, and the old audit documented that a `model_calls[]` trace of `refine*` methods can omit repetition-review model calls. Report refiner retries, total/session calls, and audio replay checks separately. A zero `refiner_executed` does not imply no ASR work.
- For speed, use the outer `timing.total_seconds` as the paired Web-session wall time. The frozen protocol says this is local connect-to-final, serial one-sample concurrency, including dictionary setup and fresh ASR/refinement; it excludes file read/decode and browser/user-proxy time. Report the timing breakdown and paired median/tail distribution. Do not use only `entity_audio_repair.elapsed_seconds` as the end-to-end audio-review cost, or infer that zero `replay_audio_seconds` means the mode had no processing overhead.
- Pair rows by `id`, and confirm there is exactly one row per arm and identical `pcm_sha256`; do not treat the two arms as independent examples. Keep corpus, cohort, and topic results stratified, particularly the 300 benchmark negatives and 32 pilot no-registered-target controls.
- The old 24-sample smoke compared `baseline`, `entity_rules`, and `entity_audio_repair`; it was not a glossary-context versus audio-review comparison. Its `entity_asr_context_requested/applied` values are false in all arms.

At audit time, the current append-only `results.jsonl` had 88 rows / 44 paired recordings out of 5,012 planned sessions. This is an in-progress operational snapshot, not a final accuracy or latency estimate. The audio arm had 14 checks/replay contexts and zero applied text changes in those 44 pairs; the remainder had no check. These counts can change as collection continues.
