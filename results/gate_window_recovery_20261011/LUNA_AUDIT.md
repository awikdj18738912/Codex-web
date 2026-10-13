# Luna audit — window recovery and fresh streaming

Read-only audit of `regression/summary.json`, `fresh_streaming/summary.json`, selected session events, the frozen fresh-bank manifest, and `system/repetition_preservation.py`. No model or service was started or changed.

## Consumed 32-case regression

The prior 32×2 WebSocket/HTTP regression is a regression set, not a new independent test. Current and learned had identical final text in all 32 pairs, exact intended script 23/32 each, and CER 0.0263157895 each. Calls were 24 vs 23 Refiner calls; PPL counts were 3 each. The four previously known cases 002/008/014/023 were restored in both policies. Case 029's distant named repetition was restored in both. Preserve the current-policy case 031 finish-tail latency outlier: 31.92394 s versus learned 1.64270 s. Its model calls and drain were each sub-second; because chunk ACKs continued until about 44.5 s and the client paced chunks without waiting for each ACK, finish-tail includes queued stream/ASR work and is not a pure refiner latency. Do not hide the single outlier behind p95 or attribute it solely to the gate.

## Fresh 20-case streaming diagnostic

`fresh_streaming/summary.json` is complete for 20 sessions per policy. The summary reports 20 equal raw-ASR pairs and `differences: []`. Both policies had 13/20 exact intended-script matches and intended micro-CER 0.0448877805. However, direct exact comparison of `sessions.jsonl` shows one final-text difference, fresh_015: current outputs `明白，明白。需要逐字记录`, learned outputs `明白，明白，需要逐字记录`. This same punctuation-only difference is present in v2 below, so the summary difference field does not represent exact final-string equality. Refiner calls were 16 vs 14, plausibility calls 2 each; median finish tails were 0.390 s vs 0.347 s and p95 0.738 s vs 0.727 s. These are synthetic streaming diagnostics against unverified synthesis scripts, not human-gold or audio-verified quality results. The bank's manifest marks `audio_script_verified: false`; do not claim the audio was heard or that the scripts were confirmed by listening. The bank is now consumed by this streaming comparison and must not be presented as a fresh test after any code change.

### Confirmed named-cue/punctuation failure

The source matcher in `system/repetition_preservation.py` allows quotes and whitespace between a named repetition and its verb, but not an ASR-inserted comma. Static evaluation of the exact ASR text for fresh_019 shows `protected_repetitions()` protects `确认，确认` as `explicit_repetition_intent`, rather than the earlier named target `已经备案，已经备案`.

This is visible in the actual streaming trace: spoken script contains `已经备案，已经备案。接着确认确认货位。`; raw ASR is `已经备案，已经备案。接着确认，确认货位资料交给仓库主管。前面的已经备案，说了两遍，需要原样保留。`; both policies finalize as `已经备案。接着确认。确认货位资料交给仓库主管。前面的已经备案，说了两遍，需要原样保留。`. The first two source spans are committed as `已经备案，已经备案。接着确认，` → `已经备案。接着确认。`; no repetition restoration is recorded. Thus the current source-span/windowing path exposes a concrete failure: the late instruction is outside the refined window, while comma-blocked named matching lets its generic `说了两遍` cue attach to the nearer, unrelated confirmation repetition. This is an engineering defect in text association, independent of whether the learned gate is used; both policies produce the same final text.

Fresh_017's comma-form named cue also fails the named regex, but its only relevant repeated unit is still restored by the generic cue, so it does not demonstrate this wrong-target outcome. The comma case should be included in a post-fix regression, while this bank remains consumed and cannot validate the fix as unseen data.

## Scope limits

The fresh summaries say `references: unverified_synthesis_scripts`; exact-script and CER figures are comparisons to those scripts only. Equal ASR across policies establishes a paired input comparison, not correctness of the transcript. This audit makes no claim that weak labels are factual errors or that a listener verified the recordings.


## Consumed v2 punctuation regression

After adding comma/colon tolerance and preventing a named-reference sentence from falling back to an unrelated nearest repetition, a paired replay ran against the same 20 audio cases. The configuration explicitly marks `data_role: consumed_regression`; these results are a regression check of the patch, not an independent evaluation. The summary reports 20 equal raw-ASR pairs, 13/20 exact script matches per policy, identical intended micro-CER 0.0349127182, 17 vs 15 Refiner calls, 2 plausibility calls each, and zero failed Refiner calls. Median finish tails were 0.483 s vs 0.419 s; p95 1.139 s vs 1.077 s. As in v1, direct session-row comparison finds fresh_015 differs in punctuation between policies despite summary `differences: []`; exact output equality should not be claimed. The v2 CER is lower than v1's 0.0448877805 on the same consumed bank, but this repeated bank does not establish a general quality gain.

Fresh_019 now restores the named target `已经备案，已经备案` in both policies. Its terminal restoration record targets `已经备案，`, and final source-span metadata covers the full transcript. The second repeated phrase is still punctuated differently from the spoken script: output has `接着确认。确认货位` where the script has `接着确认确认货位`. Its character distance improves from 6 in v1 to 2 in v2; describe this as recovery of the named repetition with a remaining local discrepancy, not a fully corrected sentence. The four other punctuation probes and remaining sessions completed without Refiner failures; do not infer broader protection from this small, consumed set.
