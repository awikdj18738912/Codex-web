# Luna audit — terminal-gain pilot

Read-only audit of the frozen manifest/sources, executed adapter, configuration, job state, run log, and final `recordings.jsonl` / `arms.jsonl`. No source/data changes or model runs were made by this audit.

## Frozen sample and evaluation scope

The run completed 36 recordings across 21 templates: filler 8, stutter 7, meaningful repeat 7, protected content 7, meaningful expand 7. All rows are `variant=clean`, `synthetic_split=train`; each has a PCM SHA-256. The manifest declares `human_gold: false`, `audio_script_verified: false`, and `new_training: false`. These are unverified synthesis scripts and ASR outputs, so the outcome labels below are weak script-distance proxies, not verified transcript quality.

This exercised the actual WebSocket finalizer using each frozen recorded ASR final payload. The adapter rejects other endpoints and does not send audio chunks or invoke ASR. It is a cold final-snapshot comparison, not an online streaming latency benchmark. Confidence was disabled; original Refiner, native local-review policy, and final-source preservation remained enabled. Caches were fresh per recording and arm.

The loaded Refiner was verified as `torch.float32`; the model weight SHA-256 was `1a9635c8…31d907d`. The adapter checks the loaded dtype and weight hashes before running. The source-row and adapter hashes are pinned in the manifest/configuration. No learned gate or training was used.

## Pairing and intervention checks

All 36 native repeat controls had the same final text and the same Refiner call inputs, outputs, and routing as their initial native run. Every arm's final `raw_text` matched its frozen source recording. There were 37 distinct native REFINE input signatures over the 36 recordings; the one recording with two distinct signatures received two separate single-signature vetoes. For each of the 37 counterfactual arms, I verified the same route-key set as native and exactly one changed key: the targeted native REFINE changed to KEEP, while other decisions stayed native. One additional `skip_all` arm was emitted for the multi-signature recording; it is an auxiliary all-window comparison and is not counted among the 37 single-signature outcomes.

The finalizer still ran its native local-review and terminal-protection paths after window vetoes. Thus each comparison measures the terminal result after a single window decision is vetoed, including any local-review fallback or extra local call that follows; it is not a direct comparison of isolated model outputs.

## Observed paired outcomes

Across the 37 single-signature comparisons, the weak label counts were 33 neutral, 4 benefit, and 0 harm. All four weak-proxy benefits came from filler windows where native refinement removed an ASR-inserted `嗯`; the vetoed final retained it. Seven counterfactuals changed the final string: four received a positive script-distance label, while three differed only in punctuation/format relative to native and remained neutral under the proxy. The 33 neutral labels therefore do not imply that all outputs were byte-identical.

Twenty-seven of 37 vetoes had a positive Refiner-compute-advantage proxy; ten did not, because the unchanged local pathway can replace or add work. The per-counterfactual `skip_saves_refiner_calls` distribution was 28 with one saved call, 6 with none, 2 with two, and 1 with three. In the six zero-saved-call cases, one native window Refiner call was replaced by one local repetition-review call: native used 8–17 Refiner forward passes, while local review used 2; recorded input-token positions were nevertheless 111–127 higher for local review. This is why neither call-count nor token-position proxies alone establish compute or latency savings.

The execution log (`arms.jsonl`) contains 110 snapshots, 96 Refiner calls, and 3 physical Plausibility API calls. For `expand_gate_20261011_00026`, `decide` returned `model_plausibility_remove` once in native, once in native-repeat, and once in the single-signature veto. The `skip_all` result aliases that same single-veto execution because `skip_all_physical_reused=true`; it is a strategy view, not a fourth API call. Thus native and skip-all strategies each show one PPL call for this recording, and the counterfactual correctly saves zero plausibility calls because the native local-finalizer path remains active. Decoder input-token positions are a compute proxy, not FLOPs; wall-clock model timings include staging and shared-GPU noise, and should not be reported as end-to-end or online latency.

The outcome supports only a narrow observation on these 36 synthetic train recordings: some native window refinements remove filler recognized by the script, while most single-window vetoes are neutral under that same unverified-script comparison. It does not establish general quality, real-speech behavior, or a deployment latency result.
