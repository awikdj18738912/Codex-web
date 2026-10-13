# Luna audit — round 4 terminal-gain collection

Read-only audit of the formal round 4 manifest, partition manifests and sources, collection logs, `recordings.jsonl`, and physical-execution `arms.jsonl`. No code, data, GPU process, or service was changed by this audit.

## Formal run and source coverage

`collection_job_state.json` is complete for `train_and_dev_counterfactuals_complete`. The official top-level manifest freezes all 818 train views and all 288 dev views from their round 3 ASR sources, with the source hashes recorded. Train is 441 clean plus 377 noise views; dev is 162 clean plus 126 noise. Each partition's `sources.jsonl` and `recordings.jsonl` have exact one-to-one coverage by `(recording_group, variant, pcm_sha256)`. The PCM hashes and per-view IDs are unique within each partition. Source metadata remains `synthetic_split=train` or `synthetic_split=dev` respectively.

The formal logs reach `GAIN 818/818` and `GAIN 288/288`. Every source view was run, including views with no native window REFINE opportunity; those still provide coverage controls. The superseded preparation is outside the official output tree at `results/gate_terminal_gain_round4_20261012_superseded_native_filter/`; it contains preparation manifests/source snapshots, but no `recordings.jsonl`, `arms.jsonl`, or collection logs. Its filtered views were not included in the formal run. The official manifest states `pilot_reexecuted_not_merged: true`: the previous pilot's 36 recordings are part of the original train views and are re-executed once here, not appended as 36 extra source rows or counted as fresh recordings.

## Native opportunities and paired execution

Native opportunities were discovered in the actual finalizer after replaying each frozen ASR final response. The prior case files' prepared opportunities were not used to filter recordings. Train had 292 actual native REFINE input occurrences across 291 recording views (290 views with one and one view with two); dev had 143 occurrences across 143 views. This differs from the train manifest's 291 prepared opportunity views and illustrates why that earlier filter must not define the collection.

The official arms contain 1,929 physical train snapshots and 719 dev snapshots. Train has 818 native, 818 native-repeat controls, 292 single-signature veto arms, and one physical all-window `skip_all` arm for the multi-signature view. The other train `skip_all` values reuse an existing single-veto/native execution and are not extra physical calls. Dev has 288 native, 288 native-repeat controls, and 143 single-signature veto arms; no extra physical all-window arm was needed. The physical `arms.jsonl` records 885 Refiner calls and 109 Plausibility calls for train, and 417 Refiner calls and 2 Plausibility calls for dev. These include the controls and counterfactual executions and should not be read as per-recording or online call totals.

All native-repeat controls were stable (818/818 train, 288/288 dev). Every arm's final raw text matched the frozen source ASR text. For each counterfactual, the native route-key set was preserved and only the targeted native REFINE key changed to KEEP; other window actions remained native. I found no route, raw-input, split, or view-coverage mismatches.

## Train/dev and test separation

Train and dev have disjoint template groups, recording groups, and PCM hashes. Actual native-window input signatures (`task_kind`, domain, raw text, baseline text, available context) had 131 distinct train signatures and 100 dev signatures, with no intersection. Across all recorded arms, exact Refiner model-call input signatures and Plausibility input signatures also had zero train/dev overlap.

The official manifest carries the consumed test-template exclusions. Neither train nor dev sources contain an excluded template. The dev partition remains development data; it includes reused validation/risk material and must not be called a new independent test. All labels derived from train recordings belong only to the train partition, and dev outcomes must remain dev-only if used for development checks.

## Weak-label scope

Terminal outcome labels are paired script-distance proxies against unverified synthetic scripts. Train had 244 neutral and 48 benefit window outcomes; dev had 129 neutral and 14 benefit outcomes. These counts are not human labels, audio verification, or real-speech quality estimates. The run is a cold actual-WebSocket-finalizer snapshot with frozen recorded ASR input; it does not provide live ASR or end-to-end streaming latency evidence.

## Linear prototype and development selection (before callback replay)

The assembled training file contains 292 train-only window rows and exactly matches `train_all.jsonl`; all rows carry `split=train`. Their source labels are 48 benefit and 244 neutral, mapped to 119 `refine_required` and 173 `skip_allowed` targets for the cost-aware model. The separate dev file has 143 rows, all `split=dev`, with 14 benefit and 129 neutral. Exact train/dev input-signature overlap and conflicting train-target exclusions are both zero. The 37 training rows tagged `seen_in_pilot` belong to 36 already-known pilot source views (one view has two window rows); they are original train examples, not 37 new recordings or appended pilot outcomes.

The training code fits each model only on `train_rows`. Development rows are used for scoring, threshold selection, and the explicit no-benefit-loss/no-degraded-view criterion. Feature construction reads only the pre-action `input` fields (`task_kind`, domain, raw text, rule baseline, and prior context); it does not read labels, action targets, terminal outcomes, costs, references, or recording metadata. Training weights normalize by recording group and template group. The two feature sets are character TF-IDF 1–3 grams with logistic regression: `terminal_quality` uses benefit versus neutral, and `conservative_cost` uses the explicit refine/skip target.

Model SHA-256 values match both `model.pkl` files, the corresponding model metadata, and the copies embedded in `selection.json`; each metadata file points to the exact dataset-manifest hash. The selected primary is `terminal_quality` at threshold 0.8; `conservative_cost` is an alternate at 0.7. Both metadata records say `calibrated: false` and `online_enabled: false`. Their scores must not be described as calibrated confidence.

On this reused development set, the 0.8 selection retains all 14 weak-proxy beneficial windows (11 old-validation, 2 new-dev, 1 risk-dev), has zero script-distance-degraded views, and vetoes 15 of 143 windows. The selected arm reconstruction reports 185 → 166 Refiner calls, with exact script 206/288 and intended micro-CER 0.0438267 unchanged. Offline CPU scoring was about 0.647 ms mean and 0.679 ms p95 per scored window; these are prototype scoring timings, not live service latency. This selection is based on previously used development material and unverified synthesis scripts, so it is neither a fresh independent test nor proof of quality generalization. The actual callback replay was completed separately; its checks and measurements are recorded below.


## Actual classifier-callback cold finalizer replay

The separate `linear_replay` run exercised the selected `terminal_quality` classifier callback inside the WebSocket finalizer for all 288 frozen dev ASR final responses. The replay configuration pins the selected policy hash, model pickle hash, Refiner weights, runtime protocol, source-row hash, and adapter hash; it records `torch.float32`, native local policy, fresh per-view caches, confidence disabled, `independent_test: false`, and `online_latency: false`. Only `window_refine` is routed through the classifier; local review and terminal guards remain native. The run did not send live ASR audio or test a live ASR connection.

I independently reconstructed the selected branch for each dev view from the frozen development prediction and executed counterfactual arms, then compared its final string/distance to the callback output. All 288 outputs matched their executed branch exactly, matching the replay's `expected_branch_verified` flag for all 288. The gate was called on 143 actual native REFINE windows; it vetoed 15 and retained 128. All 14 weak-proxy beneficial windows were retained. Against the native finalizer, 0/288 views were degraded by the script-distance proxy; exact script remained 206/288 and intended micro-CER 0.0438266967. Refiner calls were 166 and Plausibility calls 1. These results are unchanged from the selected-arm reconstruction, now verified through the actual callback path.

The classifier's measured scoring interval total was 218.985 ms across 143 callback invocations: mean 1.531 ms, p95 1.705 ms, and max 16.009 ms per invocation. The offline scorer's 0.647 ms mean is a separate measurement and must not be substituted for the callback figure. The callback interval is still only a model-scoring measurement inside a cold frozen-ASR final snapshot; it excludes live ASR and paced audio delivery and is not end-to-end or production online latency.

This replay uses the same reused development rows that selected the model and threshold, including old-validation, new-dev, and risk-dev material. It confirms implementation-to-executed-branch consistency on that dev corpus, not independent generalization. The scripts remain unverified and human gold is absent; the zero-degradation and benefit-retention findings remain weak-proxy development observations.
