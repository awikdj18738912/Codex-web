# Entity streaming budget experiment

This is a private, offline experiment. Its candidate, worker, plans, libraries, and collectors are isolated under this results directory; nothing here is wired into production. No V5 effect conclusion is recorded here.

## Data and scope

`frozen_test_plan.json` freezes 180 original short recordings: 60 development and 120 confirmation. The root `collect.py` writes `development_results.jsonl` / `confirmation_results.jsonl`; archived V2 development output is under `v2_development/`. V1 and V2 use the short-recording transport profile described below.

`long_test_plan.json` describes three concatenated sparse-development stress clips and the 322.7-second Li Feiyu development recording. The three sparse clips have component references; Li has no complete reference and is a target-presence/streaming-latency case only. These concatenations are stress inputs, not natural long-form audio or new unseen recordings.

`confirmation_long_inputs/plan.json` describes five additional concatenations from existing confirmation recordings: three news groups of eight clips, one medical group of six, and one automotive group of six. It records every source component and timeline. Each composite has one second of leading and trailing zero PCM and three seconds between clips. The original full `library_key` is retained for each group, using its frozen full-domain SQLite library. These are also existing clips concatenated for stress testing, not new or unseen data.

## Reproduction

Run commands from the repository root. The existing local Web service must be available to the private worker; these commands are documentation only and are not started by this README.

```bash
PYTHONPATH=.:.runtime/functional_test_deps python -m pytest -q results/entity_streaming_budget_20261013/test_candidate.py
python results/entity_streaming_budget_20261013/start_worker.py

python results/entity_streaming_budget_20261013/collect.py --split development
python results/entity_streaming_budget_20261013/score_trial.py --split development --partial
python results/entity_streaming_budget_20261013/score_trial.py --split development

python results/entity_streaming_budget_20261013/collect.py --split confirmation
python results/entity_streaming_budget_20261013/score_trial.py --split confirmation --partial
python results/entity_streaming_budget_20261013/score_trial.py --split confirmation

python results/entity_streaming_budget_20261013/prepare_long.py
python results/entity_streaming_budget_20261013/collect_long.py --split long
python results/entity_streaming_budget_20261013/score_long.py --partial
python results/entity_streaming_budget_20261013/score_long.py

python results/entity_streaming_budget_20261013/collect_confirmation_long.py --split confirmation_long
python results/entity_streaming_budget_20261013/score_confirmation_long.py --partial
python results/entity_streaming_budget_20261013/score_confirmation_long.py
```

Collectors run `old` and `new` serially, with deterministic arm order per sample and otherwise the same settings. The original short-recording profile sends 64,000-byte frames with up to four outstanding and no artificial inter-frame delay. Long runs use one outstanding frame and an 80 ms gap after each acknowledged frame. Collection waits for each session's late cleanup to settle before starting the next sample.

Scoring scripts read references and gold annotations only after collection, offline. They are not imported by the worker or passed through the inference inputs. The confirmation-long scorer builds each reference by concatenating component references in plan order and adds their gold mention counters; these labels remain scorer-only.

## Frozen inputs and audit trail

`frozen_test_plan.json` records the short-sample selection, source input hashes, and library snapshot hashes. The ten SQLite snapshot hashes listed by that plan were cross-checked against the files; `library_metadata.json` identifies the experiment's complete frozen SQLite set and hashes. `confirmation_long_inputs/plan.json` records the source PCM hashes, derived PCM hashes, component order, duration, and group library key; its five derived files are under the same directory. The five derived PCM hashes were rechecked against their files. The long input plans record their PCM hashes as well.

Before upload, each collector checks the sample PCM against its planned SHA-256. The worker uses the frozen `source_snapshot/` and frozen library snapshots. Its `worker.json` records source-code hashes; each split protocol records those source hashes and the active `candidate.py` hash. In the archived V1–V4 protocols, the source snapshot and candidate hashes were cross-checked against the archived files. The offline scorers retain plan, reference, results, and protocol hash identifiers. The confirmation-long plan also maps each derived input to the collector sample ID, PCM hash, duration, and full library key.

## Version archive

- `v1_smoke/` is the initial short-run wiring smoke; it contains three development pairs, not the full planned set.
- `v2_development/` archives the full 60-pair short development run and its offline score.
- `v3_long/` archives the four-pair sparse/Li long stress run with the earlier incremental policy, including a 20-second local-window limit and a two-second single-request timeout.
- `v4_long/` archives the same four long inputs and transport profile with a 60-second window limit, a five-second single-request timeout, and no scheduling of a new background local replay during finish. Its result archive is separate from V3.

These labels describe experiment versions and their recorded settings; archived runs should be assessed with their own protocols and hashes. Do not compare isolated replay-time fields across batches as user latency.

## Timing interpretation

`replay_audio_seconds` is the module-reported amount of input audio sent to entity replay. It is not a measure of total GPU computation. Discarded shadow work or an EchoGuard fallback replay may not be fully represented by that field. `total_seconds` measures the WebSocket session through final; `settled_total_seconds` extends through the worker's quiescence check; `post_return_cleanup_seconds` is the cleanup interval after final. Use these timings to understand observed user/session latency, and compare runs only when their batch, input, worker, candidate, and transport protocols match. Do not compare isolated replay-time values across batches.
