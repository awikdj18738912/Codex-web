# Entity streaming preflight V6 report

## Decision

V6 is not approved for release. It preserved the 9 independent groups' entity hits and produced a small descriptive reduction in mean total time, and it preserved the three sparse long-stress hits while retaining the Li Feiyu correction. The five known-dense groups lost one entity hit and gained one character error. That regression is already present in the same-session primary ASR output, before entity processing; it is not evidence of a false entity replacement, but the group does not pass. All results remain in this private experiment; the formal entity module and database were not changed or deployed.

## Full-Web results

Entity scores are canonical mentions hit / annotated mentions. CER is calculated on normalized clean text. Timing is mean seconds per sample within each group.

| V6 group | Old → V6 entity hits | Old → V6 CER | Old → V6 mean total (s) | Old → V6 mean tail (s) | Score and run data |
|---|---:|---:|---:|---:|---|
| 9 independent concatenation groups | 76/82 → 76/82 | 0.03250 → 0.03250 (44/1,354 errors both) | 13.3067 → 13.0160 | 3.9943 → 3.3513 | [score](independent_score.json) · [results](independent_results.jsonl) · [protocol](independent_protocol.json) |
| 5 known-dense concatenation groups | 37/40 → 36/40 | 0.02916 → 0.03038 (24 → 25 / 823 errors) | 13.2474 → 13.8672 | 4.2299 → 4.1795 | [score](known_dense_score.json) · [results](known_dense_results.jsonl) · [protocol](known_dense_protocol.json) |
| 3 sparse long-stress clips | 3/3 → 3/3 | 0.11811 → 0.11811 | 21.7933 → 19.1924 | 0.8766 → 0.4200 | [score](long_score.json) · [results](long_results.jsonl) · [protocol](long_protocol.json) |

The Li Feiyu development recording is reported separately because it has no complete gold reference. The target was absent from old final raw text and present in V6 final raw text; the first `updates.raw_text` containing 厉飞宇 arrived at 11.221 seconds, after 76 of 323 frames. Its total time was 50.470 → 41.382 seconds and tail was 1.936 → 1.336 seconds. No CER or full-transcript accuracy is claimed for this recording; see the [long score](long_score.json) and [results](long_results.jsonl).

For the nine independent groups, same-session primary ASR scored 68/82 mentions with 54 character errors in both arms. Final raw text scored 76/82 with 42 errors in both arms; final clean text scored 76/82 with 44 errors. The raw-versus-primary gain is therefore shared by old and V6, while clean includes the rest of the Web refinement pipeline. The independent group’s old/V6 scores are unchanged; the mean total-time difference is small and descriptive.

In known-dense `news_group_1`, primary ASR already scored 11/11 for old and 10/11 for V6; final raw retained the same gap. The paired score records that raw hit and error differences were already present in `primary_finish_text`, and both rows have zero confirmed entity changes. This is an upstream transcription difference, not a demonstrated entity-repair misreplacement. It still leaves known-dense performance below the old arm, so V6 is not cleared.

## V6 workload policy and limits

The preflight requires at least 45 seconds of received audio and two stable segments, permits at most two planned windows, and admits only plans expected to save at least 40% of received audio. It reuses cached selector results when the same first-pass text is seen again. These checks reduce unnecessary review work, but the known-dense entity regression remains a release blocker.

The 36 sessions are 18 paired local WebSocket runs across the three groups above. PCM was sent directly through the local WebSocket collector; this is not browser transport. The concatenation groups are assembled from existing source clips, not natural continuous long recordings or wholly new unseen audio. The Li row is an existing 322.7-second development recording.

The evaluation backend reports PCM input amounts, HTTP request counts, active requests, and measured wall time. HTTP request counts are not GPU model-call counts. Primary/replay wall values cover completed measured HTTP handlers and may omit unfinished requests at snapshot time; do not interpret them as complete GPU compute time. Compare timing only within the same run and protocol.

The next planned experiment changes only the replay-review decoding interval to 10 seconds. It will keep first-pass ASR and entity verification unchanged; no result is available yet. This plan does not constitute a V6 rollout recommendation.
