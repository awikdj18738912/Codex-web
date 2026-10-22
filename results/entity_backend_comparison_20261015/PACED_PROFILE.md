# Paced entity backend profile

Status: **complete**; 32/32 records across 8/8 fixed IDs.

The run used `realtime_160ms_wallclock_paced`. Every arm was checked for identical per-ID PCM hashes and selected-parameter hashes. Chunk submits were checked against audio availability with a 2 ms numeric tolerance. Qwen context had to remain unapplied.

| Arm | N | First text wall median (s) | Tail after last audio submit median / max (s) | Total wall − audio median (s) |
|---|---:|---:|---:|---:|
| qwen_raw | 8 | 2.186 | 0.154 / 0.371 | 0.154 |
| qwen_ctc_sidecar | 8 | 2.219 | 0.178 / 0.460 | 0.179 |
| native_hotword | 8 | 0.854 | 0.117 / 0.165 | 0.119 |
| native_broad_hotword | 8 | 0.982 | 0.042 / 0.065 | 0.044 |

## Hybrid versus raw Qwen paired finish tail

Paired samples: 8; hybrid minus raw tail median: 0.027 s; maximum absolute delta: 0.182 s.

## Compute and quality diagnostics

CTC and fusion compute summaries are in `paced_profile_summary.json`; quality metrics, if references were available, are marked small-sample diagnostics.

Eight samples are descriptive only. Their medians/maxima do not establish a service guarantee or a p95 latency. The hybrid total wall and finish tail include final fusion. No quality or latency result from this profile alone supports a broad deployment conclusion.
