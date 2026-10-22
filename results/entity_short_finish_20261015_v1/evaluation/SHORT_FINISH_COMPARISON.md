# 短音频收尾复核对照

报告状态：partial（中期子集）。

旧 run：`/home/aim0/桌面/tan/ASR/Codex-web/results/entity_challenge_scene_eval_20261014_v2`  
新 run：`results/entity_short_finish_20261015_v1/evaluation`  
共同样本：315/2000；新旧 primary 文本不同：0 条。
全量比较要求严格 2000 个唯一 ID；`--allow-partial` 只允许新 run 是旧 ID 的子集，仍拒绝重复或额外 ID。

全量运行是重新执行的评测，使用同一固定已知词库；它不是独立 blind 测试。primary 文本存在漂移时，下面同时报告相同 primary 文本子集，便于区分复核收益和首轮 ASR 差异。

| 时长组 | 样本 | 保存基线命中/提及 | 保存基线召回 | 保存基线 CER | 旧 audio_raw 命中/提及 | 旧召回 | 旧 CER | 新 audio_raw 命中/提及 | 新召回 | 新 CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全量 | 315 | 160/369 | 43.36% | 7.40% | 191/369 | 51.76% | 6.96% | 309/369 | 83.74% | 5.11% |
| <=4秒 | 86 | 49/93 | 52.69% | 9.14% | 49/93 | 52.69% | 9.14% | 76/93 | 81.72% | 6.27% |
| (4,8]秒 | 164 | 78/198 | 39.39% | 7.69% | 100/198 | 50.51% | 7.12% | 169/198 | 85.35% | 5.14% |
| >8秒 | 65 | 33/78 | 42.31% | 5.98% | 42/78 | 53.85% | 5.55% | 64/78 | 82.05% | 4.45% |

primary 文本相同的配对子集为 315 条：旧 audio_raw 191/369 命中、CER 6.96%；新 audio_raw 309/369 命中、CER 5.11%。

相对旧 audio_raw 的逐样本变化：实体改善/退化样本 107/0，CER 改善/退化样本 107/0；gold 逐词截顶后恢复/丢失 118/0 次。

新 run 收尾检查覆盖 153 条；超时样本 0，错误样本 0。只累计检查/变化/超时/错误/耗时/candidate/replay 计数：`{"short_finish_checks": 153.0, "short_finish_changes": 118.0, "short_finish_timed_out": 0.0, "short_finish_errors": 0.0, "short_finish_wall_seconds": 47.590266838204116, "short_finish_candidates": 926.0, "short_finish_replay_seconds": 880.1911875}`；配置值单列：`{"short_finish_max_seconds": 16.0, "short_finish_wait_seconds": 4.0}`。新 results 覆盖共同 comparison ID 315/315。

收尾耗时分布按 snapshot.incremental.short_finish_wall_seconds 汇总：

- 全部共同样本：sum 47.590s，mean 0.151s，p50 0.021s，p95 0.381s，max 1.500s。
- checks > 0 样本：sum 46.351s，mean 0.303s，p50 0.280s，p95 0.561s，max 1.500s。

false edit audit：旧 run 0 条候选样本/0 个候选，新 run 0 条候选样本/0 个候选。候选是自动审计结果，不代表人工确认。

这里统计的是收尾步骤 wall time，不等于浏览器端用户延迟；正式服务尚未重启。明细见 `short_finish_comparison.json` 和 `short_finish_sample_comparison.csv`。
