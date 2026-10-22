# 实体挑战评分报告

完整2000条未完成：共享ASR无响应导致采集中断，当前仅315条已完成样本。实现、验证和运行限制见 `../REPORT.md`；不能把子集命中率外推为整套新分数。

状态：**部分结果**。本次评分 315/2000 个唯一 ID。

这是离线评分。文本归一化顺序为 ZhITNNormalizer、OpenCC t2s、NFKC、casefold、删除 Unicode P/Z/C。先归一化并合并相同 gold canonical，再以非重叠 str.count 计数，命中数按 gold 次数截顶；CER 为归一化字符串上的字符编辑距离。

| 评分臂 | 覆盖样本 | 实体命中 / gold | 实体召回率 | CER |
|---|---:|---:|---:|---:|
| 保存的基线 ASR | 315 | 160 / 369 | 43.36% | 7.40% |
| 配对首轮 ASR | 315 | 160 / 369 | 43.36% | 7.40% |
| 实体音频修复结果 | 315 | 309 / 369 | 83.74% | 5.11% |
| 保存基线的 prepare_entity_segment 规则诊断 | 315 | 160 / 369 | 43.36% | 7.40% |
| 原音修复后的 prepare_entity_segment 规则诊断 | 315 | 309 / 369 | 83.74% | 5.11% |

音频修复配对首轮结果：实体召回率从 43.36% 到 83.74%，变化 +40.38 个百分点；按 gold 次数截顶后新增恢复 149 个提及、丢失 0 个提及。改善/退化样本数分别为实体 133/0、CER 133/0。
覆盖：音频主臂 315/315；保存规则诊断 315/315；当前规则诊断 315/315。规则诊断结果来自 prepare_entity_segment 输出，仅作独立规则诊断；未运行 Refiner 或完整 Web，不能视为完整流程质量。
primary 与保存 ASR 文本不同的样本 0/315。音频复核检查 209 次（187 个样本），修改 149 次（133 个样本）；row.timing.total_seconds 合计 415.88 秒。PCM replay 合计 1221.17 秒，replay wall 合计 58.76 秒；incremental.errors 合计 0。模块错误记录 0。

增量音频复核计数：ambiguous_rejected=95、budget_rejected=0、cache_reused=0、candidates=562、completed_windows=209、detections=423、late_candidates_kept=0、low_score_rejected=821、oversized_rejected=0、queue_rejected=0、unmapped=0
分阶段保守误改候选：0 个样本。候选基于输出新增的库 canonical 且标准 reference 不支持；这只是自动筛查，没有人工复核结论。

详细结果见 `domain_summary.csv`、`sample_comparison.csv`、`comparisons.jsonl`、`summary.json` 和 `scoring_verification.json`。
