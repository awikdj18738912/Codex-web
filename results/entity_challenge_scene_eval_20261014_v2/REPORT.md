# 实体挑战评分报告

本轮仅修正混合专名词库。727条真实重测，其他1,273条在源码、PCM、完整词库字节相同时复用；完整对照见 `LIBRARY_COMPARISON.md`，方法、场景标注限制和清单元数据变更记录见 `REVIEW.md`。

状态：**完整结果**。本次评分 2000/2000 个唯一 ID。

这是离线评分。文本归一化顺序为 ZhITNNormalizer、OpenCC t2s、NFKC、casefold、删除 Unicode P/Z/C。先归一化并合并相同 gold canonical，再以非重叠 str.count 计数，命中数按 gold 次数截顶；CER 为归一化字符串上的字符编辑距离。

| 评分臂 | 覆盖样本 | 实体命中 / gold | 实体召回率 | CER |
|---|---:|---:|---:|---:|
| 保存的基线 ASR | 2000 | 1342 / 2658 | 50.49% | 6.63% |
| 配对首轮 ASR | 2000 | 1342 / 2658 | 50.49% | 6.63% |
| 实体音频修复结果 | 2000 | 1440 / 2658 | 54.18% | 6.38% |
| 保存基线的 prepare_entity_segment 规则诊断 | 2000 | 1342 / 2658 | 50.49% | 6.63% |
| 原音修复后的 prepare_entity_segment 规则诊断 | 2000 | 1440 / 2658 | 54.18% | 6.38% |

音频修复配对首轮结果：实体召回率从 50.49% 到 54.18%，变化 +3.69 个百分点；按 gold 次数截顶后新增恢复 98 个提及、丢失 0 个提及。改善/退化样本数分别为实体 94/0、CER 93/0。
覆盖：音频主臂 2000/2000；保存规则诊断 2000/2000；当前规则诊断 2000/2000。规则诊断结果来自 prepare_entity_segment 输出，仅作独立规则诊断；未运行 Refiner 或完整 Web，不能视为完整流程质量。
primary 与保存 ASR 文本不同的样本 0/2000。音频复核检查 326 次（297 个样本），修改 99 次（94 个样本）；row.timing.total_seconds 合计 2327.20 秒。PCM replay 合计 2129.96 秒，replay wall 合计 98.26 秒；incremental.errors 合计 0。模块错误记录 0。

增量音频复核计数：ambiguous_rejected=2077、budget_rejected=0、cache_reused=0、candidates=8154、completed_windows=326、detections=2609、late_candidates_kept=15、low_score_rejected=3961、oversized_rejected=0、queue_rejected=0、unmapped=0
分阶段保守误改候选：0 个样本。候选基于输出新增的库 canonical 且标准 reference 不支持；这只是自动筛查，没有人工复核结论。

详细结果见 `domain_summary.csv`、`sample_comparison.csv`、`comparisons.jsonl`、`summary.json` 和 `scoring_verification.json`。
