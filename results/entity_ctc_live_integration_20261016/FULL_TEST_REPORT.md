# Live CTC + 当前原音复核：冻结 test 全量比较

冻结 test 的 1829 个唯一 ID 均已覆盖；逐臂核验 split 与 PCM SHA。实际 Qwen 首轮可能有轻微波动，off/on 相互规范化一致的样本才纳入严格配对收益；波动逐条记于 JSON。CTC bypass 固定使用 gap=-2、margin=0.1；不基于 test 结果调参。

## 六臂质量指标

| Arm | Samples | Entity hits / mentions | Recall | CER | Review checks | Short-finish checks | CTC committed | CTC errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen raw | 1829 | 1224 / 2467 | 49.61% | 6.76% | — | — | — | — |
| CTC bypass (gap=-2, margin=0.1) | 1829 | 1675 / 2467 | 67.90% | 5.43% | — | — | — | — |
| Archived old paired primary | 1829 | 1224 / 2467 | 49.61% | 6.76% | — | — | — | — |
| Archived old entity audio review | 1829 | 1314 / 2467 | 53.26% | 6.50% | 302 | None | None | None |
| Current short-finish review (CTC off) | 1829 | 1957 / 2467 | 79.33% | 4.52% | 1413 | 1166 | 0 | 0 |
| Current CTC + audio review (CTC on) | 1829 | 2037 / 2467 | 82.57% | 4.33% | 1440 | 1173 | 83 | 0 |

## CTC on 相对当前 off

按 canonical 和 gold 提及次数上限裁剪：新增命中 85 个实体提及、丢失 5 个提及，净增 80；CER 改善 79 条音频、退化 19 条。首轮规范化一致的配对样本 1827 / 1829。提及变化不是修改精确率或误改次数。逐 ID delta 见 `full_test_comparison.json`。

Qwen 首轮与保存基线逐字不一致：off 4 条、on 4 条；off/on 首轮规范化后仍不一致 2 条，已从严格配对增益中排除。

严格配对分母：2465 个实体提及、42717 个参考字符；off/on 实体命中分别为 1956 / 2036，字符错误分别为 1927 / 1847。

按 582 个来源组有放回重采样 1000 次，命中率差值的 95% 百分位区间为 +2.36 至 +4.29 个百分点；CER 差值为 -0.272 至 -0.117 个百分点。该区间不包含单次服务运行的随机性。

丢失的 5 个提及来自 4 条音频；其中 4 条的 CTC 提案数为零，不能直接归因为错误的 CTC 替换。复核输出与窗口调度的差异仍可能影响结果，需另行固定复核响应或重复运行来隔离。

## 另行引用的控制结果

先前 `results/entity_ctc_live_integration_20261016/comparison.json` 报告含 330 条合并质量样本，其中 controls 100 条；此前组合 control CER=0.010744985673352435。这组控制来自先前组合评测，数据、cohort 与口径均单独保留，不并入本次 1829 条 full-test，也不是本次 full-test off/on 的 control 对照。

实体 recall 使用规范化 canonical 子串计数，无实体边界或上下文判断，是词面命中指标；分母是 gold mentions 总数。CER 分母是规范化 reference 字符数。Review checks 是日志中的复核检查计数，单次检查可能涉及多个流式请求；CTC errors=0 仅指错误计数器为零。

词库场景划分曾借助参考文本，属于已知场景条件，不能代表未知领域或自动领域路由。两臂各运行一次、加速提交音频，耗时不能视作交互延迟；严格配对仅排除了首轮文本漂移，没有固定后续复核响应与调度。归档旧模块的两个输出臂分别列示。
