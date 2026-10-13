# 门控成对试验初步报告

完成 22/22 个决策样本；人工已标注 0 条。

11 条 always 与 rules 最终输出逐字相同，可确定为相对操作无收益；它们不证明原始转录正确。

这是冻结 ASR 最终快照的真实 Refiner 对照；不是在线速度测试或独立泛化验收。

| 任务 | 策略 | 逻辑 Refiner 调用 | 模型成本累计（秒） | 相对规则输出变化 | 参考距离改善／退化（待核对） |
| --- | --- | ---: | ---: | ---: | ---: |
| window_refine | rules | 0 | 0.000 | 0 | 0 / 0 |
| window_refine | current | 0 | 0.000 | 0 | 0 / 0 |
| window_refine | always | 0 | 0.000 | 0 | 0 / 0 |
| local_edit_review | rules | 0 | 0.000 | 0 | 0 / 0 |
| local_edit_review | current | 22 | 1.717 | 11 | 0 / 11 |
| local_edit_review | always | 22 | 1.717 | 11 | 0 / 11 |

相同 current/always 操作共享一次真实执行，表内各策略成本是映射到该策略的实测操作成本，不能将各行相加当成实验墙钟。
语义、实体与数字误改需要人工复核；未标注前不能称为已验证改善率或训练收益数据。

样本来自 21 个来源；22 条有配套参考，音频路径可用 22/22。

## 当前策略仍可筛选的无效操作

| 任务 | 当前选中的决策 | 最终无变化 | 对应实测模型成本累计（秒） |
| --- | ---: | ---: | ---: |
| window_refine | 0 | 0 | 0.000 |
| local_edit_review | 22 | 11 | 0.788 |

上述最终无变化操作对应 0.788 秒累计模型时间。这是按结果回看的局部 oracle 空间，不代表学习型门控已取得同等收益。

剔除／重跑异常记录保留在 pairs_failed_*.jsonl；辅助模型在 CPU 暂存权重，评分仍使用原 CUDA 精度，转移开销单独记录，不作为正式服务稳态成本。
本轮值得继续审核变化样本，建立真实正收益与退化标签；不能仅凭无效调用比例宣布新门控能提升准确率或速度。

产物：protocol.json、cases.jsonl、pairs.jsonl、summary.json、annotations.csv、changed_for_review.csv、neutral_pairs.jsonl 和 ANNOTATION_GUIDE.md。
