# CTC 与原音复核组合评测

冻结配对 cohort：short_test 230 条，control 100 条。输入 ID、PCM hash、保存 Qwen 主流文本及实际组合输出均已严格核对。

## short_test 四臂

| Arm | 样本数 | Entity hits / mentions | Recall | CER |
|---|---:|---:|---:|---:|
| Qwen raw | 230 | 100 / 280 | 35.71% | 8.53% |
| CTC bypass | 230 | 145 / 280 | 51.79% | 7.56% |
| 当前短收尾 | 230 | 228 / 280 | 81.43% | 5.81% |
| 实际组合 | 230 | 235 / 280 | 83.93% | 5.69% |

## Control CER

| Arm | 样本数 | CER |
|---|---:|---:|
| Qwen raw | 100 | 1.07% |
| 实际组合 | 100 | 1.07% |

## 组合相对当前短收尾

配对 230 条；gold entity recovered 7，lost 0；CER 改善 7 条、退化 0 条。

原音复核 checks/session 计数合计 201；CTC proposals 48、committed 7、errors 0、finish timeout 0。短收尾复核 checks 155、errors 0、timeouts 0。

发现 0 个新增词库 canonical 自动审计候选。该候选表未经过人工复核，不代表 precision。逐 ID paired delta、hash 与审计候选见 comparison.json。

所有指标复用 backend comparison 的 score/norm 实现。控制集比较 Qwen 与组合 CER；实体 recall 的分母为 gold mention 数，CER 分母为规范化 reference 字符数。
