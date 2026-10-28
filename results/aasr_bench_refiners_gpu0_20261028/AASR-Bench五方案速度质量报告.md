# AASR-Bench 中文集合：五种精修方案报告

## 本轮结论

在本轮510条中文AASR纯文本评估中，原AgenticASR 1B的rubric规范化得分89.98%，高于当前0.5B LoRA＋规则库的62.22%和全量＋规则库的60.83%。此前内部256句上“全量组合较好”的结论不能推广为在AASR上优于1B。

门控使本轮32句的单句平均延迟降低约11%，但510句batch 8总吞吐基本不变。全量有1条重复生成到1024token上限，异常已计入。

四个0.5B组合的内容保留得分约96%，差距主要在filter口语清理与rephrase自我修正。数字保护存在103–113条回退，原生数字格式目标与AASR参考也不完全一致；分数衡量当前组合整体，不能将所有差距直接归因于模型规模或训练方式。

五组均完成510样本、3416道rubric题，没有失败题。裁判为本机4bit Gemma4，属于机器评分；不是人工准确率，也不是论文分数复现。

本报告输入是 AASR-Bench 的纯文本 oral 转录，未送入音频，也未运行 ASR。逐字匹配 clean 仅作描述性对照，不是论文官方 rubric 分数。原始 clean 与同规则库归一化后的 clean 均单独报告。

集合共 510 条、10 个场景；76 条 oral 与 clean 相同（passthrough），434 条不同。五种方案在 GPU 上串行运行。模型为 BF16 Transformers/PyTorch 推理，不是 vLLM；LoRA 在内存中合并。四个 0.5B 方案使用 native v91 数字规则，原配置 1B 不使用数字规则。

## 单句响应时间

32 条（5 条 passthrough、27 条有改动）逐条 batch=1。该时间包含规则处理、门控（适用时）、refiner 和数字 guard。

| 方案 | n | 平均 ms | 中位数 ms | P95 ms | 规则 ms | 门控 ms | refiner ms | 数字 guard ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 0.5B | 32 | 835.50 | 726.67 | 1549.72 | 9.680 | 0.000 | 822.94 | 2.847 |
| LoRA 0.5B＋门控 | 32 | 743.07 | 703.16 | 1550.34 | 9.631 | 3.358 | 727.43 | 2.622 |
| 全量 0.5B | 32 | 819.90 | 706.46 | 1511.83 | 9.697 | 0.000 | 807.17 | 2.992 |
| 全量 0.5B＋门控 | 32 | 728.11 | 665.89 | 1520.24 | 9.677 | 3.510 | 712.33 | 2.568 |
| 原配置 AgenticASR 1B | 32 | 792.35 | 702.95 | 1551.08 | 0.000 | 0.000 | 792.33 | 0.000 |

## Batch 8 吞吐与质量

510 条完整集合以 batch size 8 处理，最后一批 6 条，共 64 批。批量吞吐表示整批执行的总时间和句/秒，不能等同单句响应时间；批内输出需等待本批生成结束。

| 方案 | 510条总批处理秒数 | 句/秒 | refiner调用 | 原始clean逐字匹配 | 归一化clean逐字匹配 | passthrough原始匹配 | passthrough归一化匹配 | 数字回退 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 0.5B | 137.53 | 3.71 | 510/510 | 96/510 | 103/510 | 70/76 | 76/76 | 112 |
| LoRA 0.5B＋门控 | 137.26 | 3.72 | 442/510 | 96/510 | 103/510 | 70/76 | 76/76 | 113 |
| 全量 0.5B | 166.42 | 3.06 | 510/510 | 89/510 | 95/510 | 70/76 | 76/76 | 103 |
| 全量 0.5B＋门控 | 166.59 | 3.06 | 442/510 | 89/510 | 95/510 | 70/76 | 76/76 | 103 |
| 原配置 AgenticASR 1B | 129.86 | 3.93 | 510/510 | 190/510 | 185/510 | 60/76 | 56/76 | 0 |

质量统计是字面匹配，不是官方 rubric 质量分。原始 clean 和规则归一化 clean 都列出，以呈现数字规范化差异。逐场景统计和 434 条非 passthrough 子集见 summary.json。

当前全量模型在 `recheck.jsonl:201` 出现重复生成，达到1024个新token上限；加门控与不加门控均发生，最后由数字保护回退为规则基线。该异常的完整耗时计入批量总时间，未删除异常样本，也未用重跑替换。它会拉低批量吞吐，不能仅按平均耗时判断模型速度。

1B最终使用 `system.refiner.SYSTEM_PROMPT` 的原始输入配置，允许模型规范数字。此前“数字已由规则处理”的离线CLI提示词不适用于1B原始输入对照，已仅补跑1B；首轮仍归档在 `first_pass_1b_offline_cli_prompt`，四个0.5B方案保持不变。

## 门控差异

| 基础模型 | 单句均值变化 | 510条批处理总时长变化 | 吞吐加速 | 因 skip 损失原始 clean 匹配 | 仍 route、批形状相关差异造成匹配损失 | 弱标签需修改但被 skip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 0.5B | 11.1% | 0.2% | 1.00× | 0 | 0 | 2 |
| 全量 0.5B | 11.2% | -0.1% | 1.00× | 0 | 0 | 2 |

“仍 route”说明该输入确实进入了精修模型。若该输出与无门控结果不同，报告将其单列为门控改变子批形状时观察到的差异，不能称为门控漏检。详细 ID 与输出在 gate_differences.jsonl。

本轮门控跳过68/510句：66句为 oral 与 clean 相同的干净样本，另2句（`recheck.jsonl:774`、`recheck.jsonl:619`）主要差在百分比或时间写法。它们仍由规则库处理，不能仅因 oral≠clean 就认定漏掉了必须由精修模型解决的问题。该集合多数句子需清理，门控跳过比例约13.3%，与先前人为50/50集合不同。

## 数字 guard 回退

numeric_fallback 表示模型正文经数字恢复与完整性检查后回退到 rule_baseline。下表给出回退数量，以及回退前 model_output 命中而最终 output 失去匹配的 ID；正文与 raw_output 保存在逐条预测中。

| 方案 | 回退数 | 丢失原始 clean 匹配 | 丢失归一化 clean 匹配 |
| --- | ---: | ---: | ---: |
| LoRA 0.5B | 112 | 9 | 10 |
| LoRA 0.5B＋门控 | 113 | 9 | 10 |
| 全量 0.5B | 103 | 5 | 6 |
| 全量 0.5B＋门控 | 103 | 6 | 7 |
| 原配置 AgenticASR 1B | 0 | 0 | 0 |

## 官方 rubric 评分（本机量化裁判）

以下评分使用原始 AASR official rubric 和本机量化 Gemma4-QAT-W4A16 judge，通过仓库官方 benchmark_io/result_store/metrics 解析与汇总。此评测直接输入 oral 文本，不是论文的全音频、全精度复现；不得称为官方论文分数。百分比是 rubric 规范化得分，不是准确率。

| 方案 | 总分 | rubric规范化得分 | content | format | filter | rephrase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 0.5B | 2689/4322 (3416题) | 62.22% | 96.85% | 61.29% | 30.75% | 22.05% |
| LoRA 0.5B＋门控 | 2684/4322 (3416题) | 62.10% | 96.79% | 61.29% | 30.35% | 22.05% |
| 全量 0.5B | 2629/4322 (3416题) | 60.83% | 96.15% | 62.38% | 28.11% | 18.67% |
| 全量 0.5B＋门控 | 2629/4322 (3416题) | 60.83% | 96.15% | 62.54% | 28.11% | 18.55% |
| 原配置 AgenticASR 1B | 3889/4322 (3416题) | 89.98% | 96.26% | 83.23% | 88.39% | 82.89% |

评分汇总文件：[rubric_scores_by_variant.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/rubric_scores_by_variant.json)；裁判协议：[judge_protocol.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/judge_protocol.json)。各方案原始汇总：
- [lora.rubric_summary.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/lora.rubric_summary.json)
- [lora_gate.rubric_summary.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/lora_gate.rubric_summary.json)
- [full.rubric_summary.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/full.rubric_summary.json)
- [full_gate.rubric_summary.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/full_gate.rubric_summary.json)
- [agentic1b.rubric_summary.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/agentic1b.rubric_summary.json)

## 测试口径与产物

四个 0.5B 方案使用 native `zh-itn-strict-conservative-v91` 数字规则。LoRA checkpoint 在内存合并，模型通过 BF16 PyTorch Transformers 加载；AgenticASR 1B 按原配置直接处理原文，没有数字规则。batch 8 总耗时只表示批处理吞吐，不能解释为单句延迟。

- [逐场景、逐组 summary.json](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/summary.json)
- [summary.csv](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/summary.csv)
- [门控输出差异 JSONL](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/gate_differences.jsonl)
- [510句逐条对照，按场景分组](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/510句逐条对照.md)
- [原始预测与 model_output](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/predictions.jsonl)
- [逐批计时](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/batch_timings.jsonl)
- [原始 rubric](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/original_rubric.json)
- [运行协议](/home/aim0/桌面/tan/ASR/Codex-web/results/aasr_bench_refiners_gpu0_20261028/protocol.json)

