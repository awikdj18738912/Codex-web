# 实体后端对比报告

| Split | 评分臂 | 样本数 | 实体命中 / mentions | 实体召回率 | CER |
|---|---|---:|---:|---:|---:|
| dev | Qwen 原始主流 | 171 | 118 / 191 | 61.78% | 5.28% |
| dev | CTC greedy | 171 | 108 / 191 | 56.54% | 8.63% |
| dev | Native greedy | 171 | 109 / 191 | 57.07% | 8.77% |
| dev | Native beam | 171 | 112 / 191 | 58.64% | 8.21% |
| dev | Native hotword（dev 选参） | 171 | 129 / 191 | 67.54% | 7.59% |
| dev | Qwen + CTC sidecar（dev 选参） | 171 | 140 / 191 | 73.30% | 4.74% |
| test | Qwen 原始主流 | 1829 | 1224 / 2467 | 49.61% | 6.76% |
| test | CTC greedy | 1829 | 1323 / 2467 | 53.63% | 9.32% |
| test | Native greedy | 1829 | 1349 / 2467 | 54.68% | 10.04% |
| test | Native beam | 1829 | 1406 / 2467 | 56.99% | 8.99% |
| test | Native hotword（dev 选参） | 1829 | 1570 / 2467 | 63.64% | 8.54% |
| test | Qwen + CTC sidecar（dev 选参） | 1829 | 1675 / 2467 | 67.90% | 5.43% |
| control | Qwen 原始主流 | 100 | 0 / 0 | 0.00% | 1.07% |
| control | CTC greedy | 100 | 0 / 0 | 0.00% | 2.22% |
| control | Native greedy | 100 | 0 / 0 | 0.00% | 2.58% |
| control | Native beam | 100 | 0 / 0 | 0.00% | 2.08% |
| control | Native hotword（dev 选参） | 100 | 0 / 0 | 0.00% | 2.08% |
| control | Qwen + CTC sidecar（dev 选参） | 100 | 0 / 0 | 0.00% | 1.07% |

实体命中数按同一规范化 canonical 的 gold mention 数截顶；召回率分母为该 split 的 gold mention 总数，CER 分母为规范化 reference 字符数。每个表中样本数是该臂存在 hypothesis 的样本数。dev、test、control 分开统计；控制集 Qwen 有实际主流记录时正常评分，真实缺失值显示为 NA。test 仅套用 dev 选择的参数。

## 配对变化

sample_comparison.csv 按样本给出 CTC、native hotword、hybrid 相对 Qwen raw 和 native beam 的 clipped 实体 recovered/lost 数及 CER 改善/退化标记；聚合指标见 summary.json。词库审计列出 CTC、native 与 hybrid 输出里标准 reference 不支持的本样本词库 canonical 命中，并比较 native hotword 对 native beam、hybrid 对 Qwen 的新增项；这是自动词法候选，不是人工 precision。

## 性能与限制

Native/CTC compute RTF = sum(compute seconds) / sum(audio seconds)；finish wall 的 p50/p95 按样本 finish 字段计算。Hybrid 的顺序 compute estimate 包含 CTC 与 fusion compute，以及可用的保存 Qwen HTTP wall，不能当作真实并发端到端 latency，也不与 native CPU wall time等同。真实按时钟送音的四臂延迟已完成，另见 PACED_PROFILE.md；完整性及计时校验见 paced_profile_summary.json。首次匹配要求 canonical 在连续两个 partial 中出现，记录第一个 partial 的音频可用位置；没有 gold 声学结束时间，因此不称为实体说完到字幕延迟。Native CJK 字符编码只覆盖全部 canonical 的一部分，需结合 hotword_encoded_count / canonical_count 及 coverage 字段解释。

这些结果只代表本次冻结语料和具体 Qwen/native/CTC/hybrid 后端，不能外推到全部 CTC 模型或所有替换 ASR。
