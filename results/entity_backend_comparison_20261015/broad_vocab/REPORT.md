# Broad-vocabulary Native 对比报告

Broad model: bilingual zh-en 2023 small, int8 encoder/joiner and FP32 decoder. Hotword score is selected from dev only; test/control use this single selected score.

| Split | Arm | Samples | Entity hits / mentions | Entity recall | CER |
|---|---|---:|---:|---:|---:|
| dev | Qwen raw | 171 | 118 / 191 | 61.78% | 5.28% |
| dev | Qwen + CTC sidecar | 171 | 140 / 191 | 73.30% | 4.74% |
| dev | Native 2025 beam | 171 | 112 / 191 | 58.64% | 8.21% |
| dev | Native 2025 hotword | 171 | 129 / 191 | 67.54% | 7.59% |
| test | Qwen raw | 1829 | 1224 / 2467 | 49.61% | 6.76% |
| test | Qwen + CTC sidecar | 1829 | 1675 / 2467 | 67.90% | 5.43% |
| test | Native 2025 beam | 1829 | 1406 / 2467 | 56.99% | 8.99% |
| test | Native 2025 hotword | 1829 | 1570 / 2467 | 63.64% | 8.54% |
| test | Broad 2023 beam | 1829 | 1079 / 2467 | 43.74% | 14.10% |
| test | Broad 2023 hotword | 1829 | 1552 / 2467 | 62.91% | 12.56% |
| control | Qwen raw | 100 | 0 / 0 | 0.00% | 1.07% |
| control | Qwen + CTC sidecar | 100 | 0 / 0 | 0.00% | 1.07% |
| control | Native 2025 beam | 100 | 0 / 0 | 0.00% | 2.08% |
| control | Native 2025 hotword | 100 | 0 / 0 | 0.00% | 2.08% |
| control | Broad 2023 beam | 100 | 0 / 0 | 0.00% | 5.44% |
| control | Broad 2023 hotword | 100 | 0 / 0 | 0.00% | 6.09% |

分母：实体召回率使用 summed gold mentions；CER 使用 normalized reference characters。所有 dev/test/control 独立统计。测试集和控制集要求分别完整覆盖 1829/100 个唯一 ID，且只评估 dev 选出的 broad hotword 权重。

Dev 选择的 broad hotword score：**3.0**。选择基准为 broad beam；优先在 CER errors 不增加的候选中最大化实体命中，完全依据 dev。

成对样本改变量、主 run CSV 来源、broad 词表请求/编码覆盖及限制见 `summary.json`。

本报告只支持对这些具体模型、词库和数据分区的比较，不推断其他 native 替换模型的效果。
