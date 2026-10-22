# Qwen2.5-0.5B 中文 Refiner 微调

模型：`/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct`。

只改变 Refiner 的基座和参数规模，继续使用筛后数据里的系统提示词和输入/输出格式。ASR 模型和项目运行时数字规则库沿用现有设置。

## 数据与准备

使用 `sft_reviewed_v5_20261021`：训练 84,875 条、验证 14,977 条；133 条人工处理项和 15 条审核异常已经排除。相同输入和相同最终意图留在同一划分中。源账本及人工清单保留供追溯。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 .runtime/llamafactory-env/bin/python -u experiments/scripts/prepare_qwen25_refiner_training.py
```

准备脚本核对导出文件摘要、全部样本 token 长度、Qwen 模板与推理模板的一致性，并按最大样本长度生成不会截断的 `cutoff_len`。随后通过实际 LLaMA Factory 数据加载器生成缓存，逐条检查仅答案参与损失、总长度与监督长度均未改变。结果写入 `results/qwen25_05b_refiner_preparation_20261021/preparation_report.json`。

## 训练

默认在 GPU0 单卡全参数 SFT，BF16 + SDPA，5 轮，学习率 2e-5，有效批量 32。0.5B 的规模允许用全参数训练；这轮不使用量化训练。每轮在完整验证集计算 loss 并保存 checkpoint，训练结束恢复验证 loss 最低的权重。首次运行需要选中 GPU 至少有 20 GiB 空闲显存。

```bash
bash scripts/train_qwen25_refiner_reviewed.sh
```

配置：`experiments/llamafactory/qwen25_05b_refiner_reviewed_v5.yaml`。

输出：`/home/aim0/data/ASR/training/qwen25_05b_refiner_reviewed_v5_full/`，项目内 `results/qwen25_05b_refiner_reviewed_v5_full` 为同一位置的快捷路径。缓存和准备报告同样放在 data 盘，通过项目 `results/qwen25_05b_refiner_preparation_20261021` 访问。

日志：`logs/qwen25_05b_refiner_reviewed_v5/`。

启动器拒绝覆盖非空训练目录。意外中断后应显式指定最后一个 checkpoint 并准备续训配置；不要把重新开始的训练写入已有目录。准备步骤不加载 GPU 权重，也不启动训练。

## 检查实际精修效果

验证 loss 用于训练期间监控。还需要对基座和各轮 checkpoint 使用相同的验证文本生成精修结果，比较字符错误率、精确匹配率、无需修改样本的误改率，并查看逐条输出。脚本固定从九个场景均衡抽取 256 条验证样本，不能替代独立真人口语测试或语义审核。

在空闲 GPU 上执行基座基线：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/llamafactory-env/bin/python -u experiments/scripts/evaluate_qwen25_refiner.py \
  --checkpoint /home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct \
  --output results/qwen25_05b_refiner_preparation_20261021/base_predictions.jsonl
```

训练后对结果目录或具体 `checkpoint-*` 目录执行同一脚本，换一个输出文件名：

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/llamafactory-env/bin/python -u experiments/scripts/evaluate_qwen25_refiner.py \
  --checkpoint results/qwen25_05b_refiner_reviewed_v5_full \
  --output results/qwen25_05b_refiner_preparation_20261021/finetuned_predictions.jsonl
```

参数变小后的效果需要这些比较才能确认；自动审核通过的训练数据和较低验证 loss 都不能保证精修语义完全正确。

参考：[Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)。
