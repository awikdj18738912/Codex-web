# 当前项目中的 LLaMA-Factory 训练环境

已重新从官方仓库拉取新版代码，位置为项目根目录 `LLaMA-Factory/`。本地旧副本没有参与安装。

- LLaMA-Factory：`0.9.6.dev0`，固定提交 `ce9dc9e072f80fa3abe0989d4ab90da25f083438`。
- 训练解释器：`.runtime/llamafactory-env/bin/python`。
- Transformers `5.8.0`、PEFT `0.18.1`、Datasets `4.0.0`、Accelerate `1.11.0`、TRL `0.24.0`。
- PyTorch `2.11.0`、TorchVision `0.26.0`、TorchAudio `2.11.0` 复用已有安装，GPU 库不重复下载。

环境中的训练依赖独立于在线 ASR 依赖，只有所需的 95 个基础包通过项目内符号链接共享。依赖闭包共 129 个包，`pip check` 已通过。原 `agentic-asr` 环境的 Transformers `5.14.1` 和 Accelerate `1.14.0` 保持不变，Web 8081 和 ASR 8766 健康检查通过。

## 直接使用

在当前项目根目录执行：

```bash
# 检查模型 tokenizer、数据编码、提示格式和 assistant-only loss。
bash scripts/llamafactory_refiner.sh preflight

# 正式 LoRA 微调；须选择有足够空闲显存的一张 GPU。
bash scripts/llamafactory_refiner.sh train --device cuda:0
```

训练配置为 `experiments/llamafactory/refiner_local_lora.yaml`。输出到独立目录 `results/refiner_lora_20261010`，不会覆盖基础模型；已有非空输出目录会被拒绝，可用 `--output` 指定新目录。

若选择 CPU 完整训练：

```bash
bash scripts/llamafactory_refiner.sh train --device cpu \
  --output results/refiner_lora_cpu_trial
```

正式训练在 GPU 上启动前会检查至少 6GiB 空闲显存。这是保守启动阈值，不是训练峰值上限。训练启动前项目内 ASR 和 Web 服务已停止。训练结束后，8081、8051、8082仍停止；8766随后由另一个Codex托管进程重新启动，目前正在运行。

官方 CLI 也已可用，例如：

```bash
.runtime/llamafactory-env/bin/llamafactory-cli version
```

本项目的 Refiner 配置需要通过上面的专用脚本启动，它仅注册实际模型需要的模板并调用官方 LLaMA-Factory `run_exp`，没有另写 Trainer。

## 为什么增加一个模板名称

新版自带 `cpm4`。当前 Refiner 推理时，还会在 assistant 开头加入关闭思考的固定前缀 `<think>\n\n</think>\n\n`。直接用普通 cpm4 会漏掉这段。

`agenticasr_local_nothink` 按实际 checkpoint 的 ChatML/BOS 格式注册，并将该前缀放入提示部分，不参与回答 loss。已对全部 1,000 条样本验证：训练 prompt 与实际 `apply_chat_template(..., enable_thinking=False)` 的 prompt **逐 token 一致**。最长完整样本 369 tokens，低于配置的 1024 tokens，不会截掉当前样本的回答。

上游模板代码和基础模型 tokenizer 文件未修改。

## 数据与监督

`experiments/llamafactory/data/dataset_info.json` 注册三个独立数据集：

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| refiner_local_train | 628 | 训练 |
| refiner_local_dev | 122 | 开发评估 |
| refiner_local_test | 250 | 合成测试；不进入本轮训练／开发评估 |

来自此前准备的 1,000 条格式样本，仍按 32 个模板来源组划分。训练文件只含 system/user/assistant messages，审核元数据单独保存在 data_audit.jsonl。全部目标已通过严格局部补丁校验。

LLaMA-Factory 已实际加载 628/122 条数据，检查了全部 750 条记录的监督 mask：提示部分为 `-100`，只有 JSON 回答及终止部分参与 loss。

这是保留／局部补丁的**文本格式预热数据**，不是真实 ASR 错误标注，也不能用合成测试结果证明真人纠错效果。正式格式微调后仍需 ASR 配对及完整会话评估。

## 已完成的真实训练检查

完整的第一轮 LoRA 训练已经完成，使用实际 Refiner 权重：

- LoRA rank 8、alpha 16、目标 q_proj/v_proj，训练 3 个 epoch、237 个优化步骤。
- 628 条训练和 122 条开发数据；三轮 dev loss 为 0.1704、0.1750、0.1625。
- 训练约 5 分 24 秒，输出独立 adapter 到 `results/refiner_lora_20261010/`；基础模型权重未改写。

#### 合成留出结果

250 条按模板来源隔离的合成留出样本上，JSON 合法率为 **91.6%**，精修目标完全匹配率为 **66.4%**，平均生成时间为 **471 ms/条**。模型能保留常规文本、有意义重复、条件句和数字；口吃和赘词清理没有泛化。模型生成了21条整句删除建议，均被保护机制拦截；通过协议的有效替换为0条。

因此当前 adapter 是训练试验产物，还不能作为可用的精修模型。这个留出集由模板生成，不含真实音频，也不是 WER/真人语音结论。逐条记录和限制见 [FORMAT_EVALUATION.md](/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_lora_20261010/FORMAT_EVALUATION.md)。

## 第二轮：增加训练侧合成重复

第二轮使用 experiments/llamafactory/data_v2/，在原 628 条训练样本外新增 188 条单点合成变形；开发集 122 条和测试集 250 条均保持逐字节不变。新增数据主要是安全词位上的首字重启、词重复，少量句首填充词；无法安全选位的 125 条候选跳过。训练前预检确认运行时 prompt 一致、816 条训练和 122 条开发监督掩码有效，测试集未用于训练。

v2 使用相同 LoRA 配置，3 epoch、306 步，耗时 6 分 33 秒，最低开发集 loss 为 0.09455。相同的 250 条冻结测试集上，JSON 合法率从 91.6% 降到 80.0%，精确目标匹配从 66.4% 降到 57.6%；虽然出现 44 条通过协议的 replace，但包括删掉干净句子主语的误改。平均生成时间为 534 ms/条。

所以 v2 也不部署。更低的开发 loss 没有转化为更好的留出效果；仅扩大文本合成训练还不够。完整对比见 [v2 FORMAT_EVALUATION.md](/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_lora_v2_20261010/FORMAT_EVALUATION.md)。adapter 保存在 results/refiner_lora_v2_20261010/，第一版 adapter 未覆盖。

基础 adapter 输出可继续复用。后续如再次训练，使用新的输出目录：

```bash
bash scripts/llamafactory_refiner.sh smoke \
  --output .runtime/llamafactory-smoke-another
```

训练开始前按要求停止了项目内ASR和Web服务。训练后8081、8051、8082仍停止；8766后来由另一个Codex托管进程启动，目前运行中。BMQ项目的其他GPU进程未停止。

## 复现安装和数据注册

```bash
bash scripts/setup_llamafactory_refiner.sh
```

安装固定源码和 constraints.txt 中的版本。首次创建项目内虚拟环境，复用已核验版本的 PyTorch/CUDA 包，安装训练依赖后限制可见范围；不会升级现有在线服务环境。基础环境版本若与核验结果不同会明确拒绝隐式替换 GPU 栈。

数据导出保留已有划分，校验源文件 SHA-256，重复执行相同导出可复用；源文件或已有导出内容不同则拒绝覆盖。

环境重建后，运行 preflight 校验实际 tokenizer 与模板。符号链接依赖于本机已有基础环境；迁移机器时需要准备相同基础 GPU 栈或重新核验安装组合。

验证证据位于 `results/llamafactory_setup_20261010/`，包括预检查、训练源码提交、配置和数据哈希、237 步完整训练摘要。格式留出评估另保存在 LoRA 输出目录中。完整依赖版本保存为 `environment.freeze.txt`。
