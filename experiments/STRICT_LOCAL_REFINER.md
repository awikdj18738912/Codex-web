# 第一步：让现有 Refiner 学会保留和局部编辑

本轮已实现可切换的严格窗口编辑协议，并用 LLaMA-Factory 完成一版格式适配 LoRA。运行时回归和合成留出集评估已经完成；现有真人 ASR 服务已按任务要求停止，适配器未接入正式服务。

## 输出契约

正确内容返回：

```json
{"action":"keep","source":"","target":"","reason":"no_change"}
```

输入 `我们我们明天去郑州开会。` 的局部修改示例：

```json
{"action":"replace","source":"我们我们","target":"我们","reason":"repetition"}
```

输入 `这些泵站一级一级地向高处提水。` 应返回 keep，保留逐级含义。正确数字、单位、实体、否定、指代及条件关系也应保留。

新模式 `strict_local` 的运行时限制：

- 仅接受一个 JSON 对象，且只有 action/source/target/reason 四个字符串字段；拒绝整段文本、代码围栏、重复键、额外字段和伪 keep。
- source 必须逐字且唯一地存在于模型输入；检查重叠出现的位置，不能任意替换一个同名片段。
- source/target 各最多 32 字符，一次只允许一个连续修改区域。
- 保留原窗口的末尾标点／引号，包括逗号，避免模型在没有后续上下文时自行封句。
- 首次处理与完整性重试都应用严格解析。遭拒后保留规则基线，关闭该次窗口响应的部分挽救和据此生成的删除复核。
- 现有实体、数字、语义完整性检查继续执行。格式合法并不证明语义正确，仍可能产生错误的局部删除。

这里的 32 字符是首版工程范围，可能拒绝正确的长补丁或多处修改，需要记录漏掉的有效编辑并做消融。它不是最佳长度或风险保证。

该开关约束 Web 普通窗口及其重试。独立重复候选复核仍使用原有 KEEP/REMOVE 契约；本轮没有训练或更改该任务的决策。最终完整会话效果还需把它一起纳入评估。

## 当前证据

对旧配对试验保存的 503 条真实窗口模型响应做格式检查，全部是整段文本，严格解析接受 0 条。这些响应来自旧提示词，**没有用新提示词重新推理**；不能据此估算新模式的纠错效果或速度。

旧试验的文本辅助预审中，发生变化的窗口包含 22 条收益候选、26 条退化候选、76 条中性改动和 13 条存疑。因此，单纯拒绝旧响应可能同时阻止退化和损失收益，不能将拒绝次数当成质量提升。这些辅助标签未经听音确认，亦不是论文真值。

第一版 LoRA 微调已完成：628 条训练数据、122 条开发数据，237 个优化步骤、3 个 epoch，耗时 5 分 23 秒。三轮开发集 loss 为 **0.1704、0.1750、0.1625**；最低 loss 在第 237 步。loss 不能单独证明纠错效果。

按模板组留出的 250 条合成文本中，严格 JSON 合法率为 **91.6%**，精修目标完全匹配率为 **66.4%**，平均模型生成时间约为 **471 ms/条**。42 条正常文本、41 条有意义重复、42 条事实保护和 41 条句法样本均保留原文；42 条口吃和42条赘词样本均没有正确清理。模型生成了21条replace建议，但全都因覆盖整句被严格解析器拒绝；没有一条补丁通过协议并产生有效替换。

测试目标由合成脚本提供，未经听音确认；它不是 WER、真人精修准确率或论文测试集。结果足以说明当前适配器还不应接入严格模式。完整逐条结果见 [合成留出集评估](/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_lora_20261010/FORMAT_EVALUATION.md)。

第二轮在训练侧新增 188 条安全单点合成变形，测试与开发划分保持不变。虽然最低 dev loss 降至 0.09455，v2 的 250 条冻结测试精确目标匹配率降至 57.6%，协议合法率降至 80.0%；它开始改写输入，却出现干净句子删主语等误修，也未能清理原测试中的赘词。v2 同样不部署，详细对照见 [第二轮评估](/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_lora_v2_20261010/FORMAT_EVALUATION.md)。现有默认 compat、门控和正式服务均未接入这两个 adapter。

训练开始前，8081、8051、8082 Web 和 8766 ASR 都已停止。训练结束后8081、8051、8082仍停；随后8766由另一个Codex托管进程重新启动，目前正在运行。BMQ项目的其他服务未停止。仓库默认仍使用 `compat`。

## 已生成的格式适配数据

目录：`results/refiner_local_edit_20261009/format_sft/`。

| 划分 | keep | replace | 总数 |
| --- | ---: | ---: | ---: |
| train | 418 | 210 | 628 |
| dev | 82 | 40 | 122 |
| test | 166 | 84 | 250 |
| 全部 | 666 | 334 | 1000 |

每条包含标准 system/user/assistant messages 和独立 metadata。正常表达、正常叠词、事实保护及正确句法样本以 keep 监督；指定赘词和口吃样本以一个可执行局部补丁监督。约一半使用人工构造的实体占位符，检查占位符保留。所有响应均已用实际严格解析器回放，精确恢复其已知目标。

相关模板、名字／地点替换、正常与口语版本按 32 个来源组划分，避免同一模板跨 split。首次生成的更短字符锚点版本留在 `format_sft_initial/` 供追溯，当前训练入口使用 `format_sft/`。

这些是虚构文本上的**格式与保留能力预热**，不是实际 ASR 错误对，也不是语音真值。此阶段无需 TTS；它不能代替真实／经质量检查的合成 ASR 纠错数据，更不能用其合成 test 宣称真人泛化。

新空目录可复现：

```bash
python experiments/scripts/prepare_local_edit_sft.py \
  --source results/gate_synthetic_20261009/texts.jsonl \
  --output results/refiner_local_edit_reproduction/format_sft

python experiments/scripts/audit_local_edit_protocol.py \
  --source results/gate_pilot_20261009 \
  --output results/refiner_local_edit_reproduction/protocol_audit
```

## 接入方式

Web 启动参数新增 `--window-edit-policy compat|strict_local`。health 和窗口审计记录会报告策略与严格协议版本。默认 compat 保留既有 checkpoint 的解析方式。

未来候选模型准备经过扩充数据和回归验收后，才在独立端口测试。当前 LoRA 输出目录是适配器，不是可直接由 Web `--refiner-model` 加载的完整 checkpoint；示例中的完整格式适配模型路径需要先另行导出：

```bash
GATE_PYTHON=/home/aim0/data/conda/envs/agentic-asr/bin/python
# 此路径须指向从通过回归的 adapter 单独导出的候选 checkpoint；本轮尚未导出。
LOCAL_EDIT_CHECKPOINT=/path/to/format-adapted-refiner
"$GATE_PYTHON" -m system.web_app \
  --refiner-model "$LOCAL_EDIT_CHECKPOINT" --refiner-device cpu \
  --asr-url http://127.0.0.1:8766 \
  --refinement-gate-mode tri_state --final-refinement-mode off \
  --enable-zh-itn --window-edit-policy strict_local --port 8091
```

CPU 仅为示例设备，速度未测；需要其他设备时显式选择有余量的 GPU。服务的词库、窗口和校验设置应与对照一致，示例未配置正式词库，不构成完整质量对照。

启动脚本也支持 `WINDOW_EDIT_POLICY=strict_local`。如果目标端口已有 compat 服务，它会明确拒绝静默复用，不会自动重启服务。

配对采集可冻结严格策略：

```bash
python experiments/scripts/collect_gate_pairs.py prepare \
  --output results/strict_local_pairs_trial \
  --window-count 50 --review-count 0 --window-edit-policy strict_local

python experiments/scripts/collect_gate_pairs.py run \
  --output results/strict_local_pairs_trial \
  --model /path/to/format-adapted-refiner --device cpu
```

run 继承 prepare 冻结的窗口策略，显式指定相反策略会报错。旧 compat 冻结试验继续使用原有接口。上例复用开发资料，不能当作独立真人验收。

## 接下来的训练与验收

1. 扩充口吃和赘词表达的语言覆盖，改进留出模板上的编辑泛化。采用编辑前缀、词边界和音频对齐多样性，避免训练只有少数固定句式。
2. 对新候选继续报告严格 JSON 合法率、逐类别 keep 率、净编辑命中率和有害修改；以留出来源组评估，不能只看格式或 loss。
3. 加入真实 ASR 或经过质量检查的 TTS→ASR 配对，覆盖实体错误、漏字、正常强调和条件句，并保留存疑参考。
4. 再用同一冻结 ASR 输入比较现有 checkpoint/compat 与候选 checkpoint/strict_local，分别统计净收益、误改、漏修和成本；随后做完整有状态回放和独立真人评估。

## 验证

严格模式、格式数据、Web finish、数字和实体保护等针对性回归共 171 项通过。随后针对训练片段锚点调整的 30 项测试也通过；启动脚本语法和 diff 格式检查通过。请求 strict_local 时，现有 8081 compat 服务的复用检查按预期拒绝配置不匹配。

完整 3 epoch 微调和 250 条合成格式留出评估已完成。结果显示协议拦截了整句删除，但模型没有学会留出集口吃／赘词清理；真人纠错质量、风险和端到端速度仍未验证。
