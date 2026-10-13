# 按项目标点片段及 K=3 窗口提取 ASR 分数（2026-10-13）

**实测成功。** 独立 GPU 进程使用本地 Qwen3-ASR-1.7B 转录 3 段既有音频，保存生成 token 的分数，并按当前项目的标点分片/合并规则统计 44 个片段、16 个窗口的分数。最终全部片段和窗口中参与统计的字词覆盖率均为 100%。

这是由 token 概率汇总得到的未校准识别分数，不是模型直接提供的“整个窗口正确的概率”，也不是“这个窗口需要精修的概率”。没有启用门控新策略，没有重启线上 ASR/Refiner，也没有调用 Refiner。

## 实际音频结果

| 音频 | 使用长度 | 流式解码更新 | 标点/合并后片段 | 窗口 | 完整评分 |
| --- | ---: | ---: | ---: | ---: | --- |
| 测试音频 1 的 16k 转换 WAV | 40.848 秒 | 21 | 13 | 5 | 13/13 片段，5/5 窗口 |
| 测试音频 3 的首个既有切片 | 30 秒 | 15 | 15 | 5 | 15/15 片段，5/5 窗口 |
| 癌症播客首个切片的前 30 秒 | 30 秒 | 15 | 16 | 6 | 16/16 片段，6/6 窗口 |

总计 100.848 秒音频、44 个片段、16 个窗口；其中 14 个完整 K=3 窗口，2 个终态不足 K 的尾窗。所有流式快照共记录 121 个可处理窗口，没有分数覆盖不完整的快照。该数量包含同一窗口在不同版本下的重复快照，不是 121 条独立样本。

测试音频 1 额外用原有 `QwenVLLMConfidenceDecoder` 在同一模型、相同采样参数下重跑：**每次流式更新和最终文本逐字相同**。其他两个音频没有做该文本对照，不能把这一结论扩大到所有输入。

音频文件、实际使用样本数和 float32 PCM 哈希记录在每个 `case_*/audio_provenance.json`。测试录音的生成来源/说话人未核验，本次只验证分数提取，不建立人工音频金标。

## 不是仅按句号分句

原型直接调用 `CumulativeWindowRefinement._fixed_chunks`，使用当前 Web 的 K=3、`one_punctuation_window=True`、`fixed_groups=True`、80 字符限制及数字改口分组合并配置。

- 边界包含中英文逗号、分号、冒号、问号、叹号和句号。
- 当前规则不以顿号 `、` 切分；数字小数点也不会作为标点边界。
- 保留项目的改口、异常断句、数字改口合并逻辑。因此一个最终片段可能包含多个标点，不能单纯数三个逗号就称为 K=3。
- 流式阶段不足三个片段的尾组继续等待；结束时按现有规则输出短尾窗。

真实输出示例（测试音频 3）：

| 原文片段 | 片段分数 | 字词覆盖率 |
| --- | ---: | ---: |
| 天南一个能打的都没有， | 0.999529 | 100% |
| 另外， | 0.432305 | 100% |
| 咱们的零食我倒装不下了。 | 0.896091 | 100% |

以上三片段构成同一个窗口：

> 天南一个能打的都没有，另外，咱们的零食我倒装不下了。

窗口汇总分数为 **0.898098**，最低片段分数为 **0.432305**。这说明只看窗口平均分可能掩盖局部低分，因此同时保留最低片段分数和低分 token 比例。这里只描述模型的不确定性，没有把低分词标为识别错误。

## 如何获得分数

新增模块 `system/asr_window_confidence.py`，独立试验入口 `experiments/scripts/probe_asr_window_confidence.py`。

1. 每次 vLLM 解码保存 chosen token ID 和对应 logprob；缺失的概率保持空值，不把后续 token 分数向前错移。
2. 使用 Qwen ByteLevel 词表恢复 UTF-8 字节范围，避免中文一个字符被多个 token 编码时用单 token decode 得到替代字符而错位。
3. 保存稳定前缀的原有分数，替换回滚后重新生成的尾部。若回滚切穿旧 token、前缀不匹配或生成字节与解码文本不同，对应部分保持未知，不假装完整覆盖。
4. 按 SDK 解析语义去除语言头等非转写内容；若 SDK 重复修复改变了文本，无法严格对应的结果标为 unsupported。
5. 按项目实际片段和窗口的原文范围计算几何平均。跨边界 token 按 UTF-8 重叠比例分配权重；仅标点、空白字符不参与评分，但混合字词和标点的 token 概率仍是整个 token 的条件概率。

每个窗口包含 `score`、`coverage`、`min_token_probability`、`min_fragment_score`、`low_probability_token_ratio` 和 `evidence_revisions`。覆盖不完整时 `score=null`，仅另存已覆盖部分的诊断分数。

前缀分数来自其原始生成版本，整个窗口可能混合多次流式解码的证据；这不等于用最新全部音频重新评分了整个窗口。`evidence_revisions` 保留了这一来源。本次三例没有触发映射失败或回滚切穿 token，但相应拒收路径已有单元测试。

分数始终对应 **原始 ASR 字词**，不是数字规则或实体规范化后的新文本。如果接入线上，必须继续使用原文位置关联，不能对改写后文本直接套用旧 token 偏移。

## 环境、验证和失败记录

使用物理 GPU 1，本地模型目录 `/home/aim0/data/models/ASR/Qwen3-ASR-1.7B`；qwen-asr 0.0.6、vLLM 0.14.0、Transformers 4.57.6、Torch 2.9.1。流式更新 2 秒，前 2 次不固定前缀，之后回滚 5 个 token，单次最多 32 个新 token，和现有解码适配器配置相同。

19 项置信度/字节映射测试与 39 项现有标点/窗口测试共 **58 项通过**。覆盖逗号/分号/冒号、小数点/顿号、数字改口合并、尾组等待、缺失概率、UTF-8 拆分、语言头、回滚替换和拒收路径。Luna 只读核对了分窗、源代码与运行完成状态。

首次使用 0.32 显存预算加载失败：8192 序列长度需要约 0.88 GiB KV cache，预算中剩余仅 0.02 GiB。该次尚未转录音频，记录保留在 `results/asr_window_confidence_probe_20261013/` 和对应日志。提高独立进程预算至 0.42 后成功加载，模型加载约 39.72 秒，试验结束后 GPU 显存恢复到原有服务占用。

本次流式调用均值包含模型推理和冷暖运行差异，没有配对耗时实验，不能据此计算新增分数功能的延迟开销或门控节时。

复现命令（需要先确认 GPU 1 有足够空闲显存，并使用尚不存在的 output 目录）：

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /home/aim0/data/conda/envs/qwen3-asr/bin/python -u \
  experiments/scripts/probe_asr_window_confidence.py \
  --physical-gpu 1 --gpu-memory-utilization 0.42 \
  --output results/asr_window_confidence_probe_new_run
```

## 当前边界和下一步

本次确认了 **获取和汇总分数的可行性**。这是独立流式 SDK 实验，每个音频作为单个 utterance，未复现线上 Silero VAD 的切段、补静音和多 ASR 状态轮转。因此不能声称已与每一次线上会话的窗口逐一对齐。

线上接入还需要在 ASR 状态轮转时保存已完成段的 token 分数，按真实 `completed_segments`/活动段组合原文，并传入相同的 VAD 来源片段。跨状态的 token 证据必须带唯一段 ID；回滚和标点变更后重新统计受影响窗口。

建议先旁路返回并记录这些字段，继续沿用原生规则路由；之后才评估“正则＋窗口分数”的误报、漏检及真实精修收益。原始高分不能证明无需清理，低分也不能证明 Refiner 能修复。当前没有校准阈值，没有启用生产跳过。

完整结果：`fragments.tsv`、`windows.tsv`、`case_*/final.json`；生成 token 与流式分数历史：`case_*/stream_trace.jsonl`；汇总与哈希检查：`summary.json`、`verification.json`。

参考：已检索并核对 [Qwen 官方 README](https://github.com/QwenLM/Qwen3-ASR) 与 [流式解码实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)，以及适用本地版本的 [vLLM 0.14.0 logprobs 文档](https://docs.vllm.ai/en/v0.14.0/api/vllm/sampling_params/)。两项目为 Apache-2.0；本次使用既有安装和权重，未引入第三方置信度模型，也没有升级依赖。
