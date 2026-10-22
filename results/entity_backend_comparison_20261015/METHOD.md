# 实体流式后端配对对照方法

本实验比较两个具体实现：保留现有 Qwen3-ASR-1.7B，增加独立增量 CTC 实体声学旁路；改用 sherpa-onnx 中文 Zipformer transducer 的原生流式热词解码。只新增实验脚本和结果，没有切换线上后端。

## 数据与隔离

沿用原有 2,000 条实体挑战样本，PCM 与标注不变；按来源组划分 171 条开发集与 1,829 条测试集。开发集与测试集的 source_group 不重叠。另取原有基准中的 100 条未标注目标实体的控制样本，配置干扰领域词库。这些控制样本并不等于经过人工确认的“完全没有实体”。

每条音频使用同一份已经拆分的场景 SQLite 词库。场景来自之前参考文本辅助的数据整理，因而这是已知领域条件下的实体识别对照，不能作为自动领域路由或完全盲测的结果。原始挑战集本来就针对 Qwen 的困难实体；结论适用于此挑战集。

推理输入 inputs.jsonl 不含参考转录和 gold_entities。标签仅在 tune 和 report/analyze 阶段读取。冻结的 PCM、词库、输入、参考、模型哈希在 protocol.json；选参及推理代码哈希在 execution_manifest.json。

## 实验臂

| 实验臂 | 用途 |
|---|---|
| Qwen raw | 保存的真实无领域提示主流转写，固定同一次输出用于质量配对；控制样本补充了实际调用 |
| CTC greedy | 旁路声学模型自身的识别水平 |
| Native greedy | 更换主识别器的基础差异 |
| Native beam | 控制 beam search 自身的收益 |
| Native hotword | 与 Native beam 使用相同模型与搜索宽度，增加领域热词 |
| Qwen + CTC sidecar | 保留 Qwen 文本，仅对有增量 CTC 声学证据、文本候选与邻近锚点支持的实体片段做替换 |

两套 Zipformer 均为 2025-06-30 中文 int8 模型，在 CPU 上设置 num_threads=2。Qwen 使用现有 RTX 3090 服务。这是本机部署方案的对照，不是相同 FLOPs 或相同硬件下的模型竞赛。

所有声学输入以 160 ms 新 PCM 块推进。CTC 使用缓存 encoder 状态，不重新解码整段旧音频；45 帧特征窗口以 32 帧步幅推进，模型要求的重叠前瞻照常保留。结束时补 0.66 秒零值用于排空模型；这不计入源音频长度，也不要求实际等待 0.66 秒。

CTC 前端使用该 ONNX 元数据指定的 Whisper fbank 特征。已与 sherpa 官方解码器校验官方测试 WAV：最终文本、每个 160 ms 输入前缀的输出和 token 时间戳一致；整段与分块输入的 posterior 一致。词检索为本次实现的增量 CTC 图搜索原型，并非完整复现论文全部筛选策略。

## 选参

Native 热词权重扫描 0.5、1、1.5、2、3；CTC 融合 minimum_gap 扫描 -0.25、-0.5、-1、-2，候选分差扫描 0.04、0.1。预先规定：相对各自无偏置基线，开发集 CER 不增加时最大化实体命中数；并列时优先更少字符错误及更保守参数。若无可行参数，选择 CER 最低值。

开发集选定 Native 热词权重 2；CTC minimum_gap=-2、minimum_margin=0.1。之后测试集只运行这一组配置。minimum_gap 是实体路径与相同时段逐帧 greedy 上界的每 token 分差，不是经过校准的概率或完整竞争句子的似然比。

## 编码覆盖

本次 Native 模型的 cjkchar 热词接口要求词的每个字都有 bare 字符 token，不支持该路径的字节回退及 `▁` 替代写法。37 个词库合计 8,990 个文件内 canonical 项中，4,538 项可编码，覆盖 50.48%；未编码项全部保留审计。CTC 旁路支持多种实际 token 路径及字节回退。报告另按 gold 实体是否可编码分组，以揭示这项不对称。不能据此断言所有原生热词模型都有相同限制。

## 指标和时间

实体命中率使用规范化后 canonical 的出现数，以 gold mention 次数截顶；CER 用同一规范化参考的字符编辑距离。所有方法使用相同规范化。另统计修复、损失、全文退化，以及输出新增但参考文本未支持的词库实体候选；后者是自动审计，不能代替人工 precision。

批量运行以尽快完成质量评测为目的，实际计算时间受并行任务竞争影响。保存的 Qwen HTTP 耗时与 CTC 加和只能作为串行估计，不能当作并发延迟。真正延迟通过另一个实时按时钟送音、同 PCM 扇出的三臂实验测量，区分首次出字、最后音频可用后的尾延迟与总时长。没有人工实体声学结束时间，不能宣称测出了“实体说完至正确字幕”的延迟。

## 原始资料

- [CTC-WS, Interspeech 2024](https://arxiv.org/abs/2406.07096)：CTC posterior 上搜索上下文图，再以帧对齐修正候选。
- [Streaming CTC-WS, 2026-05](https://arxiv.org/abs/2605.18222)：面向流式推理的 CTC 实体偏置研究；本次是工程原型，不直接引用论文实验数字作为本机收益。
- [sherpa-onnx 热词文档](https://k2-fsa.github.io/sherpa/onnx/hotwords/)：transducer 的 modified_beam_search 支持热词；因此加入无热词 beam 对照。
- [官方 Python 实现](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py)、[热词编码源码](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/utils.cc)：检查 API 与编码行为。
- [sherpa-onnx 项目](https://github.com/k2-fsa/sherpa-onnx)：持续维护，Apache-2.0；本次使用本地已安装 1.13.8 绑定。模型发布包的许可应与框架许可分别核对，不能由框架许可推定所有权重许可。

资料核查日期：2026-10-15。本次无训练、无付费 API 调用；主要新增成本是独立 CPU CTC 模型计算与缓存，保留 Qwen 的 GPU 占用。原生后端的本次 CPU 路径可减少 Qwen GPU 依赖，但是否值得替换必须同时看全文与实体质量。

## 补充的广词表原生模型

发现 2025 模型热词编码覆盖不足后，增加本地已安装的 `small-bilingual-zh-en-2023-02-16` 候选。其词库逐条编码覆盖为 8,965/8,990（99.72%），用于检查词表限制是否主导了比较结果。该补充实验独立保存在 `broad_vocab/`，不改变首轮冻结的六个实验臂。

遵照[官方模型示例](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)，使用 INT8 encoder/joiner 和 FP32 decoder。最初全 INT8 的开发集诊断出现明显识别退化，已归档且排除；它不参与选参或测试结论。

补充模型同样只用 171 条开发样本扫描五个权重，并以相对其自身无热词 beam 不增 CER 的实体命中数选参，锁定权重 3。之后对完整 1,829 条测试样本与 100 条控制样本运行 beam/hotword 两臂。选择文件在 `broad_vocab/selection.json`，模型、输入来源与代码哈希另存其 protocol/manifest。实时测试扩为四臂，在相同的固定八条音频上轮换顺序；小样本延迟用于本机工程判断，不视为所有场景的 p95 保证。
