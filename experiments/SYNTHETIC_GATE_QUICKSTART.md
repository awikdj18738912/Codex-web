# 用合成语音启动门控训练数据试验

当前已准备好 1,000 条文本及流水线，并已安装 Qwen3-TTS 0.6B CustomVoice。已完成 10 条 train-only TTS→ASR 小试点，共得到 20 条 clean/noise 转写：4/10 与朗读稿逐字相同，3/10 与整理目标逐字相同，去标点字符错误率均值约 8.7%；9/10 条的 clean/noise 转写相同。数据仍是合成弱监督，未做听音核验，也没有用这批数据训练门控。

## 已完成的内容

数据目录为 `results/gate_synthetic_20261009`。`texts.jsonl` 每条包含：

- `spoken_text`：发送给 TTS 的朗读稿。
- `intended_text`：允许清理指定口吃／赘词、保留强调和事实的整理目标，只供离线监督。
- `seed_sha256`：冻结文本及其意图的证据哈希。
- `template_group`、`synthetic_split`：用于分组划分的来源信息，不作为门控输入。
- `audio_script_verified=false`：脚本尚未被证实与实际发声完全一致。

| 类别 | 数量 | 目标 |
| --- | ---: | --- |
| 正常表达 | 167 | 提供无需精修的输入 |
| 非必要语气词 | 167 | 观察精修能否清理赘词 |
| 指定口吃 | 167 | 观察精修能否删除异常重复 |
| 有意义的重复 | 167 | 保留“一级一级”“很多很多”等表达 |
| 实体、数字、单位、否定 | 166 | 观察完整保护链后的净变化 |
| 条件句、主谓和并列结构 | 166 | 覆盖误断句风险 |

1,000 条均为人为设定的虚构转录探针，不是业务真值，也不是 1,000 个独立语义场景。共有 32 个模板来源组，合成 train/dev/test 为 628/122/250 条。基础句的正常、赘词和口吃版本，以及同一模板的名字／地点替换均归在同一组；后续音频、噪声版本继承分组。

这批少量模板用于验证流程与弱监督可行性。规模化训练还需要增加语义和说话方式的多样性；合成 test 不承担真人验收或风险校准。

## 1. 接通 TTS

当前 Qwen ASR `http://127.0.0.1:8766` 可用。Qwen3-TTS-12Hz-0.6B-CustomVoice 已从官方 ModelScope 仓库下载到 /home/aim0/data/models/ASR/Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-CustomVoice，模型约 2.4 GiB，内含 speech_tokenizer 权重。隔离 Python 环境在项目 .runtime/qwen3-tts-env；其 PyTorch 复用现有安装，Transformers/Accelerate 版本独立于 ASR 环境。

启动兼容现有生成脚本的 Qwen TTS 服务（默认 8011，设备自动选择；显存不足时使用 CPU）：

```bash
bash scripts/qwen_tts_service.sh --device auto --port 8011
```

另开终端检查 ASR/TTS 服务：

```bash
GATE_PYTHON=/home/aim0/data/conda/envs/agentic-asr/bin/python
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py preflight
```

若已有本地 MOSS 模型，服务入口是：

```bash
# 以下模型路径必须替换成实际已有的 MOSS 模型目录。
GATE_TTS_MODEL=/path/to/MOSS-TTS
"$GATE_PYTHON" -m pipeline.src.speech.moss_tts_service \
  --model-path "$GATE_TTS_MODEL" --device cpu --port 8011
```

此命令只展示项目已有服务入口，尚未验证当前环境能加载具体 MOSS 权重。CPU 只是无需占用现有 GPU 的接入方式，未测速度。选择 GPU 前需要按实际模型检查显存余量；当前两张 3090 上已有服务占用大量显存。

服务兼容现有 `/v1/tts` 输入和 `audio_path` 返回协议；音频输出必须位于项目目录，以便 TTS 服务和数据流水线共享文件。`/health` 只检查服务可访问，首次合成才会加载权重并验证实际推理。已保存的单句 smoke 音频为 results/qwen3_tts_20261010/smoke.wav，还不代表 1,000 条文本都能被模型忠实朗读。

项目原 MOSS 接口 422 修复的历史测试使用桩模型，不代表 MOSS 权重已加载。本次 Qwen 路由已用真实权重完成 HTTP 合成；逐条结果见 [QWEN_TTS_PILOT_20261010.md](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_synthetic_20261009/QWEN_TTS_PILOT_20261010.md)。

## 2. 先合成 10 条，再调用现有 ASR

在项目根目录执行：

```bash
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py synthesize --split train --limit 10
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py decode --split train --limit 10
```

成功后将生成：

- `audio/`：原始合成音、单声道 16kHz 音频及实际用于识别的 clean/noise 音频。
- `tts.jsonl`：合成响应、文本／音频哈希、声学完整性检查和成本。
- `asr.jsonl`：实际 Qwen 输出、流式 chunk 响应及 finish 响应。
- `tts_configuration.json`、`asr_configuration.json`：冻结服务参数，改变设置后会拒绝续跑。
- `failures.jsonl`：失败记录。不会将服务失败降级为文字模拟，失败项可在服务恢复后续跑。

本次试点使用 Qwen3-TTS 的 Vivian 预设音色，生成音频后由 Qwen ASR 实际转写。clean/noise 来自同一段语音，noise 是固定种子的 10dB 白噪声；这次 10 条里只有 1 条的转写因此发生变化。平均 TTS 耗时约 32.7 秒/条（CPU）。语音脚本尚未逐条听音确认；数字、同音实体等差异是候选错误，不自动视为可靠真值。后续扩大前应增加音色/声学条件并抽查代表样本，不能把噪声版本数当成独立录音数。

音频检查只核对格式、长度、非静音和严重削波，不证明 TTS 按稿读出了全部词、数字或重复。实际发声存疑的收益／退化标签属于弱监督。

ASR 复用现有 `/stream/start`、`/stream/chunk`、`/stream/finish` 接口。发送单声道 16kHz float32 PCM，每次 1 秒，不用原稿、参考或整理目标给 ASR 提示词。每次只处理一条录音，失败会取消自己创建的 session。

## 3. 用冻结的当前处理链跑实际配对

```bash
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py prepare-pairs \
  --window-count 10 --review-count 0

# 显式指定运行设备。CPU 示例保留当前模型精度，但生成速度未测。
"$GATE_PYTHON" experiments/scripts/collect_gate_pairs.py run \
  --output results/gate_synthetic_20261009/pairs --device cpu
```

prepare-pairs 从合成 ASR 文件中选择 10 个窗口决策；窗口数不是录音数。它快照化当前代码、数字动态库和词库副本，默认不会混入旧演示音频。run 使用现有真实 Refiner、校验、重试和回退，产生 rules/current/always 三条分支的最终输出与调用成本。选择有余量的 GPU 时可替换 `--device`。

第一轮以窗口任务验证链路。增加 `--review-count` 可采集局部复核，但必须确实存在足量候选，脚本不会为了凑数编造候选。原门控跳过的窗口也会进入探索性采样。

这仍是冻结最终快照的离线配对，未验证完整流式会话的质量、延迟或在线加速。后续需要有状态回放和独立真人验证。

## 4. 导出保守弱标签

```bash
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py label
```

结果是 `weak_gate_labels.jsonl` 和 `weak_label_summary.json`：

| 判定 | 标签性质 |
| --- | --- |
| 两条最终输出逐字相同 | neutral；可证明该操作没有净输出效果 |
| 词语层面确有变化，只有精修结果完整匹配整理目标 | benefit 弱标签；TTS 原稿尚未经听音证实 |
| 只有规则基线匹配整理目标，精修偏离 | harm 弱标签；TTS 原稿尚未经听音证实 |
| 仅标点／字形变化，或双方都不匹配目标 | uncertain；保留，排除弱监督预训练 |

比较先使用本轮冻结的数字规范化，再按词语变化判断。它不会仅因 CER 下降、发生修改或校验器接受就生成 benefit；标点变化可能切断条件句，因此也不会一概标 neutral。

全部记录均 `human_gold=false`、`eligible_for_real_risk_calibration=false`。`eligible_for_weak_pretraining=true` 仅允许作为合成预训练材料。`input` 只保留决策时已经可见的原始文本、规则基线、上下文及局部候选；目标文本和本次精修答案不会进入门控输入。

## 5. 根据小试结果扩大

先查看实际标签数量、拒绝／失败记录、代表性的收益和风险，以及合成音是否漏读关键词。若多数结果没有收益，先调整场景覆盖或音频条件，确认当前 Refiner 是否具有可选择的收益；文本类别本身不会决定标签。

确认小试后，对同一数据目录续跑剩余文本：

```bash
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py synthesize
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py decode

# 用新配对目录冻结扩大后的 ASR 数据，保留第一轮证据。
"$GATE_PYTHON" experiments/scripts/collect_gate_pairs.py prepare \
  --output results/gate_synthetic_20261009/pairs_full \
  --sources results/gate_synthetic_20261009/asr.jsonl --no-demo-sources \
  --window-count 1000 --review-count 0
```

首轮 pairs 目录中的输入已经冻结，追加 ASR 后不应覆写其快照。完整配对可继续使用 collector 的 run/report；label 子命令默认读取 `pairs/`，扩大轮使用以下命令，将弱标签及摘要写入 `pairs_full/`，保留首轮结果：

```bash
"$GATE_PYTHON" experiments/scripts/collect_gate_pairs.py run \
  --output results/gate_synthetic_20261009/pairs_full --device cpu
"$GATE_PYTHON" experiments/scripts/synthetic_gate_data.py label \
  --pairs-dir results/gate_synthetic_20261009/pairs_full
```

实际产出足量收益／退化样本后再训练门控原型。合成弱监督可减少逐条人工工作，最终风险阈值和真人效果仍用独立真实录音验证；既有 157 条辅助审核样本仅作开发／回归材料。

## 验证

测试覆盖模板分组、禁止目标泄漏、保留有意义的重复、数字／否定误改、保守标点标签、服务 422 修复、音频格式、续跑、取消会话、失败不伪装成成功及配对完整性。HTTP 音频测试使用正弦波作为传输夹具，不是已经完成的真实 TTS 实验。

本次合成数据、配对采集及辅助审核的针对性测试共 **46 项通过**。在同时安装了项目音频依赖和 pytest 的环境中可运行：

```bash
python -m pytest -q tests/test_synthetic_gate_data.py \
  tests/test_gate_pair_collector.py tests/test_gate_review_export.py
```

文本与分组已复核可精确再生成；准备脚本的源码快照和 SHA-256 记录在数据目录的 `preparation_sources/` 与 `preparation_verification.json` 中。后续 Refiner 配对会另行冻结当时的完整 system 代码和规则。

## 6. 弱监督门控原型训练

已完成的 GPU 实验位于 `results/learned_gate_weak_prototype_20261010/`，输入为
`results/qwen_tts_asr_gpu_pilot_20261010/weak_gate_labels_train_all.jsonl`。
它只划分原始 source train，按模板、录音和相同模型输入联合建组；原始 dev/test 未用于拟合或阈值选择。
760 条非 uncertain 标签中，内部训练/验证/测试为 444/146/170 条。
另外 73 条 uncertain 随所属组保留在政策评估分母。

在新输出目录复现：

```bash
"$GATE_PYTHON" experiments/scripts/train_weak_refinement_gate.py \
  --source results/qwen_tts_asr_gpu_pilot_20261010 \
  --output results/learned_gate_weak_prototype_fresh \
  --device cuda:0 --epochs 5 --batch-size 32
```

模型为本地 4 层中文编码器加共享三分类头，融合 CLS、局部候选跨度和任务嵌入；
未加权、训练集类别加权以及字符 TF-IDF 线性基线均在读取内部测试预测前声明。
本轮主模型为 `encoder_weighted`，checkpoint 与 tokenizer/config 在其目录内。
共享分类头用于这轮小样本原型，尚未实现设计方案中的完整实体/数字数值特征与独立任务分类头。

离线加载不改变线上服务：

```python
from system.weak_refinement_gate import WeakRefinementGate

gate = WeakRefinementGate("results/learned_gate_weak_prototype_20261010/encoder_weighted")
decision = gate.decide({
    "task_kind": "window_refine", "domain": "general",
    "raw_text": "我们我们明天去北京开会。",
    "baseline_text": "我们我们明天去北京开会。", "available_context": "",
})
print(decision)  # calibrated=False，合成弱监督原型
```

输入只裁剪最老的左侧上下文；目标本身超过 token 上限或候选偏移无效时返回 KEEP。
目标脚本、类别标签和本次精修结果不进入模型输入。
验证集选出的阈值不是风险校准阈值，日志中的概率也不能当作正确率。
当前原型将内部测试精修调用从 111 降至 24，但捕获收益从 49 降至 24；
目标精确率从现有门控的 80.3% 降至 65.9%，因此暂不替换现有门控。
CPU 单请求 p95 约 9.1 ms，只代表本轮短输入、双线程、无缓存测量。

针对性回归检查：

```bash
"$GATE_PYTHON" -m unittest discover -s tests -p test_weak_refinement_gate.py -v
```

## 7. 独立句式扩展与日志监督

`expand_gate_templates.py` 定义了 60 个新情景组：口吃/填充词对应清洁文本、正常强调/叠词、否定/数字/实体保护以及条件句。每组的所有口语变体和槽位值共用一个分区；检查与原数据去掉人名、地点和数值后的模板重合。情景角色不直接生成 gate label。

```bash
"$GATE_PYTHON" experiments/scripts/expand_gate_templates.py \
  --output results/gate_template_expansion_20261010
"$GATE_PYTHON" experiments/scripts/run_gate_template_pilot.py \
  --source results/gate_template_expansion_20261010 --device cuda:0
```

本轮冻结 108 条文本，其中 64 条 train、22 条 dev、22 条 test。驱动器仅处理 train，使用 clean ASR 输入；它只启动、停止自己创建的 8011 TTS 服务，复用 8766 ASR 服务，不停止已有 Web/ASR 服务。
阶段日志位于 `logs/`，状态原子写入 `job_state.json`，已完成 TTS/ASR/配对可以续跑。端口 8011 被其他进程占用时会明确失败，不会替换该服务。
原学生模型及阈值保持冻结，仅在新训练候选上做诊断；不会根据新诊断或旧内部测试自动调阈值。实际标签与每任务的 harm 模板覆盖记录在 `expansion_report.json`；新 dev/test 文本暂不合成，也不用于本轮拟合。

用户授权下，主代理负责实现，Luna 子代理独立只读检查阶段日志和训练完整性。上一轮的独立审计摘要见 [Luna 审计](/home/aim0/桌面/tan/ASR/Codex-web/results/learned_gate_weak_prototype_20261010/LUNA_AUDIT.md)。

2026-10-10 扩展试点已完成：64 条训练语音、64 条 clean ASR、64 个窗口配对和 22 个局部审核配对。
86 个实际弱标签中 benefit 8、neutral 63、harm 4、uncertain 11，有效弱训练记录 75 条。
窗口与局部审核各有 2 个 harm 模板组，并集是 3 个独立组；仍不足以验证各任务风险。
详情见 [新句式报告](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_template_expansion_20261010/REPORT.md) 和 [Luna 监督终检](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_template_expansion_20261010/LUNA_MONITOR_REPORT.md)。
`completion_integrity.json` 保存音频、数据和源码哈希检查，`teacher_version_check.json` 说明旧/新采集器的核心精修与局部审核函数、Refiner 权重、实体快照及数字库一致；Web 侧差异仅涉及本轮未启用的可选音频修复/UI 路径。

## 8. 第二轮：保留原门控的保守否决路由

实现入口为 `WeakRefinementGate.decide_with_fallback(input, native_action)`：
原动作 KEEP/DEFER 不运行编码器；原动作 REFINE 时仅在高 neutral 或高 harm 分数下否决。
超长目标、非法候选等不支持输入保留原动作。这仍是显式离线接口，没有接入在线 Web 路由。

第二轮只拟合原内部 train 和新增 train 的 519 条有效弱标签。
验证使用原内部 validation、新增 dev 的实际配对，以及14条额外开发风险候选的实际配对；
有效验证标签198条，窗口 harm 6条、6组，局部审核 harm 14条、7组。
额外候选只在开发集使用，既不进入训练也不充当独立测试。
旧内部 test 完全排除；新 test 的22条音频在 selection.json 冻结后才生成。

所有权重、配置、执行源码快照和固定数据位于
`results/gate_round2_20261010/training/`；报告为 [第二轮报告](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_round2_20261010/training/REPORT.md)。
本轮主策略在开发集保留原门控43/44个弱收益并将弱退化20→0，
但新测试的27个配对上，精修逻辑调用13→6，弱收益2→1，弱退化1→1。
独立决策配对未通过95%收益保留目标，因此没有替换在线门控，也没有根据测试调整冻结权重/阈值。
这些配对是分阶段独立执行，不能把2→1直接解释成最终转录质量下降。
后续实际顺序回放发现：22条录音新旧最终文本逐字相同，Refiner调用11→6；
窗口跳过的“工作工作”由重新生成的局部审核修好，旧指标重复计算了同一处收益。
两条策略仍都会误删独立测试中的强调语句，故尚不能部署。
14条已参与阈值选择的风险开发录音退化8→0，仅说明开发保护行为，不是泛化证据。
详细归因、范围与下一步见 [顺序回放审计](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_failure_audit_20261010/REPORT.md)。

后续工程优化已将共用重复保留校验接入提交、safe salvage、确定性清理及Web复核前置检查。
学习门控未重新训练；22条旧测试快照退化1→0，原门控/学生Refiner调用11→9和6→5。
24条新增文本开发探针中，学生仍漏掉5个原门控可修复的预期，不能据保护效果上线学生。
局部效果导出仅生成旧教师train-only提议，不能当新教师或音频金标直接训练。
新采集器保存`native_local_review_action`，教师签名包含新保护依赖；训练前应重采新教师结果。
详情与剩余边界见 [重复保留优化](/home/aim0/桌面/tan/ASR/Codex-web/results/repetition_preservation_20261010/REPORT.md)。

第三轮已重采新版教师：1192训练操作、396开发操作；原KEEP/DEFER不拟合，局部标签恢复38/18个弱收益。
三分类严格、三分类局部、二分类局部均完成3轮；开发预选三分类局部，策略在32条新测试语音生成前冻结。
真实WebSocket＋HTTP ASR的64会话测试，32配对raw/audio完全相同；主学生CER代理3.383%→4.323%，匹配22→18，Refiner24→19。
四处错跳不能用速度下降掩盖，学生继续不接入正式服务。迟到意图导致的重复删除是两条路径共有的基础链问题。
详情见 [第三轮报告](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_round3_20261011/REPORT.md)。

本轮原生否决使用的是“每任务”验证约束。测试结束后，仅为后续新试验代码
新增“验证来源×任务”的收益保留约束，避免大规模旧验证样本掩盖新增句式漏检；
这项后续约束没有追溯应用于本轮冻结策略。
`stage_gate_partition.py` 会拒绝将已评估的模板 test 源重新标作独立测试。
准备新 trial 时必须另外声明未评估的测试源。

```python
import torch
from system.weak_refinement_gate import WeakRefinementGate

torch.set_num_threads(2)  # 独立离线进程中的基准配置
gate = WeakRefinementGate("results/gate_round2_20261010/training/encoder_weighted")
decision = gate.decide_with_fallback({
    "task_kind": "window_refine", "domain": "general",
    "raw_text": "我们我们明天去北京开会。",
    "baseline_text": "我们我们明天去北京开会。", "available_context": "",
}, native_action="refine")  # 传入真实现有门控动作，不由Refiner结果倒推
```

开发合成阶段遇到TTS关闭等待超时，完整数据保留后从decode续跑；
驱动器新增仅针对其拥有的Popen子进程的有界关闭，以及源码快照/失败历史保存。
底层磁盘冷读取也造成过配对启动等待，子进程日志现采用无缓冲输出。
Luna监督到额度上限后，第二轮训练及最终核验由主代理接手。
