# 专业医疗音频与实体库实测

2026-10-03。结论：**已确认的完整短语别名可以规范专业术式名称；只批量添加标准医学词，当前自动模糊替换没有显示稳定收益，并产生了医学文本内部的误改。**

这次建立了独立的 229 项医疗测试库，并在正在运行的实际实体库里新增一条经过功能验证的 `TAVI手术` 记录。测试库和实际服务的新增记录分开保存，便于复现和审阅。

**录音与词库来源**

音频使用作者公开的 [MultiMed 中文医疗集](https://huggingface.co/datasets/leduckhai/MultiMed)，包含医生讲解、医患交流和医疗纪录内容。直接下载 Chinese train/eval/test 的 Parquet，并提取其中原始真人音频；没有生成合成语音。标准词来源为 [CMeEE 训练数据](https://huggingface.co/datasets/Aunderline/CMeEE)，原任务说明见 [CBLUE](https://github.com/CBLUEbenchmark/CBLUE)。

词库主要取 CMeEE 训练标注中在 MultiMed train 出现的完整术语；另纳入人工审阅的两字解剖/疾病词和训练集中出现的专业术式缩写。剔除部分不完整、普通片段，并对纯英文术语的训练集出现要求字母边界，避免把 CAT 从 transcatheter、ECM 从 ECMO 中抽出来。最终 229 项，均在 `medical` 领域，类型 TERM、normalize、priority 0。代表词包括骨质疏松、主动脉瓣、前列腺、脊髓、腹腔镜、鳞状细胞癌。词表见 [vocabulary.json](vocabulary.json)，可用 SQLite 见 [medical.db](medical.db)。

229 项标准词及空的开发错字别名方案在 held-out ASR 解码前冻结。没有根据测试集错字补词、补别名或调阈值。现有规则配置保持原样。

开发集取 88 条时长不超过 30 秒的录音，共解码 176 个干净/10 dB 白噪声版本，用于审阅候选错误。最初自动对齐提出的若干“别名”实为口癖、字幕省略或不等价措辞，没有直接写入全库，例如不将普通词“曾经”替换成“神经”，也没有把“针孔／真空”强制当成“腹腔镜”的别名。

独立主测试使用 test 中全部 215 条不超过 30 秒的录音，原音频约 37.92 分钟；每条另加一份固定种子 10 dB 白噪声版本，共 430 次音频解码、约 75.83 分钟音频。超长的 10 条按事先的时长规则排除，未按识别成败挑样本。冻词表在其中 49 条原始录音的配套转写里有标准名出现；干净与加噪共 124 次词表术语出现。选样见 [selection.json](selection.json)。

另沿用上一轮 20 条普通领域录音及其噪声版本，共 40 个负例，并做相同四组完整 Web 对照。它们不混入医疗主测试的统计。

**处理链路与结果**

真实 Qwen3-ASR-1.7B 和 Silero VAD 使用已有 8766 服务：16 kHz 单声道 float32 PCM、1 秒传输块、0.5 秒解码设置。保存每块与最终响应后，将相同真实 ASR 响应回放进项目 WebSocket 处理器。Refiner 是真实本地 GPU checkpoint，门控 tri_state、final refinement off、zh-itn v90；实体准备、掩码、恢复、K=3 调度、复核和输出校验正常运行。没有用假 Refiner。

四种配置共完成 1,720 次医疗 Web 会话。原始 ASR 输入固定，因此比较的是词库对后处理的影响。词库没有传入 Qwen 解码器，本次不能解释为 ASR 热词解码测试。回放用于质量比较，未测实时麦克风延迟，也未使用新增的模型响应缓存选项。

CER 使用相同 zh-itn 对参考和输出做数字格式归一，再去标点、空白、比较编辑距离。术语指标是词表标准名在配套转写及输出中的出现次数，不是独立人工 NER F1。模型调用使用服务端 `session_refiner_call_count`，包含复核；`model_calls` 另外保存常规精修调用的输入和输出。

| 完整 Web 配置 | ITN后 CER | 术语出现召回 | 较无库改善 / 恶化的音频版本数 | Refiner调用 |
| --- | ---: | ---: | ---: | ---: |
| 无实体库 | 32.219% | 107/124（86.29%） | 0 / 0 | 548 |
| 229项医疗库，仅精确保护 | 32.212% | 107/124（86.29%） | 4 / 2 | 544 |
| 229项医疗库，模糊提示 | 32.212% | 107/124（86.29%） | 4 / 2 | 557 |
| 229项医疗库，自动替换 | 32.295% | 107/124（86.29%） | 4 / 11 | 545 |

标准词库没有增加术语召回，四组均为 107/124（86.29%）。精确保护和提示带来的少量字符变化主要是一般文字或口癖处理，净少 1 个字符错误，不能归为专业术语识别提升。

自动组总体多 12 个字符错误：5048→5060，CER 增加约 0.077 个百分点，4 个版本改善、11 个版本恶化。按原始录音将干净与噪声成组的 2000 次 bootstrap，CER 差值 95% 区间为 [-0.018, +0.189] 个百分点，包含零。这批数据没有支持稳定总体收益；具体误改则已经逐例复现。

单独去掉 Refiner，只观察实体层，自动组没有减少字符错误，反而在 8 个版本、4 条原始录音上增加 10 个错误字。明细见 [entity_only.jsonl](entity_only.jsonl)。完整链路也没有拦住这些前置替换。

| 实际录音中的问题 | 原因 |
| --- | --- |
| 手术失败 → 手术室败 | 将普通短语里的“手术失”当成“手术室” |
| 手术是比较复杂 → 手术室比较复杂 | 将谓语结构当成名词实体 |
| 手术视野 → 手术室野 | 只匹配术语前几个字，破坏了更长的专业表达 |
| 骨碎补 → 骨髓补；骨碎了 → 骨髓了 | 将中药名及普通动词误归为解剖名“骨髓” |
| actually → a_CT_ually | 精确匹配忽略英文字母边界；CT 命中词内 ct，占位符恢复还留下额外下划线 |

CT 的问题在 fuzzy=off 的精确保护模式也能出现。当前 CER 去掉标点和大小写后会漏掉下划线这类显示异常，故必须检查逐条文本，不能只看汇总数。

40 个普通领域负例的四组 CER 均为 0.735%，没有新增字符误改。这不能覆盖医学句子内部的普通词：上述“手术是”“手术失败”等反例恰恰出现在专业语音中。

整体 CER 受配套字幕省略口癖、英语段落缺标和片段边界影响。例如 test 0013 的原音频包含较长的英语交流，配套参考主要保留中文叙述。因此整体值适合比较相同 ASR 输入的处理差异，术语计数和逐例语义检查需要同时看。说话人、来源片段没有独立逐人划分信息，置信区间仅按录音聚类；本次不是覆盖全部医疗场景的基准验收。

**明确别名的正例和实际添加**

另使用 development 0003 做功能演示，独立于主测试统计。该录音的术式标准名为 TAVI，术语拼写可在 [NHS 的 TAVI 说明](https://www.guysandstthomas.nhs.uk/health-information/tavi-aortic-stenosis/tavi-procedure) 核对。实际 ASR 输出为“凯比手术”，加噪版本为“凯蒂手术”。

加入完整短语别名后：

| 同一份开发录音 | 无库输出中的术式 | 使用明确别名后的术式 |
| --- | --- | --- |
| 原录音 | 凯比手术 | TAVI手术 |
| 10 dB 加噪版本 | 凯蒂手术 | TAVI手术 |

模糊模式 off 和 auto 都能正确规范，说明收益来自经过确认的精确别名。这一原始录音的两个版本中，术式出现召回由 0/2 变为 2/2；这是功能正例，不能作为独立测试集的泛化提升。数据与六次真实模型会话见 [context_alias_demo](context_alias_demo)。

运行中的 Web 实体库已新增：

- entity_id：12；标准名：`TAVI手术`。
- 领域：medical（医疗）；类型：TERM；策略：normalize；priority：10。
- 别名：`凯比手术`、`凯蒂手术`。保留“手术”后缀限定短语使用位置。

已向实际 8081 Web 服务发送这一录音的真实 PCM，分别选择 general 和 medical。通用领域保留“凯比手术”；医疗领域输出“TAVI手术”，返回 `EXACT_ALIAS` 规范记录。这个验证实际经过运行中的 ASR 与 Web 服务，结果见 [live_domain_comparison.json](live_domain_comparison.json)，API 保存结果见 [installed_entity.json](installed_entity.json)。两次重新解码的人名略有差别，故只比较已记录的术式规范，不用它们计算整体 CER 提升。

在页面的“术语领域”选择“医疗”，然后开始新的转录会话即可加载新增记录。原有 11 条实体及其原别名保持完整。229 项批量库保存在独立测试数据库中，实际服务启用的是上述经过功能验证的短语记录。

原始演示录音：[试听 / 下载](/home/aim0/桌面/tan/ASR/Codex-web/results/medical_entity_20261003/development/audio/MultiMed_Chinese_eval_0003.wav)。

**改进重点与复现**

当前收益依赖于词库是否覆盖真实错误写法和位置。标准名多数已被 ASR 写对，漏词、非同音错误或英语缩写的中文误写无法由现有拼音匹配自动恢复。扩大标准词数量同时会增加对普通子串的误匹配。

优先修复中文词/实体跨度和英文边界判定，并允许校验比较原始 ASR 与实体改写、撤销不可信前置替换；完整中药名也应优先于解剖短词。随后才适合扩大自动模式。明确别名应使用已核实的完整短语，按领域维护，并保留错误案例作为后续回归材料。

新增 [evaluate_medical_entities.py](../../experiments/scripts/evaluate_medical_entities.py) 负责医学生词、音频提取、开发/测试解码与实体层对照；已有回放脚本新增领域参数，汇总脚本处理无实体负例并使用服务端总调用数。业务源码哈希未变，评测没有改识别或匹配算法。所有阶段实际完成，脚本通过语法检查与 `git diff --check`；没有将原有用户源码修改回退。

从仓库根目录重算现有结果：

```bash
PYTHONPATH=. python experiments/scripts/evaluate_medical_entities.py match
PYTHONPATH=. python experiments/scripts/report_entity_library.py \
  --output results/medical_entity_20261003
```

重跑完整 Web 对照使用本机 Refiner 环境，已完成记录会跳过：

```bash
PYTHONPATH=. /home/aim0/data/conda/envs/agentic-asr/bin/python \
  experiments/scripts/evaluate_entity_web_replay.py \
  --output results/medical_entity_20261003 --domain medical \
  --conditions no_dictionary medical_off medical_hint medical_auto
```

从零准备需另建输出目录，复制/下载四个源文件（CMeEE_train.json、Chinese train/eval/test Parquet），并提供 pyarrow、soundfile、scipy。此次 pyarrow 装在项目 `.runtime/medical_entity_deps` 的隔离目录中，模型环境不增加依赖。录音提取与已冻结选择、源文件 SHA256 见 [provenance.json](provenance.json)；常规 ASR 和回放阶段不需要 pyarrow。

主要产物：[asr.jsonl](asr.jsonl)、[web_offline.jsonl](web_offline.jsonl)、[summary.json](summary.json)、[controls/summary.json](controls/summary.json)、[development_alias_review.json](development_alias_review.json)。完整记录包含实际音频 hash、配套参考、每块/最终 ASR 响应、实体候选、规范、回退和服务端调用统计。
