# 实体库功能实测报告

2026-10-03，工作区 `/home/aim0/桌面/tan/ASR/Codex-web`。

实体库确实能生效：精确别名会被规范化，已识别的术语会被保护，模糊匹配也能纠正部分同音错字。但这次真实录音对照没有证明当前 `auto` 能稳定提高总体准确率。扩大词库以后，纠正收益和误改同时出现；使用界面支持的类型配置时，本次总体字符错误率反而上升。建议先保留显式别名和保护能力，修复实体边界及上下文判定后，再扩大自动模糊替换。

**项目审阅范围与实体库实际位置**

已梳理 `system/`、`pipeline/`、`experiments/`、`scripts/`、`tests/` 的目录、配置、入口和调用关系，重点通读实体库所有调用入口、匹配、掩码、恢复、门控、窗口缓存及后置校验。Python 文件的模块、导入、符号和行号清单见 [project_inventory.json](project_inventory.json)，启动时的源码哈希和工作区状态见 [provenance.json](provenance.json)。这是对主项目及实体库链路的审阅；附带的 `audio.cpp/` 原生框架和其第三方依赖只检查了主要说明、入口及关联，没有逐行审计其全部模型实现、ggml、sentencepiece 等源码。

| 模块 | 实际职责及与实体库的关系 |
| --- | --- |
| `pipeline/` | 生成 Seed、Oral、Clean 和 ASR 模拟数据，做 QC、去重及 SFT 导出；训练记录的 keyword 与运行时 SQLite 词库不是同一套数据 |
| `system/qwen_asr_stream_server.py` | 接收 float32 PCM，做 VAD、分段、Qwen3-ASR 解码和置信度记录；没有加载实体库 |
| `system/entity_store.py` | SQLite 实体与别名管理，按所选领域加 `general` 过滤，转录开始时加载快照 |
| `system/entity_matcher.py` | 字符/拼音 bigram 召回、滑动子串评分、候选间隔及实体类型限制 |
| `system/protection.py`、`entity_pipeline.py` | 将精确命中和获准自动替换的实体掩码；恢复标准名；生成基线及术语提示；校验 Refiner 输出 |
| `system/web_app.py`、`window_refinement.py` | WebSocket、三态门控、K=3 窗口、缓存与最终一致性处理；实体改写在模型调用前就可以生效 |
| `system/refinement_guard.py`、`zh_itn_bridge.py` | 内容、数字、占位符等检查；当前服务启用 zh-itn v90 |
| `experiments/` | 批处理与 CER/WER/MER、Judge、校准及规则实验；批处理和 Web 使用的提示词、数字路径、默认模糊模式并不完全相同 |
| `system/live_asr.py` | sherpa-onnx + MLX 的旧入口，未接 SQLite 实体库；`live_qwen_refiner.py` 和 Web 已接入 |

实体库目前位于 ASR 之后，因此本次测试的是文本后处理效果。数据库和术语提示没有传给 Qwen 的音频解码器，不能把收益解释为 ASR 热词解码能力。

实际运行的 Web 参数是 `entity-fuzzy-mode=auto`、`refinement-gate-mode=tri_state`、`final-refinement-mode=off`、zh-itn v90。命令行 `create_app`、批处理的默认模糊模式仍是 `shadow`。`off` 只关闭模糊候选，不会关闭精确别名规范化；真正的无库基线必须不加载实体定义。

**数据和实验方法**

使用本机已有的 [AISHELL-1](https://www.openslr.org/33/) 真人录音与人工转写，下载作者公开的 [AISHELL-NER 标注](https://github.com/Alibaba-NLP/AISHELL-NER)。NER 标注与本次候选测试录音的原始转写一致，没有发现文本不一致的记录。

词库由 train + dev 的实体标注生成，按频次取 500 个至少 3 个字的标准名：239 个机构、181 个人名、80 个地点。没有用测试转写或测试 ASR 错误补充词库、别名、调阈值。所有库都使用 `normalize`、priority 0，无补造的同音别名。

固定随机种子 20261003，从 test 选择 60 条含库内实体、20 条仅含库外实体、20 条无实体的录音，并按说话人轮转抽样。共 100 条原始录音、20 位说话人。每条另生成一份 10 dB 白噪声版本，组成 200 次音频解码，累计约 17.78 分钟音频。噪声来自固定随机种子；这 200 次不是 200 条独立原始录音。选样详见 [selection.json](selection.json)。

真实 Qwen3-ASR-1.7B、Silero VAD 使用现有 8766 服务；16 kHz 单声道 float32 PCM，1 秒传输块，0.5 秒解码设置。保存每个块和最终响应，再将同一份实际 ASR 响应回放进未修改的 WebSocket 处理器，执行真实 GPU AgenticASR-Refiner、门控、zh-itn、掩码恢复和校验。这个设计固定了 ASR 输入，避免不同次解码影响词库对照；它用于质量评估，不用于端到端延迟评估。

主对照为六种配置，共 1200 次 Web 会话。随后补充 200 次符合界面实体类型的配置；这组只做 LOC→TERM 类型映射，没有改词表或阈值。主结果和补充结果均完整保留。

| 配置 | 词库和处理方式 |
| --- | --- |
| `no_dictionary` | 不加载实体库，保留相同 Refiner、数字和安全链路 |
| `current_snapshot_auto` | 当前实际实体库的独立副本；共 11 项，本次 `general` 会话加载其中 6 项 |
| `typed_shadow` | 500 项保留 PERSON/ORG/LOC 标注，只记录模糊候选 |
| `typed_hint` | 500 项保留类型，模糊候选作为 Refiner 提示 |
| `typed_auto` | 保留类型；PERSON/ORG/LOC 不在 auto 类型白名单中，模糊命中会退为提示 |
| `native_auto` | PERSON/ORG 保留，LOC 映射为界面提供的 TERM，地名可以自动改写 |
| `all_term_auto` | 全部标为 TERM，单独观察自动纠错上限及风险；这是诊断配置，不是建议将人名机构全部这样配置 |

**完整 Web 链路结果**

下表主指标是 ITN 后 CER：参考与输出都用相同 zh-itn v90 做数字格式归一，再去标点和空白，累加编辑距离并除以参考字符总数。这样不会把“七月二十日→7月20日”误算成实体识别错误。它仍受数字规则覆盖限制，不是人工语义评分。原始 CER 也保存在 [summary.json](summary.json)。

实体指标是标注标准名的出现次数召回：按标准名逐项检查输出出现次数，不是 NER F1，也不评价实体角色。200 次输出共 254 个标注实体，其中 126 次属于 500 项测试库的覆盖范围。

| 配置 | 干净 CER | 加噪 CER | 合并 CER | 全部实体召回 | 库内实体召回 | 较无库改善/恶化的音频次数 | 实际 Refiner 调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 不使用实体库 | 2.809% | 4.311% | 3.560% | 85.83% | 92.86% | 0 / 0 | 5 |
| 当前实际库 auto | 2.809% | 4.311% | 3.560% | 85.83% | 92.86% | 0 / 0 | 5 |
| 扩充库 typed shadow | 2.809% | 4.311% | 3.560% | 85.83% | 92.86% | 0 / 0 | 5 |
| 扩充库 typed hint | 2.743% | 4.246% | 3.494% | 85.83% | 92.86% | 2 / 0 | 14 |
| 扩充库 typed auto | 2.743% | 4.246% | 3.494% | 85.83% | 92.86% | 2 / 0 | 14 |
| 符合界面类型的 native auto | 2.874% | 4.376% | 3.625% | 85.83% | 92.86% | 2 / 4 | 10 |
| 全部 TERM auto，诊断组 | 2.678% | 4.246% | 3.462% | 87.80% | 96.83% | 5 / 4 | 5 |

当前库与这批新闻、体育和生活录音的实体不匹配，未产生规范化收益。这说明领域覆盖不足，不能据此说实体库本身不工作。

全部 TERM 的诊断组修正 5 次、误改 4 次，总编辑错误从 109 降到 106，净减少 3 个字符，下降 0.098 个百分点。改善实际来自 3 条原始录音，恶化来自 2 条原始录音；干净和加噪版本有重复贡献。按原始录音成组重采样的 2000 次配对 bootstrap，CER 差值 95% 区间为 [-0.471, +0.199] 个百分点，包含 0，不能据此宣称稳定提升。该区间没有进一步按说话人聚类，仍只是这次小样本的参考。

符合界面类型的 `native_auto` 净增加 2 个字符错误，CER 从 3.560% 升到 3.625%。人名、机构的 auto 限制仍在，但地名允许自动替换就能触发普通词误改。

库外实体和无实体两组在本次对照中没有新增字符错误。四次模糊误改都发生在同时包含已正确识别实体的句子中。因此，仅测试“完全不含实体”的负例不够，还需要测实体旁边的普通词。

**真实录音中的纠正和误改**

| 录音 ID | 人工参考/ASR/输出 | 结果 |
| --- | --- | --- |
| `BAC009S0907W0403` | 参考“刘艾林”，ASR“刘爱林”，全部 TERM auto 输出“刘艾林” | 干净和加噪各纠正 1 字 |
| `BAC009S0904W0350` | 参考“曾春蕾”，干净 ASR“曾春雷”，全部 TERM auto 输出“曾春蕾” | 干净版纠正 1 字；加噪版本来正确 |
| `BAC009S0907W0388` | 参考“吕会会”，ASR“吕慧慧”，全部 TERM auto 输出“吕会会” | 两个版本各纠正 2 字 |
| `BAC009S0905W0412`、`BAC009S0902W0430` | 正确的“北京时间”被改成“北京市间” | 两条录音的两个版本均误改；普通子串“北京时”被匹配为“北京市” |

typed hint/auto 只把“吕慧慧”改为“吕会慧”，没有恢复完整标准名，所以 CER 略有下降，实体召回没有改善。真实模型收到“刘艾林”“曾春蕾”的 KEY 提示后，仍保留原错名并在 KEY 中重复原错名。记录见 [web_offline.jsonl](web_offline.jsonl) 中的 `model_calls`，并非仅凭提示已传入就判断有效。

**功能诊断与原因**

构造文本测试独立保存在 [text_diagnostics.json](text_diagnostics.json)，不纳入上面的音频指标。

| 功能 | 实测 |
| --- | --- |
| 精确别名 | 当前库“乌冬得→乌东德”正确生效，连 fuzzy=off 也会生效 |
| 模糊纠错 | 当前库“鬼灵们→鬼灵门”正确生效 |
| preserve 策略 | “北航”保留原表面，不扩展标准名 |
| 领域过滤/停用/删除 | 独立库验证通过，未修改原库 |
| 同音负例 | 正确的“外出务工人员回乡”被 auto 改为“外出误工人员回乡” |
| 低 ASR 置信度 | confidence=0.01 时仍会执行上述误替换；自动评分不使用 ASR 置信度作为拦截条件 |
| shadow | 记录“务工→误工”候选，但保持原文 |
| 损坏占位符后的回退 | 回退到已经做过实体替换的基线，不能撤销“务工→误工”这样的错误 |

关键原因：

1. 匹配器使用滑动子串，没有中文词边界或实体边界判定。“北京时”的拼音与“北京市”相同，字符也重合 2/3，得分 0.8967，超过 0.80 阈值。它没有检查这三个字实际属于“北京时间”。
2. 相似度分数不是正确率。“务工→误工”得分 0.855，拼音为 1.0；缺少竞争候选时 margin 直接为 1.0，也不能证明语义正确。
3. 后置检查比较的是 `prepared.baseline_text`。错误的自动替换先进入基线并被掩码保护，校验会把这个错误当作需要保留的实体。
4. 当前 Web 的 `prepare_entity_segment(... allow_auto=True)` 对中间窗口和 pending 文本同样生效。64 次流式诊断里，中间 update 已出现“北京时间→北京市间”。`system/README.md` 的“仅最终阶段 auto”说明与现代码不一致，候选理由中的 `final_event` 也不能证明那是最终事件。
5. Refiner 的 KEY 是标准名列表，没有显式的 observed→canonical 候选映射。这次多数同音提示没有纠正完整实体；是否能改进需要新的提示词/模型对照，不能只靠增加词条。

**链路验证、局限与改进顺序**

回归测试：`python -m pytest -q`，346 passed，1 条 Starlette/httpx 弃用提醒，详见 [pytest.log](pytest.log)。新增评测脚本通过语法检查，所有评测阶段均实际执行完成。没有改动业务源码、原实体库或用户已有源码修改；数据库两张表与初始副本一致，原有源码哈希未变，两个服务保持健康，检查见 [workspace_preservation.json](workspace_preservation.json)。直接调用运行中 Web 的 smoke 测试会正常追加会话日志和两份服务报告。

流式诊断选择 8 条原始录音及其加噪版本，对无库、当前库、native 和全部 TERM 四组共做 64 次回放会话。64/64 的最终文本与对应离线输出一致，但正确纠正和误改都已经在中间 update 生效。该子集是在看到主结果后用于诊断选取，不作为独立统计样本，也没有模拟真实麦克风时间间隔。

另将两条真实录音的 PCM 直接发送给现有 8081 Web 服务，实际连接 8766 ASR，最终原始转写与保存的 ASR 结果一致，结果见 [live_web_smoke.json](live_web_smoke.json)。批量对照中的 Refiner 也是真实本地 GPU 模型；回放只替换 ASR HTTP 返回值，没有用假 Refiner。

这是短句、普通话、20 位说话人的小规模测试，且特意提高了库内实体样本比例，不能外推为整个 AISHELL 测试集、长会议、方言或当前小说音频的总体提升。白噪声也不能代表所有现实噪声。没有用合成语音或构造错字替代真实录音评价，没有因收益小而省略误改。

建议按以下顺序处理：先给候选增加实体边界/分词及上下文验证，并让后置校验能比较原始 ASR 与实体改写，必要时撤销改写；通用领域避免让短词自动替换普通词；先用 shadow 审查真实候选，再按具体领域维护显式别名；人名机构维持类型限制；随后试验 observed、canonical、跨度和备选项的结构化提示。最后扩大独立音频集，同时报告纠正、误改、实体召回和模型调用。

**复现和产物**

新增脚本：

- [evaluate_entity_library.py](../../experiments/scripts/evaluate_entity_library.py)：固定选样、词库构建、真实 ASR、13/17 组实体层对照与构造文本诊断。
- [evaluate_entity_web_replay.py](../../experiments/scripts/evaluate_entity_web_replay.py)：真实 Refiner 的完整 Web 回放，对照离线和流式。
- [report_entity_library.py](../../experiments/scripts/report_entity_library.py)：CER、实体召回、改善/恶化、配对置信区间及逐例结果。

从仓库根目录重新计算已有结果：

```bash
PYTHONPATH=. python experiments/scripts/evaluate_entity_library.py match
PYTHONPATH=. python experiments/scripts/evaluate_entity_library.py diagnostics
PYTHONPATH=. python experiments/scripts/report_entity_library.py
```

ASR 和 Web 阶段需要该机的 `agentic-asr` 环境，已完成记录会跳过。用新的输出目录开始一个完整实验时，指定现有标注文件；请保留相同模型、配置与数据路径。脚本 `prepare` 拒绝覆盖现有测试库。

```bash
PYTHONPATH=. python experiments/scripts/evaluate_entity_library.py prepare \
  --output results/entity_library_repeat \
  --annotations results/entity_library_20261003/source/aishell_ner_transcript.txt

PYTHONPATH=. /home/aim0/data/conda/envs/agentic-asr/bin/python \
  experiments/scripts/evaluate_entity_library.py asr --output results/entity_library_repeat

PYTHONPATH=. python experiments/scripts/evaluate_entity_library.py match \
  --output results/entity_library_repeat

PYTHONPATH=. /home/aim0/data/conda/envs/agentic-asr/bin/python \
  experiments/scripts/evaluate_entity_web_replay.py --output results/entity_library_repeat \
  --conditions no_dictionary current_snapshot_auto typed_shadow typed_hint typed_auto native_auto all_term_auto

PYTHONPATH=. python experiments/scripts/report_entity_library.py \
  --output results/entity_library_repeat
```

[asr.jsonl](asr.jsonl) 保留实际音频 hash、人工参考及每块/最终 ASR 响应；[entity_only.jsonl](entity_only.jsonl) 保留实体层各模式候选；[web_offline.jsonl](web_offline.jsonl) 保留 1400 次完整链路的输出及模型调用；[web_streaming.jsonl](web_streaming.jsonl) 保留 64 次流式诊断；[summary.json](summary.json) 是机器可读汇总。测试库、标注来源、配置和全部日志均保存在本目录。
