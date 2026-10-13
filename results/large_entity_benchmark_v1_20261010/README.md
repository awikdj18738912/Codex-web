# 千实体多领域测试集 v1

数据集已经构建，主代理独立验证通过：**1,346 个去重标准名称、2,410 条真实音频**，每条带来源参考转录，每个入库名称都有音频与实体跨度证据。Luna 完成来源整理、参考审核和汇总构建；主代理接手额度限制后的独立验证及运行检查。正式用户实体库与生产功能未改。

## 规模

| 项目 | 实际数量 |
|---|---:|
| 全局去重标准名称 | **1,346** |
| 人名 PERSON | 357 |
| 机构 ORG | 384 |
| 地名 LOC | 310 |
| 专业术语 TERM | **295** |
| 正例音频 | 2,110 |
| 词库未命中负例 | 300 |
| 主集总音频 | **2,410** |
| 主集总时长 | 约 3 小时 44 分钟 |
| 另列已见 FAQ 回归音频 | 103，不计主集 |

正例原目标为 2,500 条，最终少 390 条；保留实际有证据的录音，没有重复同一音频来填数量。名称按 NFKC＋简体标准字形去重，同一名称的多个来源或话题不重复计数。这里不是完整真实世界实体链接统计；尚未全面统一所有合法别称，未核实别名均未加入。

## 话题分布

| 话题组 | 标准名称数量 |
|---|---:|
| 新闻人物、组织、地理 | 1,051 |
| 医疗健康 | 101 |
| 科技、计算机、通信 | 24 |
| 工业交通 | 23 |
| 金融经济 | 26 |
| 法律政策 | 21 |
| 自然科学、化学 | 24 |
| 农业生态 | 28 |
| 历史教育 | 24 |
| 体育文化 | 26 |

少量名称跨话题，所以分组数不能直接相加。各组不均衡，新闻中的人名、机构、地名占多数；不能把总体效果当成每个专业领域都同样可靠。后续评估同时报告每组结果和跨组平均。

## 质量检查与边界

音频来自本地 AISHELL、Common Voice、SpeechIO、WenetSpeech；医学词源另参考 CMeEE，但只有实际出现在语音参考中的词才计入覆盖。AISHELL-NER 参考标注来自 Alibaba-NLP/AISHELL-NER，术语部分属于参考匹配与人工文字审核，标注层次分开保存。

主集排除了已知旧实体评估的 AISHELL 328 条、Common Voice 112 条和 SpeechIO 205 条底层录音。疑似泛词、参考拼写、类型冲突、词内误匹配及两个待听审的高字速录音进入隔离清单，不计入主库。未根据 ASR 表现挑选或删除音频。

独立验证已核对：全部冻结输入／输出哈希、2,410 条原始音频与 float32 PCM 哈希、1,346 个名称的完整参考跨度与支持证据、全库和 10 份话题词库的 SQLite 内容、正负例计数、参考／金标旁文件一致性，以及不存在重复原始或解码音频。

这些检查不等于逐条人工听审，也不等于所有姓名和术语已逐一进行外部权威考证。词库是根据标准参考提前建立，测试的是已知词汇的辅助作用，不是未知词汇泛化；同一来源或说话人仍可能与历史数据重叠。

## 文件入口

- [方案](/home/aim0/桌面/tan/ASR/Codex-web/experiments/LARGE_ENTITY_BENCHMARK_PLAN.md)
- [标准名称及证据](/home/aim0/桌面/tan/ASR/Codex-web/results/large_entity_benchmark_v1_20261010/dataset/entities.json)
- [音频、转录、实体跨度](/home/aim0/桌面/tan/ASR/Codex-web/results/large_entity_benchmark_v1_20261010/dataset/samples.jsonl)
- [独立实验实体库](/home/aim0/桌面/tan/ASR/Codex-web/results/large_entity_benchmark_v1_20261010/dataset/library/entities.db)
- [隔离条目](/home/aim0/桌面/tan/ASR/Codex-web/results/large_entity_benchmark_v1_20261010/dataset/quarantine.json)
- [独立验证报告](/home/aim0/桌面/tan/ASR/Codex-web/results/large_entity_benchmark_v1_20261010/independent_verification.json)

dataset/ 内文件已经冻结。AISHELL、CV、Wenet 音频路径引用本地原数据；SpeechIO 选中音频已抽取到 dataset/audio；所有主音频的统一 16 kHz 单声道 float32 原始 PCM 均在 dataset/pcm。

## 含精修的评估

评估输入独立存放在 evaluation_inputs/input_manifest.json，不含参考文本或金标名称；参考只在评分阶段读取。条件固定为无实体库、仅现有实体规则、实体规则＋原音复核，均使用现有精修模型和门控，保留数字及重复处理规则。

先用固定哈希选出的 24 条跨话题样本检查流程，再执行全量。正式报告区分真实 ASR 请求耗时与缓存响应进入真实 Refiner 的耗时。烟雾检查不能替代全部 2,410 条效果结论。

运行入口：

```bash
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/verify_large_entity_benchmark.py
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/evaluate_large_entity_benchmark.py prepare
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/evaluate_large_entity_benchmark.py collect
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/evaluate_large_entity_benchmark.py pipeline --device cuda:0
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/evaluate_large_entity_benchmark.py report
```

当前烟雾数据已准备，因此再次 prepare 会拒绝覆盖。collect／pipeline 支持跳过已完成样本。完整集合对各阶段加 `--all`，输出到独立 full_evaluation/，不会覆盖烟雾检查。GPU 精修阶段需要足够空闲显存；本轮运行管理器暂时释放备用 Web 8081 的显存，完成后恢复服务。

### 全量评估结果

全量评估已完成，覆盖 2,410 条音频和 7,230 条唯一 Web 会话。三组目标实体字面命中率分别为：无实体库 88.56%（2431/2745），仅实体规则 88.63%（2433/2745），实体规则加原音复核 94.83%（2603/2745）。对应 CER 为 2.71%、2.72%、2.16%。相较仅实体规则，原音复核改善 167 条、CER 退化 3 条；审计发现 4 处新增库表面名称不在来源参考中。

| 条件 | 字面命中 | CER |
|---|---:|---:|
| 无实体库 | 88.56% (2431/2745) | 2.71% |
| 仅实体规则 | 88.63% (2433/2745) | 2.72% |
| 实体规则 + 原音复核 | 94.83% (2603/2745) | 2.16% |

详细结果见[全量报告](full_evaluation/REPORT.md)和[独立核对记录](full_evaluation/LUNA_CHECK.md)。
