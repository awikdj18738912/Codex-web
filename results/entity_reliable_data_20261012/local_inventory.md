# 本地可用语料与参考质量盘点

盘点范围是 `/home/aim0/data/datasets/ASR` 和项目已有的来源清单、staging 统计。没有播放或逐条列出音频，没有下载数据；181 留出只读了 `unseen/summary.json` 的来源去重及 hash 汇总，没有打开参考文件。语料名字或存在 `text` 字段都不等于人工转录证据。

## 本地数据

| 来源 | 本地形式与规模 | 转录/参考证据与主要风险 |
|---|---|---|
| AISHELL-1 | WAV 压缩包、JSONL 清单；train 120,098、dev 14,326、test 7,176 行；目录约 49 GB | 项目 staging 指向官方 transcript 与 AISHELL-NER 标注，6,848 条 test 对齐、0 个文本对齐不一致。现有本地说明没有证明原始转录是人工逐字制作。优先级高，但现有测试音频已大量复用，应按 ID 和音频 hash 去重。 |
| Common Voice zh-CN 26.0 | 852,010 个 MP3；README 报告 190,017 条 validated clips、1,074.32 小时；train/dev/test 共 51,029 条；约 25 GB | README 说明社区对音频与给定句子投赞成/反对票，是明确的人类匹配校验证据。它是预设文本朗读，不代表自由语音人工转写或实体边界标注。validated 的未用 ID 仍很多。 |
| WenetSpeech TEST_NET | 24,774 条 JSONL 段记录和 24,774 个 WAV；约 2.6 GB | 清单有音频/文本信息，没找到本地文件能证明人工转写或逐条审核。项目此前用过 25 条基准样本和 26 条 medical staging 样本；同一语料的其余片段仍可在去重后考虑。 |
| WenetSpeech 其他分区 | JSONL manifests；dev 13,825 行、train_s 151,599 行、另有约 346 MB 的 train_m manifest；manifest 目录约 705 MB | 本地只确认清单，未确认相应音频可用，也未找到人工转录证明。 |
| FLEURS zh-CN test | 两个 Arrow shard，合计约 709 MB；dataset_info 记录 945 个测试样本；整个本地 FLEURS 树约 4.2 GB | schema 有原始和规范化 transcription 字段，但本地资料没有交代人工转录/校验流程。可作为新来源候选，需人工听音。 |
| MultiMed Chinese test | 单个 Parquet，225 行、约 33 MB；字段包括 bytes、path、text、duration | schema 没有标注来源说明。已有项目观察提示至少有音频与参考不匹配风险；这应作为逐项复核的警示，不能泛化成全部 225 条都错。 |
| AASR-Bench | 本地有 JSONL benchmark 和 rubric；README 声称 917 条及 917 个 WAV，但数据目录内未找到音频；约 5.5 MB | README 没有说明转录/标注者，且明确将音频来源、标注来源、许可和隐私审查列为待确认。任务侧重口语清理、纠错、格式和改写，不是已证实的逐字 ASR 金标准。 |

## 项目已有来源与去重

冻结的 2,410 条 benchmark 来源计数为 AISHELL-1/AISHELL-NER 2,131、Common Voice test 192、SpeechIO 62、WenetSpeech TEST_NET 25。2026-10-11 的三份 holdout manifest 合计 400 个唯一音频 ID，manifest 限制说明来源是 AISHELL 和 Common Voice。181 条 unseen 的 summary 记录与这 2,410 条、此前 400 条、328 个 AISHELL 历史 ID 和 112 个 Common Voice source ID 去重；未读取其参考。

SpeechIO 虽然不在 ASR 根目录下单独建库，但项目内 `results/speechio_professional_entity_eval_20261010/source/` 有 6 个 Parquet shard，共 12,254 行、约 2.18 GB，schema 有 `segment_id`、`speaker`、`text`、`duration`、`bytes`、`path` 和 `original_full_path`。本地资料没有说明这些文本由谁转录或如何核验。之前 professional staging 使用其中 65 条，并排除了 416 个 SpeechIO identifier；这不等于整个 SpeechIO 来源都不可再用。

同一语料来源不应被整体丢弃。先用原始 source ID 和实际音频 hash 剔除已用录音，再从未用分区/ID抽样。AISHELL-1 的 train/dev 与 Common Voice validated 还留有大量未用记录。对新来源优先级，AISHELL 的官方 transcript/NER 来源信息和 Common Voice 明确的社区音频匹配投票是目前更好的本地证据；二者仍需按具体音频核验。WenetSpeech TEST_NET、FLEURS 可以纳入听音审核队列，但当前本地来源材料不足以把它们标成已人工验证。MultiMed 要先定位并复核已知不匹配项，AASR-Bench 则需先补齐本地音频和标注来源说明。

## 请求检查的语料

SpeechIO 的 Parquet 数据在项目结果目录内；WenetSpeech TEST_NET 在本地 ASR 目录。ASCEND、GigaSpeechMandarin、AISHELL-4 和 KeSpeech 均未在 `/home/aim0/data/datasets/ASR` 的目录名中找到。本结论只表示该指定本地目录未发现，不表示数据集不存在或不可获得。
