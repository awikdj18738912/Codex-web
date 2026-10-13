# 编辑检测器：来源核对、公开材料与验证对照

日期：2026-10-13。继续上一轮高误报问题，本次落实了来源核对、3,000 对公开口语文本导入和来源可追溯编码器对照。**验证结果有改善，但尚未达到上线条件。现有服务及其门控档位没有改动。**

## 上一轮失败暴露的两个问题

原训练材料并没有看起来那么多样：105 条合成填充词正例只用了 5 种固定删除串，每种 21 条；105 条口吃正例同样只有 5 种固定删除串，主要通过换人名和地名扩充。每种删法与单一模板绑定，验证、测试还分别留出了单一词形。无模板交叉并不等于足够丰富的口语监督。

本地 `models/entity_ranker/encoder` 是完整可加载的 4 层、hidden=256 BERT，但缺少上游模型、revision 或预训练记录。现有资料既不能证明它具备可追溯的语言预训练，也不足以断言它是随机权重。它的词表与 HFL 相同，权重却不是 HFL 的直接截层/截维结果。不能把“能加载 checkpoint”直接等同于“已核实预训练来源”。

## 已导入的公开材料

从作者 [Translation_Disfluency_Detection 仓库](https://github.com/dqqcasia/Translation_Disfluency_Detection) 下载了配对文本与标注规则，固定提交 `0456baded348fa9f8617c6e0a14757fe7b7c31d0`。原始文件在 `data/public_edit_sources_20261013/chinese_disfluency/`，下载 URL、文件列表和 SHA256 已保存。

作者说明发布了 3,000 条标注口语文本。实际 `dis.raw.3k` 与 `flu.raw.3k` 各 3,000 行。标注规则面向翻译前的口语标准化，包含断句、识别错误、语法简化及不流畅处理，其目标比当前忠实精修更宽。因此不将整句目标直接当本项目金标，也不把未改字符标成已确认 KEEP。依据：[作者说明](https://github.com/dqqcasia/Translation_Disfluency_Detection/blob/0456baded348fa9f8617c6e0a14757fe7b7c31d0/README.md)、[作者标注规则](https://github.com/dqqcasia/Translation_Disfluency_Detection/blob/0456baded348fa9f8617c6e0a14757fe7b7c31d0/data/chinese_disfluency/chinese_annotation_rules_2018.pdf)。

导入结果位于 `results/public_edit_candidates_20261013/`：

| 分类 | 配对数 |
|---|---:|
| 符合小范围、低删除比例及当前完整性检查的删除候选 | 358 |
| 目标原样保留，不能自动充当全局 KEEP 金标 | 449 |
| 当前完整性保护冲突 | 1,734 |
| 删除比例过大，需核对 | 265 |
| 源或目标含破损/未知角括号标注 | 91 |
| 数字值、单位或表面完整性冲突 | 90 |
| 单次删除跨度过大 | 12 |
| 其它非删除标准化改写 | 1 |
| 合计 | 3,000 |

358 条候选包含 528 个删除操作、111 种删除串。候选的删除位置是正位置提议，其余字符用 `-100` 表示未知，不产生 KEEP 负标签。所有记录都为 review-only，`training_eligible=false`、`human_audio_gold=false`。通过算法筛选不代表其删除意图已经与本项目对齐。

源仓库没有声明数据许可证，暂不自动用于训练或生产；也没有可核实的录音、说话人或文档边界。当前保留的是可追溯的文本核查资料，而不是项目目标对齐且带录音出处的人工金标。

另检查了 [Decopy 作者数据入口](https://github.com/atri45/Decopy/blob/ab77eb0d40a69ae5a8b0a7d65d230873333f42bd/build/get_require.py) 及其 [Dryad 存档](https://datadryad.org/dataset/doi:10.5061/dryad.q573n5ttw)。作者说明材料混合了人工标注与自动生成拼写样本，不能把全体都标成人工金标。完整训练文件没有下载成功：Dryad 下载 API 返回 401，作者 ModelScope 镜像在此环境 TLS 连接失败；未导入或用于训练。GitHub 中下载到的 1,100 条 SIGHAN2015 生成测试变体只进入 `spelling_benchmark_quarantine.jsonl`，没有混入训练。Dryad 的存档许可不能推及其它仓库材料。

## 来源可追溯编码器的相同数据对照

改用已经缓存的 [HFL Chinese RoBERTa WWM 模型](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，仍通过 BERT 类加载。模型卡标为 Apache-2.0。固定 revision `5c58d0b8ec1d9014354d691c538661bf00bfdb44`；本地权重 SHA256 `1ded5a5a1c7841dee6e47942f7b5bf2bcf6f73ff19197580f852f7f638f86b35` 与官方 Git LFS 元数据一致。

证据位于 `data/public_edit_sources_20261013/encoder_registry.json` 与 `encoder_official_metadata.json`。训练器可接受 `--encoder-provenance` 并核对路径及文件哈希，结果记录是否提供了来源证明。

此次使用物理 GPU 1，数据、分组、训练轮数与上一轮保持一致：417 条训练、126 条验证，18 条任务定义只进训练。公开候选没有进入拟合。模型为 12 层、hidden=768，含检测头共 101,682,437 个参数；encoder 的来源和容量同时改变，因此这不是只隔离某一个因素的因果证明。

结果位于 `results/edit_detector_traceable_encoder_20261013/`。只评估验证组和训练 probe；已有内部测试组没有再次运行，`test_evaluated=false`，`internal_text_test=null`。

| 验证指标 | 上一轮小编码器 | 来源可追溯大编码器 |
|---|---:|---:|
| 应修改窗口检出 | 42/42 | 42/42 |
| 正常 KEEP 被误报 | 84/84 | 23/84 |
| 候选位置 precision | 11.0% | 77.9% |
| 填充词有意义编辑命中 | 21/21 | 20/21 |
| 重复编辑命中 | 21/21 | 21/21 |

选择第 2 轮、阈值 0.15。它显示当前基础编码器组合更有效，但 23/84 的 KEEP 误报仍高；验证组已经用于选择，不能当成独立验收。后四类只有少量训练 probe，没有独立验证；插入任务没有正训练例，推理仍禁用。

CPU 2 线程、预热后单窗口接口平均 **113 ms**、p95 **120 ms**，包括 tokenization。参数和 CPU 开销明显增加；不能把验证定位改善说成端到端提速。来源可追溯大模型更适合作为离线教师候选，当前没有替代线上门控。

## 新来源的部分诊断

在读取模型预测前，从 358 条公开候选中固定随机选出 128 条，共 200 个作者删除操作、239 个标注正位置；checkpoint 和阈值冻结，没有使用这批数据训练或调参。结果保存于 `results/public_edit_detection_probe_20261013/`。

| 指标 | 小模型 | 大模型 |
|---|---:|---:|
| 作者删除操作命中 | 194/200 | 159/200 |
| 标注正位置覆盖 | 231/239 | 173/239 |
| 建议候选位置总数 | 3,842 | 842 |
| 49 种删除词形的宏平均命中 | 89.1% | 62.2% |

大模型提出的候选更少，同时漏掉了更多作者删除。小模型高命中也伴随极宽的候选范围。这里没有可信的负位置或 KEEP 标签，未标注位置只能算未知，**不能将 669/3,611 个未知建议位置算作误报，也不能据此计算总体精度**。

这批作者目标可能仍含超出忠实精修的简化意图，且没有音频和独立说话人/文档分组。它是新来源的正位置诊断，不是整体 ASR 精修质量或速度验收。128 条在这次运行后已消费，后续不能当作新鲜测试。

## 本次结论和后续重点

正向：查明了原材料的词形/模板单一问题；固定了可追溯的基础编码器；验证集误报和定位精度有改善；真实公开口语文本及候选位置已导入，避免继续只看少数合成模板。

负向：大模型仍有明显 KEEP 误报与新词形漏检，CPU 开销上升；公开标准化数据不能直接替代本项目忠实精修金标；正常口语 KEEP、后四类和插入任务仍缺可信独立监督。

下一步应做同词形在正确/口吃、引用/强调、指代/填充语境里的对照监督，并核对公开候选的意图与来源许可。大模型先用于离线辅助核查，经过新来源数据检验后再考虑蒸馏到小模型；不能把当前 checkpoint 直接接入实际跳过策略。

## 验证与文件入口

15 项针对性测试通过。公开候选完成了 3,000 行计数、正位置掩码、非训练/非音频金标、未知 KEEP 和产物哈希检查。Luna 做了编码器、词形分组、作者标注规范与最终数字的只读复核。

- `experiments/scripts/import_public_edit_candidates.py`：严格导入与正位置候选筛选。
- `experiments/scripts/train_edit_location_detector.py`：来源校验及 `--evaluation-scope validation_only`。
- `experiments/scripts/probe_public_edit_detection.py`：冻结模型的新来源部分诊断。
- `results/public_edit_candidates_20261013/manifest.json`：来源与筛选统计。
- `results/edit_detector_traceable_encoder_20261013/assessment.json`：验证改善、尚未上线的处置。

所有输出均使用新目录保存，旧实验与模型未覆盖。此轮没有重新启动服务、切换门控或执行新的音频转录。
