# 复现与文件说明

本轮是已使用开发资料上的冻结 ASR 最终快照试验。真实模型为现有 AgenticASR-Refiner，保持 float32 精度；没有重新解码音频、修改正式词库或重启正式服务。

输入共 500 条：400 个窗口、100 个局部复核候选，来自 195 个原始来源。包含真人医疗 153 条、真人新闻 90 条、公开 TTS 116 条及没有参考的既有演示音频 141 条。均衡探索性抽样与自然业务分布不同；噪声版本、相邻窗口不能视为独立录音。

在新的空目录准备同样的试验：

```bash
python experiments/scripts/collect_gate_pairs.py prepare \
  --output results/gate_pilot_reproduction \
  --window-count 400 --review-count 100 --seed 20261009
```

prepare 会快照化当前 system 源码、数字动态库、ASR 输入和正式词库副本。当前工作区若已改变，就不是本轮已冻结的相同代码；完全复核本轮须使用这里的 frozen/、native/、inputs/、entities_snapshot.db 和 cases.jsonl。

使用真实模型运行，GPU 选择必须满足显存余量检查：

```bash
/home/aim0/data/conda/envs/agentic-asr/bin/python \
  experiments/scripts/collect_gate_pairs.py run \
  --output results/gate_pilot_reproduction --device cuda:1
```

run 支持续跑并校验冻结文件；同一 current/always 决策确实选择同一神经操作时共享真实执行，保证两条分支不会因重复采样产生伪差异。成本表把该次实测成本映射到两种策略，物理执行次数另记，不可将策略成本相加当作实验时间。

共享 GPU 首轮产生的异常已归档到 pairs_failed_*.jsonl 并剔除重跑。辅助 masked LM 的权重在 CPU 暂存，需要评分时移动到 CUDA；评分设备与精度不变，权重搬运单独记录。这是隔离试验的内存管理方式，正式服务无需采用该方式；本轮耗时不能作为完整转录加速结论。

```bash
python experiments/scripts/collect_gate_pairs.py report \
  --output results/gate_pilot_20261009
```

| 文件 | 用途 |
| --- | --- |
| protocol.json | 冻结设置、输入哈希、采样池和来源统计 |
| cases.jsonl | 500 条决策输入；参考只用于离线审核 |
| pairs.jsonl | 三条策略的最终输出、调用明细、forward 次数、恢复／回退信息 |
| run_configuration.json / execution_runs.jsonl | 权重哈希、精度、适配器版本与执行批次信息 |
| summary.json / REPORT.md | 成本和输出差异摘要；参考距离仅是审核辅助 |
| annotations.csv | 全部样本与空白人工审核字段；report 会保留已填写内容 |
| changed_for_review.csv | 157 条发生文字变化的待审核样本 |
| neutral_pairs.jsonl | 343 条最终输出逐字相同的 neutral 标签；输入不含参考答案 |
| ANNOTATION_GUIDE.md | 意图、事实错误、存疑参考与标签填写规则 |

343 条 neutral 只证明该次神经操作相对规则结果没有净输出效果，不证明原 ASR 正确。其余样本需要核验原音、参考及最终意图后再填写 benefit/harm；不得用 edit distance 或 refiner_accepted 自动代替语义标签。

所有 500 条音频路径均可访问，其中 359 条有配套参考；缺失或存疑的 intended 参考应补充审核，不能为了训练凑三类标签。

## 文本辅助审核与原音复核

157 条发生变化的样本现已逐例完成助手文本预审，结果保存在 `assistant_review_decisions.json`。这不是人工听音确认：32 条收益候选、76 条中性改动、27 条退化候选、22 条存疑。退化候选包含 26 条格式／句法问题及 1 条可能删除逐级含义的内容问题，不能将这些数量解释为事实错误率或当前门控线上错误率。

导出工具校验证据哈希与配对结果，生成独立的合并视图和原音队列：

```bash
python experiments/scripts/export_gate_reviews.py --output results/gate_pilot_20261009
```

`ASSISTANT_REVIEW_REPORT.md` 说明判断目标与局限，`assistant_changed_for_review.csv` 提供逐条原因，`audio_review_queue.csv` 将全部 157 条按内容风险、格式风险、收益和中性改动排序。26 条有明确音频疑点；全部辅助标签仍为 `auxiliary_unconfirmed`、`assistant_audio_checked=false`、`assistant_train_eligible=false`。所有 157 条对应音频文件已检查存在，但没有声称听过原音。

人工仍在原始 `annotations.csv` 填写 `human_label`、`rationale`、`reviewer` 和 `reference_status`。重新生成 collector 报告后再次运行导出即可刷新合并副本；工具不会改写人工原表，也不会将辅助建议自动升级成训练真值。配对输出、参考或审核上下文改变时，会拒绝复用旧判断。

本次导出前后对 `annotations.csv`、`changed_for_review.csv`、`neutral_pairs.jsonl`、`cases.jsonl`、`pairs.jsonl` 的 SHA-256 校验均相同。针对审核导出与既有配对采集的测试共 **25 项通过**：

```bash
python -m pytest -q tests/test_gate_review_export.py tests/test_gate_pair_collector.py
```
