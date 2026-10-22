# 数字 / 英文词面 guard 消融（2026-10-17）

执行者：Luna delegated analysis。语义教师仍是本机 Qwen `refiner-teacher`（Qwen3-30B-A3B-Instruct-2507 AWQ 4bit，`127.0.0.1:8010/v1`）。

## 同批 300 条结果

同一批 Oral、已保存 Clean 目标及 ASR 响应被重放；20 条 Oral 生成失败保留在 300 分母。有效 Oral 分母是 280。四组中只运行时 monkeypatch 两项词面检查； construction 精确证明和后续流水线检查均保持原状。

| 组别 | 接受 / 300（去重前） | 最终保留 / 300 | 有效 Oral 口径 | 状态简表 |
|---|---:|---:|---:|---|
| 两个 guard 开 | 257 | 255（85.0%） | 255/280 = 91.1% | Oral拒20，ASR拒6，组装拒6，语义拒11 |
| 仅数字 guard 关 | 260 | 258（86.0%） | 258/280 = 92.1% | Oral拒20，ASR拒3，组装拒6，语义拒11 |
| 仅英文 guard 关 | 259 | 257（85.7%） | 257/280 = 91.8% | Oral拒20，ASR拒4，组装拒6，语义拒11 |
| 两个 guard 都关 | 262 | 260（86.7%） | 260/280 = 92.9% | Oral拒20，ASR拒0，组装拒6，语义拒12 |
| 两个 guard 关，且从审查提示删去数英句 | 260 | 258（86.0%） | 258/280 = 92.1% | Oral拒20，ASR拒0，组装拒6，语义拒14 |

完整 construction 对 280 条都通过未修改的 proof：重建口语与 normalize 后 source 相等，且 `expected_clean == target`。因此这些完整目标在四组的词面 guard 开关之间没有差异。词面 guard 的独立诊断虽显示 280 条中 53 条数字规则不通过、43 条英文保留不通过（其中16条两者都不通过），construction 分支仍按既有精确证明放行。

真正受两个词面 guard 影响的是 6 条已保存 ASR partial 候选。它们全部通过固定 target 与生产 `valid_partial_pair`。数字 guard 关时3条进入后续，英文 guard 关时2条进入，两者都关时6条进入。两者都关后，5条通过语义审查、export 与去重，最终多保留5条；第6条 `scale_daily_chat_0090` 未通过原样保留检查。该输入有“2024年，不对，120Hz刷新率，最后改成，2024年好像快到了”；审查的三个布尔值均为 true、issues 为空，但 `cleaned_target` 是 `2024年好像快到了`，而固定 partial target 带结尾逗号，因逐字不一致被拒。这是 exact-target 边界，不应描述成语义错误或数值错误。

两 guard 都关时，最终保留由255增至260（300分母增加1.7个百分点；有效 Oral 分母由91.1%到92.9%）。这说明 guards 对当前批次的 ASR partial 分支有可测影响；它们不解释全部保留损失。

## 再移除审查提示中的数英句

额外组只从 `AUDIT_SYSTEM` 删除这句：`数字格式由外部规则负责。英文名称和型号沿用原文。` 其余 prompt、三个布尔项、精确 construction proof、固定 partial target、surface、unsupported-edit 规则及 export 检查保持不变。对268条原本进入审查的 source/target 和6条新 ASR partial 候选共274条重新审查，去重后为258条；比“两 guard 都关”少2条，不是进一步提升。

去重 ID 集合差为：prompt 移除组新增 ID 无；仅“两 guard 都关”保留 `scale_meeting_0031` 与 `scale_vibe_coding_0205`。前者在重新审查时未通过 `cleaned_target` 逐字一致性；后者的 partial target 也未通过该一致性检查。新增候选 `scale_daily_chat_0090` 的三个布尔值均为 true、issues 为空，只因片段结尾逗号不一致被拒。模型输出有采样/判断波动；此集合差是本次观测，不单独归因于删去该句。

另一个正式 Clean 生成提示仍含 `数字格式由外部规则库负责；英文名称和型号沿用原文。` 本实验没有重新生成 Clean，而是复用已保存目标，所以没有测试删除 Clean 提示句的效果，也没有把实验描述成整批重新生成。

## 其它损耗与历史对照

基线11条审查拒收中，10条原因是 `audit_target_not_clean_or_not_exact`，三个布尔项均为 true，逐项检查是标点差异：约7条片段末尾逗号被审查输出去掉，2条英文句点与中文句号重复（`.。`），另1条 Git 命令目标末尾的中文句号被去掉。另1条才是 `source_valid=false`。因此这10条不能概括成语义错误样本。6条组装拒收全部是 `skipped_short`。另有20条 Oral 失败、2条重复被去重移除。上述损耗都不因关闭数字或英文 guard 而改变。

旧 `refiner_teacher_scale1000_20261017` frozen run 的最终保留为404/1000（40.4%）。原始状态为 Oral拒347、Clean拒17、ASR拒80、语义拒132、组装拒10、接受414，另去重10条。在旧规则快照中，636条有已保存 source/target 的样本均通过当时词面 guard 诊断；这636条是已通过前置检查并留下source/target的幸存样本；347条Oral拒收没有留下最终Oral，因此该诊断不能否定早期的数字／英文误拦，也不能分解686次混合grounding告警的原因。此诊断只加载旧 `numeric_policy.py`，而它引用的 `system.refinement_guard.py` 使用当前文件；该历史 system 依赖未被快照保存，所以它不是完整旧环境重现。

上游 `480/500 = 96%` 与当前基线在有效 Oral 口径下的 `255/280 = 91.1%` 仍是不同批次，差约4.9个百分点，不能作因果比较。当前两 guard 都关为 `260/280 = 92.9%`，高约1.8个百分点；仍不能把剩余差额归因到 guard 或提示词。

## 请求与缓存记录

教师是 Qwen，而非 Luna。ASR 和既有审查响应按原 request 完全匹配后复用；新进入的 ASR 样本只在各 prompt 版本实际需要时请求审查，并按 request hash 跨组共享。

- `live_api_calls.jsonl` 与 `judge_api_calls.jsonl` 保留全部请求/响应记录。
- 初次直接 API 探测的279个 HTTP 400 请求未获得可用审查响应，不算审查结果；随后成功响应才进入缓存。
- 两份日志合计453条成功响应行，精确请求哈希285个，168行为重复成功响应；相同精确请求没有响应冲突。额外有279条HTTP400错误行。
- 正式分组结果由 `run_pipeline_replay.py` 写出；早期离线 probe 和 `run_judge_replay.py` 仅保留作请求调试记录。`offline_300.json` 是 guard 词面诊断，不是最终保留率。

## 复现与边界

执行入口：

```bash
/home/aim0/data/conda/envs/vllm/bin/python results/refiner_guard_ablation_20261017/run_pipeline_replay.py
```

数字关闭时仅把 `lock_numeric_target(source, target)` 运行时替换为 `target`；英文关闭时仅把 `english_entities_preserved(source, target)` 替换为 `True`。代码不改 `normalize_source`，也不改/调用关闭版 `restore_numeric_surfaces`。`_construction_plan_matches` 和 `expected_clean` 精确等式始终原样运行。ASR simulator、`valid_partial_pair`、`04_assemble`、semantic QC、`export_sft` 和阈值0.75的 dedup 使用实际生产模块。

结果保存在本目录；没有由实验脚本写入生产源文件或历史结果。工作区已有生产文件 dirty/untracked 状态，清单记录实际载入模块路径与 SHA-256，不能把这些既有工作区状态归因到本实验。详细逐条状态、输入哈希、模块哈希和运行时 patch 见 `summary.json`、`manifest.json`。
