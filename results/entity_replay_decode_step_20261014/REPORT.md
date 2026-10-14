# 实体复核解码间隔上线报告

## 上线内容

本次只把实体音频复核的第二遍解码间隔从 **2 秒改为 10 秒**。首次 ASR 仍保留请求的间隔，本轮测试为 2 秒；筛选、替换确认、上下文、精修和数据库均未调整。生产代码改动位于 [entity_audio_repair.py](/home/aim0/桌面/tan/ASR/Codex-web/system/entity_audio_repair.py) 和 [entity_streaming_replay.py](/home/aim0/桌面/tan/ASR/Codex-web/system/entity_streaming_replay.py)。Web 服务已重启并完成冒烟验证；ASR 服务没有重启。**V6 预判方案未部署，本次上线范围只有实体复核解码间隔。** 本次改动接入现有流程，没有增加开关。它加快复核，没有接入 V6 的提前发布实体改字功能。

## Web 实测

新方案的 6 组、48 个片段共有 53 个参考实体，实体命中 **53/53**，与旧方案相同；两边清理后字符错误也同为 32/993。6 组均无新旧实体命中或字符错误差异。平均 Web 总时长由 **13.6399 秒降至 11.8960 秒**，平均尾段时长由 **4.0753 秒降至 2.3432 秒**。

| 指标 | 旧方案 | 新方案（2→10 秒） |
|---|---:|---:|
| 片段 / 组数 | 48 / 6 | 48 / 6 |
| 参考实体命中 | 53/53 | 53/53 |
| 清理后字符错误（参考字符数 993） | 32 | 32 |
| 平均总时长 | 13.6399 秒 | 11.8960 秒 |
| 平均尾段时长 | 4.0753 秒 | 2.3432 秒 |

独立 Web 集的首次 ASR 初稿为 49/53 实体、35 个字符错误；两方案的最终 raw 输出均为 53/53、29 个字符错误。首稿与两臂结果均来自这批 Web 测量，不能据此把实体复核增益归因于不同的首次 ASR 输入。详细数据见 [independent_score.json](independent_score.json)。

## 固定初稿的实体复核对照

206 对输入固定为相同 ASR 预测初稿和真实音频 replay；初稿由旧结果中的实体替换反向还原，并通过重建断言。它是从既有 2506 项基准池选出的已知成功回归集与负控，**不是新长音频或未见数据上的验证，也不是总体 ASR 准确率评估**。

| 指标 | 旧方案 | 新方案（2→10 秒） |
|---|---:|---:|
| 配对样本 / 参考实体 | 206 / 298 | 206 / 298 |
| 输出实体命中 | 285/298 | 287/298 |
| 字符错误（参考字符数 3272） | 62 | 60 |
| 平均实体复核耗时 | 0.64986 秒 | 0.41870 秒 |
| 实体变更数 | 178 | 180 |
| 变更中参考文本不存在的数量 | 4 | 4 |

178 个旧变更签名全部保留，新方案另有 2 个变更；32 个负控两边均为零变更。旧、新两边各有 4 个变更在参考文本中找不到。平均复核耗时只统计实体音频复核阶段，不包括首次 ASR、精修、浏览器或完整 Web 耗时。受控结果显示同一初稿上新方案净增 2 个实体命中并减少 2 个字符错误；这组结果仍属于困难样本回归集，不能外推为总体准确率变化。计划及重建校验见 [controlled_regression_plan.json](controlled_regression_plan.json)，评分见 [controlled_score.json](controlled_score.json)。

## 已知回退与实测边界

必须披露：已知 dense 集的 5 组、40 个参考实体，命中从 **37/40 降至 36/40**；raw 字符错误从 23 增至 24，clean 字符错误从 24 增至 25。唯一实体差异在首次 ASR 的 `primary_finish_text` 中已经存在，该组两臂没有已确认实体替换。206 条固定同初稿的另一次受控回归保留了全部旧确认结果，支持本次只调整复核参数的选择；它不能直接证明已知 dense 集中的这条错词也能修好。不能据此声称所有 full-Web 结果都没有下降，或将已经存在于 ASR 初稿的差异称作规则误替换。具体配对记录见 [known_dense_score.json](known_dense_score.json)。

新增 48 个 Web 测试片段与旧冻结集合的 180 条及 V6 的 72 个片段没有 ID 重叠。它们仍来自以前测试过的 2506 条底池，按元数据重新选样；6 组拼接压力音频不代表天然未知的长音频。

上线后的 32 秒固定切片冒烟请求完成于 11.6016 秒，输出把“厉飞羽”基于音频复核归一为“厉飞宇”（1 项 `AUDIO_REPLAY_CONFIRMED`），实体复核阶段为 1.316 秒。ready/final 字段确认首次 ASR chunk 为 2 秒、复核 decode 为 10 秒，复核已完成；该切片没有完整参考，不能作为 CER 或配对时延结论。实验统计的 ASR `reported_model_ms` 可能没有覆盖 `/stream/finish` 最后一次解码，不能据此宣称 GPU 提速 95%，也没有逐次统计模型调用数。对用户的收益按上述墙钟总时长、尾段时长和实体复核阶段实测值报告。见 [live_web_smoke.json](live_web_smoke.json) 和 [web_rollout.json](web_rollout.json)。

关联的 40 项 pytest 已通过，补充首遍/复核参数隔离断言后又通过了其中 24 项；本报告不将其解释为端到端准确率证据。正式实体库哈希未变化，首次 ASR/Web/实体检索及匹配源文件未修改，仅两个实体复核模块改变，见 [最终验证](final_verification.json)。

已联网核对 [Qwen 官方流式实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)：每积累一个解码间隔的音频会进行一次解码，结束时会处理未满间隔的剩余音频。因此复核间隔调到 10 秒不会丢弃短于 10 秒的尾部音频；实际性能和准确率仍以上述本地测试为证据。官方项目为 Apache-2.0 许可，本轮未引入新依赖或外部付费服务。

## 数据文件

- [long_score.json](long_score.json)：稀疏拼接与单条无完整参考长音频的压力结果。
- [known_dense_score.json](known_dense_score.json)：已知实体密集组配对结果及首次 ASR 初稿对照。
- [independent_score.json](independent_score.json)：6 组 Web 配对结果、时延及参考评分。
- [controlled_score.json](controlled_score.json)：固定初稿与真实音频的 206 对受控复核结果。
- [controlled_regression_plan.json](controlled_regression_plan.json)：样本选择、输入还原和校验说明。
- [web_rollout.json](web_rollout.json)：Web 服务上线进程及健康状态。
- [live_web_smoke.json](live_web_smoke.json)：上线后固定切片的 Web 冒烟响应。
