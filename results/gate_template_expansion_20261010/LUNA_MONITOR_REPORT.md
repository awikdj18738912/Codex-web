# Luna 日志监督与独立终检

日期：2026-10-10。主代理负责新生成器和管线代码，`gpt-6-luna` 子代理只读检查训练审计与本轮阶段日志，没有修改代码、控制服务、重新训练模型或调阈值。

- 约每 45 秒检查阶段状态和日志，在合成半程、解码及完成时报告；未发现失败/OOM/停滞信号。
- 8 个阶段完成。TTS 64 条均为 cuda:0，ASR 64 条均为 train/clean；数据中的种子 ID、音频哈希及录音组无重复。
- 108 个文本记录属于 60 个独立模板组，train/dev/test 为 64/22/22 条，模板组无跨分区。仅 train 合成、解码和配对。
- 旧原型的权重、metadata 与冻结 selection 保持一致；本轮只使用原模型做新训练候选诊断，refit=false、threshold_retuned=false。
- 新窗口任务有 2 条 weak harm，来自 batch_checks、each_compare；局部审核有 2 条 weak harm，来自 each_compare、emphasis_detail。并集为 3 个独立模板组，不能把跨任务的 each_compare 当成两个独立语料来源。
- 场景类型不等于 gate label。86 条实际配对标签为 benefit 8、neutral 63、harm 4、uncertain 11；有效弱训练记录 75 条。
- 所有音频与目标脚本保持未核验标志，人工金标为零；每任务只有 2 个 harm 模板组，仍不能用于风险校准。

口径说明：synthesize.log 中 `TTS id: Xs` 表示音频时长。真正的请求耗时由 tts.jsonl 的 elapsed_seconds / tts_response.elapsed_seconds 记录。监督过程中已纠正这一口径，不用音频时长冒充推理延迟。
