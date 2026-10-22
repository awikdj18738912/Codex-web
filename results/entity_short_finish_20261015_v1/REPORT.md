# 流式实体模块：短录音收尾复核

已实现短录音收尾路径。音频到来时继续按既有稳定前缀检测与局部音频确认；在结束时对不超过16秒（还受窗口/缓存限制）的完整录音补检最终文本。对于符合原检索阈值、歧义和短词保护的未复核候选，最多再做一次、最多8个提示词的原音复核，额外步骤总时限4秒。短音频回放完整PCM后才使用最终识别确认，避免先修好前面一个词就跳过末尾实体。已经确认的流式替换保留；已经尝试过的候选不反复补跑。更长录音保留原来的流式收尾策略。

超时、错误、提示词复述和无法确认的结果保留原文。只写回验证通过且位置/邻文匹配的实体修改，不整句采用提示识别输出。新状态字段记录收尾检查、替换、超时、错误、候选数、音频量和耗时。

## 测试

相关控制器、既有流式、取消/超时、Web收尾测试115项通过，17个subtests通过；随后新增的finish-only WebSocket集成测试也通过（新增测试文件共11项通过）。指定环境命令及源码哈希在test_validation.json；测试文件快照见tests/。

## 实际评测：315条已完成子集，完整2000条未完成

使用同一固定测试集、同一领域词库和同一首轮ASR口径，旧/新实体模块仅在新增短录音收尾路径上变化。已完成的315条首轮ASR与原记录完全一致。

|同315条样本|实体命中/提及|实体命中率|CER|
|---|---:|---:|---:|
|原始ASR|160/369|43.36%|7.40%|
|旧流式实体模块|191/369|51.76%|6.96%|
|新增短录音收尾复核|309/369|83.74%|5.11%|

相对旧模块新增恢复118次实体提及，丢失0次；107条样本实体命中改善，0条净命中退化。新增收尾复核153次；这315条的收尾复核步骤平均0.151秒、P95 0.381秒；仅实际触发检查的153条平均0.303秒、P95 0.561秒，最大1.500秒。这里是控制器新增步骤耗时，不是浏览器端总延迟。

自动审核尚未发现新增的参考不支持实体；未逐条人工听审，不能声称总体误改率为零。这315条包含已知诊断样本及固定顺序已完成部分，并非随机代表性子集，不能把83.74%当作完整2000条的新命中率。

## 中断与运行状态

全量采集在第315条之后中断：共享8766 ASR服务的健康检查持续超时，采集器最终在会话清理时抛出httpx.ReadTimeout。原因尚未确定；观察到系统IO压力高和服务线程等待文件页。使用同模型在独立8768端口的恢复启动也超过180秒未成功，已向这次私有启动进程发送终止信号。共享ASR与正式Web进程未重启，新的源码尚未部署到这两个正在运行的Web进程。

已完成315条通过独立审计：2014个主ASR PCM传输帧与源音频一致、149处最终实体修改可由首轮文本和实体替换逐字重建、153次新增收尾回放确实发送了完整原始短音频。源码、词库、输入和原数据集清单哈希匹配。失败的第316条HTTP尝试单独保留在evaluation/failed_asr_trace_20261015.jsonl，未写入成功结果。

ASR恢复后，可以继续同一冻结cohort：

```bash
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/evaluate_current_entity_challenge.py collect --run-dir results/entity_short_finish_20261015_v1/evaluation
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/score_current_entity_challenge.py --run-dir results/entity_short_finish_20261015_v1/evaluation
/home/aim0/data/conda/envs/agentic-asr/bin/python experiments/scripts/report_short_finish_evaluation.py --run-dir results/entity_short_finish_20261015_v1/evaluation
```

恢复后应重新执行完整审计；本次partial_verification只验证已完成子集。

原始短录音没有删除、加长或重新筛选；分层诊断位于../entity_streaming_testset_diagnostics_20261015。新的场景词库仍为原有固定v2版本，目标标注与参考没有进入推理。

官方流式ASR源码已查阅：[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)。模块行为与本轮性能结论以本地源码和实际记录为准。
