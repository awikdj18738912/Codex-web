# Qwen3.8 本地教师候选生成（200条）

- 依据排除先前开发样本后的固定新样本 `sample_200.jsonl`，完成了200条口语输入到精修文本生成。
- 模型：Qwen3.8-27B AWQ INT4，本地GPU0推理；提示词为当前恢复的生产中文精修提示词。
- 调用只传system prompt和口语原文。旧目标是在模型返回以后才附加到结果记录，供后续审阅。
- 生成错误0，缺失输出0；每条输入与样本按ID逐字核对一致。服务现已关闭。
- 输出在 `teacher_candidates.jsonl`。它们是教师提案，尚未逐条质量审核，不能直接覆盖训练目标。
- 抽样、数据源SHA及限制见 `sample_manifest.json`；模型、提示词和采样参数见 `provenance.json`；结构核对见 `candidate_integrity_report.json`。
