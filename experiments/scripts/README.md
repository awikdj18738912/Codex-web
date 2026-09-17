# 推理与评测脚本索引

- postprocess_asr.py：对 ASR JSONL 运行批量 Refiner 推理。
- postprocess_contract.py：定义及检查推理输入/输出契约。
- transcribe_and_refine.py：选择 Qwen3-ASR 或 Whisper，对音频执行 ASR + Refiner 端到端推理。
- transcribe_and_refine_qwen3.py：Qwen3-ASR 专用端到端入口。
- main.py：AASR-Bench Judge 入口。
- benchmark_io.py：读取基准 JSON/JSONL、匹配样本及校验 schema。
- client.py：OpenAI 兼容 Judge 服务客户端。
- result_store.py：保存可续跑的逐条 Judge 结果和摘要。
- metrics.py：聚合评测指标。
- text_metrics.py：CER、WER、MER 文本错误率。
- calibrate_confidence.py：根据开发集标签拟合置信度校准参数。
- __init__.py 等辅助文件：由 Python 导入模块使用。

各入口支持的模型参数以对应脚本的 --help 和当前代码为准。评测应固定输入、checkpoint、rubric 和环境，并保留逐样本结果。
