# 数据流水线源码

src/ 是五阶段生成流程共用的实现层：

- generators/：生成种子、Oral/Clean 文本和 ASR 模拟输入。
- pipeline/：执行语义质量控制、重复检测和去重。
- services/：调用 OpenAI 兼容 LLM 并管理流水线启动的服务进程。
- speech/：可选 TTS 与 ASR 适配器，用于更贴近语音识别的模拟。
- utils/：JSON/JSONL、提示词、关键词、日志与续跑工具。
- __init__.py：Python 包标记。

顶层 run_pipeline.py 负责阶段编排；configs/ 中保存场景、提示词和运行设置。
