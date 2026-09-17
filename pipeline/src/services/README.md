# 模型服务客户端与进程管理

- llm_client.py：提供同步和异步 OpenAI 兼容 LLM 客户端，供种子、Oral、Clean、ASR 模拟及质控流程复用。
- vllm_lifecycle.py：管理流水线启动的 vLLM 进程，包括启动、健康检查和停止。
- __init__.py：包标记。

请求地址、模型名和并发等运行参数由 pipeline/configs/settings.py 与环境变量提供。连接失败时先检查服务是否启动、API 路径是否兼容以及模型标识是否正确。
