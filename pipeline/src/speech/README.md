# 语音合成与识别适配器

这些可选适配器用于把生成文本转换为语音，再调用 ASR 获得更接近真实识别器输出的训练输入：

- tts_asr_backend.py：TTS → ASR 模拟后端及缓存。
- moss_tts_service.py：本地 MOSS-TTS HTTP 服务接口。
- qwen3_asr_service.py：本地 Qwen3-ASR HTTP 服务接口。
- __init__.py：包标记。

这是数据流水线的可选路径，不是浏览器在线服务本身。外部模型权重、服务和硬件需单独准备。
