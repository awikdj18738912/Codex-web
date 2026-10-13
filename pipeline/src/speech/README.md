# 语音合成与识别适配器

这些可选适配器用于把生成文本转换为语音，再调用 ASR 获得更接近真实识别器输出的训练输入：

- tts_asr_backend.py：TTS → ASR 模拟后端及缓存。
- moss_tts_service.py：本地 MOSS-TTS HTTP 服务接口。
- qwen_tts_service.py：本地 Qwen3-TTS CustomVoice HTTP 服务接口，兼容项目的 /v1/tts 音频路径协议。
- qwen3_asr_service.py：本地 Qwen3-ASR HTTP 服务接口。
- __init__.py：包标记。

这是数据流水线的可选路径，不是浏览器在线服务本身。Qwen3-TTS 0.6B 模型保存在 /home/aim0/data/models/ASR/Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-CustomVoice，项目专属推理环境位于 .runtime/qwen3-tts-env。从项目根目录运行 bash scripts/qwen_tts_service.sh --device auto --port 8011 可启动；auto 仅在 GPU 0 有至少 8 GiB 空闲显存时使用 GPU，否则用 CPU。权重不纳入 Git。
