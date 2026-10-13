# 首批合成门控文本

本目录已生成 1,000 条虚构文本探针，详见 `texts.jsonl`、`manifest.json`。包含正常表达、赘词、口吃、正常叠词、实体数字否定和句法边界六类；没有按类别预先赋收益标签。

当前只完成文本准备和服务就绪检查：ASR 8766 可用，TTS 8011 尚未启动。本轮还没有真实合成音、ASR 配对结果、可用训练标签或训练后的门控模型。

接入中文 TTS 后，按项目内 [运行说明](/home/aim0/桌面/tan/ASR/Codex-web/experiments/SYNTHETIC_GATE_QUICKSTART.md)先跑 10 条。新增脚本入口为 `experiments/scripts/synthetic_gate_data.py`。

已有 32 个模板来源组，合成 train/dev/test 为 628/122/250；这些划分用于合成开发试验，不能替代独立真人测试。`audio_script_verified=false`，实际朗读存疑的标签仅作弱监督。
