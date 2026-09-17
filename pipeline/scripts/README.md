# 数据流水线脚本

本目录包含分阶段数据生成、格式转换和服务辅助脚本。

- 01_seed_and_oral.py：生成场景种子池及 Oral 口语转写。
- 02_clean_text.py：为 Oral 生成 Clean 书面目标。
- 03_simulate_asr.py：创建对齐的 ASR 风格输入。
- 04_assemble.py：整理不同场景产物为统一记录。
- 05_finalize.py：语义质量检查、去重、分布统计并导出最终数据。
- export_sft.py：将最终 JSONL 导出为 LLaMA Factory SFT JSON。
- start_vllm.sh：启动本地 OpenAI 兼容生成服务。
- convert_to_mlx.py：将 Hugging Face Refiner checkpoint 转为 macOS MLX-LM 可用格式。
- reset_data.sh：重置生成数据。执行前务必阅读目标路径并确认其中内容可清理。

通常从根目录运行 run_pipeline.py，由它编排五个生成阶段；单阶段脚本可用于局部续跑或调试。
