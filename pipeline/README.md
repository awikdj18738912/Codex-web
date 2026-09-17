# 训练数据流水线

pipeline/ 负责构造 Refiner 训练数据并导出 SFT 文件。它将场景与种子文本扩展为口语表达（Oral），生成对应的书面目标（Clean），再模拟 ASR 识别文本。之后对记录执行语义质量控制、去重和统计，最终产物可供 LLaMA Factory 使用。

## 数据流程

~~~text
场景配置 / 种子池
  → 01 生成 seed 与 Oral
  → 02 生成 Oral → Clean
  → 03 模拟 ASR 假设
  → 04 按共同 schema 组装记录
  → 05 语义 QC、去重、分布检查
  → data/final/train.jsonl
  → export_sft.py 导出 SFT JSON
~~~

从仓库根目录启动主流程：

~~~bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL_NAME=/path/to/instruct-model
python run_pipeline.py
~~~

环境变量和生成参数位于 pipeline/configs/settings.py。LLM 服务需要兼容 OpenAI Chat Completions API；如果使用本地 vLLM，可参考 pipeline/scripts/start_vllm.sh。流水线可能调用多个模型请求，建议先查看每一阶段脚本的输入、输出路径和续跑行为。

## 导出 SFT

~~~bash
python pipeline/scripts/export_sft.py \
  --inputs data/final/train.jsonl \
  --train-output /path/to/llamafactory-data/train_sft.json \
  --val-output /path/to/llamafactory-data/val_sft.json
~~~

随后在 LLaMA Factory 的 dataset_info.json 注册 train/validation 文件，并在 refiner.yaml 中配置基础模型、数据目录和输出目录。不要覆盖仓库中的机器专用训练配置；先复制后改路径。

## 目录说明

- run_pipeline.py：流水线编排入口。
- configs/：场景、提示词和运行参数。
- scripts/：五阶段脚本、SFT 导出、服务启动、MLX 转换和数据重置工具。
- src/generators/：种子、Oral、Clean 与 ASR 假设生成器。
- src/pipeline/：语义质量控制和去重。
- src/services/：LLM 客户端与 vLLM 生命周期控制。
- src/speech/：可选 TTS、ASR 服务适配器。
- src/utils/：JSONL、提示词、关键词与日志工具。

## 运行注意事项

生成数据和调用模型会产生外部 API/GPU 成本。reset_data.sh 会清理生成数据，运行前应阅读脚本并确认要处理的目录。最终数据和中间产物通常属于本地运行输出，不应误认为仓库自带的正式训练集。
