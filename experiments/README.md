# 推理与评测

experiments/ 提供 Refiner 批处理、ASR + Refiner 端到端推理、基准 Judge、文本指标和置信度校准工具。该目录主要提供可运行脚本；论文中的全部训练过程、消融图表和最终结果归档不一定包含在仓库中。

## 工作流概览

~~~text
ASR JSONL ──→ postprocess_asr.py ──→ 精修 JSONL
音频文件 ──→ transcribe_and_refine.py ──→ 端到端结果
系统输出 + rubric ──→ main.py Judge ──→ 逐条结果与摘要
参考文本 + 假设 ──→ CER / WER / MER
开发集 confidence + 标签 ──→ temperature scaling / ECE
~~~

## 批量精修

每条输入至少包含 source_record_id 和 output.raw_text：

~~~bash
python experiments/scripts/postprocess_asr.py \
  path/to/asr_output.jsonl \
  path/to/refined_output.jsonl \
  --model /path/to/AgenticASR-Refiner \
  --entity-db data/entities.db
~~~

实体库参数可选。处理入口会应用适用的保护、实体候选、安全校验和回退，并将结果及审计字段写入 JSONL。具体 JSON 字段以 postprocess_contract.py 和当前实现为准。

## 音频端到端推理

~~~bash
python experiments/scripts/transcribe_and_refine.py \
  path/to/example.wav results/e2e/result.jsonl \
  --asr-backend qwen3 \
  --asr-model /path/to/Qwen3-ASR-0.6B \
  --refiner-model /path/to/AgenticASR-Refiner
~~~

可选择 qwen3 或 whisper 后端。Whisper 可使用本地 Hugging Face checkpoint；端到端推理依赖对应后端的独立环境，选项见脚本的 --help。

## AASR-Bench Judge

准备基准发布的 rubric.json，然后运行：

~~~bash
python experiments/scripts/main.py \
  path/to/system_output.jsonl \
  --rubric /path/to/rubric.json \
  --results path/to/judge_results.jsonl \
  --summary path/to/judge_summary.json \
  --api-url http://127.0.0.1:8000/v1 \
  --model /path/to/judge-model
~~~

Judge 结果依赖评审模型及提示词；应保留原始逐样本输出，不要只报告聚合分数。数据入口：[Hugging Face](https://huggingface.co/datasets/Andrew0425/AASR-Bench) 与 [ModelScope](https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench)。

## 置信度与文本指标

- calibrate_confidence.py：拟合温度参数并生成校准相关结果。校准只针对开发数据，不能用测试集调阈值。
- text_metrics.py：CER、WER、MER 等文本错误率工具。
- metrics.py：基准结果聚合。
- benchmark_io.py、result_store.py：结果 schema、匹配、断点续跑和摘要。
- client.py：OpenAI 兼容 Judge 服务客户端。

置信度适配器支持若干 ASR 侧分数字段；Qwen 的 token-logprob 派生分数默认不是校准概率。详见 [ASR置信度复现实验](../ASR置信度复现实验.md)。

## 复现实验建议

固定 ASR 输出来比较后处理策略；评估在线延迟时则必须重新跑音频链路。保存模型版本、配置、代码版本、硬件、命令和逐样本结果。实验设计与当前实现边界见 [AgenticASR实验设计方案](../AgenticASR实验设计方案.md) 和 [CURRENT_WORK_AND_NEXT_STEPS.md](../CURRENT_WORK_AND_NEXT_STEPS.md)。
