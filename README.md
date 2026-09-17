# AgenticASR：面向真实场景的语音识别文本精修

AgenticASR 将自动语音识别（ASR）的原始转写进一步整理成可读、忠实于说话者最终意图的文本。它不是另一个语音识别器：ASR 负责“听见并写出”，Refiner 负责在不改变关键信息的前提下清理口语表达、处理自我修正、整理格式，并通过规则与校验降低误改风险。

本仓库包含研究代码、数据生成与评测工具，以及可在本地运行的服务原型。它与打包发布的桌面产品 VibeXASR 有关联，但不是同一份发行物。

## 项目能力

- **口语转书面文本**：处理填充词、明显口吃和连续重复，整理标点，尽量保留说话者最终表达的意思。
- **理解自我修正**：结合上下文处理“我想要苹果——不对，是梨”这类改口，而不是机械保留错误版本。
- **适配不同 ASR 前端**：批处理和浏览器服务可接入 Qwen3-ASR 或 Whisper；本地流式示例还支持 sherpa-onnx。
- **支持离线与流式使用**：批量精修已有转写、对音频文件端到端处理，或通过浏览器麦克风和分块上传音频。
- **保护重要内容**：术语、缩写、URL 和标识符可先掩码再恢复；数字由上下文规则处理；模型候选经校验，失败时回退到规则处理后的 ASR 文本。
- **支持研究复现**：包含训练数据生成、SFT 导出、AASR-Bench Judge、CER/WER/MER 指标和置信度校准脚本。

AASR-Bench 公开版本包含 917 条样本和 6,637 项原子评分标准，覆盖 Content、Format、Filter、Rephrase。

## 系统工作流

~~~text
语音文件 / 浏览器麦克风 / ASR JSONL
                 │
                 ▼
       ASR 前端（Qwen3 / Whisper / sherpa-onnx）
                 │ 原始假设 raw_text
                 ▼
   分段与窗口 → 实体/术语保护 → 数字规则
                 │
                 ▼
      off / conservative / tri_state 门控 → Refiner（按状态调用）
                 │
                 ▼
        恢复占位符 → 输出安全检查
                 │
          接受候选或回退
                 ▼
     clean_text + 可审计运行记录

训练数据：口语/书面文本生成 → ASR 假设模拟 → 质量控制 → 去重 → SFT
实验评测：批量推理 → 基准 Judge / CER、WER、MER
~~~

精修不是单一模型调用。当前系统会组合分段、实体保护、确定性数字转换、可选门控、Refiner 和输出校验。原始 ASR 文本与最终 clean_text 分开保存；占位符或数值校验失败时，不会直接采用有问题的候选。

## 仓库模块

| 路径 | 职责 |
| --- | --- |
| [pipeline/](pipeline/README.md) | 生成 Oral/Clean 数据、模拟 ASR 输入、质量控制、去重和 SFT 导出 |
| [experiments/](experiments/README.md) | 批量精修、音频端到端推理、置信度校准和基准评测 |
| [system/](system/README.md) | Web 服务、ASR 后端、本地流式客户端、Refiner、安全规则和术语库 |
| [tests/](tests/) | 核心逻辑及服务交互测试 |
| [scripts/](scripts/) | 当前工作机的 Qwen 和 Web 启动脚本 |
| [data/](data/) | 生成数据、SQLite 术语库等本地运行数据 |
| [results/](results/) | 推理输出、Web 会话和日志等运行产物 |
| [MediaSup/](MediaSup/)、[assets/](assets/README.md) | 演示媒体和论文插图 |

## 快速启动：浏览器服务

仓库当前提供两个启动脚本。请在两个终端分别运行：

终端一启动 Qwen3-ASR 流式后端：

~~~bash
bash scripts/start_qwen_asr.sh
~~~

终端二启动 Web 前端与 Refiner：

~~~bash
bash scripts/start_web.sh
~~~

启动成功后打开 [http://127.0.0.1:8081](http://127.0.0.1:8081)。脚本默认让 Qwen 监听 8766、Web 监听 8081。端口冲突时可设置 QWEN_PORT 或 WEB_PORT；修改 ASR 端口后也要同步设置 ASR_URL。

这两个脚本针对当前工作机配置了 Conda、模型和 GPU 默认路径。在其他机器上可覆盖 CONDA_EXE、QWEN_ENV、QWEN_MODEL、REFINER_MODEL、ENTITY_DB、QWEN_GPU、WEB_GPU 等环境变量。Qwen 服务还需要 Silero VAD 文件（默认 models/silero_vad.onnx）；Web 服务需要 Refiner checkpoint 和可用的 SQLite 实体库。脚本会检查依赖路径，但不会自动下载模型或创建 Conda 环境。详细参数见 [system/README.md](system/README.md) 和仓库根目录的 [启动指令](启动指令)。

## 环境准备

建议将数据工具、ASR 后端和 Refiner 放在隔离环境中。根目录依赖覆盖流水线和实验工具；system/requirements.txt 覆盖服务端的基础依赖：

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r system/requirements.txt
~~~

不同后端有不同硬件和依赖要求。Qwen vLLM 服务需要兼容的 CUDA/GPU 环境；Transformer Refiner 通常使用 CUDA；system.live_asr 的 MLX-LM 路径面向 macOS。安装基础依赖不代表已安装模型权重或全部后端依赖。

## 常用命令

### 精修已有 ASR JSONL

输入至少要有 source_record_id 和 output.raw_text：

~~~bash
python experiments/scripts/postprocess_asr.py \
  /path/to/asr_output.jsonl \
  /path/to/refined_output.jsonl \
  --model /path/to/AgenticASR-Refiner \
  --entity-db data/entities.db
~~~

### 音频文件端到端推理

以 Qwen3-ASR 为例：

~~~bash
python experiments/scripts/transcribe_and_refine.py \
  path/to/example.wav results/e2e/result.jsonl \
  --asr-backend qwen3 \
  --asr-model /path/to/Qwen3-ASR-0.6B \
  --refiner-model /path/to/AgenticASR-Refiner
~~~

切换到本地 Whisper checkpoint 时，将参数改为 --asr-backend whisper 并传入 Whisper 模型路径。

### 生成训练数据

启动 OpenAI 兼容的 LLM 服务并设置 VLLM_BASE_URL、VLLM_MODEL_NAME 后，从仓库根目录运行：

~~~bash
python run_pipeline.py
~~~

最终 JSONL 写入 data/final/。使用 pipeline/scripts/export_sft.py 可以导出 LLaMA Factory SFT 格式。详细说明见 [pipeline/README.md](pipeline/README.md)。

### 运行测试

~~~bash
pytest -q
~~~

当前测试结果为 **215 passed，1 条 Starlette/httpx 弃用提醒**。测试通过表示仓库测试集通过，不等于真实 GPU 模型和浏览器音频链路已完成验收。

## 当前实现边界

本项目目前是**研究型可运行原型**。已有批处理、音频推理、浏览器模式、术语保护、安全校验和回退能力；方向 A 研究方案中的完整系统尚未实现。

当前精修路由支持 off / conservative / tri_state。tri_state 已接入浏览器在线、离线和离线流式链路：中间假设按 KEEP（保留）、DEFER（等待稳定）、REFINE（调用模型）路由，逗号和句末标点都作为窗口边界，活动窗口默认最多保留最近 3 个标点块（仍受窗口字符上限约束）；模型省略的窗口内原始标点会触发重试或安全回退，未结束片段不会凭空补标点；Refiner 已支持 JSON 局部补丁并在 source 不唯一或校验失败时回退；前端会显示状态和原因。当前仍未完成统一 ASR 假设协议、稳定前缀与完整动态活动区、双参考数据和完整流式评测。详细状态见 [CURRENT_WORK_AND_NEXT_STEPS.md](CURRENT_WORK_AND_NEXT_STEPS.md)，实际精修规则见[当前精修规则与实现说明](当前精修规则与实现说明.md)。

叠词保护已扩展为独立量词安全词表（`system/quantifiers.py`）和结构模式兜底：`一朵朵`、`一层层`、`一本本`、`一艘艘`、`一群群` 等“数词/限定词 + 量词重叠”不会被去重；即使遇到词表之外的地域量词，只要符合该结构也会保留。Refiner 删除重叠量词时会被语义校验拦截并回退。

当前样本中确认的字幕错断、固定搭配和重复字错误由 `system/transcript_repairs.py` 执行精确短语修复；规则不对普通句号做全局删除，未审核的句子仍由 Refiner 和安全校验处理。

## 演示

- 中文演示：[视频](MediaSup/CH_demo.mp4) · [预览图](MediaSup/CH_demo_preview.jpg)
- English demo: [video](MediaSup/en_demo.mp4) · [preview](MediaSup/en_demo_preview.jpg)

## 项目资料

- 论文：[arXiv](https://arxiv.org/html/2607.28175v1)
- 项目页：[AgenticASR](https://anxmuy.github.io/blog/agenticasr/)
- 桌面应用：[VibeXASR](https://vibexasr.speech.wiki/)
- 基准数据：[Hugging Face](https://huggingface.co/datasets/Andrew0425/AASR-Bench) · [ModelScope](https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench)
- Refiner：[Hugging Face](https://huggingface.co/Andrew0425/AgenticASR-Refiner/tree/main) · [ModelScope](https://www.modelscope.cn/models/MuyuanJ/AgenticASR-Refiner)

## 引用

~~~bibtex
@misc{jiang2026agenticasrrefiningspeechrecognition,
  title={AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach},
  author={Zixuan Jiang and Binghao Qiang and Jiaying Chi and Yanqiao Zhu and Kai Yu and Xie Chen},
  year={2026},
  eprint={2607.28175},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2607.28175}
}
~~~

## 许可

本项目采用 [Apache License 2.0](LICENSE)。
