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

Oral 生成包含可独立抽样的中文误断句类别：`BOUNDARY_PUNCTUATION_RATIO` 默认 `0.10`，可通过环境变量设置，设为 `0` 关闭。它只在非 passthrough 的中文或中英混合样本中按该比例抽样；英文和 clean 直通样本不会注入。抽中时在完整依存语义短语中放置恰好一个错误句号，记录标签 `boundary_punctuation`，可与 explanation 控制同时出现。Clean 修复该句界时只改标点，例如“真。让我感到惊讶。”→“真让我感到惊讶。”；其他位置仍照常清理口癖，例如“嗯真。让我感到惊讶。”→“真让我感到惊讶。”。正常独立句“我到了。你在哪里？”以及“真好。让我看看。”应保留句号，不盲目合并。

组装阶段对中文或中英混合记录中带 `boundary_punctuation` 或 `clean`/`passthrough` 标签的样本允许至少 4 个字符的短目标，支持短错误修复目标和正常短句对照；英文和其他记录继续使用原有场景长度门槛。

## 当前精修数据策略

源文本只写实际说出的内容，不添加编者括号释义、旁白、舞台说明或 Markdown；真实口头解释自由表达，明确口述的字面代码和符号仍可保留。

Stage 1 以原版提示为基础，让老师直接生成自然的 `oral_text`，不由程序拼接口癖或改口词。提示分别控制无改口、单次改口、回退改口、多次改口，以及轻度/中度/较明显的口语状态；解释类场景可带实体读音/拼字提示。中文与中英混合场景约有10%额外请求老师自然生成一次意外断句，标点种类和位置不固定。`boundary_punctuation` 标签记录这项控制。

Oral 场景生成采用持续请求池：每完成一条即保存结果并补入下一条，避免整批等待最慢请求。每个场景的 `raw/<scene>/oral.state.sqlite3` 持久记录已分配编号、完成结果和待重试请求，并通过文件锁防止两个进程同时生成同一场景。续跑会先从 SQLite 修复 JSONL 与检查点，再重试未完成编号；已有旧版 JSONL 和检查点会自动迁移。`generated_count` 是实际保存条数，`attempted_count` 是已分配编号数（包含在途请求），`completed_attempts` 才是已完成尝试数，`pending_attempts` 是待完成数量。

Stage 2 先用数字规则规范化输入，再由老师根据精修提示生成 `Clean` 目标；目标中的数字也经同一规则规范化。对这条原版自由生成路径，数字值和英文实体不再由词面对齐规则单独否决，交由后续语义审查检查是否忠实。仍保留旧 `synthetic-oral-v1` 历史样本的构造证明路径；它只适用于旧的、可精确重建的记录。

Stage 3 对口语文本模拟 ASR，包含完整文本和可安全对齐的自然前缀。若无法证明截断前缀与目标对齐，则回退完整样本，不把未听到的后文放进标签。数字规则在 ASR 文本入模前处理数字格式。

Stage 4 组装后按中文/中英样本内容长度、字段完整性和已有规则进行筛选。对 `upstream-teacher-v1` 样本，最终阶段沿用原版单一语义通过判定，由老师检查是否保留说话者的完整最终意思、去除无意义口语和废弃改口且没有添加事实。ASR 字符替换、误断标点、数字值或英文实体变化不再经过额外词面拦截，交由语义判定；审核失败仍会拒收。之后执行冲突检查、相似度去重和单一意图数量限制。

SFT 导出会再次运行数字规则，避免学生模型承担 ITN。新生成的 `itn_types` 固定为空数组。精修任务覆盖口癖、重复、口吃和废弃开头清理；单次/回退/延迟/连续改口收敛；仅识别用的拼字/读音提示删除；结合上下文修复误插标点造成的意外断句；中英混用文本整理；保持干净输入不变。

自由生成样本最终审核先独立检查 `oral_text` 是否符合纯口语 ASR 数据用途，通过后再判断输入与精修目标的语义一致性。源文本不合格时不会仅因目标看起来干净而放行；正常口头解释、干净短句、改口和明确口述的代码仍可通过。两次回复都要求完整 JSON 和实际布尔 `pass`；缺字段、字符串布尔值、重复键及损坏 JSON 均拒收，原始回复、解析版本和独立源审核证据写入 `quality_audit`。旧数据的历史审核不会因更新代码而自动升级，仍需复审。

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

## 大批量自动验收

量产不依赖逐条人工或代理阅读。每条样本先经过构造／数值／实体／片段校验和独立语义审查，再去重。`run_pipeline.py` 的第5步完成后自动调用 `audit_dataset.py`：

```bash
python pipeline/scripts/audit_dataset.py \
  --input data/final/train.jsonl \
  --report data/final/acceptance_report.json \
  --sample-output data/final/review_sample.jsonl \
  --sample-size 300 --seed 20261017
```

该命令逐行检查全部记录的ID、输入与目标、已存审查证据、构造证明和片段前缀；同输入不同标签、数值或实体漂移、未按规则保存、缺证据等会令进程返回非零。最终JSONL保留ID、完整口语和构造记录，以便复验及追溯；SFT导出只保留训练所需字段。报告保存输入文件SHA-256、错误计数、前50个错误例子以及场景、现象、片段和标签来源分布。报告是结构与审查证据复验，不能把它解读为新增独立语义审查或学生效果评估。

抽检采用固定随机种子的reservoir抽样，另补入各场景及稀有现象的代表记录，故输出可能略多于`--sample-size`。它用于开发期或批次抽检，不需要打开每一条。若抽检发现系统性问题，应修复对应构造／规则并重新处理该批，避免逐条手改训练标签。第5步语义审查拒收记录默认保存在`data/final/rejects.jsonl`，不计入可训练样本数。

`TOTAL_SAMPLES`是流程各阶段的目标配置；当前流程在最终去重或筛选后仍可能不足该数量，应查看`stats.json`的`total_samples`与`target_samples`。自动验收通过表示已有记录满足检查，不表示已经凑够10万条。后续补充生成时须沿用检查、去重与场景配额，不能用重复记录填满数量。

若要求至少10万条实际保留记录，可在总流程中加`--min-final-records 100000`，或单独验收时加`--min-records 100000`。数量不足会自动返回失败，不会把不足额数据标成完成；分阶段run_pipeline本身不自动补足缺额；按最终数量补齐请使用下述generate_corpus.py。跨场景合并时，原始场景内数字ID会加场景前缀，原ID另存`meta.source_record_id`。

## 按最终保留数量生成大批量数据

新增`pipeline/scripts/generate_corpus.py`，按场景配额持续补充新候选，只有通过构造／片段配对、关键词、独立语义质检、导出检查及增量近似去重的记录才计入目标。原有分阶段`run_pipeline.py`接口继续可用。

```bash
/home/aim0/data/conda/envs/vllm/bin/python pipeline/scripts/generate_corpus.py \
  --output /home/aim0/data/datasets/ASR/refiner_zh_100k_20261017 \
  --target 100000 --base-url http://127.0.0.1:8010/v1 \
  --model refiner-teacher --concurrency 4 --batch-size 128
```

首次启动会冻结pipeline、所需system模块及原生数字库到输出目录的`source_snapshot`，实际运行该副本。工作区代码后续变化不会混入同一数据集；续跑需要配置及源码指纹与SQLite记录一致。SQLite WAL持久保存已提交样本、各候选拒收原因、逐请求响应及检查点。同candidate/stage/完整prompt的缓存可供中断恢复，不跨新candidate复用，避免重复请求固定住整个语料分布。重复执行同一命令即可续跑；同输出目录有进程锁保护。

`progress.json`报告实际保留数、各场景缺额、自动拒收／去重原因、种子数、请求活动与运行速度。按剩余场景配额补量，不把生成尝试数当成训练条数。默认最多500000候选，连续12批完全无新增会停止并保留证据，不无限循环。种子从小池启动，后续随使用量补充到目标；可传`--seed-source`复用合法种子，旧平铺数组会保留为明确的legacy类别。

最终达到目标才写`final/train.jsonl`、执行全量自动验收，保存`acceptance_report.json`和抽检文件；随后按完整最终意图分组导出`final/train_sft.json`、`val_sft.json`及`dataset_info.json`。默认85%／15%分配近似按场景与长度平衡，但同意图的完整／片段／噪声变体不可拆到训练和验证两边。`summary.json`仅在全量验收及导出完成后标记完成。进度为generating不代表已生成足额；最终计数需以complete状态的summary为准。

标点审查只对末尾一个普通中文分隔标点的等价差异容错，目标文字保持原样；问叹号、数字、型号、内部标点与代码点号不因此放宽。`.。`等异常重复句末标点仍拒收，新生成会避免多加中文句号。中文短目标最低4个内容字符，纯口癖不作为短训练目标。真实无噪声片段可用完整构造记录及精确观察前缀证明移除旧数字／英文候选，有字词变化的ASR片段继续通过原数英保护。
