# AgenticASR 当前项目状态与后续计划

> 更新日期：2026-09-16
> 依据：当前仓库代码与本地 pytest 回归结果。
> 定位：研究型可运行原型，不等同于完整研究方案或桌面产品发行包。

## 项目简介

AgenticASR 对 ASR 生成的文本假设进行二次精修，目标是在清理口语表达、重复和自我修正时，保留说话者最终意图及术语、数字等关键信息。当前仓库横跨训练数据生成、模型推理、在线/离线服务、安全校验和研究评测五部分。

## 当前能力

| 范围 | 当前实现 |
|---|---|
| 数据 | 五阶段 Oral/Clean 生成、ASR 假设模拟、组装、语义 QC、去重及 SFT 导出 |
| 批处理 | 接受约定格式的 ASR JSONL，调用 Transformer Refiner 输出精修记录 |
| 音频推理 | 可串联 Qwen3-ASR 或 Whisper 与 Refiner |
| 浏览器服务 | Qwen3-ASR 流式或 Whisper rolling-window 后端；麦克风、整段文件、分块流式文件三种模式 |
| 本地流式 | sherpa-onnx VAD/ASR、ChunkManager、滑动窗口与 MLX Refiner；另有 Qwen 麦克风客户端 |
| 结果保护 | 实体准备/保护、恢复、数字校验、精修输出检查、重试与失败回退；结构化补丁 source 定位 |
| 术语管理 | SQLite CRUD、领域/别名/策略、模糊候选模式与有界会话记忆 |
| 评测 | AASR-Bench Judge、CER/WER/MER、置信度校准工具及自动化测试 |

## 处理链路

~~~text
ASR 原始文本
  → 按标点/长度分段或组成活动窗口
  → 术语候选、实体保护与数字上下文规范化
  → off / conservative / tri_state 路由判断
  → Refiner（如被跳过则保留基线）
  → 占位符恢复、数值及内容安全检查
  → 必要时严格重试
  → 接受候选或回退到规则处理后的原始文本
  → 结果和决策写入 JSONL
~~~

浏览器入口将 WebSocket、ASR HTTP 流、前端和 Refiner 服务接在一起；Qwen/Whisper ASR 服务可单独运行。原始文本 raw_text 与最终文本 clean_text 分开，运行记录保留门控决定、延迟、实体候选、规范化和拒绝原因等字段。

## 仓库入口

| 路径 | 职责 |
|---|---|
| scripts/start_qwen_asr.sh | 启动 Qwen3-ASR vLLM 流式后端 |
| scripts/start_web.sh | 启动 Web 服务、Refiner 和实体管理 API |
| system/web_app.py | WebSocket、在线/离线处理、术语接口与 Refiner 调度 |
| system/qwen_asr_stream_server.py | Qwen3-ASR、VAD、分段状态和可选 logprob |
| system/whisper_stream_server.py | Whisper rolling-window 后端 |
| system/refinement_gate.py | off / conservative / tri_state 精修前路由与稳定性跟踪 |
| system/entity_pipeline.py | 保护、恢复、验证、重试和回退流程 |
| system/entity_store.py、entity_matcher.py、session_memory.py | SQLite 术语库、候选匹配和有限会话记忆 |
| system/numeric_normalizer.py | 中文数字、金额、日期、量词和计量单位转换 |
| system/quantifiers.py | 广覆盖量词词表与“数词/限定词 + 量词重叠”保护 |
| system/transcript_repairs.py | 已审核的字幕错断、错字和固定搭配短语修复 |
| system/refinement_guard.py | 占位符、数值、内容损失、重复和截断等检查 |
| pipeline/、experiments/ | 数据生成、推理及评测工具 |

## 启动本机浏览器服务

分别在两个终端启动：

~~~bash
bash scripts/start_qwen_asr.sh
REFINEMENT_GATE_MODE=tri_state bash scripts/start_web.sh
~~~

脚本默认 Qwen 端口 8766、Web 端口 8081，门控默认使用 tri_state，浏览器地址为 http://127.0.0.1:8081。脚本内有当前工作机的 Conda、模型和 GPU 默认值。更换机器时按需覆盖 QWEN_MODEL、REFINER_MODEL、CONDA_EXE、QWEN_ENV、ENTITY_DB、QWEN_GPU、WEB_GPU、QWEN_PORT、WEB_PORT、ASR_URL。Qwen 还要求 Silero VAD 文件；Web 要求 Refiner checkpoint 和实体 SQLite 文件。参数详情见 [system/README.md](system/README.md)。

## 已实现与实验性能力

- **实体保护管线**：Web 主路径会在送入 Refiner 前准备/掩码输入，再执行恢复和校验；不合格候选会拒绝，必要时严格重试，最终回退。其他入口的字段语义及集成覆盖仍应继续统一核验。
- **模糊术语匹配**：支持 off、shadow、hint、auto。auto 只允许严格阈值下的低风险 TERM、PROJECT、MODEL 候选；不应把它理解为任意实体的模糊全局替换。
- **数字规则**：明确上下文可转为阿拉伯数字；带量词的概数也统一为阿拉伯数字并保留“多”（如“三千多个”→“3000多个”）；连续逐位数字和固定成语保守保持。小数点不会被切成窗口边界，输出会比较数值、顺序及单位语境。
- **量词叠词保护**：独立量词安全词表覆盖名量词、集合/容器/成形量词、时间/动量词和度量词中的常见形式；结构规则保护所有已登记的“数词/限定词 + AA”形式（如“一朵朵、一层层、一本本、一艘艘、一群群”），模型删掉其中一个重叠字时由语义校验拒绝并回退。
- **字幕错断修复**：对已审核的跨标点完整短语执行精确补丁。补丁在窗口合并后再次运行，因此不会因 tri_state 的 `KEEP` 或一个标点一个块而漏掉跨块错断；同时修正“华夏。在文明”“滔滔。香水”“最大径流量。更是”“一旦决。口”等已确认错误。`黄河园区`因可能指“黄河源区”或“黄河源园区”而保留为上下文候选。未知句号不会被全局删除。
- **流式调度**：浏览器文件流式模式的 latest-only 任务可避免精修积压；tri_state Web 链路按逗号/句末标点切分，每个标点是一个块，活动窗口最多保留最近 3 个块（仍受字符上限约束）。自我修正标记出现时，会临时合并前置错误块和后续修正块，避免错误内容在 `不对/不是/而是/应该是` 到来前被提交。本地窗口模块仍提供 K=3 等兼容配置。两者属于工程机制，不等同于统一动态 active-span 研究方法。
  tri_state 还会恢复 Refiner 省略的原始窗口末尾标点；只有明确的口癖/重复确定性清理才允许删除冗余标点。
- **精修门控**：Web 路径支持 off / conservative / tri_state。tri_state 使用 HypothesisTracker 观察跨次假设，在中间阶段输出 KEEP、DEFER 或 REFINE；对未校准且无明确问题信号的完整片段采用安全 KEEP，避免把整段文本反复交给不稳定模型；跨块句号、口癖、自我修正、实体提示仍可触发 REFINE。规则清理和安全保护不随模型跳过而关闭。
- **ASR confidence**：Qwen 可选 token-logprob 适配和温度校准工具已经存在，但默认分数未校准，也没有完成独立评测集的门控阈值验证。

## 尚未完成的方向 A 研究工作

1. 定义可被 Qwen 与 Whisper 共用的 ASRHypothesis 契约，统一时间戳、sequence、token 分数、最终状态和后端元数据。
2. 完善基于跨次假设的稳定前缀、动态 active span 和三态门控；当前 tri_state 第一版已覆盖 WebSocket 中间假设路由。
3. 扩展结构化 Refiner 输出的 schema、受保护 span、编辑范围和语义层 Validator；当前已落地 keep/replace 局部补丁和非法 source 回退。
4. 建立 verbatim 与 intended 双参考标注和固定数据划分。
5. 实现事件回放、always-refine/final-only/gated 对比与调用量、延迟、稳定性和误改风险的统一报告。
6. 在固定模型、术语库和音频条件下完成跨后端、消融及端到端测试。

## 验证

当前运行结果：

~~~text
pytest -q
215 passed, 1 warning
~~~

唯一提醒为 Starlette/httpx TestClient 相关 deprecation warning。此结果是测试集回归，不代表启动了本地 GPU ASR/Refiner 或完成真实浏览器录音验收。

继续开发后建议执行：

~~~bash
pytest -q
git diff --check
~~~

工作区已有 results/logs/asr_stream.log 和 results/web/session.jsonl 运行产物；文档更新过程中应保留，不要当作源码改动回退。

## 后续优先级

1. 统一各入口的 protect → refine → restore → validate → fallback 结果语义，补齐 fake Refiner 集成测试和异常/超时覆盖。
2. 落地跨后端 ASR 协议、事件日志和 replay，并继续完善稳定性跟踪、动态 active span 和三态门控评测。
3. 建立双参考数据和安全的实体正负例，固定开发集、验证集和测试集。
4. 用独立测试集报告 CER/WER/MER、意图质量、误替换/回退率、调用数和 p50/p95 延迟。

## 延伸阅读

- [项目总览与运行方法](README.md)
- [当前精修规则](当前精修规则与实现说明.md)
- [方向 A 架构计划](DIRECTION_A_IMPLEMENTATION_PLAN.md)
- [流式门控设计](流式ASR精修门控模块设计方案.md)
- [实体匹配设计](实体领域候选匹配与模糊替换方案.md)
- [置信度复现实验](ASR置信度复现实验.md)
- [实验设计](AgenticASR实验设计方案.md)
