# 数字精修测试集

`numeric_refinement_gold.jsonl` 是面向“Refiner 清理非数字内容 + zh-itn 处理数字”的人工期望集。每行一条 JSON，包含 `id`、`category`、`behavior`、`input`、`expected`。`expected` 是建议的完整输出，不代表当前模型或规则库已经能够生成该结果。

`behavior` 的含义：

- `convert`：有足够上下文，应转换数字形式。
- `preserve`：保留原文，包括成语、专名和语义不明的表达。
- `context`：同一句里分别处理可转换与应保留的数字。
- `cleanup`：需要 Refiner 处理口误或自我修正，不能只用 zh-itn 运行。

本集特意区分时刻“一点十分”和程度表达“这一点十分重要”；孤立的“一点十分”按保守原则保持原文。成语中的数字也保持原样。比较结果时应先按 `category` 汇总，再逐条查看差异，不能把规则库对未覆盖表达的保留直接算作数值错误。纯规则库评测应排除 `cleanup` 类；完整链路评测应包含所有类别。

运行 `python experiments/scripts/evaluate_numeric_refinement_gold.py --itn-cli <zh_itn程序路径>` 可只评估规则库；增加 `--with-refiner` 可运行实验性数字遮盖、Refiner、zh-itn 链路。报告同时保存逐条输入、期望和实际结果。实验性遮盖与正式 Web 链路不同，不能把它的通过率当成生产通过率。

`experiments/scripts/evaluate_current_web_numeric.py` 直接调用当前 Web 的 `refine_update` 最终文本处理函数，使用三态门控、模型数字模式和无实体数据库配置。它跳过音频识别与 WebSocket 流式调度，因此报告的是 Web 核心处理函数的离线表现。

`experiments/scripts/evaluate_numeric_diff_guard.py` 是源位置差异保护原型：模型先返回全文，程序只合并未碰数字及规则保护短语的局部编辑，再运行 zh-itn。它用于离线验收，尚未接入 Web 服务。
