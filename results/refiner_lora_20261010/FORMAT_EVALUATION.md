# Strict-local LoRA 格式留出评估

**第一轮不通过纠错泛化检查。** 训练 628 条、dev 122 条、测试 250 条，所有数据按模板来源组隔离。250 条全部是构造的合成文本，没有音频；口语清理目标来自脚本，不是人工听音标注。

| 类别 | 样本 | JSON 合法 | 目标精确匹配 | 模型 keep | 模型 replace |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 42 | 42 | 42 | 42 | 0 |
| filler | 42 | 21 | 0 | 21 | 21 |
| stutter | 42 | 42 | 0 | 42 | 0 |
| meaningful_repeat | 41 | 41 | 41 | 41 | 0 |
| protected_content | 42 | 42 | 42 | 42 | 0 |
| syntax_boundary | 41 | 41 | 41 | 41 | 0 |
| 合计 | 250 | 229 | 166 | 229 | 21 |

**模型生成了 21 条 replace 建议，但全部被协议拒绝；通过协议的有效替换为 0 条。** 保留率高来自 keep 和拒绝后的回退，不能把 JSON 合法率解释为修复质量。

平均生成时间约 471ms/样本。它只包含模型单次补丁生成时间，不代表整条 Web、流式会话或真人转录的端到端延迟。

21 条整句删除均因 strict_local_edit_consumes_entire_window 被拒绝，最终回退为原句。其余63条目标不匹配是21条赘词样本和42条口吃样本被模型原样保留。


逐条seed、类别、预测、严格解析状态和合成目标见 format_eval_guarded_20261010/predictions.jsonl；对应输入和审计字段分别见 experiments/llamafactory/data/test.jsonl 与 data_audit.jsonl。adapter SHA-256 为 5408a42944150a4917781193db8b80b74d02feb6ccd07c2133163823483af4c7。

**结论：当前 adapter 还不适合使用。** 它学会了保留文本和输出 JSON，但没有学会将删除编辑泛化到留出模板。结果不能代替真实 ASR/语音评估，不能用于声称 WER、CER 或真人纠错效果。
