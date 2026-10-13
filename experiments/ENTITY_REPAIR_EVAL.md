# 实体规则修复验证：2026-10-09

入口：`experiments/scripts/evaluate_entity_repair.py`。

本轮复用整段癌症播客及 225 条 MultiMed 中文测试音频的真实 ASR 响应，实际本地精修模型重新推理；原音哈希、逐块及最终响应固定，不重新识别音频。无库控制与修复前的结果均保留，词库辅助识别关闭。

修复仅涉及实体模块：

- 医疗缩写：完整字母＋相同数值，检查注册冲突、分子语境和边界后规范为标准名。已验证写法不重复要求模型确认。
- 单抗药名：沿用词条级内部漏字权限、唯一候选和插入边界，补充独立分词词典未收录药名的明确药物语境验证。
- 精修失败：保留已验证的实体规范，不依赖模型是否采纳提示。

专题库保留此前 141 个标准名，没有新增实体或错字别名。所有已登记的单抗标准名按统一药物配置开启内部漏字权限；其他近音和漏字权限保持原配置。新库为独立文件，正式用户库没有自动合并。

复跑已有完整结果可执行：

```bash
PYTHONPATH=.:experiments/scripts:.runtime/medical_entity_deps \
  /home/aim0/data/conda/envs/agentic-asr/bin/python -u \
  experiments/scripts/evaluate_entity_repair.py run
```

已完成记录会跳过。模型需要足够 GPU 显存；仅重新生成报告无需加载模型：

```bash
PYTHONPATH=.:experiments/scripts:.runtime/medical_entity_deps \
  /home/aim0/data/conda/envs/agentic-asr/bin/python \
  experiments/scripts/evaluate_entity_repair.py report
```

`run-final` 用于本轮最终方案：首次将已完成的初版缩写／单抗回放归档到 `initial_with_redundant_hints/`，然后使用去除重复提示的当前代码重新运行。已有归档时不再次覆盖。

输出在 `results/entity_repair_20261009/`：

- `REPORT.md`：各阶段收益、撤回方案、医疗回归。
- `podcast/comparison.csv`、`podcast/changes.json`、`podcast/repaired.txt`：时间窗对照、紧凑差异和最终全文。
- `podcast/*.jsonl`：实际模型调用、候选、替换及真实识别输入。
- `medical/REPORT.md`：原 225 样本参考计分。
- `medical/differences_from_prior_rules.json`：与修复前的最终文字差异，本轮为空。
- `reference_negative_check.json`：优化专题库处理 225 条参考文本，本轮修改 0 条。
- `tests_final.log`、`scope_verification.json`、`service_reload.json`：测试、改动范围和服务重载记录。

优化词库为 `data/entity_benchmarks/oncology_repaired_20261009/entities.db` / `entities.json`。
已验证现有 `system.entity_context_cli import` 可完整导入 141 词及单抗漏字权限：

```bash
python -m system.entity_context_cli --db /目标测试库/entities.db import \
  --input data/entity_benchmarks/oncology_repaired_20261009/entities.json
```

播客是已见开发材料，没有人工标准转录。本轮的标准写法恢复数不能解释为实体准确率，或作为独立音频泛化收益；药名恢复也不能单凭词库认定原音就是 ASR 读错。
