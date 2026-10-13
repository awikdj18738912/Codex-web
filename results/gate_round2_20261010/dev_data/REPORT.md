# 新句式配对试点

本目录只合成和解码声明的 dev 分区；原有原型权重与阈值保持冻结。情景类别不是 benefit/harm 标签，下面标签由实际 rules/always 配对产生。

完成 22 条 GPU TTS 与 clean ASR，得到 30 个配对案例、12 个模板组。

| 路径 | benefit | neutral | harm | uncertain | harm 模板组数 |
|---|---:|---:|---:|---:|---:|
| window_refine | 3 | 16 | 0 | 3 | 0 |
| local_edit_review | 2 | 5 | 0 | 1 | 0 |

## 原模型在新训练候选上的诊断

| 政策 | 精修调用 | 捕获 benefit | weak harm | 目标精确率 |
|---|---:|---:|---:|---:|
| rules | 0 | 0 | 0 | 0.433 |
| current | 15 | 3 | 0 | 0.533 |
| always | 33 | 5 | 0 | 0.600 |
| frozen_student | 11 | 3 | 0 | 0.533 |

本目录的分区角色为 dev；原模型保持完全冻结，没有利用本结果调阈值。阈值选择只允许使用开发分区，test 不能用于拟合或重选策略。必须核对实际 harm 的任务/模板覆盖和 teacher 版本。不能通过把正常重复情景直接标作 harm 来凑数。

job_state.json 与 logs/ 保存可续跑阶段与日志。TTS 服务仅在合成期间运行；原 ASR/Web 服务没有被本脚本停止。人工金标及已核验音频均为零。
