# 新句式训练扩展试点

只合成和解码新数据的 train；原有数据、原型阈值及两批 dev/test 均保留。情景类别不是 benefit/harm 标签，下面标签由实际 rules/always 配对产生。

完成 64 条 GPU TTS 与 clean ASR，得到 86 个配对案例、35 个模板组。

| 路径 | benefit | neutral | harm | uncertain | harm 模板组数 |
|---|---:|---:|---:|---:|---:|
| window_refine | 4 | 52 | 2 | 6 | 2 |
| local_edit_review | 4 | 11 | 2 | 5 | 2 |

## 原模型在新训练候选上的诊断

| 政策 | 精修调用 | 捕获 benefit | weak harm | 目标精确率 |
|---|---:|---:|---:|---:|
| rules | 0 | 0 | 0 | 0.384 |
| current | 45 | 8 | 4 | 0.430 |
| always | 96 | 8 | 4 | 0.430 |
| frozen_student | 20 | 3 | 0 | 0.419 |

这是未用于拟合原模型的新训练候选诊断，不是新的最终验收集。原模型保持完全冻结，没有利用新结果调阈值。下一轮必须先核对实际 harm 的任务/模板覆盖和 teacher 版本，再固定训练/开发/独立验收集合。不能通过把正常重复情景直接标作 harm 来凑数。

job_state.json 与 logs/ 保存可续跑阶段与日志。TTS 服务仅在合成期间运行；原 ASR/Web 服务没有被本脚本停止。人工金标及已核验音频均为零。
