# 正则保底与置信度补充送审

2026-10-14：新增可选的 `off / shadow / on` 补充门控，默认 off。入口为
`system/confidence_supplement.py`，接入 Web 的固定 K=3 窗口。

## 决策

- 原生门控要求精修：保留原动作，不允许高 ASR 分数取消。
- 原生保留的稳定窗口：检查与原始 ASR 文本对齐的片段分数，任一片段分数低于阈值时补充送审。
- 分数缺失、残缺、原文版本不匹配、重复窗口定位歧义或无效分数：on 保守送审，shadow 只记录建议。
- 原生 DEFER、空文本和已精修假设：保留原动作。

阈值默认 0.8，使用 ASR 生成 token logprob 的片段加权几何均值，不是精修必要性的概率，也未经独立校准。
高分干净窗口仍可能包含真实口语清理需求。送审不保证 Refiner 改对；结果仍经过实体、数字和正常重复的原有保护。
为保证原生送审不被取消，on 不允许同时开启学习门控 on 或实验否决路由。
数字分数风险可以增加审核，但不授权修改被规则保护的数字值或格式。

## 配置

ASR 必须提供 source-aligned evidence，汇总的 generated_suffix 分数不能替代：

```bash
QWEN_CONFIDENCE_WINDOW_SCORES=1 QWEN_PORT=8876 QWEN_GPU=1 bash scripts/start_qwen_asr.sh
CONFIDENCE_SUPPLEMENT_MODE=shadow LEARNED_WINDOW_GATE_MODE=off \
  ASR_URL=http://127.0.0.1:8876 WEB_PORT=8085 ZH_ITN_ENABLED=1 bash scripts/start_web.sh
```

服务已有配置不同则脚本拒绝复用。直接 Python 启动对应参数：

```text
ASR: --confidence-logprobs 5 --confidence-window-scores
Web: --refinement-gate-mode tri_state --learned-window-gate-mode off
     --confidence-supplement-mode off|shadow|on --confidence-supplement-threshold 0.8
```

ASR `/health` 返回 `confidence_window_scores`；Web `/health` 和 ready 返回
`confidence_supplement` 配置，窗口决策与 trace 保存原生动作、建议动作、是否应用、理由和源分数证据。
缺少证据时 on 会增加调用，这一策略优先防漏，成本需要另行评估。

## 测试音频5的核验

实验入口为 `experiments/scripts/collect_audio5_gate_review.py`，分别进行新 ASR 采集和真实 Refiner 对照。
整段音频使用现有 VAD 分段规则，ASR 不接收实体提示上下文。冻结这一次的完整原文、VAD 段和源分数后，
各策略使用同一 K=3 分窗、模型、实体词库、zh-itn、保护、局部审核、缓存和终态重试条件。
回放阶段不再次识别音频，不声称测得音频端到端实时延迟。

review 清单中的建议和模型改动均为待核验意见。用户确认前不生成训练金标，也不依据此音频调整阈值。
测试音频5是已有开发材料，不能充作独立验收；未经听音核对的文本判断不称为识别错误事实。

## 最新输出验收变更

用户于2026-10-14要求移除 `semantic_content_loss`。当前源码不再以中文删除未命中允许条件、
多字缩成单字或局部缩短为由产生这一拒绝原因，其重试关联和旧删除词表检查也已移除。
此前的句首“而在”专用允许条件已不再需要。

独立的严重整体缩水、截断、数值/单位/格式、实体/占位符、正常叠词、特定替换与无依据插入检查继续执行。
局部模型审核仍能处理已确认的词内口吃。历史实验记录中的此原因保留并标为旧规则，
保存的W002/W003模型候选已用新校验复核接受；没有重跑模型或改写旧的四策略统计。
