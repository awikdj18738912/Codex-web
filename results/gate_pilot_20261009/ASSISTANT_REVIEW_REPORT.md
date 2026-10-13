# 门控配对样本：文本辅助审核

共 500 条配对结果；343 条最终输出逐字相同，其余 157 条已逐例完成文本辅助审核，来自 64 个原始录音来源。

标签比较的是强制执行该神经操作的 always 分支与只运行规则的 baseline 分支。current_output 供核查现有门控是否执行该操作；这里的分类数量不是当前线上错误率。

| 辅助标签 | 数量 | 含义 |
| --- | ---: | --- |
| benefit | 32 | 候选收益：清理明显重复、赘词或修复断开的语法结构 |
| neutral | 76 | 主要是等价标点、数字或字形变化，未见明确净收益 |
| harm | 27 | 候选退化：切断句法结构或删除有意义的表达 |
| uncertain | 22 | 强调、说话人边界、事实改写或参考失配，需核对原音 |

候选 harm 程度：{"minor": 26, "major": 1}。minor 指标点／可读性退化；major 指可能改变信息的修改，两者不可混报为事实错误率。

26 条存在局部或整句音频疑点，需听原音进一步判定；所有辅助审核的 audio_checked=false。当前人工标签 0 条。辅助标签可作为审核建议，不能直接作为训练真值或论文实验结论。

收益判断采用忠实保留信息、允许清理口吃与非必要语气词的精修目标。逐字转录目标下，部分清理会增加 CER；已在相应理由中说明。不审核医学观点正确性，不把参考距离下降或模型接受当作收益证明。

high / medium / low 是辅助审核的定性把握，不是校准概率，也不是 ASR 置信度。assistant_needs_audio=false 仅表示相对文本变化可以初判；转为人工真值仍须按原标注协议确认。

## 如何复核

1. 按 audio_review_queue.csv 的 review_priority 审核：1 为存疑或内容风险，2 为格式退化或需听音，3 为收益候选，4 为等价改动。
2. 队列覆盖全部发生变化的样本，audio_path 指向完整录音，没有伪造局部音频时间戳。同一录音、相邻窗口及噪声版本应结合查看。
3. 仅在 annotations.csv 填写 human_label、rationale、reviewer、reference_status，存疑可填 uncertain；辅助合并表只是可刷新副本。
4. 重跑 collector 的 report 后，再运行下方导出命令；原人工字段会原样复制，配对输出、参考或审核上下文改变会拒绝复用旧判断。

```bash
python experiments/scripts/export_gate_reviews.py --output results/gate_pilot_20261009
```

## 文件

- assistant_review_decisions.json：157 条逐例判断原件；含证据哈希，不按正则自动赋标签。
- assistant_reviews.jsonl：校验后的辅助标签；全部 auxiliary_unconfirmed、assistant_train_eligible=false。
- assistant_annotations.csv：500 条人工表与辅助字段的合并副本；人工原表不会被改写。
- assistant_changed_for_review.csv：157 条变化样本的合并审核视图。
- audio_review_queue.csv：全部 157 条变化样本的原音复核队列。
- assistant_review_summary.json：辅助审核统计，按任务和来源分别列出。

## 对门控设计的影响

大量中性改动提示门控需要预测净收益，而不是检测句子有无可改之处。条件句、主谓关系、并列结构及有意义的重叠词应进入内容保护和 harm 学习目标。纯格式 harm 与实体、数字、否定或事实 harm 需要分层报告。

典型样本（均为尚未确认的文本辅助结论）：

| 原文本 → 操作后 | 辅助判断 |
| --- | --- |
| `均面临。不均衡的复苏前景` → `均面临不均衡的复苏前景` | benefit：修复动宾连接 |
| `如果脚气一直不好，该怎么解决` → `如果脚气一直不好。该怎么解决` | minor harm：条件句误断 |
| `一级一级地向高处提水` → `一级地向高处提水` | major harm 候选：可能删除逐级含义，需听音 |
| `学了很多很多` → `学了很多` | uncertain：可能是有意强调 |
| `五连冠` → `5连冠` | neutral：数字写法变化 |

这批资料属于已用于开发的探索性样本。后续先确认收益／风险标签与精修目标，再按原始录音来源分组划分；独立测试录音需另采集。尚未训练或启用学习门控，不能据此声称已验证泛化、校准或端到端加速。
