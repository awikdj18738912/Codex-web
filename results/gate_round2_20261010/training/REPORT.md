# 第二轮：保守否决门控试验

旧内部测试不参与本轮选择；新扩展 test 在模型和阈值冻结之后才生成、解码、配对。模型只否决原门控准备执行的操作，KEEP/DEFER 不评分，未知输入保留原动作。

主策略：encoder_weighted_native_veto。新测试配对数 27，有效标签 24。

| 策略 | 精修调用 | 捕获 benefit | weak harm | 目标精确率 | 目标 CER | 估算串行 ms |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0 | 0 | 0 | 0.519 | 0.0537 | 0.0 |
| current | 13 | 2 | 1 | 0.556 | 0.0421 | 3014.9 |
| always | 30 | 3 | 1 | 0.593 | 0.0374 | 8609.9 |
| linear_tfidf_direct | 0 | 0 | 0 | 0.519 | 0.0537 | 18.3 |
| linear_tfidf_native_veto | 13 | 2 | 1 | 0.556 | 0.0421 | 3022.2 |
| encoder_unweighted_direct | 1 | 1 | 0 | 0.556 | 0.0491 | 304.5 |
| encoder_unweighted_native_veto | 7 | 2 | 0 | 0.593 | 0.0397 | 1706.7 |
| encoder_weighted_direct | 3 | 1 | 0 | 0.556 | 0.0491 | 1091.4 |
| encoder_weighted_native_veto | 6 | 1 | 1 | 0.519 | 0.0514 | 1288.2 |
| previous_frozen_direct | 3 | 1 | 0 | 0.556 | 0.0491 | 1089.4 |

验证约束：每任务至少保留现有门控95%的 exact weak benefit，weak harm 不增加；满足约束后最小化验证操作成本。测试表现不保证继续满足该约束，不能据测试重选阈值。uncertain 保留在评估分母，不参与拟合。

训练/开发/测试每任务标签与独立 harm 模板数见 dataset_manifest.json、test_report.json。所有音频未经人工核验，模型概率未校准，线上继续使用原门控。


## 本轮结论

主策略弱收益保留 1/2，未达到95%的目标；精修逻辑调用 13→6，weak harm 1→1。因此保持现有门控。新测试只有27个配对、3条benefit和1条harm，窗口路径无harm，不能证明真实风险控制。

已为后续新试验增加按验证来源分别保留收益的约束与已用测试集禁复用检查，本轮模型、权重、阈值不动，也不据测试重新拟合。Luna在开发日志监督期间达到额度上限，后续训练和最终评估由主代理核查。

## 后续归因对结论的修正

以上表格是各操作独立执行的配对指标。window与local的“工作工作→工作”来自同一录音，不能把1/2弱收益保留直接解释成整体转录质量下降。
冻结策略的真实顺序回放（22条最终快照）发现：新旧最终文本逐字相同，Refiner调用11→6，plausibility各2次；局部审核补上了学生跳过的窗口修正。
独立测试中两者仍都误删明确的强调语句，所以继续不部署。细节见 [归因报告](/home/aim0/桌面/tan/ASR/Codex-web/results/gate_failure_audit_20261010/REPORT.md)。

冻结selection.json的`selection_partition`字符串漏写risk_dev；实际执行代码及validation.jsonl显示风险开发28个case均参与验证和选阈值。
保留冻结文件原样以维持证据哈希，后续训练脚本已修正该字段。风险开发集回放不能当作独立风险测试。
