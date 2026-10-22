# 分库修正的评测说明

原混合专名库已改为10个有语义线索的场景库和1个未定库。原6,808个不同词全部保留在场景词库并集中，各库仍包含全部候选池中未入选录音的干扰词。其余26库字节不变。最大4,177词，37个库均通过原默认5,000词上限检查，全部目标实体覆盖无缺失。

原始2,000条录音及参考和baseline均不变。原混合专名组的727条重新调用真实ASR；其他1,273条只有在实体模块源码、实际PCM SHA256和完整词库SHA256均相同时才复用上轮真实结果及HTTP记录。复用条件和来源写入protocol.json及reused_samples.json，结果逐条有reused_from字段。为隔离词库变化，本轮使用上轮的source_snapshot/system；不是测试重测期间其他正在修改的生产源码。

全部2,000个首轮文本与原baseline完全一致。原音修复命中从旧库1,433/2,658（53.91%）变为新库1,440/2,658（54.18%）；原ASR仍为1,342/2,658（50.49%）。新分库比旧库多命中7次，7个样本命中增加，没有样本净命中减少。对原ASR新增恢复98次提及、丢失0次。文字匹配容量错误727→0，规则诊断覆盖1273→2000。神经Refiner和完整Web均未运行。

## 两个原误改样本

- chall_aishell_BAC009S0707W0383：旧库将“许昕、生”改为参考不支持的“许敏、生”；新库保留“许昕、生”，没有把“徐引生”修对，不能计作正确修复。
- leb1_c0f03a62bfeab44c3bc3：标准参考为“孙永杰”；旧库从“孙颖杰”误改为“孙英杰”，新版正确改为“孙永杰”。

新版自动审计未发现新增的、标准参考不支持的库实体。未逐条人工听审，不能因此声称总体误改率为零。

## 场景标注限制

场景由遮蔽目标实体后的参考文本按通用语义关键词离线标注，不是官方逐条topic、也不是在线自动场景识别。433/727条仍缺乏足够场景证据，保留“综合新闻_场景未定”，该库4,177词。其余场景与全部词条来源证据在新dataset的route_annotations.jsonl和vocabulary_provenance.jsonl。此次属于已知场景条件下的词库对照，不能作为独立盲topic泛化结论。

## 清单元数据变更记录

dataset/manifest.json在评测启动后因准备脚本快照哈希更新发生变化。原protocol中记录的manifest哈希保留原值，没有改写；当时的旧manifest没有备份，不能声称已恢复原始清单。逐文件检查确认变更涉及manifest和准备脚本快照，实际推理使用的输入、完整词库及运行时实体源码冻结副本不变；参考和baseline与原版字节一致，所有当前manifest文件hash均匹配。详细记录见manifest_metadata_review.json，最终审计明确标记PASS_WITH_DOCUMENTED_MANIFEST_METADATA_CHANGE。

原版目录和结果均保留；后续工作区评测/评分脚本默认指向results/entity_challenge_20261014_scene_v2/dataset。

数据源官方项目于2026-10-14查阅：[AISHELL-NER](https://github.com/Alibaba-NLP/AISHELL-NER)、[SpeechIO](https://github.com/SpeechColab/Leaderboard)。本轮性能数值来自本地实际测试。
