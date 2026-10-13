# Luna独立只读复核

模型：gpt-6-luna；仅阅读源码和产物，没有运行GPU、训练、修改策略或停止服务。
主代理根据Luna两轮审计回复整理本记录。

- 回放窗口分组、VAD源分段、最终实体归一化、重建局部候选、审核与清理顺序符合冻结Web路径。
- 学生输入只含操作前文本与实际候选上下文，没有将spoken/intended脚本传入模型。
- fresh cache明确属于cold final snapshot，不能解释成带先前流式缓存的真实finish，也不能用其耗时声称在线提速。
- 22条测试回放完整。新旧规范化结果22/22相同；11→6 Refiner、各2 plausibility；CER代理0.04885，各2改善/1退化。
- “工作工作→工作”跨window/local重复计数得到实证；学生窗口跳过后，局部候选重建并修好。独立配对收益行不能直接当作端到端增量收益。
- 独立测试中的“很重要，很重要……连续强调了两遍”在current/student/always中均被删成一次，学生局部门控未拦住。按未核验意图脚本是唯一退化，不能宣称已证实实际音频事实错误。
- 风险开发14组的8条新旧差异及摘要一致：current退化8，student退化0；Refiner26→2；其余6条规范化文本相同。
- risk_dev的28/28个case_id确实参与validation与阈值选择，8→0属于开发集回看，不能称独立风险评估或泛化。
- 冻结selection.json的selection_partition字段漏掉risk_dev，应在报告勘误；保留冻结哈希，未来输出修正字段。
- 配对归因保留audio_script_verified=false和human_gold=false；neutral只表示操作未改变输出，不表示文本正确。删除投影只能当诊断，不能自动升格为金标。

Luna未发现会推翻顺序回放摘要的实质性顺序或输入泄漏缺陷。上线判断仍需新的独立数据与完整流式会话测试。
