# Luna独立审计记录

由主代理根据gpt-6-luna只读回复整理。子代理检查划分、状态、日志、产物和源码，没有修改文件、运行GPU或停止服务。

- 已用旧测试必须按整个模板排除，包含未被早期配对选中的同模板兄弟；stage排除原内部4模板和第二轮12模板。
- 训练441来源818ASR、开发162来源288ASR；clean/noise同来源不能当独立录音。旧开发继续复用，不称独立holdout。
- 首次10000窗口采样失败被及时发现，成功重采12/12阶段完成；train931+261、dev288+108，case/pair/label逐ID匹配，无训练开发交叉或调用失败。
- 局部效果标签保留strict_label；只处理投影删除一致或单词汇编辑距离改变的弱监督，不能当音频金标。
- native_refine过滤正确；group×task多视图权重正确，但不等于两个任务总权重均衡。
- 阈值按cohort×task、recording_group所有有益视图约束，保持至少95%；实际各来源的原收益组全部保留。
- 三个模型三轮结束，无NaN/OOM；MLM head的UNEXPECTED载入提示是base encoder架构差异，不是失败。
- 训练仅2 harm、开发0 harm，harm阈值1.01，没有真实风险控制证据。二分类API要走decide_with_fallback保留原KEEP/DEFER。
- 32×2流式会话完成，32/32配对raw ASR字符串和音频SHA相同，无Refiner调用失败。
- 学生四处错跳，全部是原策略更近意图脚本；不得省略、按测试重选阈值或声称上线。
- finish-tail是请求finish到最终事件的时间；32样本P95没有包含current的3.832s最大值，应同时报告max。该例Refiner评分约133ms，首次辅助模型评分/加载约3515ms。
- 本轮指标基于未核验的合成脚本，不是人耳金标或真实场景错误率。
