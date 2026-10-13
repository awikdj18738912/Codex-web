# 人工收益标注说明

此表的 human_label 初始为空，不存在自动生成的真实收益标签。
automatic_pair_label 只对两条分支最终输出逐字一致的样本填 neutral；这是相对操作无效果的证据，不代表原 ASR 正确。
标签评价 always_output 相对 baseline 的变化：benefit、neutral、harm、uncertain。
先核对原音及参考，再判断最终意图；不得只按 CER 下降、发生修改或验证器接受打标签。
错误实体、数字、否定变化、关键内容删除或新增事实优先记 harm；参考／原音存疑记 uncertain。
reference_status 可填写 verified_verbatim、verified_intended、uncertain 或 missing_reference。
candidate_context 是局部复核时已有上下文；reference_full 仅用于离线审核，不进入模型输入。
窗口输出的 comparison_text 是把该局部结果替入共同全文基线后的距离代理，不能当作完整有状态会话结果。
已有资料已用于开发，公开 TTS 与真人数据须分别报告；噪声版本不是独立录音。
report 子命令会保留 human_label、rationale、reviewer 和 reference_status 的已有填写。
