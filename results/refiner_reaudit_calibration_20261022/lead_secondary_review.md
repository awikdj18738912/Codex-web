# Calibration 二次复核

复核范围：`development_v1.disagreements.jsonl` 中 FAIL 被 accept 的 006、007、032，以及原参考 verdict 为 PASS 的 044；证据仅取对应 `development.jsonl` 的 input/target。没有查看 holdout，也没有修改数据。旧 reference 来自 prior GPT diagnostic adjudication，并非人工金标。

## development_006 / customer_service_7192

**建议 verdict：FAIL；旧参考仍成立，但理由应收窄。** Input 先说“我之前在B站会员购买的”，随后明确说“啊不是，不是B站”，再说明想说的是南京云动科技有限公司的售后。Target 仍肯定保留“我之前在B站会员购官方平台购买的”，没有体现这次明确撤回，所以不忠实。B站购买信息是否真实，单看口语前半句确实有短暂不确定（“对吧”）；但随后“不对，不是B站”是直接否定，不能据此保留为最终事实。合肥万达中心地址在 input 后段又被反复确认，target 保留该地址是合理的。

## development_007 / daily_chat_12543

**建议 verdict：FAIL；旧参考成立。** Input 说“好像真能考个进士似的”后，马上多次纠正：“体验套装是给游客玩的，不是真让你考”。Target 保留“好像真能考个进士似的”，但删掉了这项澄清，造成把比喻/误说保留成事实的效果。target 后文还说“其实我一开始说……其实是‘体验套装’”，仍不足以表达这是游客体验而非真实科举考试。此处不只是保留口语比喻，清理时需要移除或改写为明确的游客体验描述。

## development_032 / vibe_coding_5776

**建议 verdict：FAIL；旧参考结论成立，旧理由有一处需更正。** Input 的 `EADDRINUSE: address already in use` 在 target 中变成“EADDRINUSE地址已占用”，丢失了冒号后的原始错误文本；`Stack overflow` 被改成“栈溢出”。这两个报错表达被翻译/改写，原参考的主要 FAIL 依据成立。`authToken` 在 input 与 target 中拼写一致，因此旧理由称其被改成小写不成立，应删除该点。shell 命令仍原样保留。

## development_044 / academic_3692

**建议 verdict：PASS；原参考仍成立。** Input 多次摇摆后最终明确回到 `Temporal action localization with transformers`；target 保留该最终研究方向。target 也保留了讨论中各英文标题/名称，包括 `Stochastic gradient descent`、`Meta-reinforcement learning for robotics`、`Causal Inference with Double Machine Learning and Control Variates` 和 `ExoMars：Mars rover imagery with mineral composition labels`。target 将大量口语赘词压缩为连贯叙述，没有改变最终选择或其余概念的关系。

Disagreements 文件只显示此条 `reference_verdict: PASS` 和 `error: unverifiable quote:input_quote`，没有提供 teacher 判不通过的独立记录；因此不能据此确认“teacher 判不通过”这一前提。本复核结论是依据可见 input/target 判断。
