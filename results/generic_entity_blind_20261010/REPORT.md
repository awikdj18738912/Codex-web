# 新来源预建词库验证

规则和词库在任何 ASR 请求前冻结；标准转录只用于提前建词库和最后评分。两来源此前未发现用于实体规则开发的记录，但不能证明模型预训练从未见过。FLEURS 未提供说话人 ID，因此不宣称说话人独立。

全部样本新调用无词库 ASR；正式原音复核对原始 PCM 实际重识别。三个 Web 条件共享相同初稿，使用实际 GPU Refiner；Web 阶段回放本轮真实 ASR 响应以控制解码波动。

|来源/分组|条件|样本|实体命中|CER|额外未参考实体次数|改善/退化|
|---|---|---:|---:|---:|---:|---:|
|all|raw_initial|128|131/141|8.93%|0|0/0|
|all|raw_repair|128|137/141|8.77%|0|5/0|
|all|no_dictionary|128|131/141|8.75%|0|5/0|
|all|dictionary_rules|128|131/141|8.75%|0|5/0|
|all|audio_repair|128|137/141|8.59%|0|5/0|
|commonvoice_zh-CN_test|raw_initial|64|50/52|8.46%|0|0/0|
|commonvoice_zh-CN_test|raw_repair|64|50/52|8.46%|0|0/0|
|commonvoice_zh-CN_test|no_dictionary|64|50/52|8.37%|0|1/0|
|commonvoice_zh-CN_test|dictionary_rules|64|50/52|8.37%|0|1/0|
|commonvoice_zh-CN_test|audio_repair|64|50/52|8.37%|0|0/0|
|commonvoice_zh-CN_test:lexicon_absent|raw_initial|16|0/0|13.45%|0|0/0|
|commonvoice_zh-CN_test:lexicon_absent|raw_repair|16|0/0|13.45%|0|0/0|
|commonvoice_zh-CN_test:lexicon_absent|no_dictionary|16|0/0|13.03%|0|1/0|
|commonvoice_zh-CN_test:lexicon_absent|dictionary_rules|16|0/0|13.03%|0|1/0|
|commonvoice_zh-CN_test:lexicon_absent|audio_repair|16|0/0|13.03%|0|0/0|
|commonvoice_zh-CN_test:lexicon_present|raw_initial|48|50/52|7.13%|0|0/0|
|commonvoice_zh-CN_test:lexicon_present|raw_repair|48|50/52|7.13%|0|0/0|
|commonvoice_zh-CN_test:lexicon_present|no_dictionary|48|50/52|7.13%|0|0/0|
|commonvoice_zh-CN_test:lexicon_present|dictionary_rules|48|50/52|7.13%|0|0/0|
|commonvoice_zh-CN_test:lexicon_present|audio_repair|48|50/52|7.13%|0|0/0|
|fleurs_zh-CN_test|raw_initial|64|81/89|9.12%|0|0/0|
|fleurs_zh-CN_test|raw_repair|64|87/89|8.90%|0|5/0|
|fleurs_zh-CN_test|no_dictionary|64|81/89|8.90%|0|4/0|
|fleurs_zh-CN_test|dictionary_rules|64|81/89|8.90%|0|4/0|
|fleurs_zh-CN_test|audio_repair|64|87/89|8.68%|0|5/0|
|fleurs_zh-CN_test:lexicon_absent|raw_initial|16|0/0|2.64%|0|0/0|
|fleurs_zh-CN_test:lexicon_absent|raw_repair|16|0/0|2.64%|0|0/0|
|fleurs_zh-CN_test:lexicon_absent|no_dictionary|16|0/0|2.64%|0|0/0|
|fleurs_zh-CN_test:lexicon_absent|dictionary_rules|16|0/0|2.64%|0|0/0|
|fleurs_zh-CN_test:lexicon_absent|audio_repair|16|0/0|2.64%|0|0/0|
|fleurs_zh-CN_test:lexicon_present|raw_initial|48|81/89|10.69%|0|0/0|
|fleurs_zh-CN_test:lexicon_present|raw_repair|48|87/89|10.42%|0|5/0|
|fleurs_zh-CN_test:lexicon_present|no_dictionary|48|81/89|10.42%|0|4/0|
|fleurs_zh-CN_test:lexicon_present|dictionary_rules|48|81/89|10.42%|0|4/0|
|fleurs_zh-CN_test:lexicon_present|audio_repair|48|87/89|10.15%|0|5/0|

audio_repair 的改善/退化相对 dictionary_rules；其他行相对原始 ASR，不能将精修导致的变化归因于实体功能。

原音修正 5 个样本、6 次替换。逐次独立替换的字符距离改善/持平/变差：6/0/0；此指标不是语义人工正确率。
总额外带词库 ASR 请求耗时 57.97 秒，均摊每样本 0.45 秒；仅发生复核的样本耗时见 comparisons.json。

负例仅表示所建词库实体不在参考中，仍使用对应完整词库；不会逐句预告实体。所有完整参考参与 CER，括号英文是否口述等数据标注风险需单独人工检查；不得按结果删样。
生产代码冻结哈希通过，正式数据库保持原样：True。本轮不修改生产规则。
