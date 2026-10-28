# 配对试验固定盲审标准

先评自然度与语义，再识别现象。审核者只看目标和口语，不看版本与指定类别。模型审核不是人工金标。

每条记录：blind_id, source_quality(pass/borderline/fail), fidelity(pass/fail), observed(列表：fillers, repetition_false_start, self_correction, boundary_punctuation, none), note。

source_quality：完整文本应是可理解的中文说话内容，不能靠目标猜测拼接段落；边界样本单独计数。目标本身矛盾或拼接也应标记，不因两侧一致就放行。

fidelity：口语最终有效内容应覆盖目标全部事实、条件与意图，无补写。明确被撤回的局部错误表达不算新增有效事实。

self_correction：必须有先前局部表达被后续明确替换或撤回；只有呃、重复、解释或同义重述不算。

boundary_punctuation：标点在完整词之间错误截断连贯句法关系。自然分句、正常停顿、斜杠、拆词不算。

fillers：有可删除且不影响意图的自然填充。repetition_false_start：有可删除的冗余重复或未完成起头，不把强调和有效内容当冗余。

解盲后严格候选需 source_quality=pass、fidelity=pass、实际现象匹配请求类别、表面校验通过。中英混用类必须有中文侧可清理现象且含原有ASCII内容。直接复制控制样本12条不计生成成功率。报告所有类别的请求数、输出数、严格候选数；拒答与格式错误保留在分母里，不复写为合格样本。
