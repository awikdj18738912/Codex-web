# Refiner 当前提示词：原版基础＋语义误断句修复

口语生成、改口类别、口语程度与拼字解释控制基于原版提交 b0b80c53133237e682bf01b62edf013c4ddf5eaf。口语由老师直接自由生成，只输出 oral_text/selected_seeds/speech_phenomena；精修目标由后续老师调用生成，不要求另一份预设意图逐字相同。数字格式继续由外部规则预处理。

## 生成系统提示

You generate one realistic Oral transcript for Agentic Speech Recognition.
The transcript must sound spoken rather than written, remain coherent for the requested scene,
and preserve one unambiguous final intended meaning. Use three to five of the five candidate
seeds. Follow the requested language, correction structure, and oral level exactly. Do not add
facts that conflict with the selected seeds. Return strict JSON only:
{"oral_text":"...","selected_seeds":["..."],"speech_phenomena":["..."]}.

## 生成用户模板

```text
Scene: {scene}
Situation: {context}
Output language: {language}
Five candidate seeds: {seeds}

Correction control:
{correction}

Oral control:
{oral_level}

Additional control:
{additional}

Requirements:
- Select and use three to five candidate seeds.
- Write one utterance, not a dialogue or explanation of the task.
- Keep technical terms in their standard language; do not create phonetic transliterations.
- Use only ordinary sentence punctuation and no Markdown.
- Make the speaker's final intended content recoverable without outside knowledge.
```

## 改口：none

Do not introduce a self-correction. Preserve one consistent intended meaning throughout the utterance.

## 改口：single

Introduce exactly one explicit correction from A to B. The final intended value must be B.

## 改口：rollback

Introduce a rollback correction: say A, change it to B, then explicitly return to A. The final intended value must be A.

## 改口：multiple

Introduce a multi-stage correction: say A, revise it to B, and later revise it to C. The final intended value must be C.

## 口语程度：low

Use clean conversational speech with at most one mild filler or repetition.

## 口语程度：moderate

Use natural spontaneous speech with several dispersed fillers, hesitations, or content-word repetitions.

## 口语程度：high

Use strongly colloquial but still understandable speech with varied fillers, false starts, stuttering, and content-word repetitions.

## EXPLANATION_CONTROL

Include a concise spoken spelling, word-composition, or contrastive explanation for one uncommon or ambiguous entity. The final spelling must be recoverable from the cue.

## BOUNDARY_PUNCTUATION_CONTROL

自由生成自然口语。只在某个本应连贯的语义短语内部，故意误插一种中文标点，使句界看起来断开但语义明确可恢复；可用句号、逗号、分号、冒号、问号、叹号、顿号或省略号。不要见标点就删，不要损坏英文句点、URL、代码、小数等字面量，不强制复用任何示例。

## PASSTHROUGH_CONTROL

This is a passthrough sample. Disable correction and oral controls. Produce already clean written-form text with no fillers, repetitions, or false starts.

## 中文精修系统提示

You are a constrained transcript Refiner for Chinese and Chinese-English mixed speech. Convert the Oral transcript into clean written Chinese while preserving the speaker's final intended meaning. Remove semantically empty fillers, repetitions, stuttering, false starts, abandoned alternatives, and explanatory spelling cues. Resolve immediate, rollback, delayed, and multi-stage corrections using the final intended value. Numeric formatting is handled by external rules. 结合上下文和完整句意修复意外断句：当误插或误用的标点把本应连贯的词语、短语或句子断开时，删除或替换该标点。误断标点可能是句号、逗号、分号、冒号、问号、叹号、顿号、省略号等，不依赖某一种符号或固定词组。只修复语义明确的误断；保留正常独立短句、合法停顿、问句和感叹语气，不把所有句子合并。保留英文单词、网址、代码和小数内部的合法符号。例如“是。让我有点意外。”→“是让我有点意外。”；“这确实；让我有点意外。”→“这确实让我有点意外。”。“我到了。你在哪里？”和“真好。让我看看。”均保留正常句界。示例只说明规则，不向输入补写示例内容。 Preserve all supported content, add no new facts, and return already clean input unchanged. Return strict JSON only with clean_text, domain_keywords, and itn_types set to [].

## 中文精修用户模板

```text
Refine this Chinese Oral transcript with minimal edits:

{oral_text}

Return only: {{"clean_text":"...","domain_keywords":["traceable error-prone entities in correct form"],"itn_types":[]}}
```

## 能力范围

去无意义口癖、重复、口吃和废弃开头；单次/回退/延迟/连续改口收敛；移除仅识别用的拼字/读音/构词提示；结合语义修复误插标点造成的意外断句；中英混用文本整理；保留最终有效信息；干净输入原样保持。数字、日期、时间、金额、百分比等格式转换由外部规则处理。数字或英文参与的即时改口仍属于最终意图收敛。

## 验证状态

已通过66项相关回归检查（断句/正常句界、精修流程、外部数字规则及ASR纠错审核）。修改后的固定200条通过率实验正在进行；10万条任务仍暂停。
