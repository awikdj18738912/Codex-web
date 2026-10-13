# 专业术语跨域样本 staging

仅包含人工挑选的标准中文术语种子，且每个词条都在 Common Voice zh-CN test 或 SpeechIO 本地源参考中逐条找到完整字面证据。`review_method=reference_curated` 表示核对了参考文本和实体跨度，不表示人工听审音频。`draft_candidates.json` 明确不计入审核集合。

`samples.jsonl` 给出音频文件路径或 Parquet 文件与零基行号定位、参考、跨度和时长。源文件未经修改。Common Voice 音频路径逐条检查并用 ffprobe 获取时长；SpeechIO 样本按源 parquet 精确行定位并使用其时长字段。排除了既有 Common Voice 两批样本及旧 SpeechIO domain protocol 的 sample ID/source ID 和完整参考句。

查看 `stats.json` 的实际领域覆盖和缺失种子；只有其中 `reviewed_unique_entities` 可作为本 staging 已审核参考候选计数。
