# 本地癌症视频播客实体库对照

输入为项目根目录的《癌症概念101：早期晚期，靶向药，免疫治疗》MP3，约 37 分 29 秒。
整段按连续 60 秒窗口处理，共 38 窗口，不挑选片段，不添加噪声。

比较真实无库首次转录、无库回放、此前 42 词医疗库回放和 141 词肿瘤专题库回放。
回放固定每块及最终真实 ASR 响应，精修模型重新推理，不缓存精修结果。
使用当前流式 Web 流程、2 秒解码间隔、真实本地精修、现有数字规则；关闭词库辅助识别。
每个窗口独立建立 ASR 会话，跨窗口的上下文不延续，因此不等同于网页中一次连续播放完整长音频。

两份词库在首次识别前固定。专题库包含之前的 42 词及脚本中提前列出的肿瘤相关标准名，
使用固定肿瘤语境词；未写入识别错字别名，不开启漏字和近音权限。
专题库位于 `data/entity_benchmarks/oncology_podcast_20261008/`，不修改正式用户词库。

没有人工标准转录。本轮报告只展示文字差异、候选、替换和模型调用，不能计算 CER、
召回率或正确纠错数；中间替换事件可能重复，不能直接解释为准确率收益。

在项目根目录运行，需要 ASR 服务和足够的 GPU 显存：

```bash
PYTHONPATH=.:experiments/scripts:.runtime/medical_entity_deps \
  /home/aim0/data/conda/envs/agentic-asr/bin/python -u \
  experiments/scripts/benchmark_oncology_podcast.py run --device cuda:0
```

已完成窗口自动跳过。仅重新生成报告无需加载精修模型：

```bash
PYTHONPATH=.:experiments/scripts:.runtime/medical_entity_deps \
  /home/aim0/data/conda/envs/agentic-asr/bin/python \
  experiments/scripts/benchmark_oncology_podcast.py report
```

输出在 `results/oncology_podcast_20261008/`：

- `manifest.json`：原文件哈希、完整窗口清单、原生产代码哈希。
- 各条件 JSONL：真实 ASR 输入、候选决策、替换记录及实际精修调用。
- `comparison.csv` / `comparison.json`：全部窗口文字和差异。
- 各条件 TXT：带时间窗的完整输出。
- `summary.json` / `REPORT.md`：汇总及具体变化。
- `scope_verification.json`：输入一致性及生产代码、正式词库保护检查。
