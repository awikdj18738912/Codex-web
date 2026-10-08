# MultiMed 中文实体库评估

入口：`experiments/scripts/benchmark_multimed_entities.py`。默认使用本机
`/home/aim0/data/datasets/ASR/MultiMed/Chinese/test-00000-of-00001.parquet` 的全部原音。

流程：提取音频并保存哈希 → 真实无实体库转录 → 仅从参考转写生成医学词表 → 无库响应回放控制 → 同一 ASR 响应的有库回放 → 报告。

基线使用真实 ASR、真实本地 Refiner、现有数字规则及 WebSocket 处理。无库条件设置 `entity_db=None`、模糊模式 off。第二遍和控制使用同一组真实 ASR 响应；精修模型照常推理，没有假模型，也没有模型响应缓存。各条件关闭“词库辅助识别”。

库只存参考中实际出现、由 CMeEE 标注或审核完整短语支持的标准名；没有错字别名。中文名附固定医疗语境词，漏字／近音权限默认关闭。每个词可追溯参考句和字符位置。词库来自被测参考，这是用户指定的已知词表覆盖实验；不能用来宣称未知实体泛化能力。

## 运行

在项目根目录执行，选一个有足够显存的 GPU：

```bash
PYTHONPATH=.:experiments/scripts:.runtime/medical_entity_deps \
  /home/aim0/data/conda/envs/agentic-asr/bin/python -u \
  experiments/scripts/benchmark_multimed_entities.py run
```

这个入口按顺序完成各阶段。已完成样本保存在 JSONL 中，重跑命令可续跑；已有样本不重复处理。切换模式、解码间隔、模型或数据文件时应使用新的输出目录。

也可以分阶段执行：

```bash
python experiments/scripts/benchmark_multimed_entities.py prepare
python experiments/scripts/benchmark_multimed_entities.py baseline
python experiments/scripts/benchmark_multimed_entities.py build-entities
python experiments/scripts/benchmark_multimed_entities.py control
python experiments/scripts/benchmark_multimed_entities.py paired
python experiments/scripts/benchmark_multimed_entities.py report
```

以上分步命令同样需要合适的 Python 环境及 `PYTHONPATH`。`build-entities` 会检查无库基线是否全部完成；回放前检查冻结词表哈希。

参数包括 `--dataset`、`--output`、`--library`、`--medical-lexicon`、`--model`、`--device`、`--asr-url`、`--mode` 和 `--chunk-seconds`。默认 streaming、2 秒解码间隔，每次传输 1 秒 float32 PCM；评估无需修改线上服务配置。

## 输出

- `results/multimed_chinese_test_eval_20261008/manifest.json`：225 条样本、音频与源码哈希、固定参数。
- `baseline.jsonl`：真实首遍原始 ASR 响应、最终输出和精修调用。
- `control_replay.jsonl`：相同 ASR 响应的无库控制。
- `entity_library.jsonl`：有库回放，包含候选、决策、替换与模型调用。
- `comparison.csv` / `comparison.jsonl`：所有样本的参考、无库、有库和差异。
- `changes.json`：有变化的样本。
- `missing_entities.json`：未保留的参考实体及原始转录、最终输出、召回候选和纠正提示。
- `control_replay_differences.json`：真实首遍与无库回放的差异；不归因于实体库。
- `summary.json` / `REPORT.md`：汇总和逐条效果。
- `data/entity_benchmarks/multimed_chinese_test/entities.db` / `entities.json`：项目中的独立评估实体库。
- `vocabulary_evidence.json` / `sample_entities.json` / `provenance.json`：词表来源与计分标注。

报告同时给出参考文字 CER、名称保留次数、实体增益／丢失、文字改善／退化及精修调用数。数字表面形式使用项目现有 ITN 统一后计分。后端未检测到语音的样本按空转录计分，仍包含在总数中。参考可能省略英语、背景语音和口头语；名称统计是最长非重叠词表出现，不是完整人工 NER 精确率。

评估词库通过项目现有 `EntityStore` 写入，随后由真实应用实例加载。需要在独立页面使用时，可用启动参数 `--entity-db` 指向该文件；它也可通过实体 CLI 导入到另一个测试库。
