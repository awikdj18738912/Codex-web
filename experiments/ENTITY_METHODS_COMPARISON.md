# 实体后处理与词库辅助识别的四组对照

入口：`experiments/scripts/compare_entity_methods.py`。本轮不修改生产代码或正式实体库。

| 条件 | ASR 提示 | 后处理词库 |
|---|---|---|
| baseline | 关闭 | 无 |
| post_only | 关闭 | 完整场景库 |
| context_only | 开启 | 无 |
| combined | 开启 | 完整场景库 |

所有条件都使用真实精修模型、同样的 tri_state 门控、final off 和 zh-itn v90。
“仅辅助识别”通过实验进程中冻结的提示构造器独立提供词表，后处理的保护器／匹配器仍无数据库。
这样可单独测提示收益，不把已有实体替换或保护算到提示效果中。

## 输入与冻结

- MultiMed 中文测试集全部 225 条原音及配套参考，约 50 分钟；此前从这些参考提取的 42 词库。这是用户指定的已知词表覆盖实验。
- 癌症播客的 11 个连续 60 秒窗口：6 个已见修复窗口及 5 个对照窗口，约 11 分钟。没有字幕；不作为独立泛化验证。
- 保持当前默认 16 个提示词、1024 字符，按领域／优先级／ID 选择。医疗提示覆盖参考中 27/63 次词表名称出现；其余名称不直接列入提示。
- 两份词表和全部生产 Python 源码哈希在识别前冻结，没有加入识别错字别名或逐句参考。

每段音频分别进行两次新的真实 ASR，交替 off/on 的执行顺序，PCM 哈希完全相同。
随后每份识别响应分别运行无库／有库的真实精修流程；只回放识别响应，不缓存精修结果。
这避免了四组重复识别带来的额外差异。质量回放耗时不作为完整转录速度指标。

## 运行

项目根目录，使用有依赖和足够显存的 Python 环境：

```bash
PYTHONPATH=.:experiments/scripts:.runtime/medical_entity_deps \
  /home/aim0/data/conda/envs/agentic-asr/bin/python -u \
  experiments/scripts/compare_entity_methods.py run
```

支持 `prepare`、`decode`、`replay`、`report` 分步执行和已完成记录续跑。
`decode` 使用现有 ASR 服务，不加载额外 GPU 模型；`replay` 加载真实本地 Refiner。
重新生成报告无需加载精修模型。

## 输出与计分

输出位于 `results/entity_methods_comparison_20261009/`：

- `REPORT.md`、`summary.json`：两类素材的四组结果。
- 各素材 `protocol.json`：固定样本、音频哈希、实际提示名单、库及源码哈希。
- `asr_off.jsonl` / `asr_on.jsonl`：真实启动确认、逐块和最终 ASR 响应。
- 四个条件 JSONL：最终文字、候选与替换、实际模型调用。
- `comparison.csv` / `comparison.json`、各条件 TXT：逐条原始识别、参考和最终文字。
- `raw_summary.json`：医疗原始 ASR 的参考 CER 及名称保留数。

医疗评分使用相同 ITN、NFKC 和标点／空白处理，统计 CER、词表内最长非重叠名称保留、
每个名称的增益／丢失、提示目标覆盖、改善／退化样本、精修调用。名称交换同时记为增益与丢失，不用净值掩盖退化。

非实体区 CER 按参考与输出的字符对齐排除参考实体区域，实体边界后的口头词插入仍计入非实体错误。
这是参考文字对照，不是完整人工语义误改标签。参考字幕存在省略、英语缺标或个别错误，不能将绝对 CER 当成经听辨核准的准确率。
播客只展示字面变化和标准名称出现，不计算 CER／准确率。
