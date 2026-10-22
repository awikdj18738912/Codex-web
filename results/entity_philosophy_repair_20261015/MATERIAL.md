# 可用于实体修复开发回归的 10 分钟材料

本例用于检查“有错时是否修复、已对时是否误改”。它已经由当前无提示 Qwen 实测筛查出错例，并在运行修复前固定标注与词库。没有人为改坏转写、替换音频、加噪或拼接短句。

## 输入与来源

- 自然对谈：本地 `测试音频5.mp3` 的前 600 秒，完整本地文件约 14 分钟。
- 对应候选切片页：[罗翔与易中天探讨人性](https://www.bilibili.com/video/BV1ACGHz2Ec4/)。该页是二次发布，不能称为已核验的官方原始节目。
- 输入：`philosophy_0_600.wav`（试听）、`philosophy_0_600.f32le`（16kHz mono float32 实验输入）。
- 领域库：`philosophy_entities.db`，53 个哲学/历史人物、作品和术语，包含未出现的合理干扰项；无 ASR 错写别名，未按修复结果调参。
- 逐处实体参考：`entity_annotations.json`。共 34 处标注，核心 31 处，疑似口误/词序异型 3 处单列。

## 先验错例

时间是筛查时该错写首次完整可见的音频位置，不是人工对齐的实体发音边界。每处均提供约 15 秒上下文音频，见 `review.html` / `review_clips/`。

| ASR 可见位置约 | 原始 Qwen 拼写 | 参考实体拼写 | 当前模块最终结果 |
|---|---|---|---|
| 02:27 | 毛中刚 | 毛宗岗 | 未修复 |
| 02:29 | 毛中刚 | 毛宗岗 | 未修复 |
| 07:03 | 适当去马斯洛斯 | 色拉叙马霍斯 | 未修复 |
| 07:13 | 适当和马修斯 | 色拉叙马霍斯 | 未修复 |
| 07:50 | 苏格拉你 | 苏格拉底 | 未修复 |

“毛宗岗”的规范拼写见[广东政协文史资料](https://www.gdszx.gov.cn/zxkw/tzgj/2022/05/content/post_29678.html)；“苏格拉底”“色拉叙马霍斯”的规范拼写见[商务印书馆图书目录](https://www.cp.com.cn/book/0f04e3c1-f.html)。身份由对话上下文判断。

标注尚无独立人工听审。上述是资料核准后的诊断拼写参考，不是逐字人工 gold。正式用于论文准确率之前，应逐处听审；尤其外国姓名的说话人口误、词序异型不能全部归因 ASR。本例将另外 3 处不确定异型排除出核心分母。

## 当前模块对照

同一次无提示主转写与实体模块输出逐处比较，未用“某个名字在别处出现过”代替该处命中。

| 诊断口径 | 原始主转写 | 实体模块输出 |
|---|---:|---:|
| 核心 31 处实体提及的参考命中 | 26/31 | 26/31 |
| 其中 5 处核心错例恢复 | 0/5 | 0/5 |
| 原本正确的 26 处对照 | 26/26 | 26/26 |

模块产生两次 `audio_replay` 补丁，把一处“色拉霍马叙斯”和一处“色拉霍马去世”规范为“色拉叙马霍斯”；它们位于不确定组，不据此声称恢复了两处确定 ASR 错误。核心 5 处仍未修复，26 处正确对照未见丢失。

当前结果说明材料具有修复测试价值，也明确暴露了模块漏修。以后的修改应在同一冻结输入、同一词库上比较这 5 处恢复数及 26 处误改数，不能通过修改测试转写或新增错词别名来让指标变好。

这轮回放加速提交音频，并在每包后等待 CTC 实际音频计算完成，以排除文件发送过快造成的队列过载；它只用于质量诊断，运行耗时不是实时延迟。CTC 处理记录为 600 秒、1876 个块事件、错误计数 0，CTC 提案和最终补丁均为 0；最终两处修改来自原音复核。没有运行下游 Web 精修。

## 复用

筛查无提示主转写（不读取词库/标注）：

```bash
.runtime/entity_ctc/bin/python experiments/scripts/screen_entity_repair_audio.py \
  --pcm results/entity_philosophy_repair_20261015/philosophy_0_600.f32le \
  --output /tmp/philosophy_raw_screen.json
```

修复质量回放（选择新输出目录，脚本拒绝覆盖已有结果）：

```bash
.runtime/entity_ctc/bin/python experiments/scripts/evaluate_entity_repair_material.py \
  --manifest results/entity_philosophy_repair_20261015/manifest.json \
  --output-dir /tmp/philosophy_entity_rerun \
  --mode on
```

当前完整逐提及结果见 `REPORT.md` / `entity_repair_report.json`，原始与修复文本见 `raw_screen.json` / `quality/on.json`。`entity_reference.txt` 只校正标注实体，其他文字保留主转写，禁止将其用于全文 CER。
