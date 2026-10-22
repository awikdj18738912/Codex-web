# 实体流式修复实验报告

## 范围和结论

本报告汇总版本归档中的离线评分：`versions/v1/*score.json`、`versions/v2/*score.json`，以及 `versions/v3_before_status/independent_score.json`。推理时没有注入 gold/reference；gold 仅由 scorer 离线计分。跨领域组由 48 个 source clips 拼成 6 条组级音频，每组 8 个 source clips。表中的 CER 是字符错误率；命中数是对固定目标词提及数的计数。

| 场景 | V1 old → new | V2 old → new | V3（部署前状态归档）old → new |
|---|---:|---:|---:|
| 3 条 sparse 长拼接：目标命中 / 3 | 3/3 → 3/3 | 3/3 → 3/3 | 未评分 |
| 48 clips、6 组跨领域：目标命中 / 53 | 53/53 → 51/53 | 53/53 → 53/53 | 53/53 → 53/53 |
| known_dense：目标命中 / 40 | 37/40 → 36/40 | 37/40 → 36/40 | 未评分 |
| Li 322 秒音频：末尾 raw 是否出现“厉飞宇” | 否 → 是 | 否 → 是 | 未评分 |

跨领域组的版本变化值得单独看：V1 的 new 臂漏掉了 2 个新闻实体修复；V2 加入 deferred 候选重试和中分候选预算预留后，new 臂在 53 个目标提及上达到 53/53。V3 部署前状态归档的离线分数也为 53/53。这个语料已经被用于回归检查和阈值/预算调整，因此结果不能称为独立盲测，也不能视为未调参的泛化估计。

## 详细评分

### 三条 sparse 长拼接

V1 和 V2 的 sparse 汇总相同：old/new 都命中 3/3；clean CER 都为 11.81%，raw CER 都为 11.42%。V2 new 的音频重放累计为 24.24 秒，old 为 271.916 秒；每臂只有 3 个样本。这里报告 scorer 使用的 sparse 固定参考，不把它外推成一般长音频准确率。

### 48 clips、6 组跨领域拼接

| 归档 | 臂 | 命中 | raw CER | clean CER | 平均总耗时 / 组 | 重放音频秒数 | 实体变化数 |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | old | 53/53 | 2.92% | 3.22% | 12.433 s | 429.592 | 4 |
| V1 | new | 51/53 | 3.12% | 3.42% | 12.295 s | 139.000 | 2 |
| V2 | old | 53/53 | 2.92% | 3.22% | 12.031 s | 429.592 | 4 |
| V2 | new | 53/53 | 2.92% | 3.22% | 11.932 s | 108.000 | 4 |
| V3 部署前状态 | old | 53/53 | 2.92% | 3.22% | 11.996 s | 429.592 | 4 |
| V3 部署前状态 | new | 53/53 | 2.92% | 3.22% | 11.883 s | 108.000 | 4 |

V1 new 在 `news_group_1` 为 5/6、`news_group_2` 为 8/9；V2 new 两组分别回到 6/6 和 9/9。这与 deferred 重试及中分预算预留后的实体变化记录一致。

整体均值不能掩盖单组变慢。V2 `news_group_2` 的 new 臂总耗时为 16.850 秒，old 为 13.501 秒；同版本 known_dense 的 new 臂平均总耗时为 12.154 秒，old 为 11.652 秒。重放音频减少，并不保证每组的端到端耗时都下降。

### Known dense，40 个目标提及

V1 与 V2 的汇总相同：old 命中 37/40，new 命中 36/40；两臂的实体变化数都是 0。V2 的首轮 `primary_hits` 已经是 old 37、new 36，最终 raw 仍为 37 和 36。因此这一分差在实体后处理之前就已存在，不能归因于实体修改；报告保留这个差异，不用别的分数代替。

V2 old/new clean CER 分别为 2.92% 和 3.04%。new 的平均总耗时 12.154 秒，old 为 11.652 秒。重放音频从 old 的 336.390 秒降至 new 的 91.000 秒，但部分组耗时增加。

### Li 322 秒音频

没有完整标准稿，只报告目标词和时间，不计算 CER、WER 或一般准确率。V2 old/new 的总耗时分别为 46.1567 秒和 40.8084 秒；old 最终 raw 未出现“厉飞宇”，clean 中出现；new 的最终 raw 与 clean 都出现，raw 流首次出现时间为 8.5516 秒。V1 归档中的 53.0602 秒是较早一轮测量，不作为最终比较值。

## 当前冻结规则

本轮冻结配置记录为 `frozen_parameters.json` 中的 `production_bounded_incremental_entity_v2`：

- 单个重放窗口上限 32 秒，环形音频缓存 96 秒。
- 候选最低检索分数 0.63，同 span 候选差距至少 0.04；分数达到 0.85 归为高分候选。
- 总重放音频预算为当前音频长度的一半，并受单窗口下限与 300 秒上限约束；分数低于 0.85 的中分候选合计最多使用总预算的一半。
- deferred 候选最多保留 16 个。finish 最多等已有 review 0.8 秒，不在 finish 时启动新 review。
- 关闭整段音频重放。回放 echo 检查失败或回放无法确认时保留原文，不把未经音频确认的猜测写回转录。
- 推理只读取冻结实体库与流式 ASR 文本。默认公共控制器是 `system.entity_audio_repair.EntityAudioRepair`；私有旧类只保留给历史回归，不作为运行时回退。离线 gold/reference 不进入推理输入。

## 计时口径

配对总耗时从本地 WebSocket 连接到完整 final 返回，包含真实 ASR 和精修、ACK 等待，以及每发送 1 秒 PCM 后固定等待 0.08 秒的测试节奏；最多一个未确认帧。不包含文件读取、浏览器解码、前端上传逻辑和代理延迟，不是浏览器计时，也不是实时麦克风录音时长。报告中的 backend HTTP 调用数是请求数，不等于模型调用数；`reported_model_ms` 是 ASR payload 报告的累计模型阶段时间，可能包含缓冲和处理工作，不是 GPU 硬件计时。backend handler wall time 也可能不覆盖尚未完成的请求。版本 JSON 的 timing note 对这些口径有明确说明。

## 最终生产类复测与部署

最终私有 worker 使用的两个实体模块与正式部署文件 SHA256 完全一致，见 `final_verification.json`。状态回调恢复了现有 Web 协议，没有修改前端或 Web 功能代码。

| 最后一次配对，322.7 秒 Li 音频 | 旧策略 | 新生产控制器 |
|---|---:|---:|
| 本地完整 Web 总耗时 | 48.7368 秒 | 44.8348 秒 |
| 收尾耗时，包含 ASR finish 和精修 | 1.1439 秒 | 1.4257 秒 |
| 复核音频量 | 322.688 秒 | 28.000 秒 |
| 最终 raw 中包含“厉飞宇” | 否 | 是 |
| 最终 clean 中包含“厉飞宇” | 是 | 是 |
| raw 首次正确的上传音频位置 | 未出现 | 第 60 秒（墙钟 9.5417 秒） |

数据来自最终 `long_score.json` 与 `long_results.jsonl`。这一轮总耗时下降，但收尾并没有下降，不能声称所有收尾等待都已缩短。旧 clean 中的正确词不能算旧原音复核成功；新策略有实际音频确认及 raw 修订记录。三条 sparse 的最终目标命中仍为 3/3，clean CER 仍相同。

另用同一录音的 40–72 秒片段，以一个 32 秒 PCM 二进制包测试大包与 offline 缓冲路径。两臂 raw/clean 都正确，新策略只进行局部复核，没有整段补跑。短包总耗时：streaming 3.581→3.816 秒；offline 3.257→3.850 秒。这是小样本功能回归，不是总体速度结论，见 `bulk_smoke.json`。

正式 Web 已重启：8081 PID 2107784，8051 PID 2108194。启动 argv、门控模型与配置、ASR 地址及数字规则保持原样；门控运行计数重启后归零，校验仅比较配置。原 ASR PID 1455505 未重启，三端口健康检查通过，私有 worker 已停止，见 `web_rollout.json` 和 `final_verification.json`。

正式服务额外复测完整 322.7 秒录音，trace `562841d8e8ca44e28a0d68c21f5ed3fe`：上传到第 60 秒时 raw 已将“厉飞羽”改成“厉飞宇”，复核音频 28 秒，整段补跑 0 次。总耗时 41.0181 秒、收尾 1.4259 秒；本次没有同时跑旧策略，不能拿这个数字计算配对提速。见 `live_web_full_smoke.json`。正式服务 32 秒片段也通过：第 20 秒输入时发布正确 raw，见 `live_web_smoke.json`。

仅修改生产 `system/entity_audio_repair.py`，新增 `system/entity_streaming_live.py`；非实体 Web、ASR、前端、检索/匹配规则文件与正式 SQLite 的前后哈希一致。主要检查 146 项通过、81 个 subtests 通过，另有 9 项历史局部复核检查通过。历史旧类测试不代表新默认策略；新公共类的状态、取消、预算、批量拆分及 Web 集成另有覆盖。

使用时刷新页面，选择流式转录，开启“实体原音复核”并关闭“词库辅助识别”。结束时不新启动复核：最终才出现、分数不足、超过窗口、定位失败或来不及确认的实体保持原文。不能据本轮回归宣布已经完成全量 2506 条新策略评估，也不能保证尾部实体与所有领域都无漏改。

## 外部技术背景

Qwen 官方 README 说明其流式推理当前由 vLLM 后端支持，且流式模式不提供时间戳；官方源码为单条流维护累积文本和回滚 token 状态。[Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md)、[流式推理源码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)。

2026 年 5 月提交的 CTC word spotting 论文提出跨 chunk 保留 keyword path 与增量提交的 contextual biasing 思路。[论文 arXiv:2605.18222](https://arxiv.org/abs/2605.18222)。这是研究方向参考，不是当前接口可直接接入的实现：论文方法使用 CTC 概率，当前集成可见的是分块后的解码文本更新；要采用该方法，需增加能提供 CTC posterior 的后端接口或替换相应解码路径。这是根据论文输入需求与当前接口数据形态作出的推断。

## 数据来源

- V1：[`independent_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v1/independent_score.json)、[`known_dense_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v1/known_dense_score.json)、[`long_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v1/long_score.json)。
- V2：[`independent_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v2/independent_score.json)、[`known_dense_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v2/known_dense_score.json)、[`long_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v2/long_score.json)。
- V3 部署前状态：[`independent_score.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/versions/v3_before_status/independent_score.json)。
- 规则快照：[`frozen_parameters.json`](/home/aim0/桌面/tan/ASR/Codex-web/results/entity_streaming_live_20261014/frozen_parameters.json)。

Qwen3-ASR 官方 README 和当前源码于 2026-10-14 查阅；CTC word spotting 论文页面列出的当前版本为 v2（2026-05-19 修订）。本报告未复制外部项目代码。

本机保持原有 qwen-asr 0.0.6 / vLLM 后端；官方 main 文档用于核对接口，不在本次升级或替换 ASR。
