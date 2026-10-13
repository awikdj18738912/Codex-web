# 学习窗口门控开关

实现日期：2026-10-12。启动默认值为关闭。

随后按用户要求完成试运行部署：ASR 8766、Web 8081 和 Web 8051 均已重启，两个 Web 实例的实际档位为 `on`，模型 `ready=true`、阈值 0.8。原 GPU、zh-itn 数字后端及实体配置保留。两个实例均通过一段已消费训练音频的真实 WebSocket/ASR 检查，各记录一次窗口否决，无评分错误。这是服务功能检查，不是新的质量评测。

之后曾按用户要求关闭两层门控；最新一次又按要求启用学习门控。当前状态：Web 8081 和 8051 都已重启，`refinement_gate_mode=tri_state`、`learned_window_gate.mode=on`、`ready=true`、阈值 0.8。ASR 8766 保持健康并继续运行，GPU、实体修复和 zh-itn 配置保留。最新部署目录和 PID 记录在 `.runtime/services/current-gate-deployment.json`。

当前可打开 [8081 页面](http://127.0.0.1:8081) 或 [8051 页面](http://127.0.0.1:8051)。最新启动日志、完整健康状态和原参数位于 `.runtime/services/current-gate-deployment.json` 指向的部署目录；历史开关记录仍保留。

## 三个档位

| 环境变量 `LEARNED_WINDOW_GATE_MODE` | 行为 |
|---|---|
| `off` | 默认值。不加载学习模型，按原生门控处理 |
| `shadow` | 原生窗口准备精修时评分，记录是否建议跳过，仍执行原策略 |
| `on` | 原生窗口准备精修时评分；`skip_allowed >= 0.8` 才跳过这个窗口 |

学习门控只作用于窗口精修，原生 KEEP/DEFER 不会升级，局部重复审核不受该开关控制。开启 shadow/on 要求原生模式为 `tri_state`。

模型目录默认：`results/gate_terminal_gain_round4_20261012/linear_training`。加载选中的 `terminal_quality/model.pkl` 及 `selection.json` 中的冻结阈值，校验模型 SHA256 和分类标签。只使用项目自己训练的可信本地文件。

模型缺失、哈希不匹配、加载失败、评分异常或分数无效时，保留原生精修动作；健康接口和窗口审计中会记录失败原因。

## 启动配置

从项目根目录执行，原生门控默认为 `tri_state`：

```bash
# 关闭学习门控
LEARNED_WINDOW_GATE_MODE=off bash scripts/start_web.sh

# 先观察建议，不改变转录路由
LEARNED_WINDOW_GATE_MODE=shadow bash scripts/start_web.sh

# 实际允许跳过部分窗口精修
LEARNED_WINDOW_GATE_MODE=on bash scripts/start_web.sh
```

这是启动时配置，不能热切换。已有服务档位不同，脚本会提示并退出，不会自动重启，也不会声称开关已生效。需要在对应 Web 服务受控重启时设置档位；同时保留原来的 GPU、端口、数字后端等配置。例如当前 8081 服务使用 zh-itn，重启时应继续传 `ZH_ITN_ENABLED=1`。

自定义本地模型目录：

```bash
LEARNED_WINDOW_GATE_MODE=shadow \
LEARNED_WINDOW_GATE_MODEL_DIR=/绝对路径/linear_training \
bash scripts/start_web.sh
```

直接启动 Python 服务时对应参数为：

```text
--refinement-gate-mode tri_state
--learned-window-gate-mode off|shadow|on
--learned-window-gate-model-dir /绝对路径/linear_training
```

## 确认实际状态

```bash
curl -s http://127.0.0.1:8081/health | python -m json.tool
```

查看 `learned_window_gate`：

- `mode`：配置档位；
- `ready`：是否成功加载模型；off 时为 false，属于正常状态；
- `threshold`、`model_sha256`：实际加载的阈值及模型；
- `load_error`：加载失败原因；
- `scored_windows` / `would_skip_windows` / `vetoed_windows`：当前进程累计评分、建议跳过和实际跳过数；shadow 的实际跳过数应为 0；
- `scoring_errors`：评分失败次数。

只有 `mode=on` 且 `ready=true` 才具备学习筛选能力；没有原生窗口 REFINE 机会时不会评分。计数是进程级窗口决策次数，不是独立录音数或节省的全部 Refiner 调用数，重启后清零。

窗口的 `refinement_gate_decisions` 增加 `learned_window_gate` 审计记录，包含实际动作、建议动作、skip 分数、阈值、是否应用、原因及评分时间。shadow 保留原门控动作；on 的实际否决理由为 `learned_window_gate_veto`。

旧服务还没有这些字段时，启动脚本按学习门控 off 处理。磁盘代码改动不会让旧进程自动拥有新开关。

2026-10-13 起，磁盘代码统一使用 `fixed_k3_v1` 分窗策略：原生 off/conservative/tri_state 共用 K=3 分组、待处理尾部和源文本缓存。流式阶段不重跑已拒绝的相同窗口，终态是否重跑只由 `FINAL_REFINEMENT_MODE` 决定；不足 K 的尾部显示 DEFER，不计为门控跳过。`/health`、ready、final 与保存的 JSONL 都记录 `window_pipeline`。

启动脚本现在还会核对原生门控、终态重试模式、完整窗口策略和 zh-itn 开关。缺少新窗口策略字段的旧进程会拒绝复用，需要受控重启。本次代码验证没有重启现有 8081/8051 服务；它们仍是此前的学习门控 on 部署。

建议先使用 shadow，核对建议跳过的窗口，再决定是否使用 on。近期实验收益和限制见 [门控实现与近期改进](GATE_IMPLEMENTATION_AND_RECENT_CHANGES_20261012.md)。

## 本次验证

开关实现阶段有 124 项相关测试通过，覆盖默认关闭、shadow 不改路由、on 窗口否决、局部审核保留、缺失/损坏模型及无效分数回退、启动脚本拒绝复用不同档位。另用真实冻结模型在 CPU 上加载并核对评分与实验适配器一致，阈值仍为 0.8；Shell 语法和文档链接检查通过。之后才按用户要求重启并完成前述服务检查。
