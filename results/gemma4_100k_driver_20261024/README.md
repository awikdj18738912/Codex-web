# 当前数据状态：13类候选，A04/B03/C03/C04停用（2026-10-26）

当前候选共38,233条，13类各2,941条；来源为CS2W 391条、FCGEC 4,159条、Gemma4 33,683条。不补回5万。A04、B03、C03、C04不生成、不训练、不评估、不补量；原候选和样本保留作历史追溯。人工复核队列21条，仍待人工复核；训练尚未启动。当前状态入口见 `results/current_refiner_dataset.json`，候选文件为 `/home/aim0/data/datasets/ASR/refiner_balanced_13classes_20261026_no_A04_B03_C03_C04/candidates.pending_review.jsonl`。

以下GPU部署和旧数据范围均为历史记录，不代表当前生成任务。

# 历史部署：恢复双卡（2026-10-26）

按用户最新要求，首轮5万任务恢复GPU0/1 TP2、总32路。当前会话 `gemma4_generate_first50k_tp2_32_resume`，日志 `generation_first50k_tp2_32_resume.log` / `server_first50k_tp2_32_resume.log`。输出目录不变，保留全部已入库候选，未启动训练。以下是历史记录。

# 历史部署：GPU1暂停（2026-10-26）

GPU1已按用户要求释放，当前首轮5万任务改为GPU0单卡16路。服务脚本 `start_teacher_gpu0_16.sh`，生成脚本 `run_background_first50k_gpu0_16.sh`，日志 `server_first50k_gpu0_16.log` / `generation_first50k_gpu0_16.log`。输出目录不变，切换时已保存35,022条候选。下方双卡部署为历史记录。

# 历史任务：首轮约5万条（2026-10-26）

用户将首轮实验目标调整为17类各2,941条，共49,997条。旧生成任务正常排空后保留46,184条候选；冻结复用29,196条，补齐20,801条缺口。已筛选通过且原生标签明确的公开数据优先复用，超额样本保留旧账本。全部仍待人工复核，未启动微调。

- 当时输出：`/home/aim0/data/datasets/ASR/refiner_balanced_50k_20261026`
- 冻结输入及统计：`/home/aim0/data/datasets/ASR/refiner_balanced_50k_inputs_20261026`
- 启动：`run_background_first50k_tp2_32.sh`
- 日志：`generation_first50k_tp2_32.log`
- tmux：`gemma4_generate_first50k_tp2_32`
- 推理服务保持8014端口、GPU0/1 TP2，总32路。
- 当时缺口类别：A01/A02/A03/A04/C01/C02/C03/C04/D01/E05，达到每类上限即跳过。

以下为历史任务记录。

# 每类5882的均衡候选筛选：当前仅GPU0

## 最新：GPU1重新投入，双卡合计32路

2026-10-25 23:36（北京时间）用户要求GPU1重新投入。已完成单卡在途批次，备份42302条已保留候选，并改为GPU0/GPU1共同加载一份TP2模型，客户端合计32路；每卡KV缓存10 GiB、服务max-num-seqs32、上下文3072、预填充4096，预读加载。

- 服务脚本`start_teacher_balanced_tp2_32.sh`；tmux `gemma4_teacher_balanced_tp2_32`；日志`server_balanced_tp2_32.log`。
- 八类监督脚本`run_background_balanced_eight_tp2_32.sh`；tmux `gemma4_generate_balanced_eight_tp2_32`；日志`generation_balanced_eight_tp2_32.log`。
- 账本继续使用`/home/aim0/data/datasets/ASR/refiner_balanced_eight_gaps_20261025`。A02/A03/A04/C01/D01/E03/E05/E06和每类5882上限保持不变。
- 原生成器状态显示GPU IDs固定为[0]，此次新增显式`--gpu-ids 0,1`，只修正硬件运行状态；生成规则、审核提示、配额与种子输入保持原样。已保存旧源码／config及完整账本备份，并在输出目录`config_revision_history.jsonl`记录新旧指纹；旧候选payload原指纹保留。
- 切换证据见`balanced_eight_gpu1_reenabled_transition.json`。当前硬件分配以`GPU_POLICY.json`和实际runtime为准；此前GPU1暂停记录不再代表当前用户决定。

以下单卡配置和运行记录保留用于追溯，当前以本节为准。


## 当前任务：八类连续补量，GPU0单卡16路

2026-10-25 23:08（北京时间）按用户“该生的继续生成”启动连续任务。当前只生成A02/A03/A04/C01/D01/E03/E05/E06，每类最终保留上限5882；一类满额自动轮转其他未满类别。旧的E03=4000、E06=3000等配额不再限制新任务。

- 新账本与实时状态：`/home/aim0/data/datasets/ASR/refiner_balanced_eight_gaps_20261025`，工作区入口`results/refiner_balanced_eight_gaps_20261025/运行状态.md`。
- 启动脚本：`run_background_balanced_eight_gpu0_16.sh`；tmux `gemma4_generate_balanced_eight_gpu0_16`；日志`generation_balanced_eight_gpu0_16.log`。
- 生成入口：`experiments/scripts/generate_balanced_refiner_gaps.py`。新配额和来源/源码指纹写入独立config及code_snapshot；原始旧账本和原候选ID保持可追溯。
- 种子来自严格标签筛选已选文件，保留原生标签及质量证据；D02完成批次以冻结文件补入并重新去重至5882，超量/重复仍保留旧账本。种子导入情况以`seed_import_summary.json`为准。
- 八类原筛选缺口合计38919条；导入与新产出仍待人工复核，不表示人工验收训练量。A01/C02/C03/C04优先继续筛选明确标签公开素材；B01/B02/B03/E07及D02不启动新生成。
- GPU0服务仍是TP1/max-num-seqs16；GPU1继续不使用。没有512条阶段限制，直到八类达到目标、用户停止或发生需处理异常。
- 规则检查、单独老师审核、同源冲突隔离、精确与词面近重复、官方验证测试重合排除均保留；占位符ID在去重键中归一，原文不改。公开没有明确对应小类原生标签的素材不导入。
- 导入断点通过幂等ID恢复，新序号从1000000以上开始，避免与旧生成派生组相撞。TERM停止派发并完成在途批次提交。

以下为历史运行记录，当前状态与路径以上述新任务为准。


## 最新运行配置：GPU0单卡总16路

2026-10-25 20:21（北京时间）按用户要求，把GPU0服务max-num-seqs和客户端worker／请求上限同步从2改为16。GPU1继续不用于本任务。模型和KV精度沿用原配置，KV缓存3 GiB，上下文3072；请求会按可用缓存调度。

- 服务：`start_teacher_gpu0_16.sh`；tmux `gemma4_teacher_gpu0_16`；日志`server_gpu0_16.log`。
- 生成：`run_background_balanced_d02_gpu0_16.sh`；tmux `gemma4_generate_balanced_d02_gpu0_16`；日志`generation_balanced_d02_gpu0_16.log`。
- 保持D02绝对目标5882；每次恢复均从账本计算剩余量，避免重复追加旧的1681条阶段额度。在途批次会提交，最终均衡筛选仍以每类5882为上限。
- 切换时已完成在途批次，保存40387条机器通过候选（D02=5166），剩余D02=716。检查点与配置摘要见`gpu0_concurrency16_transition.json`。
- 当前GPU分配和并发以`GPU_POLICY.json`及实际runtime为准，以下旧2路／双卡记录仅供追溯。


> 最新公开准入规则：必须有明确对应项目小类的原生标签，否则只保留未完成素材。旧45915条统计已被取代，CS2W推断口吃和ChineseHP推断错字不再计入。当前结果见`results/refiner_balanced_17x5882_strict_labels_20261025/筛选统计.md`，规则见本目录`PUBLIC_DATA_POLICY.json`。当前只GPU0补D02，不改变GPU1停用安排。


## 最新：均衡筛选与按缺口补量

用户已要求每类5882条。当前第一轮筛选暂选45915条、备用19151条、缺口54079条；公开待复核84694条不计入暂选，详见`results/refiner_balanced_17x5882_20261025/筛选统计.md`和`balanced_source_plan.json`。全体候选仍待人工复核，未开训。

旧六类40000上限不再作为最终训练配比。B01/B02/B03/C01/E07已足量并停止补量。当前GPU0仅有界续补D02差额1681条，脚本`run_background_balanced_d02_gpu0.sh`，tmux `gemma4_generate_balanced_d02_gpu0`，日志`generation_balanced_d02_gpu0.log`；其余类别先处理公开待复核素材。新生成不会自动进入本次冻结的暂选文件，下一版需重新合并去重。

GPU1仍保持本任务停用。以下为历史硬件、配额与运行记录，当前筛选目标以本节为准。

更新：2026-10-25 03:45（北京时间）。用户要求继续生产，已从3798条accepted接续，取消首轮512条停止阈值。用户已调整范围：优先复用已有数据，仅合成公开来源尚未覆盖的内容。原17类全量合成已停止，3213条机器通过候选完整备份保留；不再执行全部10万条合成。

## 硬件记录：GPU1按用户要求暂停

2026-10-25 14:02（北京时间）已完成在途批次并备份39329条accepted、33041条rejected和3条duplicate；随后卸载TP2模型，释放GPU1。当前GPU1保持空闲，后续仅在用户重新要求使用时恢复。

- GPU0单卡TP=1，使用此前验证通过的配置：上下文3072、max-num-seqs2、KV缓存3 GiB。
- 生成客户端同步降到2个worker／2个并发请求；保留六类范围和原配额，不启动训练。
- 服务tmux：`gemma4_teacher_gpu0`；脚本`start_teacher.sh`；日志`server_gpu0_resume_20261025.log`。
- 生成tmux：`gemma4_generate_gaps_gpu0`；监督脚本`run_background_gaps_gpu0.sh`；日志`generation_gaps_gpu0.log`。
- 8014接口启动后自动从原账本继续。当前GPU分配决定见`GPU_POLICY.json`；切换及备份记录见`gpu1_pause_transition.json`。
- GPU1释放时实测15 MiB占用、0%利用率、无计算进程。单卡服务已通过health HTTP200并成功完成新的生成请求，runtime已确认2个worker；验证记录见`gpu1_pause_verified.json`。

## 此前配置：双卡总32路（已停止）

- 两张GPU共同加载一份Gemma4 teacher，TP=2；总32个客户端worker，服务max-num-seqs=32，总并发上限32。
- API仅保留`http://127.0.0.1:8014/v1`，模型名`gemma4-31b-teacher`；旧8015独立副本已停止。
- 每卡模型约9.59 GiB，KV预算10 GiB；总KV容量23718 token。32路长请求会由服务按可用缓存调度，不能据此承诺32倍吞吐。
- 服务tmux：`gemma4_teacher_tp2_32`，启动脚本`start_teacher_tp2_32.sh`，日志`server_tp2_32.log`。
- 定向补缺tmux：`gemma4_generate_gaps_32`，监督脚本`run_background_tp2_32.sh`，日志`generation_gaps_tp2_32.log`。
- 当前只选B01/B02/B03/D02/E03/E06六类；A/C等已有来源覆盖类别暂停新增。六类旧配额总上限40000，不是必须生成量；最终数量由公开数据适配和缺口统计决定。
- 首轮512条阈值试跑已自然完成（新增585条通过候选）。用户要求继续生产后，当前脚本改为`--stop-after 0`，持续补充选定六类直至各类上限或受控停止；原试跑脚本保留为`run_background_tp2_32_trial512.sh`。六类累计40000是历史保护上限，不代表最终必须合成数量，公开数据覆盖后可受控缩减。
- 原生成／审核代码与原始候选config保持不变；新增范围控制代码`experiments/scripts/refiner_scoped_runner.py`和传输代码分别存入`runtime_snapshots`。当前范围和并发由输出目录`runtime_current.json`记录。
- 新的范围控制器收到TERM/INT后停止派发，并等待在途批次提交；不要再同时启动旧单卡／旧双副本监督脚本。

## 公开来源与状态

CS2W train 5789条配对、FCGEC过滤版train 35354条已获取并固定revision。其中FCGEC正确训练句16224条已导出恒等候选，错误句编辑还原仍待完成。训练素材与验证／测试隔离，以上均是未验收候选，不等于可用训练量。

来源目录：`/home/aim0/data/datasets/ASR/public_refiner_sources_20261025`；工作区入口`results/public_refiner_sources_20261025`，详见manifest。保留已有参考，不用大模型批量重写现成正确答案。

生成账本继续在`/home/aim0/data/datasets/ASR/gemma4_selfgen_all17_100k_v2_20261024`；工作区`results/gemma4_selfgen_all17_100k_v2_20261024/运行状态.md`自动更新。目录名保留用于兼容历史记录，当前执行范围已改变。总混合候选规模暂沿用约10万条规划，公开可用量待筛选后统计；未启动训练。

以下为历史双副本说明，服务与监督脚本均已由上述当前配置替代。

更新：2026-10-24 23:43（北京时间）。GPU0/GPU1两套独立服务健康检查均HTTP 200，生成调度器已切换为双卡。GPU0此前中文精修请求已返回预期结果；输出含Markdown围栏，去除后JSON解析通过，验证记录见`gpu0_smoke_response.json`。此目录保存启动脚本与日志，最新状态以健康检查和数据状态文件为准。

切换后已确认两端口均成功返回生成／审核响应，候选继续入库；原779条accepted全部保留，原配置文件未改变。验证快照见`multigpu_verification.json`。新增传输测试3项通过，相关测试共22项通过。

## 历史：独立单卡服务

- 模型：本地Google Gemma4 31B QAT W4A16 compressed-tensors。
- GPU0/GPU1各加载一份相同模型，各自tensor parallel=1，纯文本，BF16激活。
- GPU0 API：`http://127.0.0.1:8014/v1`；GPU1 API：`http://127.0.0.1:8015/v1`；两者模型名均为`gemma4-31b-teacher`。
- tmux：`gemma4_teacher_gpu0`、`gemma4_teacher_gpu1`。
- GPU0启动脚本：`start_teacher.sh`；日志：`server_gpu0.log`。
- GPU1启动脚本：`start_teacher_gpu1.sh`；日志：`server_gpu1.log`。
- 上下文3072，最多2条序列，单批预填充1024 token，KV缓存固定3 GiB。
- `gpu-memory-utilization`不会覆盖显式KV缓存预算。日志报告KV总容量3555 token；多请求可能排队或抢占，不能理解为两条3072 token请求均可同时完整驻留。
- 本次引擎加载记录权重18.7 GiB；加载峰值接近整卡容量。

健康检查：

```bash
curl --fail http://127.0.0.1:8014/health
curl --fail http://127.0.0.1:8015/health
```

## 历史：17类生成

- tmux：`gemma4_generate_100k`。
- 当前监督脚本：`run_background_multigpu.sh`；日志：`generation_multigpu.log`。
- 原单卡监督脚本`run_background.sh`与日志`generation_gpu0.log`保留用于追溯，不应与当前任务同时启动。
- 当前输出：`/home/aim0/data/datasets/ASR/gemma4_selfgen_all17_100k_v2_20261024`。
- 工作区入口：`results/gemma4_selfgen_all17_100k_v2_20261024`。
- 阅读`运行状态.md`或`progress.json`了解当前状态与各类数量。
- 4个worker，每个服务最多同时2个请求，每批生成4条；每条进行规则检查、模型审核及去重，错误和原始回答写入同一SQLite账本。
- 入口`experiments/scripts/generate_refiner_multigpu.py`只替换当前进程的请求传输，不修改原生成器、分类规则、配额或审核逻辑。原配置中的`api`保留为逻辑地址，实际双端口配置记录在输出目录`runtime_current.json`和`runtime_history.jsonl`；传输源码单独保存到`runtime_snapshots`，成功的原始响应附有`_local_transport.endpoint`。
- 仍由一个Runner统一分配序号、提交候选和执行去重，原账本的配置校验与文件锁继续生效。
- 切换前有779条accepted、329条rejected；一致性备份和切换记录见输出目录`backups/before_multigpu_20261024T154332Z`、`multigpu_switch.json`。切换时尚未提交的工作不计入accepted，原始调用记录保留。
- 旧版阶段结果保留在`gemma4_selfgen_all17_100k_20261024`，未并入v2配额。
- 目标是100000条机器通过、待人工审核候选。此次任务不启动训练。

后台会先等待两套teacher健康检查通过，并确认返回相同模型名。遇到临时审核传输失败时保留待审记录并有限重试；其他异常停止并保留错误，不把失败误报成完成。tmux使进程脱离当前终端持续运行，不提供机器重启后的自动恢复或无限自动重启。

查看运行终端：

```bash
tmux attach -t gemma4_generate_100k
```

按`Ctrl+B`再按`D`退出查看，后台继续运行。需要停止任务时应向生成脚本发送TERM，避免直接中断正在写入的账本。

参考：[vLLM官方Gemma4示例](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md)、[vLLM启动参数](https://docs.vllm.ai/en/latest/configuration/engine_args/)。具体可运行性以本机验证为准。

本次续跑备份和起点见输出目录`continuation_history.jsonl`及本目录`continue_gaps32_baseline.json`；账本备份在输出目录`backups/continue_gaps32_20261024T194530Z`。

严格标签重筛结果：暂选40021条，公开9214、生成30807；不含无原生小类标签的ChineseHP或按R-type推断的A02/A03/A04。C01不再视为满额。详情及缺口见新版summary.json与balanced_source_plan.json，旧版45915仅作历史。

16路首次冷启动期间CPU匿名内存换出导致长时间初始化；已清理本任务旧API残留，并顺序预热纯文本权重，当前单卡16启动脚本使用官方`--safetensors-load-strategy prefetch`。旧冷启动日志保留为`server_gpu0_16_coldstart.log`，恢复记录见`gpu0_16_coldstart_recovery.json`。此项只调整加载方式，未改变模型或数据标签规则。

GPU0单卡16已完成运行验收：health HTTP200、实际runtime workers16／per-server-limit16、新POST成功，原40387条机器通过候选完整保留。缓存占用较高，已观察到队列和抢占，不能承诺16条满长请求始终同时驻留；实际指标见gpu0_concurrency16_verified.json。

八类续跑已确认开始产出：新序号超过1000000，A03/A04/C01/E03等类别已有新accepted；公开导入9214条均保留明确原生标签，GPU1无计算进程。运行验收快照见balanced_eight_restart_verified.json；实时数量以新目录progress.json为准。

双卡续跑已通过实际验收：health HTTP200、runtime active_gpu_ids=[0,1]/workers32、新模型请求200、切换前42302条候选完整保留，八类配额／审核规则和种子不变。证据见balanced_eight_gpu1_reenabled_verified.json；实时数量见八类新账本progress.json。
