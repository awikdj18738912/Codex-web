2026-10-04 项目功能复测报告

结论：当前 Qwen3-ASR + Web + Refiner 配置的主要功能检查通过。发现并修复旧版接口的非法参数处理及控制栏布局问题；真实音频中仍存在精修语义误改，因此本次结果不等于文本质量全面验收。

自动化测试：361 passed，726 subtests passed，1 条 jieba/pkg_resources 弃用提醒。原生 C++ 的 itn_safe_test 和 itn_zh_conservative_v2_test 编译运行成功；启动脚本 bash -n 通过。数字测试包含 163 条 v90 样例，验证输出、幂等性与变更位置映射。

当前配置：Qwen3-ASR-1.7B，Silero VAD，vad_finalize；AgenticASR-Refiner，tri_state 门控；zh-itn-strict-conservative-v90；最终整篇精修关闭。Web 8081 与 ASR 8766 最终健康检查通过，ASR 修复已加载到服务。

实测范围：

| 功能 | 验证方式和结果 |
| --- | --- |
| 在线转写 | Chromium 使用真实音频文件模拟麦克风输入，运行约 11 秒后点击停止；收到唯一 final，计时停止，控件恢复 |
| 离线整段 | 浏览器文件选择器上传完整《测试音频1.mp3》（40.848 秒），原始和精修文本与服务器 final 一致 |
| 离线流式 | 同一完整音频逐块上传，收到中间事件、精修更新和 final；启用词库辅助识别，确认服务应用术语上下文 |
| 术语库 | HTTP 与浏览器新增、编辑、别名搜索、类型与状态筛选、停用、启用、删除；重复创建 409，非法内容 422，不存在的删除 404 |
| 数字和内容保护 | 单元测试及真实音频验证日期、百分比、金额；原有数字成语“三番五次”保留，苹果改口为梨得到处理 |
| 精修统计 | 浏览器模型调用次数与服务器 session_refiner_call_count 一致，历史修改与质量回退可显示 |
| 精修日志 | HTTP 获取成功；浏览器显示、下一窗口、上一窗口与跳转正常 |
| 两种 ASR 协议 | current 和 legacy_v1 分别创建会话、传输真实音频、结束、取消；重复结束返回 400 |
| 异常输入 | 非法模式、非有限分块参数、非法词库开关被拒绝；非 float32 长度的 PCM 被拒绝；损坏文件显示错误并恢复控件 |
| 静音 | 三种 WebSocket 模式仅发送静音时正常结束，没有识别文本且模型调用为 0 |
| 页面 | 1280px 和 390px 视口；移动端无整页水平溢出，浏览器无 JavaScript 异常 |
| 数据工具 | SFT 导出 9 条训练、3 条验证并跳过 1 条无效记录；基础去重、分布统计、CER/WER/MER 同文为 0；批处理 validate-only 通过 |
| 服务复用 | start_services.sh 复用健康服务，不重复加载模型 |

本次修复：

- system/qwen_asr_stream_server.py：旧版 /v1/stream/start 对 NaN/Infinity 缺少有限性校验，导致底层状态初始化抛错并返回 500；现在返回 400，同时捕获浮点转整数及超大整数转换的 OverflowError。新增 3 项 HTTP 回归测试，合法参数继续正常创建会话。
- system/web/index.html：新增词库辅助控件后，控制栏空间不足导致标签逐字竖排；增加自动换行并限制控件宽度，已通过浏览器截图复测。

仍需处理的质量问题：

离线整段模式可稳定复现 ASR 片段“然后哈哈哈，我一无一实的告诉你，你觉得我这个人怎么样？”被精修为“然后我一无所有地告诉你觉得我这个人怎么样？”。这里出现了新的“一无所有”含义，目前保护规则未拦截。流式模式在该样本上触发 semantic_content_loss 回退并保留相应原文，说明两种模式的处理结果仍存在差异。原始 ASR 也有疑似成语识别错误，本次没有人工标注参考文本，未计算识别准确率，也没有为此新增未经审核的短语替换规则。

验证边界：本次麦克风测试使用 Chromium 模拟设备，不包含实际麦克风硬件及环境噪声验收；完整文件测试使用一条 40.848 秒音频，不包含五条音频的全量评测、多人高并发和长时间稳定性压测。Whisper、sherpa-onnx、完整训练数据生成、模型训练以及外部 Judge 没有端到端运行。

测试实体均已删除，HTTP CRUD 前后现有实体列表完全一致；正常转写测试向现有会话输出追加了测试记录。

证据文件：pytest.log、pytest.xml、integration.json、data_tools.json、sft_export.log、postprocess_validate.log、browser_*_events.json 与 browser_*.png。integration_before_fix.json 保留初次 500 的复现；browser_offline_events_before_fix.json 保留首次真实音频结果。
