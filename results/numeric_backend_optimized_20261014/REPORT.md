# 数字后端耗时对比

同一模型预热后交错运行，单窗口文本回放；不包含 ASR、模型加载、音频上传或 WebSocket 调度。

每类 12 句，每句 3 轮；所有模型条件复用同一个实例。独立学习门控和实体数据库关闭，隔离数字后端影响。

| 类别 | 模式 | 平均 ms | 中位 ms | P95 ms | 模型调用 | 重试 | 拒绝 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nonnumeric_control | zh_itn_web_chain | 174.399 | 115.622 | 463.177 | 21 | 3 | 3 |
| nonnumeric_control | zh_itn_only | 1.263 | 1.022 | 3.886 | 0 | 0 | 0 |
| nonnumeric_control | refiner_only | 349.045 | 359.431 | 451.803 | 36 | 0 | 0 |
| nonnumeric_control | model_web_chain | 171.699 | 111.336 | 456.488 | 21 | 3 | 3 |
| all | zh_itn_web_chain | 188.192 | 115.324 | 484.689 | 57 | 3 | 6 |
| all | zh_itn_only | 4.209 | 3.231 | 8.914 | 0 | 0 | 0 |
| all | refiner_only | 404.428 | 381.188 | 829.042 | 108 | 0 | 0 |
| all | model_web_chain | 564.426 | 509.274 | 1675.946 | 147 | 54 | 54 |
| numeric_only | refiner_only | 455.858 | 449.505 | 835.402 | 36 | 0 | 0 |
| numeric_only | model_web_chain | 734.684 | 738.439 | 1682.858 | 57 | 21 | 21 |
| numeric_only | zh_itn_web_chain | 6.776 | 7.564 | 15.660 | 0 | 0 | 0 |
| numeric_only | zh_itn_only | 6.173 | 6.926 | 14.990 | 0 | 0 | 0 |
| numeric_cleanup | zh_itn_web_chain | 383.401 | 372.408 | 526.486 | 36 | 0 | 3 |
| numeric_cleanup | zh_itn_only | 5.192 | 4.742 | 8.976 | 0 | 0 | 0 |
| numeric_cleanup | refiner_only | 408.382 | 348.005 | 836.186 | 36 | 0 | 0 |
| numeric_cleanup | model_web_chain | 786.894 | 755.666 | 1689.968 | 69 | 30 | 30 |

纯规则组不承担口语清理，与混合句/非数字精修组不构成同任务质量比较。model_web_chain 含 Python 数字兜底，refiner_only 才是直接纯模型。

输出与轨迹见 observations.jsonl；速度结果没有独立人工质量评分。GPU 上还有现有服务，交错多轮可减小顺序影响，不能代表所有线上负载。
