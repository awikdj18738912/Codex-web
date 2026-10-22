# Refiner 生成提速记录

检查时间：2026-10-18T12:43:15.814257+08:00

## 已部署

- 保留双卡 DP=2；启用 FULL_DECODE_ONLY CUDA Graph，捕获规模 1、2、4，编译模式为 NONE。
- Graph 捕获完成，额外显存约 0.09 GiB；服务端点继续使用 8010。
- Oral 从固定批次 gather 改为持续并发 8：完成后立即补位。
- SQLite 持久记录尝试编号和完成结果；JSONL、旧检查点是可修复的镜像。
- 已有 3566 条记录与修改前备份逐条一致；目前完整 JSONL 3838 条，编号唯一，SQLite integrity_check=ok。
- 本次保持教师模型、提示词、采样温度和已有 1024-token Oral 上限。

## CUDA Graph 对照

固定16个输入，并发8；8次预热后计时；每请求固定256输出tokens（ignore_eos=true），每轮共4096 tokens。这些输出仅用于性能比较，不进入训练集。

| 配置 | 吞吐 tokens/s | 计时秒数 | API错误 |
|---|---:|---:|---:|
| Eager | 144.63 | 28.32 | 0 |
| Graph | 570.97 | 7.17 | 0 |
| Graph复测 | 575.84 | 7.11 | 0 |

本次固定负载 Graph/Eager 吞吐比约 3.95。这不直接等于全量数据端到端加速倍数。

## 真实生成观察

约50秒新增95条有效格式 Oral（约6840条/小时），另有12次失败尝试；两张卡均观察到约99%利用率。该窗口较短，长度分布和生成失败会影响长期速度。当前仍处于 Oral 阶段，不代表已经完成 Clean、ASR 或最终语义审核。

## 检查

20项通过：连续补位、旧数据迁移、写入中断恢复、取消后续跑、编号唯一、防止同场景双写、失败补量、现有误断句与量产状态相关检查。

## 追溯文件

- benchmark.py / requests.json：基准与冻结输入。
- eager.json / graph.json / graph_repeat.json：原始响应与计时。
- oral_before.jsonl / oral_checkpoint_before.json：正式数据备份。
- migration_check.json / live_integrity_check.json：迁移核对。
- summary.json：本次结果摘要。
- serve_teacher_eager.sh / serve_teacher_graph.sh：服务配置。
- teacher_graph.log：当前服务日志（数据目录中的 teacher_dp2.log 指向它）。
