# CTC + 原音复核 paced 配对诊断

固定 8 个 short-finish/test ID，每条各跑 off 与 on；ID 按输入-only SHA256 排序选择。严格核对 16 条完整覆盖、split、配对 PCM SHA256、两臂实际 Qwen raw 主流文本一致，并逐包验证提交时间不早于 `audio_clock_wall + audio_seconds - 2 ms`。

| Arm | First nonempty text 中位数 | Finish tail 中位数 | Finish tail 最大值 | 原音复核 checks | CTC review skips | CTC committed | CTC errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 2.240s (8/8) | 0.412s | 0.661s | 8 | 0 | 0 | 0 |
| on | 2.238s (8/8) | 0.459s | 0.704s | 7 | 2 | 2 | 0 |

配对 on−off finish-tail 差中位数：0.023s。

## 八条样本的事后质量诊断

| Arm | Entity hits / mentions | Recall | CER |
|---|---:|---:|---:|
| off | 9 / 11 | 81.82% | 4.10% |
| on | 11 / 11 | 100.00% | 3.28% |

这些质量数字只描述这 8 条已选录音，不用于估计总体精度或推广到其他语料。该 8 对小样本也不是服务 p95；finish tail 是最后一块音频提交到最终文本返回的 wall time。逐样本、逐包核验与完整统计见 `paired_paced_summary.json`。
