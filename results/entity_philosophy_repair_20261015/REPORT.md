# 哲学音频实体修复材料报告

对照只在同一次 `quality/on.json` 内比较 `primary_text` 与 `text`。
标注 reference 只校正实体拼写，不是全文人工 gold；本报告不计算 CER。
实体命中通过规范化后的 reference 与 hypothesis 全文序列对齐，把每个 reference_start/end 独立投射到对应文本跨度；其他文字仍是原始 Qwen 输出。

## 按提及指标

| 臂 / 字段 | core 31 | targeted hard 5 | uncertain 3（单列） |
|---|---:|---:|---:|
| on / `primary_text` | 26/31 命中；5 miss；0 未投射 | 0/5 命中；5 miss；0 未投射 | 0/3 命中；1 miss；2 未投射 |
| on / `text` | 26/31 命中；5 miss；0 未投射 | 0/5 命中；5 miss；0 未投射 | 2/3 命中；0 miss；1 未投射 |

同一 on run 的 core 转换：初始错误 5，修复 0，仍错 5；初始命中后丢失 0。无法投射：primary 0，text 0。


筛选基线 `raw_screen.json` final text：core 26/31，hard 0/5，uncertain 0/3。它用于说明错误富集筛选起点，不纳入 on 同次配对修复统计。


## 8 个错例及逐处前后

每格为该 mention 对应局部跨度，不是全文替换。uncertain 3 不计入 core denominator。

| mention | 类别 | reference 表面 | primary_text | text |
|---|---|---|---|---|
| mention_009 | core_spelling | 毛宗岗 | 毛中刚 | 毛中刚 |
| mention_010 | core_spelling | 毛宗岗 | 毛中刚 | 毛中刚 |
| mention_024 | order_or_spoken_variant_uncertain | 色拉叙马霍斯 | — | 色拉叙马霍斯 |
| mention_026 | order_or_spoken_variant_uncertain | 色拉叙马霍斯 | — | — |
| mention_029 | core_spelling | 色拉叙马霍斯 | 适当去马斯洛斯 | 适当去马斯洛斯 |
| mention_031 | core_spelling | 色拉叙马霍斯 | 适当和马修斯 | 适当和马修斯 |
| mention_033 | core_spelling | 苏格拉底 | 苏格拉你 | 苏格拉你 |
| mention_034 | order_or_spoken_variant_uncertain | 色拉叙马霍斯 | 色拉霍马去世 | 色拉叙马霍斯 |

## 26 个正确对照

| mention | reference 表面 | primary_text | text | 是否误改 |
|---|---|---|---|---|
| mention_001 | 柏拉图 | 柏拉图 | 柏拉图 | 否 |
| mention_002 | 理想国 | 理想国 | 理想国 | 否 |
| mention_003 | 色拉叙马霍斯 | 色拉叙马霍斯 | 色拉叙马霍斯 | 否 |
| mention_004 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |
| mention_005 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |
| mention_006 | 商鞅 | 商鞅 | 商鞅 | 否 |
| mention_007 | 商鞅 | 商鞅 | 商鞅 | 否 |
| mention_008 | 罗翔 | 罗翔 | 罗翔 | 否 |
| mention_011 | 孟子 | 孟子 | 孟子 | 否 |
| mention_012 | 孟子 | 孟子 | 孟子 | 否 |
| mention_013 | 荀子 | 荀子 | 荀子 | 否 |
| mention_014 | 荀子 | 荀子 | 荀子 | 否 |
| mention_015 | 荀子 | 荀子 | 荀子 | 否 |
| mention_016 | 荀子 | 荀子 | 荀子 | 否 |
| mention_017 | 韩非子 | 韩非子 | 韩非子 | 否 |
| mention_018 | 韩非 | 韩非 | 韩非 | 否 |
| mention_019 | 孔子 | 孔子 | 孔子 | 否 |
| mention_020 | 韩非 | 韩非 | 韩非 | 否 |
| mention_021 | 韩非子 | 韩非子 | 韩非子 | 否 |
| mention_022 | 康德 | 康德 | 康德 | 否 |
| mention_023 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |
| mention_025 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |
| mention_027 | 色拉叙马霍斯 | 色拉叙马霍斯 | 色拉叙马霍斯 | 否 |
| mention_028 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |
| mention_030 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |
| mention_032 | 苏格拉底 | 苏格拉底 | 苏格拉底 | 否 |

## 快照、错误与 CTC fallback

### on
incremental.errors=0；CTC errors=0；fallback_reason=None；acoustic_events=1876；compute_seconds=106.49109710892662；submitted_audio_seconds=600.0。
完整 entity_audio_repair snapshot：
```json
{
  "applied": true,
  "candidate_count": 0,
  "change_count": 2,
  "changes": [
    {
      "boundary_issue": null,
      "canonical": "色拉叙马霍斯",
      "character_score": 0.666667,
      "end": 1771,
      "entity_id": 41,
      "entity_type": "PERSON",
      "evidence_source": "audio_replay",
      "evidence_surface": "色拉叙马霍斯",
      "length_score": 1.0,
      "match_type": "AUDIO_REPLAY_CONFIRMED",
      "observed": "色拉霍马叙斯",
      "phonetic_score": 0.761111,
      "retrieval_score": 0.775556,
      "source_left": "么你的相信苏格拉底跟",
      "source_right": "他们的一个著名的讨论那",
      "start": 1765
    },
    {
      "boundary_issue": null,
      "canonical": "色拉叙马霍斯",
      "character_score": 0.5,
      "end": 2124,
      "entity_id": 41,
      "entity_type": "PERSON",
      "evidence_source": "audio_replay",
      "evidence_surface": "色拉叙马霍斯",
      "length_score": 1.0,
      "match_type": "AUDIO_REPLAY_CONFIRMED",
      "observed": "色拉霍马去世",
      "phonetic_score": 0.688889,
      "retrieval_score": 0.701111,
      "source_left": "学所以苏格拉你就提醒",
      "source_right": "但是我告诉你什么叫好",
      "start": 2118
    }
  ],
  "checks": 12,
  "elapsed_seconds": 12.391,
  "enabled": true,
  "entity_count": 53,
  "incremental": {
    "admitted": 15,
    "ambiguous_rejected": 0,
    "audio_cache_seconds": 0.0,
    "budget_rejected": 1333,
    "cache_reused": 0,
    "candidate_margin": 0.04,
    "candidate_score": 0.63,
    "candidates": 932,
    "completed_windows": 12,
    "ctc": {
      "acoustic_events": 1876,
      "committed": 0,
      "compute_seconds": 106.49109710892662,
      "errors": 0,
      "finish_timed_out": 0,
      "mode": "on",
      "proposals": 0,
      "shadow_proposals": 0,
      "streaming_review_skips": 0,
      "submitted_audio_seconds": 600.0
    },
    "deferred_retried": 3930,
    "detection_seconds": 5.9595435610972345,
    "detections": 589,
    "echo_rejected": 0,
    "errors": 0,
    "finish_starts_review": true,
    "finish_wait_seconds": 0.8,
    "full_audio_fallbacks": 0,
    "late_candidates_kept": 4,
    "long_recording_whole_audio_replay_enabled": false,
    "low_score_rejected": 627,
    "medium_budget_rejected": 2724,
    "medium_replay_seconds": 141.0,
    "oversized_rejected": 9,
    "pending": false,
    "pending_candidates": 0,
    "queue_rejected": 0,
    "queued_windows": 0,
    "review_wall_seconds": 12.391048484481871,
    "rule": "bounded_incremental_entity_v3_short_finish",
    "short_finish_candidates": 0,
    "short_finish_changes": 0,
    "short_finish_checks": 0,
    "short_finish_errors": 0,
    "short_finish_max_seconds": 16.0,
    "short_finish_replay_seconds": 0.0,
    "short_finish_timed_out": 0,
    "short_finish_wait_seconds": 4.0,
    "short_finish_wall_seconds": 0.0,
    "source_scope": "stable_segments_and_stable_partial_prefix",
    "stream_revisions": 2,
    "unmapped": 42,
    "whole_audio_replay_enabled": true,
    "whole_audio_replay_max_seconds": 16.0,
    "window_limit_seconds": 32.0
  },
  "local_replay_enabled": true,
  "local_replay_reason": null,
  "local_replay_rule": "segment_windows_v1",
  "max_candidate_score": 0.0,
  "minimum_replay_score": 0.63,
  "phase": "complete",
  "prompt_terms": [
    "法家"
  ],
  "reason": null,
  "replay_audio_seconds": 159.0,
  "replay_context_applied": true,
  "replay_decode_seconds": 10.0,
  "replay_mode": "streaming_local",
  "replay_windows": [
    {
      "end_seconds": 14.0,
      "start_seconds": 0.0
    },
    {
      "end_seconds": 18.0,
      "start_seconds": 0.0
    },
    {
      "end_seconds": 133.0,
      "start_seconds": 120.0
    },
    {
      "end_seconds": 177.0,
      "start_seconds": 164.0
    },
    {
      "end_seconds": 207.0,
      "start_seconds": 197.0
    },
    {
      "end_seconds": 249.0,
      "start_seconds": 238.0
    },
    {
      "end_seconds": 243.0,
      "start_seconds": 225.0
    },
    {
      "end_seconds": 385.0,
      "start_seconds": 374.0
    },
    {
      "end_seconds": 409.0,
      "start_seconds": 399.0
    },
    {
      "end_seconds": 474.0,
      "start_seconds": 461.0
    },
    {
      "end_seconds": 511.0,
      "start_seconds": 503.0
    },
    {
      "end_seconds": 528.0,
      "start_seconds": 508.0
    }
  ],
  "revision": 2,
  "short_expansion_rule": "spelling_or_complete_edge_sound_v1",
  "source_audio_seconds": 600.0,
  "streaming_cache_used": false,
  "streaming_replay": null
}
```
### raw_screen
incremental.errors=None；CTC errors=None；fallback_reason=None；acoustic_events=None；compute_seconds=None；submitted_audio_seconds=None。
完整 entity_audio_repair snapshot：
```json
{}
```

## 解释范围

- 拼写来源核准的是实体规范写法；没有独立人工听审确认每个实际发音。
- source 身份是推断，来源原件未核实；异序或疑似口误的 3 处仍待定，并从 core denominator 排除。
- `reference_text` 是逐处实体修正代理，其他文字保留原始转写；禁止据此计算全文 CER。
- 本段材料按初始错误富集，是开发诊断，不是盲测或总体准确率估计。
- runner 的加速只用于质量跑数；elapsed_seconds 不代表真实用户延迟。
