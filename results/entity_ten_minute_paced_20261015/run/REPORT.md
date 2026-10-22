# Long paced entity profile

This report compares wall-clock behavior for the frozen manifest. It makes no accuracy claim.

- Manifest rows: 1
- Requested rounds: 1
- Completed arms: 2
- Complete paired sample-rounds: 1
- Qwen health captured before run: `{"api_styles": ["current", "legacy_v1"], "asr_send_chunk_seconds": 0.5, "confidence_calibrated": false, "confidence_logprobs": 5, "context_supported": true, "max_context_chars": 2048, "mode": "streaming", "ok": true, "segmentation_mode": "vad_finalize", "silence_rms_threshold": 0.002, "vad": "silero", "vad_min_silence": 0.7, "vad_min_speech": 0.25, "vad_model": "/home/aim0/桌面/tan/ASR/Codex-web/models/silero_vad.onnx", "vad_preroll": 0.7, "vad_tail_pad": 1.0, "vad_threshold": 0.5}`

## Paired deltas

Positive values mean entity took longer than raw. These are per-item observations, not accuracy results.

| ID | Round | Total wall delta (s) | First text delta (s) | Finish tail delta (s) |
|---|---:|---:|---:|---:|
| water_natural_600s | 0 | 4.793 | -0.032 | 0.311 |

## Limitations

- Timing describes this host, this Qwen service, and these PCM inputs.
- The run does not estimate transcription accuracy or generalize beyond this manifest.
- Target-term timestamps are exact substring observations in returned text, not phonetic onset times.
- Each arm is sequential and can be affected by service load and order effects.

## Source hashes

```json
{
  "experiments/scripts/profile_entity_long_paced.py": "5fcab9cec2f5f5f6f9eef24a4e7eb5a95ea1a56e8138fd70c41056398e036a7e",
  "system/entity_audio_repair.py": "f786be2813121230c4f9eea380a166fea8c5fbc67c676fc08ed28144fafb1566",
  "system/entity_streaming_live.py": "d42fbe84ddbcd0f8e768ef7e1679999bc5447045bba7a725b3d39f2b86d8c963",
  "system/entity_ctc_sidecar.py": "ddb73806974b781e77d07f4c13438b517ca466dea4b0e4a362f0d6054168ff3d",
  "experiments/scripts/evaluate_entity_ctc_review_combination.py": "6332163a23813ff3d4b66c5c060f57ffa6816bb05c8db702c53db0d00e92119b",
  "experiments/scripts/entity_ctc_frontend.py": "40546f10adc5ad91cf89b5edd52420e3ba660552b3cbd5968aec57b1eef94b44",
  "experiments/scripts/entity_ctc_word_spotter.py": "655eaa31dd6fd6e1d0774d30b693a38e5bd66d8ba6696d429f9f925e4713177f",
  "experiments/scripts/evaluate_entity_ctc_fusion.py": "2131d100c8f42b3e464a617c9d78f2e3a9bd3dbae9264bb490f6fd7bbb9ce052"
}
```
