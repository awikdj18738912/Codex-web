# Qwen3-TTS train-only pilot

## Setup

- Model: Qwen3-TTS-12Hz-0.6B-CustomVoice, fixed Vivian preset voice.
- Model directory: /home/aim0/data/models/ASR/Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-CustomVoice.
- Runtime: project-isolated .runtime/qwen3-tts-env; Transformers 4.57.3, Accelerate 1.12.0, qwen-tts 0.1.1, CPU PyTorch inference.
- TTS service: http://127.0.0.1:8011, compatible with the existing /v1/tts request and audio_path response.
- ASR service: existing streaming Qwen ASR at http://127.0.0.1:8766.
- Selection: 10 records from the frozen train split only; no dev/test record was synthesized or decoded.
- Variants: one clean and one fixed-seed 10 dB white-noise input per utterance.

The model output passed basic file, duration, non-silence, and clipping checks. Those checks do not prove the audio faithfully reads every entity, number, or repetition; audio_script_verified remains false and human_gold remains false.

## Results

- 10 TTS audio files and 20 ASR transcripts were produced.
- Exact transcript-to-spoken-script match: 4/10.
- Exact transcript-to-intended-target match: 3/10.
- Mean character error rate against the spoken script, after removing punctuation, spaces, and symbols: 8.7%.
- Clean and noise ASR text was identical for 9/10 utterances.
- Mean TTS wall time on CPU: 32.7 seconds per utterance.

Examples show useful ASR-like differences: a stutter remained in the transcript, the names 田露 and 唐昕 became 田璐 and 唐鑫, and 8310元 was rendered as the erroneous text 八餐三百一十元. The synthetic noise changed only one of ten transcripts, so it added little diversity in this small sample.

## Interpretation

This verifies that the Qwen TTS service can feed the existing Qwen ASR data pipeline and produce varied text/audio pairs. It does not establish real-speech performance or validate the transcript as human gold. The first batch is too small to train on or to estimate generalization; use it to inspect error types, then add more voices and acoustic conditions before scaling.

Data files are tts.jsonl, asr.jsonl, and audio/ in this directory. The selected seed IDs and split metadata remain attached to each row.
