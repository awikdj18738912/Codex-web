#!/usr/bin/env python3
"""Isolated, stateful Zipformer2 CTC posterior frontend for experiments.

No production recognizer is imported. One ORT session is shared by independent
streams; each stream runs only the next feature window and carries all caches.
The 45-frame input windows overlap by 13 frames, as in sherpa-onnx, but outputs
and cache updates never replay an earlier chunk.

Model metadata selects Whisper or ordinary Kaldi features. In particular the
Chinese 2025-06-30 checkpoint says ``feature=whisper``: ordinary OnlineFbank is
not interchangeable with OnlineWhisperFbank for that checkpoint.

Reference implementations (Apache-2.0): k2-fsa/sherpa-onnx, csrc files
online-zipformer2-ctc-model.cc, online-recognizer-ctc-impl.h, features.cc, math.cc.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort


_BYTE_TOKEN = re.compile(r"<0x([0-9a-fA-F]{2})>")


class StreamingCtcModel:
    """Shared CPU session and immutable model description (batch size one)."""

    sample_rate = 16000
    feature_frame_shift_seconds = 0.01

    def __init__(self, modelpath: str | Path, tokenspath: str | Path,
                 num_threads: int = 2):
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(modelpath), options,
                                            providers=["CPUExecutionProvider"])
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.chunk_length = int(self.metadata["T"])
        self.chunk_shift = int(self.metadata["decode_chunk_len"])
        self.feature_dim = int(self.inputs[0].shape[-1])
        self.vocab_size = int(self.outputs[0].shape[-1])
        self.feature_type = self.metadata.get("feature", "fbank")
        if self.feature_type not in ("fbank", "", "whisper"):
            raise ValueError(f"Unsupported feature type: {self.feature_type!r}")
        self.subsampling_factor = int(self.metadata.get("subsampling_factor", 4))
        self.frame_shift_seconds = (self.feature_frame_shift_seconds *
                                    self.subsampling_factor)
        if not 0 < self.chunk_shift <= self.chunk_length:
            raise ValueError("Invalid model chunk lengths")
        if self.inputs[0].shape[1] != self.chunk_length:
            raise ValueError("Model T metadata disagrees with input shape")
        self.tokens = [""] * self.vocab_size
        for line in Path(tokenspath).read_text(encoding="utf-8").splitlines():
            token, index = line.rsplit(maxsplit=1)
            index = int(index)
            if 0 <= index < self.vocab_size:
                self.tokens[index] = token
        if any(not token for token in self.tokens):
            raise ValueError("Token table does not cover the model vocabulary")
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.blank_id = next((self.token_to_id[t] for t in
                              ("<blk>", "<eps>", "<blank>")
                              if t in self.token_to_id), None)
        if self.blank_id is None:
            raise ValueError("Token table has no CTC blank")
        self._state_specs = self._initial_state_specs()
        if len(self.outputs) != len(self.inputs):
            raise ValueError("Expected log_probs followed by one output per cache")

    def _initial_state_specs(self):
        """Resolve cache dimensions following official InitStates metadata."""
        def values(key):
            return [int(x) for x in self.metadata[key].split(",")]

        dims = values("encoder_dims")
        layers = values("num_encoder_layers")
        heads = values("num_heads")
        qdims = values("query_head_dims")
        vdims = values("value_head_dims")
        contexts = values("left_context_len")
        kernels = values("cnn_module_kernels")
        shapes = []
        for i, num_layers in enumerate(layers):
            for _ in range(num_layers):
                shapes.extend([
                    (contexts[i], 1, qdims[i] * heads[i]),
                    (1, 1, contexts[i], 3 * dims[i] // 4),
                    (contexts[i], 1, vdims[i] * heads[i]),
                    (contexts[i], 1, vdims[i] * heads[i]),
                    (1, dims[i], kernels[i] // 2),
                    (1, dims[i], kernels[i] // 2),
                ])
        shapes.extend([(1, 128, 3, 19), (1,)])
        if len(shapes) != len(self.inputs) - 1:
            raise ValueError("Unsupported Zipformer2 cache layout")
        result = []
        for node, expected in zip(self.inputs[1:], shapes):
            if len(node.shape) != len(expected):
                raise ValueError(f"Unexpected cache rank: {node.name}")
            for actual, size in zip(node.shape, expected):
                if isinstance(actual, int) and actual != size:
                    raise ValueError(f"Cache metadata/shape mismatch: {node.name}")
            dtype = {"tensor(float)": np.float32,
                     "tensor(int64)": np.int64}.get(node.type)
            if dtype is None:
                raise ValueError(f"Unsupported cache dtype: {node.type}")
            result.append((node.name, expected, dtype))
        return result

    def create_stream(self) -> "StreamingCtcStream":
        return StreamingCtcStream(self)

    def reset(self, stream: "StreamingCtcStream | None" = None):
        if stream is None:
            return self.create_stream()
        if stream.model is not self:
            raise ValueError("Stream belongs to another model")
        return stream.reset()

    def decode_tokens(self, ids) -> str:
        """Decode byte fallback as UTF-8 across token and chunk boundaries."""
        data = bytearray()
        for index in ids:
            token = self.tokens[int(index)]
            match = _BYTE_TOKEN.fullmatch(token)
            if match:
                data.append(int(match.group(1), 16))
            else:
                data.extend(token.replace("▁", " ").encode("utf-8"))
        return data.decode("utf-8", errors="replace").strip()


class StreamingCtcStream:
    """Per-utterance features, model caches, and cross-chunk greedy CTC state.

    ``accept_audio`` consumes new normalized mono float32 samples at 16 kHz and
    returns zero or more block dicts. ``finish`` adds exactly 0.66 seconds of
    silence once, marks feature input finished, and drains ready windows.
    ``frame_offset`` indexes CTC frames, and ``frame_times_seconds`` follows
    sherpa's approximate 40 ms CTC frame grid (not forced acoustic alignment).
    Availability excludes synthetic padding; padding flags expose the tail.
    """

    def __init__(self, model: StreamingCtcModel):
        self.model = model
        self.reset()

    def reset(self):
        model = self.model
        if model.feature_type == "whisper":
            options = knf.WhisperFeatureOptions()
            options.dim = model.feature_dim
            # WhisperFeatureComputer also enforces these internally.
            options.frame_opts.samp_freq = model.sample_rate
            options.frame_opts.dither = 0
            options.frame_opts.snip_edges = False
            self.fbank = knf.OnlineWhisperFbank(options)
        else:
            options = knf.FbankOptions()
            options.frame_opts.samp_freq = model.sample_rate
            options.frame_opts.dither = 0
            options.frame_opts.snip_edges = False
            options.mel_opts.num_bins = model.feature_dim
            options.mel_opts.high_freq = -400
            self.fbank = knf.OnlineFbank(options)
        self.states = {name: np.zeros(shape, dtype=dtype)
                       for name, shape, dtype in model._state_specs}
        self.processed_feature_frames = 0
        self.output_frames = 0
        self.audio_samples = 0
        self.padding_samples = 0
        self.finished = False
        self.token_ids = []
        self.token_frames = []
        self._previous_token = model.blank_id
        self.compute_seconds = 0.0
        self.feature_compute_seconds = 0.0
        self.num_blocks = 0
        self.raw_logsumexp_max_abs = 0.0
        self.normalized_blocks = 0
        self.last_feature_frame = None
        return self

    @property
    def text(self) -> str:
        return self.model.decode_tokens(self.token_ids)

    @property
    def final_text(self) -> str:
        return self.text

    @property
    def finaltext(self) -> str:
        return self.text

    @property
    def stats(self) -> dict:
        return {
            "audio_seconds": self.audio_samples / self.model.sample_rate,
            "padding_seconds": self.padding_samples / self.model.sample_rate,
            "compute_seconds": self.compute_seconds,
            "feature_compute_seconds": self.feature_compute_seconds,
            "num_blocks": self.num_blocks,
            "output_frames": self.output_frames,
            "processed_feature_frames": self.processed_feature_frames,
            "last_feature_frame": self.last_feature_frame,
            "raw_logsumexp_max_abs": self.raw_logsumexp_max_abs,
            "normalized_blocks": self.normalized_blocks,
            "finished": self.finished,
        }

    def is_ready(self) -> bool:
        # Match sherpa's strict '<', including at utterance end.
        return (self.processed_feature_frames + self.model.chunk_length <
                self.fbank.num_frames_ready)

    def accept_audio(self, samples: np.ndarray) -> list[dict]:
        if self.finished:
            raise RuntimeError("Cannot accept audio after finish(); reset first")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Expected finite mono samples at 16000 Hz")
        started = time.perf_counter()
        if samples.size:
            feature_samples = (samples if self.model.feature_type == "whisper"
                               else samples * np.float32(32768))
            self.fbank.accept_waveform(self.model.sample_rate, feature_samples)
        self.audio_samples += samples.size
        self.feature_compute_seconds += time.perf_counter() - started
        return self.step()

    def step(self) -> list[dict]:
        """Drain every currently ready window without accepting more audio."""
        model = self.model
        blocks = []
        # Explicit bound catches a readiness/cache cursor regression.
        max_blocks = self.fbank.num_frames_ready // model.chunk_shift + 1
        while self.is_ready():
            if len(blocks) >= max_blocks:
                raise RuntimeError("CTC drain failed to advance feature cursor")
            feature_start = self.processed_feature_frames
            started = time.perf_counter()
            features = np.stack([self.fbank.get_frame(i) for i in
                                 range(feature_start,
                                       feature_start + model.chunk_length)])
            if model.feature_type == "whisper":
                features = np.log10(np.maximum(features, np.float32(1e-10)))
                features = np.maximum(features, features.max() - np.float32(8))
                features = (features + np.float32(4)) / np.float32(4)
            # Standard Kaldi features use PCM amplitude as sherpa does.
            features = np.ascontiguousarray(features[None], dtype=np.float32)
            self.feature_compute_seconds += time.perf_counter() - started
            feed = {model.inputs[0].name: features, **self.states}
            started = time.perf_counter()
            output = model.session.run(None, feed)
            elapsed = time.perf_counter() - started
            log_probs = np.asarray(output[0][0], dtype=np.float32)
            if (log_probs.ndim != 2 or log_probs.shape[1] != model.vocab_size
                    or not np.isfinite(log_probs).all()):
                raise RuntimeError("Model returned invalid CTC frame scores")
            if log_probs.shape[0] * model.subsampling_factor != model.chunk_shift:
                raise RuntimeError("Output frame count disagrees with CTC time grid")
            maximum = log_probs.max(axis=1, keepdims=True)
            lse = maximum + np.log(np.exp(log_probs - maximum).sum(axis=1,
                                                                      keepdims=True))
            norm_error = float(np.max(np.abs(lse)))
            self.raw_logsumexp_max_abs = max(self.raw_logsumexp_max_abs, norm_error)
            if norm_error > 1e-4:
                log_probs = log_probs - lse
                self.normalized_blocks += 1
            self.states = {node.name: value for node, value in
                           zip(model.inputs[1:], output[1:])}
            offset = self.output_frames
            greedy = log_probs.argmax(axis=1)
            emitted = []
            for i, token in enumerate(greedy):
                token = int(token)
                if token != model.blank_id and token != self._previous_token:
                    self.token_ids.append(token)
                    self.token_frames.append(offset + i)
                    emitted.append(token)
                self._previous_token = token
            frame_times = ((offset + np.arange(log_probs.shape[0])) *
                           model.frame_shift_seconds)
            audio_seconds = self.audio_samples / model.sample_rate
            blocks.append({
                "log_probs": log_probs,
                "frame_offset": offset,
                "feature_frame_offset": feature_start,
                "frame_shift_seconds": model.frame_shift_seconds,
                "frame_times_seconds": frame_times,
                "available_audio_seconds": audio_seconds,
                "compute_seconds": elapsed,
                "raw_logsumexp_max_abs": norm_error,
                "greedy_token_ids": greedy,
                "new_token_ids": emitted,
                "token_ids": list(self.token_ids),
                "tokens": [model.tokens[i] for i in self.token_ids],
                "text": self.text,
                "is_padding": frame_times >= audio_seconds,
                "after_input_finished": self.finished,
            })
            self.processed_feature_frames += model.chunk_shift
            self.output_frames += log_probs.shape[0]
            self.compute_seconds += elapsed
            self.num_blocks += 1
            # Retain only the lookahead features required by the next window.
            self.fbank.pop(model.chunk_shift)
        return blocks

    def finish(self, padding_seconds: float = 0.66) -> list[dict]:
        if self.finished:
            return []
        if not np.isfinite(padding_seconds) or not 0 <= padding_seconds <= 2:
            raise ValueError("Padding must be between 0 and 2 seconds")
        self.padding_samples = round(padding_seconds * self.model.sample_rate)
        started = time.perf_counter()
        if self.padding_samples:
            self.fbank.accept_waveform(self.model.sample_rate,
                                      np.zeros(self.padding_samples, np.float32))
        self.fbank.input_finished()
        self.finished = True
        self.last_feature_frame = self.fbank.num_frames_ready - 1
        self.feature_compute_seconds += time.perf_counter() - started
        return self.step()


def _read_pcm16(path):
    with wave.open(str(path)) as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Smoke input must be mono PCM16 WAV")
        if wav.getframerate() != 16000:
            raise ValueError("Smoke input must be 16 kHz")
        return np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(
            np.float32) / 32768


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--compare-sherpa", action="store_true")
    args = parser.parse_args()
    model_path = args.model_dir / "model.int8.onnx"
    tokens_path = args.model_dir / "tokens.txt"
    model = StreamingCtcModel(model_path, tokens_path, args.num_threads)
    samples = _read_pcm16(args.wav)
    stream = model.create_stream()
    shapes = []
    for start in range(0, samples.size, 2560):
        shapes.extend(tuple(b["log_probs"].shape) for b in
                      stream.accept_audio(samples[start:start + 2560]))
    shapes.extend(tuple(b["log_probs"].shape) for b in stream.finish())
    report = {"text": stream.text, "stats": stream.stats,
              "block_shapes": sorted(set(shapes)), "metadata": model.metadata}
    if args.compare_sherpa:
        import sherpa_onnx
        recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            model=str(model_path), tokens=str(tokens_path),
            num_threads=args.num_threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", provider="cpu")
        reference = recognizer.create_stream()
        for start in range(0, samples.size, 2560):
            reference.accept_waveform(16000, samples[start:start + 2560])
            while recognizer.is_ready(reference):
                recognizer.decode_stream(reference)
        reference.accept_waveform(16000, np.zeros(round(0.66 * 16000), np.float32))
        reference.input_finished()
        while recognizer.is_ready(reference):
            recognizer.decode_stream(reference)
        report["sherpa_text"] = recognizer.get_result(reference)
        report["matches_sherpa"] = stream.text == report["sherpa_text"]
        report["sherpa_version"] = sherpa_onnx.__version__
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
