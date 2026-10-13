"""Local HTTP service for Qwen3-TTS CustomVoice synthesis."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import time
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = Path(
    "/home/aim0/data/models/ASR/Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-CustomVoice"
)

logger = logging.getLogger(__name__)


class QwenTTSRequest(BaseModel):
    text: str = Field(min_length=1)
    output_path: str
    language: str = "Chinese"
    speaker: str | None = None
    instruct: str | None = None
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)


class QwenTTSService:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        default_speaker: str = "Vivian",
        minimum_free_gib: float = 6.0,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.default_speaker = default_speaker
        if minimum_free_gib <= 0:raise ValueError("minimum_free_gib must be positive")
        self.minimum_free_gib = float(minimum_free_gib)
        self._model: Any | None = None
        self._resolved_device: str | None = None

    def _choose_device(self, torch: Any) -> tuple[str, Any]:
        if self.device == "auto":
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info(0)
                if free >= 8 * 1024**3:
                    return "cuda:0", torch.bfloat16
            return "cpu", torch.float32
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            free, _total = torch.cuda.mem_get_info(index)
            if free < self.minimum_free_gib * 1024**3:
                raise RuntimeError(
                    f"Only {free / 1024**3:.2f} GiB is free on GPU {index}; "
                    f"The requested TTS startup reserve is {self.minimum_free_gib:g} GiB."
                )
            return f"cuda:{index}", torch.bfloat16
        if self.device == "cpu":
            return "cpu", torch.float32
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Qwen TTS model directory not found: {self.model_path}")

        import torch
        from qwen_tts import Qwen3TTSModel

        torch.set_num_threads(min(8, os.cpu_count() or 1))
        device_map, dtype = self._choose_device(torch)
        logger.info(
            "Loading Qwen TTS model=%s device=%s dtype=%s",
            self.model_path, device_map, dtype,
        )
        self._model = Qwen3TTSModel.from_pretrained(
            str(self.model_path),
            device_map=device_map,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self._resolved_device = device_map
        logger.info(
            "Qwen TTS ready; supported speakers=%s",
            sorted(self._model.model.get_supported_speakers()),
        )
        return self._model

    def synthesize(
        self,
        *,
        text: str,
        output_path: str | Path,
        language: str = "Chinese",
        speaker: str | None = None,
        instruct: str | None = None,
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("text must not be empty")
        destination = Path(output_path).expanduser().resolve()
        if not destination.is_relative_to(PROJECT_ROOT):
            raise ValueError("audio output must stay inside the project workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        generation = dict(
            text=text,
            language=language or "Chinese",
            speaker=speaker or self.default_speaker,
            max_new_tokens=max_new_tokens,
        )
        if instruct:
            generation["instruct"] = instruct

        started = time.perf_counter()
        wavs, sample_rate = model.generate_custom_voice(**generation)
        if not wavs or wavs[0] is None or len(wavs[0]) == 0:
            raise RuntimeError("Qwen TTS returned an empty audio waveform.")

        import soundfile as sf
        import numpy as np

        waveform = np.asarray(wavs[0], dtype=np.float32)
        sf.write(str(destination), waveform, int(sample_rate), format="WAV", subtype="PCM_16")
        return {
            "audio_path": str(destination),
            "sample_rate": int(sample_rate),
            "duration_seconds": len(waveform) / int(sample_rate),
            "speaker": generation["speaker"],
            "backend": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "device": self._resolved_device,
            "elapsed_seconds": time.perf_counter() - started,
        }


def create_app(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str = "auto",
    default_speaker: str = "Vivian",
    minimum_free_gib: float = 6.0,
):
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install FastAPI and Uvicorn in the Qwen TTS environment.") from exc

    service = QwenTTSService(model_path, device, default_speaker, minimum_free_gib)
    app = FastAPI(title="AgenticASR Qwen3-TTS Service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": service._model is not None,
            "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "device": service._resolved_device or service.device,
        }

    @app.post("/v1/tts")
    def synthesize(request: QwenTTSRequest) -> dict[str, Any]:
        return service.synthesize(**request.model_dump())

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(
        os.environ.get("QWEN_TTS_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    ))
    parser.add_argument(
        "--device", default=os.environ.get("QWEN_TTS_DEVICE", "auto"),
        help="auto, cpu, cuda, or cuda:N; auto uses CPU unless GPU 0 has at least 8 GiB free",
    )
    parser.add_argument("--speaker", default=os.environ.get("QWEN_TTS_SPEAKER", "Vivian"))
    parser.add_argument("--host", default=os.environ.get("QWEN_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN_TTS_PORT", "8011")))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--minimum-free-gib",type=float,default=6.0,help="Explicit CUDA startup reserve; does not change model precision")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    import uvicorn

    app = create_app(args.model_path, args.device, args.speaker, args.minimum_free_gib)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
