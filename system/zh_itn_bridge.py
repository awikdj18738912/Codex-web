"""In-process adapter for the standalone conservative zh-itn rule library."""

from __future__ import annotations

import ctypes
from pathlib import Path

from .numeric_normalizer import NumericNormalization, NumericNormalizationResult


DEFAULT_LIBRARY = (
    Path(__file__).resolve().parent.parent
    / "zh-itn" / "zh-itn" / "build" / "libzh_itn_bridge.so"
)


class ZhITNNormalizer:
    """Normalize one source window once and preserve the rule's source spans."""

    def __init__(self, library: Path = DEFAULT_LIBRARY) -> None:
        if not library.is_file():
            raise FileNotFoundError(f"zh-itn library not found: {library}")
        native = ctypes.CDLL(str(library))
        native.zh_itn_normalize_text.argtypes = [ctypes.c_char_p]
        native.zh_itn_normalize_text.restype = ctypes.c_void_p
        for name in ("zh_itn_result_valid", "zh_itn_result_output", "zh_itn_result_error",
                     "zh_itn_result_mapping_count"):
            getattr(native, name).argtypes = [ctypes.c_void_p]
        native.zh_itn_result_valid.restype = ctypes.c_int
        native.zh_itn_result_output.restype = ctypes.c_char_p
        native.zh_itn_result_error.restype = ctypes.c_char_p
        native.zh_itn_result_mapping_count.restype = ctypes.c_size_t
        for name in ("zh_itn_mapping_kind", "zh_itn_mapping_rule_id",
                     "zh_itn_mapping_input_start", "zh_itn_mapping_input_end",
                     "zh_itn_mapping_output_start", "zh_itn_mapping_output_end"):
            getattr(native, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            getattr(native, name).restype = (
                ctypes.c_char_p if name.endswith(("kind", "rule_id")) else ctypes.c_size_t
            )
        native.zh_itn_rule_pack.argtypes = []
        native.zh_itn_rule_pack.restype = ctypes.c_char_p
        native.zh_itn_result_free.argtypes = [ctypes.c_void_p]
        native.zh_itn_result_free.restype = None
        self._native = native
        self.rule_pack = native.zh_itn_rule_pack().decode("utf-8")

    def normalize(self, text: str) -> NumericNormalizationResult:
        if not text:
            return NumericNormalizationResult(text, ())
        native = self._native
        handle = native.zh_itn_normalize_text(text.encode("utf-8"))
        if not handle:
            raise RuntimeError("zh-itn failed to normalize source window")
        try:
            if not native.zh_itn_result_valid(handle):
                error = native.zh_itn_result_error(handle)
                raise RuntimeError(f"zh-itn rejected source window: {error!r}")
            output = native.zh_itn_result_output(handle).decode("utf-8")
            changes = []
            for index in range(native.zh_itn_result_mapping_count(handle)):
                if native.zh_itn_mapping_kind(handle, index) != b"replace":
                    continue
                start = native.zh_itn_mapping_input_start(handle, index)
                end = native.zh_itn_mapping_input_end(handle, index)
                output_start = native.zh_itn_mapping_output_start(handle, index)
                output_end = native.zh_itn_mapping_output_end(handle, index)
                rule_id = native.zh_itn_mapping_rule_id(handle, index)
                changes.append(NumericNormalization(
                    text[start:end], output[output_start:output_end],
                    rule_id.decode("utf-8") if rule_id else "zh-itn",
                    start, end,
                ))
            return NumericNormalizationResult(output, tuple(changes))
        finally:
            native.zh_itn_result_free(handle)
