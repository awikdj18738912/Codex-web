# -*- coding: utf-8 -*-
"""Build aligned ASR-style input and Clean-target pairs for Step 3."""
import json
import logging
import random
import re
import unicodedata
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    ASR_CACHE_DIR,
    ASR_DEVICE,
    ASR_ENABLE_CACHE,
    ASR_FALLBACK_TEXT_SIM,
    ASR_HTTP_TIMEOUT,
    ASR_MODE,
    ASR_SIMULATION_RATIO,
    ASR_STRENGTH,
    ASR_STT_BASE_URL,
    ASR_STT_PROVIDER,
    ASR_TTS_BASE_URL,
    ASR_TTS_PROVIDER,
    KEYWORD_MODE,
    RAW_DIR,
)
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.io_utils import Checkpoint, append_jsonl, read_jsonl
from src.utils.keyword_utils import extract_keywords, normalize_review_targets
from src.utils.numeric_policy import prepare_fragment_training_pair, prepare_training_pair

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Punctuation removed by the diagnostic rule-based simulator.
_REMOVABLE_PUNCT = set(
    "\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f"
    "\uff08\uff09\u3010\u3011\u300a\u300b\u3008\u3009"
    "\u2014\u2026\uff1a\uff1b\u3001\u00b7\u2015()[]{}\"'"
)

_ALLOWED_STRENGTHS = {"light", "medium", "heavy"}
_ALLOWED_MODES = {"llm_sim", "text_sim", "tts_asr"}


class _UnsafeFragmentAlignment(ValueError):
    """A requested fragment cannot be proven safe, so a full pair may be used."""


def valid_partial_pair(full_input: str, full_target: str, partial_input: str, partial_target: str) -> bool:
    """A fragment must actually end before the full clean semantic content."""
    compact = lambda text: re.sub(r"[\s，,。.!！?？:：;；、‘’“”\"'…—]+", "", text)
    source, target, inp, out = map(compact, (full_input, full_target, partial_input, partial_target))
    return bool(len(inp) >= 4 and 4 <= len(out) < len(target)
                and len(inp) < len(source) and target.startswith(out)
                and out in {compact(prefix) for prefix in _natural_prefixes(full_target, 4)})

ASR_SIMULATION_SYSTEM = """You create an ASR-corrupted version of an observed partial transcript.
The provided clean target is fixed and already aligned to the observed transcript. Add plausible
non-numeric Chinese ASR omissions or substitutions only within the observed transcript. Do not
invent any unobserved continuation, and copy the fixed clean target exactly.
Return strict JSON only: {"input": "...", "target": "..."}. Do not add explanations."""

ASR_SIMULATION_USER = """Observed oral prefix:
{oral}

Fixed clean target for this prefix (copy exactly):
{clean}

Return the corrupted observed prefix and the unchanged fixed target."""


def _is_punctuation_or_space(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def _align_observed_prefix(
    oral_text: str,
    clean_prefix: str,
    generation_plan: dict | None = None,
    full_clean: str | None = None,
) -> str | None:
    """Map clean text to observed speech without stitching abandoned candidates."""
    if generation_plan is not None:
        if generation_plan.get("slot_policy") != "single-slot-v1":
            return None
        return _align_constructed_prefix(oral_text, clean_prefix, full_clean, generation_plan)

    oral_chars = [
        (char, index) for index, char in enumerate(oral_text)
        if not _is_punctuation_or_space(char)
    ]
    clean_chars = [char for char in clean_prefix if not _is_punctuation_or_space(char)]
    if not clean_chars:
        return None
    source_pos = 0
    last_index = None
    for wanted in clean_chars:
        while source_pos < len(oral_chars) and oral_chars[source_pos][0] != wanted:
            source_pos += 1
        if source_pos >= len(oral_chars):
            return None
        last_index = oral_chars[source_pos][1]
        source_pos += 1
    return oral_text[:last_index + 1] if last_index is not None else None


def _compact_indexed(text: str) -> tuple[str, list[int]]:
    indexed = [(char, index) for index, char in enumerate(text) if not _is_punctuation_or_space(char)]
    return "".join(char for char, _ in indexed), [index for _, index in indexed]


def _find_unique(haystack: str, needle: str) -> int | None:
    if not needle:
        return None
    first = haystack.find(needle)
    if first < 0 or haystack.find(needle, first + 1) >= 0:
        return None
    return first


def _greedy_end(oral: str, wanted: str, start: int, stop: int) -> int | None:
    cursor = start
    last = None
    for char in wanted:
        while cursor < stop and oral[cursor] != char:
            cursor += 1
        if cursor >= stop:
            return None
        last = cursor
        cursor += 1
    return last


def _align_constructed_prefix(
    oral_text: str,
    clean_prefix: str,
    full_clean: str | None,
    generation_plan: dict,
) -> str | None:
    """Constrain pre-slot text before the correction chain and final slot after it."""
    if not isinstance(full_clean, str) or not full_clean:
        return None
    values = generation_plan.get("correction_values")
    if not isinstance(values, list):
        return None
    from src.utils.numeric_policy import _construction_plan_matches, normalize_source
    if not _construction_plan_matches(oral_text, full_clean, generation_plan):
        return None
    from src.utils.synthetic_oral import validate_correction_slots
    try:
        validate_correction_slots(generation_plan.get("raw_intent", ""), values)
    except (TypeError, ValueError):
        return None

    oral, oral_indices = _compact_indexed(oral_text)
    clean, _ = _compact_indexed(full_clean)
    wanted, _ = _compact_indexed(clean_prefix)
    if not wanted or not clean.startswith(wanted):
        return None
    normalized_full, _ = normalize_source(full_clean)
    if normalized_full != full_clean:
        return None  # Numeric normalization can change offsets; use a full sample.

    oral_limit = len(oral)
    parts = generation_plan.get("spelling_parts", [])
    if parts:
        cue_suffix = "，" + "，".join(f"{part['word']}的{part['character']}" for part in parts)
        cue_start = oral_text.rfind(cue_suffix)
        if cue_start < 0:
            return None
        oral_limit = sum(index < cue_start for index in oral_indices)

    if not values:
        end = _greedy_end(oral, wanted, 0, oral_limit)
        return None if end is None else oral_text[:oral_indices[end] + 1]

    if any(not isinstance(value, str) or not value for value in values):
        return None
    normalized_values = [normalize_source(value)[0] for value in values]
    if normalized_values != values:
        return None
    slot = _compact_indexed(values[-1])[0]
    slot_start = _find_unique(clean, slot)
    if slot_start is None:
        return None

    compact_values = [_compact_indexed(value)[0] for value in values]
    if any(not value for value in compact_values):
        return None
    chain = compact_values[0]
    markers: list[tuple[int, int]] = []
    for index, value in enumerate(compact_values[1:], start=1):
        marker = "不对" if index == 1 else "最后改成"
        marker_start = len(chain)
        chain += marker + value
        markers.append((marker_start, marker_start + len(marker)))
    chain_start = _find_unique(oral, chain)
    if chain_start is None:
        return None

    # If the target ends before the final value begins, it must be fully
    # observed before the first abandoned candidate is spoken.
    if len(wanted) <= slot_start:
        end = _greedy_end(oral, wanted, 0, chain_start)
        return None if end is None else oral_text[:oral_indices[end] + 1]

    pre_slot = wanted[:slot_start]
    pre_end = _greedy_end(oral, pre_slot, 0, chain_start)
    if pre_end is None and pre_slot:
        return None
    final_marker_start, final_marker_end = markers[-1]
    final_candidate_start = chain_start + final_marker_end
    if (final_candidate_start >= oral_limit
            or oral[final_candidate_start:final_candidate_start + len(slot)] != slot):
        return None
    tail = wanted[slot_start:]
    tail_end = _greedy_end(oral, tail, final_candidate_start, oral_limit)
    if tail_end is None:
        return None
    return oral_text[:oral_indices[tail_end] + 1]


def _natural_prefixes(clean_text: str, minimum: int) -> list[str]:
    candidates = []
    for match in re.finditer(r"[,，;；。！？!?]", clean_text):
        index = match.start()
        if match.end() < minimum or match.end() >= len(clean_text):
            continue
        if index and clean_text[index - 1].isdigit() and clean_text[match.end()].isdigit():
            continue  # A thousands separator is not a semantic boundary.
        prefix = clean_text[:match.end()]
        if any(prefix.count(left) > prefix.count(right) for left, right in (("(", ")"), ("[", "]"), ("{", "}"), ("（", "）"))):
            continue
        candidates.append(prefix)
    return candidates


def _select_partial_target(clean_text: str, rng: random.Random) -> str:
    """Select a strict, naturally bounded prefix shorter than the full target."""
    candidates = _natural_prefixes(clean_text, 8)
    if not candidates:
        raise ValueError("Clean target has no eligible natural fragment boundary")
    return rng.choice(candidates)


def _remove_punct(text: str, strength: str = "medium") -> str:
    """Remove selected punctuation for diagnostic ASR degradation."""
    if strength == "light":
        keep = {"\uff0c", "\u3002", "\uff01", "\uff1f", ",", ".", "!", "?", "\uff1a", ":"}
    elif strength == "heavy":
        keep = {",", "."}
    else:
        keep = {"\uff0c", "\u3002", "\uff01", "\uff1f", ",", ".", "!", "?"}

    chars = []
    for c in text:
        if c in keep:
            chars.append(c)
        elif c in _REMOVABLE_PUNCT:
            chars.append(" ")
        else:
            chars.append(c)
    return "".join(chars)


def simulate_asr(text: str, strength: str = "medium") -> str:
    """Apply lightweight rule-based ASR degradation for diagnostics."""
    if strength not in _ALLOWED_STRENGTHS:
        strength = "medium"
    text = _remove_punct(text, strength=strength)
    text = re.sub(r"\s+", " ", text).strip()
    return text



class ASRSimulator:
    """Convert Clean records to aligned Refiner training records."""

    def __init__(
        self,
        async_client=None,
        mode: str | None = None,
        strength: str | None = None,
        tts_provider: str | None = None,
        stt_provider: str | None = None,
        tts_base_url: str | None = None,
        stt_base_url: str | None = None,
        device: str | None = None,
        enable_cache: bool | None = None,
        fallback_text_sim: bool | None = None,
        simulation_ratio: float = ASR_SIMULATION_RATIO,
        rng: random.Random | None = None,
    ):
        self.raw_dir = Path(RAW_DIR)
        self.mode = mode or ASR_MODE
        self.strength = strength or ASR_STRENGTH
        self.tts_provider = tts_provider or ASR_TTS_PROVIDER
        self.stt_provider = stt_provider or ASR_STT_PROVIDER
        self.tts_base_url = tts_base_url or ASR_TTS_BASE_URL
        self.stt_base_url = stt_base_url or ASR_STT_BASE_URL
        self.device = device or ASR_DEVICE
        self.enable_cache = ASR_ENABLE_CACHE if enable_cache is None else enable_cache
        self.fallback_text_sim = ASR_FALLBACK_TEXT_SIM if fallback_text_sim is None else fallback_text_sim
        self.simulation_ratio = simulation_ratio
        self.rng = rng or random.Random()
        self.llm = async_client
        self._backend = None
        if self.mode not in _ALLOWED_MODES:
            raise ValueError(f"Unsupported ASR mode: {self.mode}. Allowed: {sorted(_ALLOWED_MODES)}")
        if self.strength not in _ALLOWED_STRENGTHS:
            raise ValueError(f"Unsupported ASR strength: {self.strength}. Allowed: {sorted(_ALLOWED_STRENGTHS)}")
        if not 0 <= self.simulation_ratio <= 1:
            raise ValueError("simulation_ratio must be between 0 and 1")

    def _get_lang(self, scene: str) -> str:
        return "en" if scene.startswith("english_") else "zh"

    def _get_backend(self):
        if self._backend is None:
            from src.speech.tts_asr_backend import TTSASRBackend, TTSASRConfig

            self._backend = TTSASRBackend(TTSASRConfig(
                tts_provider=self.tts_provider,
                stt_provider=self.stt_provider,
                device=self.device,
                cache_dir=Path(ASR_CACHE_DIR),
                enable_cache=self.enable_cache,
                strength=self.strength,
                tts_base_url=self.tts_base_url,
                stt_base_url=self.stt_base_url,
                timeout=ASR_HTTP_TIMEOUT,
            ))
        return self._backend

    async def _simulate_with_llm(
        self,
        oral_text: str,
        clean_text: str,
        generation_plan: dict | None = None,
        *,
        return_observed_prefix: bool = False,
    ) -> tuple[str, str] | tuple[str, str, str]:
        if self.llm is None:
            self.llm = AsyncLLMClient()
        fixed_target = _select_partial_target(clean_text, self.rng)
        observed_prefix = _align_observed_prefix(
            oral_text, fixed_target, generation_plan, clean_text if generation_plan is not None else None,
        )
        if observed_prefix is None:
            raise _UnsafeFragmentAlignment("Clean fragment cannot be aligned to a safe observed prefix")
        response = await self.llm.generate(
            system_prompt=ASR_SIMULATION_SYSTEM,
            user_prompt=ASR_SIMULATION_USER.format(oral=observed_prefix, clean=fixed_target),
            temperature=0.7,
            max_tokens=1024,
        )
        try:
            parsed = LLMClient._parse_json(response)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("ASR simulator returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("ASR simulator response must be a JSON object")
        input_text = parsed.get("input")
        target_text = parsed.get("target")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("ASR simulator response is missing `input`")
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("ASR simulator response is missing `target`")
        if target_text.strip() != fixed_target:
            raise ValueError("ASR simulator changed the fixed fragment target")
        result = (input_text.strip(), fixed_target)
        return (*result, observed_prefix) if return_observed_prefix else result

    async def _generate_pair(
        self,
        oral_text: str,
        clean_text: str,
        *,
        language: str | None = None,
        passthrough: bool = False,
        generation_plan: dict | None = None,
    ) -> tuple[str, str, bool, bool, str | None]:
        if passthrough:
            return clean_text, clean_text, False, False, None
        selected = self.rng.random() < self.simulation_ratio
        if self.mode == "llm_sim":
            if selected:
                if not _natural_prefixes(clean_text, 8):
                    # A complete, valid sentence is still useful when it has
                    # no safe place to make a partial example.
                    return oral_text, clean_text, False, False, None
                try:
                    input_text, target_text, observed_prefix = await self._simulate_with_llm(
                        oral_text, clean_text, generation_plan, return_observed_prefix=True,
                    )
                except _UnsafeFragmentAlignment as error:
                    logger.info("Using full pair because fragment alignment is unsafe: %s", error)
                    return oral_text, clean_text, False, False, None
                return input_text, target_text, True, False, observed_prefix
            return oral_text, clean_text, False, False, None
        if self.mode == "text_sim":
            return simulate_asr(oral_text, strength=self.strength), clean_text, False, False, None
        try:
            input_text = await self._get_backend().transcribe(oral_text, language=language)
            return input_text, clean_text, False, False, None
        except Exception as e:
            if not self.fallback_text_sim:
                raise
            logger.warning("tts_asr failed; fallback to text_sim: %s", e)
            return simulate_asr(oral_text, strength=self.strength), clean_text, False, True, None

    async def simulate_one(self, scene: str, item: dict) -> dict:
        """Generate one ASR-style input and aligned target."""
        oral_text = item.get("oral_text", "")
        clean_text = item.get("clean_text") or item.get("output", {}).get("refined_text", "")
        lang = item.get("language") or self._get_lang(scene)

        if not oral_text or not clean_text:
            return {}

        speech_phenomena = item.get("speech_phenomena") or item.get("meta", {}).get("speech_phenomena", [])
        passthrough = "clean" in speech_phenomena
        meta_in = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        plan = meta_in.get("generation_plan")
        try:
            asr_output, aligned_target, is_fragment, used_fallback, observed_prefix = await self._generate_pair(
                oral_text,
                clean_text,
                language=lang,
                passthrough=passthrough,
                generation_plan=plan if isinstance(plan, dict) else None,
            )
        except ValueError as error:
            logger.warning("Rejecting invalid ASR pair id=%s: %s", item.get("id"), error)
            return {}
        if is_fragment:
            pair = prepare_fragment_training_pair(
                asr_output, aligned_target, oral_text, clean_text,
                plan if isinstance(plan, dict) else None,
                observed_prefix=observed_prefix,
            )
        else:
            pair = prepare_training_pair(asr_output, aligned_target, plan)
        if pair is None:
            logger.warning("Rejecting numeric/entity drift in ASR pair id=%s", item.get("id"))
            return {}
        asr_output, aligned_target, numeric_meta = pair
        keywords = normalize_review_targets(
            extract_keywords(item),
            input_text=oral_text,
            refined_text=aligned_target,
        )
        itn_types = []

        meta = dict(item.get("meta", {})) if isinstance(item.get("meta"), dict) else {}
        meta.update(numeric_meta)
        meta.update({
            "scene": scene,
            "language": lang,
            "speech_phenomena": speech_phenomena,
            "itn_types": itn_types,
            "is_fragment": is_fragment,
            "asr_mode": self.mode,
            "asr_strength": self.strength,
            "keyword_mode": KEYWORD_MODE,
            "tts_provider": self.tts_provider if self.mode == "tts_asr" else None,
            "stt_provider": self.stt_provider if self.mode == "tts_asr" else None,
            "asr_fallback_text_sim": used_fallback,
            "passthrough": passthrough,
        })
        if is_fragment:
            meta["fragment_reference_clean"] = clean_text
            meta["fragment_observed_prefix"] = observed_prefix

        result = {
            "input": asr_output,
            "oral_text": oral_text,
            "clean_text": aligned_target,
            "output": {
                "refined_text": aligned_target,
                "keyword_list": keywords,
            },
            "meta": meta,
        }
        if "id" in item:
            result["id"] = item["id"]
        return result

    async def simulate_scene(self, scene: str, resume: bool = True, input_source: str = "clean") -> int:
        """Generate ASR-style inputs for one scene with checkpointing."""
        scene_dir = self.raw_dir / scene
        source_name = "clean.jsonl" if input_source == "clean" else "oral.jsonl"
        source_path = scene_dir / source_name
        asr_path = scene_dir / "asr_sim.jsonl"
        ckpt_path = scene_dir / "asr_sim.ckpt.json"

        if not source_path.exists():
            if input_source == "clean":
                logger.error("Clean data not found: %s", source_path)
            else:
                logger.error("Oral data not found: %s", source_path)
            return 0

        source_data = read_jsonl(source_path)
        total = len(source_data)

        ckpt = Checkpoint(ckpt_path)
        processed_count = ckpt.get("processed_count", 0) if resume else 0

        remaining_items = source_data[processed_count:]

        if not remaining_items:
            logger.info("Scene '%s': all %d items already processed.", scene, total)
            return processed_count

        logger.info(
            "Scene '%s': mode=%s strength=%s source=%s total=%d processed=%d remaining=%d",
            scene, self.mode, self.strength, source_name, total, processed_count, len(remaining_items),
        )

        pbar = tqdm(total=len(remaining_items), desc=f"ASR-Sim[{scene}]", unit="item")

        for item in remaining_items:
            result = await self.simulate_one(scene, item)
            if result:
                append_jsonl([result], asr_path)
            processed_count += 1
            ckpt.set("processed_count", processed_count)
            pbar.update(1)

        pbar.close()
        return processed_count
