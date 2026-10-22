# -*- coding: utf-8 -*-
"""Step 4: materialize the common training-record schema."""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import KEYWORD_MODE, PROCESSED_DIR, RAW_DIR, TOTAL_SAMPLES
from configs.scenes import get_enabled_scenes, validate_enabled_scene
from src.utils.io_utils import read_jsonl, setup_logging, write_jsonl
from src.utils.keyword_utils import extract_keywords, normalize_review_targets
from src.utils.numeric_policy import prepare_fragment_training_pair, prepare_training_pair


def _min_length(scene: str) -> int:
    if scene.startswith("english_"):
        return 6
    if scene in ("vibe_coding", "academic", "dictation_memo", "voice_search", "passthrough"):
        return 8
    return 10


def _speech_phenomena(item: dict) -> list[str]:
    """Merge labels from both supported record locations without losing either."""
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    phenomena: list[str] = []
    for value in (item.get("speech_phenomena"), meta.get("speech_phenomena")):
        if isinstance(value, list):
            phenomena.extend(str(label) for label in value)
        elif isinstance(value, str) and value:
            phenomena.append(value)
    return list(dict.fromkeys(phenomena))


def _record_min_length(scene: str, item: dict) -> int:
    language = _lang_for(scene, item)
    return 4 if language in {"zh", "zh-en-mix"} else _min_length(scene)


def _record_content_length(text: str, scene: str, item: dict) -> int:
    if _lang_for(scene, item) in {"zh", "zh-en-mix"}:
        # Ignore whitespace and punctuation: a short target still needs four
        # letters or digits carrying content, regardless of its speech tags.
        import re
        text = re.sub(
            r"(?:^|(?<=[，,。！？!?；;\s]))(?:嗯|呃|额|那个|就是说|哎呀|(?:哈哈|呵呵|嘿嘿|嘻嘻)+)"
            r"(?=[，,。！？!?；;\s]|$)", "", text,
        )
        content = "".join(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))
        # A determiner inside a real sentence (e.g. 那个女孩) is meaningful.
        # Reject only an entire filler-only target, never strip such words
        # globally from the content-length calculation.
        if re.fullmatch(r"(?:嗯|呃|额|那个|就是说|哎呀|哈|呵|嘿|嘻)+", content):
            return 0
        return len(content)
    return len(text.strip())



def _lang_for(scene: str, item: dict) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return item.get("language") or meta.get("language") or ("en" if scene.startswith("english_") else "zh")


def _copy_record_id(record: dict, item: dict, scene: str) -> None:
    if "id" not in item:
        return
    original = item["id"]
    # Raw generators count independently inside each scene. Namespace these
    # local counters before all scenes are combined into one training file.
    if type(original) is int or (isinstance(original, str) and original.isdecimal()):
        record["id"] = f"{scene}_{original}"
        record["meta"]["source_record_id"] = original
    else:
        record["id"] = original


def _assemble_from_asr(scene: str, raw_data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    assembled = []
    stats = {"skipped_empty": 0, "skipped_short": 0, "skipped_numeric_drift": 0}
    for item in raw_data:
        input_text = item.get("input", "")
        oral_text = item.get("oral_text", "")
        output = item.get("output") if isinstance(item.get("output"), dict) else {}
        clean_text = output.get("refined_text") or item.get("clean_text", "")
        keywords = normalize_review_targets(
            extract_keywords(item),
            input_text=oral_text or input_text,
            refined_text=clean_text,
        )

        if not input_text or not clean_text:
            stats["skipped_empty"] += 1
            continue
        original_meta = item.get("meta") or {}
        plan = original_meta.get("generation_plan")
        if original_meta.get("is_fragment") is True:
            reference = original_meta.get("fragment_reference_clean") or (plan or {}).get("expected_clean")
            pair = prepare_fragment_training_pair(
                input_text, clean_text, oral_text or input_text, reference,
                plan if isinstance(plan, dict) else None,
                observed_prefix=original_meta.get("fragment_observed_prefix"),
            ) if isinstance(reference, str) else None
        else:
            pair = prepare_training_pair(input_text, clean_text, plan)
        if pair is None:
            stats["skipped_numeric_drift"] += 1
            continue
        input_text, clean_text, numeric_meta = pair
        if len(input_text.strip()) < 2 or _record_content_length(clean_text, scene, item) < _record_min_length(scene, item):
            stats["skipped_short"] += 1
            continue

        meta = dict(item.get("meta", {})) if isinstance(item.get("meta"), dict) else {}
        meta.update(numeric_meta)
        meta.setdefault("scene", scene)
        meta.setdefault("language", _lang_for(scene, item))
        meta["speech_phenomena"] = _speech_phenomena(item)
        meta["itn_types"] = []
        meta.setdefault("is_fragment", False)
        meta.setdefault("keyword_mode", KEYWORD_MODE)

        record = {
            "input": input_text,
            "oral_text": oral_text or input_text,
            "output": {
                "refined_text": clean_text,
                "keyword_list": keywords,
            },
            "meta": meta,
        }
        _copy_record_id(record, item, scene)
        assembled.append(record)

    return assembled, stats


def _assemble_from_clean(scene: str, raw_data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    assembled = []
    stats = {"skipped_empty": 0, "skipped_short": 0, "skipped_numeric_drift": 0}
    for item in raw_data:
        oral_text = item.get("oral_text", "")
        clean_text = item.get("clean_text", "")
        keywords = normalize_review_targets(
            extract_keywords(item),
            input_text=oral_text,
            refined_text=clean_text,
        )
        speech_phenomena = _speech_phenomena(item)
        itn_types = []

        if not oral_text or not clean_text:
            stats["skipped_empty"] += 1
            continue
        original_meta = item.get("meta") or {}
        plan = original_meta.get("generation_plan")
        if original_meta.get("is_fragment") is True:
            reference = original_meta.get("fragment_reference_clean") or (plan or {}).get("expected_clean")
            pair = prepare_fragment_training_pair(
                oral_text, clean_text, oral_text, reference,
                plan if isinstance(plan, dict) else None,
                observed_prefix=original_meta.get("fragment_observed_prefix"),
            ) if isinstance(reference, str) else None
        else:
            pair = prepare_training_pair(oral_text, clean_text, plan)
        if pair is None:
            stats["skipped_numeric_drift"] += 1
            continue
        oral_text, clean_text, numeric_meta = pair
        if len(oral_text.strip()) < 2 or _record_content_length(clean_text, scene, item) < _record_min_length(scene, item):
            stats["skipped_short"] += 1
            continue

        lang = _lang_for(scene, item)
        meta = dict(item.get("meta", {})) if isinstance(item.get("meta"), dict) else {}
        meta.update(numeric_meta)
        meta.update({
            "scene": scene,
            "language": lang,
            "speech_phenomena": speech_phenomena,
            "itn_types": itn_types,
            "is_fragment": bool(meta.get("is_fragment", False)),
            "asr_mode": meta.get("asr_mode", "clean_fallback"),
            "keyword_mode": meta.get("keyword_mode", KEYWORD_MODE),
        })

        record = {
            "input": oral_text,
            "oral_text": oral_text,
            "output": {
                "refined_text": clean_text,
                "keyword_list": keywords,
            },
            "meta": meta,
        }
        _copy_record_id(record, item, scene)
        assembled.append(record)

    return assembled, stats


def _choose_source(scene_dir: Path, requested: str) -> tuple[Path | None, str]:
    asr_path = scene_dir / "asr_sim.jsonl"
    clean_path = scene_dir / "clean.jsonl"

    if requested == "asr":
        return (asr_path if asr_path.exists() else None), "asr"
    if requested == "clean":
        return (clean_path if clean_path.exists() else None), "clean"
    if asr_path.exists():
        return asr_path, "asr"
    if clean_path.exists():
        return clean_path, "clean"
    return None, requested


def assemble_scene(scene: str, input_source: str = "auto") -> list[dict]:
    """Load one scene, apply basic checks, and create training records."""
    logger = logging.getLogger(__name__)
    scene_dir = Path(RAW_DIR) / scene
    source_path, actual_source = _choose_source(scene_dir, input_source)

    if source_path is None:
        logger.error("No input data found for scene '%s' (requested=%s)", scene, input_source)
        return []

    logger.info("Scene '%s': assembling from %s", scene, source_path.name)
    raw_data = read_jsonl(source_path)
    if actual_source == "asr":
        assembled, stats = _assemble_from_asr(scene, raw_data)
    else:
        assembled, stats = _assemble_from_clean(scene, raw_data)

    logger.info(
        "Scene '%s': assembled %d from %s, skipped_empty=%d, skipped_short=%d",
        scene, len(assembled), source_path.name, stats["skipped_empty"], stats["skipped_short"],
    )
    return assembled


def main():
    parser = argparse.ArgumentParser(description="Materialize common training records")
    parser.add_argument("--scene", type=str, default=None, help="one internal scene key")
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLES, help="compatibility option")
    parser.add_argument("--input-source", type=str, default="auto", choices=["auto", "asr", "clean"],
                        help="source: auto prefers asr_sim.jsonl, then clean.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    output_dir = Path(PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_assembled = 0

    if args.scene:
        try:
            validate_enabled_scene(args.scene)
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(2)
        logger.info("--- Assembling: %s ---", args.scene)
        records = assemble_scene(args.scene, input_source=args.input_source)
        if records:
            out_path = output_dir / f"{args.scene}.jsonl"
            write_jsonl(records, out_path)
            total_assembled += len(records)
            logger.info("Saved %d records to %s", len(records), out_path)
    else:
        raw_dir = Path(RAW_DIR)
        if not raw_dir.exists():
            logger.error("Raw dir not found: %s", raw_dir)
            return
        enabled_scenes = set(get_enabled_scenes())
        scenes = sorted([
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and d.name in enabled_scenes
            and ((d / "asr_sim.jsonl").exists() or (d / "clean.jsonl").exists())
        ])
        for scene in scenes:
            logger.info("--- Assembling: %s ---", scene)
            records = assemble_scene(scene, input_source=args.input_source)
            if records:
                out_path = output_dir / f"{scene}.jsonl"
                write_jsonl(records, out_path)
                total_assembled += len(records)
                logger.info("  Saved %d records to %s", len(records), out_path)

    logger.info("All done. Total assembled: %d", total_assembled)


if __name__ == "__main__":
    main()
