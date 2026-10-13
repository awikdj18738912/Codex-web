"""Controlled text -> local TTS HTTP service -> current streaming Qwen ASR.

Outputs are resumable synthetic development data, never human gold. There is
no text-simulation fallback: a speech-service failure remains an explicit error.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Each template stays wholly in one synthetic split, including all slot values,
# oral/clean counterparts and audio/noise versions. Values are fictional probes.
TEMPLATES = {
    "clean": [
        ("明天去{place}开会。", "明天去{place}开会。"),
        ("请把文件交给{name}。", "请把文件交给{name}。"),
        ("{name}负责整理会议记录。", "{name}负责整理会议记录。"),
        ("我们将在{place}讨论下一步的安排。", "我们将在{place}讨论下一步的安排。"),
        ("报告已经提交。请{name}确认收到。", "报告已经提交。请{name}确认收到。"),
        ("今天先检查设备。明天再去{place}。", "今天先检查设备。明天再去{place}。"),
        ("{name}的检查报告已经打印好了。", "{name}的检查报告已经打印好了。"),
        ("请到{place}的服务台领取资料。", "请到{place}的服务台领取资料。"),
    ],
    "filler": [
        ("嗯，明天去{place}开会。", "明天去{place}开会。"),
        ("请把那个，文件交给{name}。", "请把文件交给{name}。"),
        ("呃，{name}负责整理会议记录。", "{name}负责整理会议记录。"),
        ("我们呢，将在{place}讨论安排。", "我们将在{place}讨论安排。"),
        ("报告，呃，已经提交。请{name}确认收到。", "报告已经提交。请{name}确认收到。"),
        ("今天先，嗯，检查设备。明天去{place}。", "今天先检查设备。明天去{place}。"),
        ("{name}的检查报告呢，已经打印好了。", "{name}的检查报告已经打印好了。"),
        ("请到，呃，{place}的服务台领取资料。", "请到{place}的服务台领取资料。"),
    ],
    "stutter": [
        ("我们我们明天去{place}开会。", "我们明天去{place}开会。"),
        ("请请把文件交给{name}。", "请把文件交给{name}。"),
        ("{name}负责负责整理会议记录。", "{name}负责整理会议记录。"),
        ("我们要讨讨论去{place}的安排。", "我们要讨论去{place}的安排。"),
        ("报报告已经提交。请{name}确认。", "报告已经提交。请{name}确认。"),
        ("今天先检检查设备。明天去{place}。", "今天先检查设备。明天去{place}。"),
        ("{name}的的检查报告已经打印好了。", "{name}的检查报告已经打印好了。"),
        ("请到{place}的服服务台领取资料。", "请到{place}的服务台领取资料。"),
    ],
    "meaningful_repeat": [
        ("这些泵站一级一级地向高处提水。{name}负责记录。", "这些泵站一级一级地向高处提水。{name}负责记录。"),
        ("我们一件一件地核对{place}的设备。", "我们一件一件地核对{place}的设备。"),
        ("{name}慢慢走到门口。", "{name}慢慢走到门口。"),
        ("今天从{name}那里学到了很多很多。", "今天从{name}那里学到了很多很多。"),
        ("{name}对此特别特别重视。", "{name}对此特别特别重视。"),
        ("请把{place}的材料逐页核对。一页一页地看。", "请把{place}的材料逐页核对。一页一页地看。"),
        ("{name}刚刚离开这里。", "{name}刚刚离开这里。"),
        ("{place}的事情要清清楚楚地说明。", "{place}的事情要清清楚楚地说明。"),
    ],
    "protected_content": [
        ("{name}说不是今天，而是明天去{place}。", "{name}说不是今天，而是明天去{place}。"),
        ("这笔费用是{amount}元，不是{other_amount}元。", "这笔费用是{amount}元，不是{other_amount}元。"),
        ("地址是{place}{number}号，不是{other_number}号。", "地址是{place}{number}号，不是{other_number}号。"),
        ("预约的是{name}，不是{other_name}。", "预约的是{name}，不是{other_name}。"),
        ("{name}没有同意这个方案。", "{name}没有同意这个方案。"),
        ("报表记录的数量是{number}，单位是毫升。", "报表记录的数量是{number}，单位是毫升。"),
        ("报告中的数值是{decimal}。请{name}核对。", "报告中的数值是{decimal}。请{name}核对。"),
        ("{name}说文件提到了奥希替尼，没有提到吉非替尼。", "{name}说文件提到了奥希替尼，没有提到吉非替尼。"),
    ],
    "syntax_boundary": [
        ("如果{name}明天到场，我们就讨论下一步安排。", "如果{name}明天到场，我们就讨论下一步安排。"),
        ("当我们到达{place}，还需要核对设备。", "当我们到达{place}，还需要核对设备。"),
        ("{name}需要了解是否安全，以及需要注意哪些方面。", "{name}需要了解是否安全，以及需要注意哪些方面。"),
        ("这些材料都是{name}准备的。", "这些材料都是{name}准备的。"),
        ("我们在{place}完成检查之后，还要整理报告。", "我们在{place}完成检查之后，还要整理报告。"),
        ("即便{name}不能到场，我们也会继续讨论。", "即便{name}不能到场，我们也会继续讨论。"),
        ("开发商面临不均衡的复苏前景。{name}正在整理报告。", "开发商面临不均衡的复苏前景。{name}正在整理报告。"),
        ("在{place}，我们会看到新的设备。", "在{place}，我们会看到新的设备。"),
    ],
}
NAMES = ("王明", "李安", "陈晓", "赵宁", "刘洋", "周晴", "吴越", "郑林",
         "孙悦", "徐平", "马青", "朱远", "胡兰", "郭新", "何晨", "高峰",
         "林夏", "罗凡", "梁宇", "宋丽", "唐昕", "许然", "韩松", "冯洁",
         "曹颖", "邓辉", "彭月", "蒋舟", "蔡阳", "潘秋", "田露", "董文")
PLACES = ("北京", "上海", "杭州", "成都", "南京", "武汉", "西安", "苏州",
          "广州", "深圳", "天津", "重庆", "长沙", "郑州", "济南", "青岛",
          "福州", "厦门", "南昌", "合肥", "太原", "石家庄", "沈阳", "大连",
          "长春", "哈尔滨", "贵阳", "昆明", "南宁", "海口", "兰州", "银川")


def digest(value) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_row(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def make_seeds(count: int, seed: int) -> list[dict]:
    if not 1 <= count <= 1000:
        raise ValueError("The template pilot supports 1..1000 texts; diversify templates before scaling further")
    rng = random.Random(seed)
    schedule = [(kind, index) for index in range(8) for kind in TEMPLATES]
    split_by_template = {}
    family_for_kind = {kind: "basic" if kind in {"clean", "filler", "stutter"} else kind for kind in TEMPLATES}
    for family in sorted(set(family_for_kind.values())):
        indices = list(range(8))
        rng.shuffle(indices)
        split_by_template.update({(family, index): "train" if rank < 5 else "dev" if rank == 5 else "test"
                                  for rank, index in enumerate(indices)})
    seen, rows = set(), []
    for index in range(count):
        kind, template = schedule[index % len(schedule)]
        for _attempt in range(10000):
            name, other_name = rng.sample(NAMES, 2)
            number, other_number = rng.sample(range(1, 1000), 2)
            slots = dict(name=name, other_name=other_name, place=rng.choice(PLACES),
                         number=number, other_number=other_number, amount=number * 10,
                         other_amount=other_number * 10, decimal=f"{number // 10}.{number % 10}")
            spoken, intended = [text.format(**slots) for text in TEMPLATES[kind][template]]
            identity = (kind, spoken, intended)
            if identity not in seen:
                seen.add(identity)
                break
        else:
            raise ValueError(f"Template {kind}:{template} exhausted; add templates rather than duplicating texts")
        family = family_for_kind[kind]
        domain = "medical" if (kind in {"clean", "filler", "stutter"} and template == 6) or (kind == "protected_content" and template == 7) else "general"
        record = dict(seed_id=f"synth_gate_{index:05d}", category=kind,
                      template_group=f"{family}:{template:02d}", spoken_text=spoken, intended_text=intended,
                      synthetic_split=split_by_template[family, template], domain=domain,
                      audio_script_verified=False, content_role="fictional_controlled_transcription_probe")
        record["seed_sha256"] = digest(record)
        rows.append(record)
    return rows


def prepare(args) -> None:
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refuse to overwrite a prepared dataset; use a new output directory")
    rows = make_seeds(args.count, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        append_row(out / "texts.jsonl", row)
    manifest = dict(schema_version=1, count=len(rows), seed=args.seed,
                    texts_sha256=file_hash(out / "texts.jsonl"), generator_sha256=file_hash(Path(__file__)),
                    categories=dict(Counter(row["category"] for row in rows)),
                    synthetic_splits=dict(Counter(row["synthetic_split"] for row in rows)),
                    unique_template_groups=len({row["template_group"] for row in rows}),
                    split_unit="template_family_including_all_slot_values_and_audio_variants",
                    reference_role="unverified_synthesis_scripts_not_human_gold",
                    intended_policy="clean_filler_and_marked_stutter_preserve_emphasis_and_facts",
                    scope="controlled_synthetic_development_not_real_speech_acceptance")
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def seeds(out: Path) -> list[dict]:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(out / "texts.jsonl") != manifest["texts_sha256"]:
        raise RuntimeError("Frozen seed texts changed; use a new output directory")
    rows = read_rows(out / "texts.jsonl")
    if len(rows) != manifest["count"] or len({row["seed_id"] for row in rows}) != len(rows):
        raise ValueError("Seed coverage is invalid")
    for row in rows:
        if digest({key: value for key, value in row.items() if key != "seed_sha256"}) != row["seed_sha256"]:
            raise ValueError("Seed identity changed")
    return rows


def assert_configuration(path: Path, value) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Resume configuration changed: {path.name}; use a new output directory")
    if not path.exists():
        write_json(path, value)


def preflight(args) -> None:
    import httpx
    status = dict(texts=len(seeds(args.output.resolve())), tts=None, asr=None)
    with httpx.Client(timeout=5) as client:
        for name, url in (("tts", args.tts_url), ("asr", args.asr_url)):
            try:
                payload = client.get(url.rstrip("/") + "/health").raise_for_status().json()
                if payload.get("ok") is not True and payload.get("status") != "ok":
                    raise ValueError("Health did not report readiness")
                status[name] = dict(available=True, url=url, health=payload)
            except Exception as error:
                status[name] = dict(available=False, url=url, error=str(error))
    status["tts_health_scope"] = "HTTP readiness only; first synthesis verifies actual model loading"
    write_json(args.output.resolve() / "preflight.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def audio_quality(path: Path) -> dict:
    import numpy as np
    import soundfile as sf
    wave, rate = sf.read(path, dtype="float32")
    if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError("Expected finite mono 16kHz speech")
    duration = len(wave) / rate
    rms = float(np.sqrt(np.mean(wave ** 2))) if len(wave) else 0
    if not 0.3 <= duration <= 90 or rms < 1e-5:
        raise ValueError("Empty, silent or out-of-range synthesis")
    clipping = float(np.mean(np.abs(wave) >= 0.999))
    if clipping > 0.2:
        raise ValueError("Severely clipped synthesis")
    return dict(duration_seconds=duration, rms=rms, clipping_fraction=clipping,
                acoustic_sanity_passed=True, audio_script_verified=False)


def synthesize(args) -> None:
    import httpx
    out = args.output.resolve()
    rows = seeds(out)
    if getattr(args, 'split', None):
        rows = [row for row in rows if row["synthetic_split"] == args.split]
    rows = rows[:args.limit] if args.limit else rows
    config = dict(tts_url=args.tts_url, mode="direct", language="Chinese", max_new_tokens=1024,
                  speaker="Vivian", voice_scope="fixed_qwen_custom_voice_speaker_not_verified_identity")
    with httpx.Client(timeout=args.timeout) as client:
        client.get(args.tts_url.rstrip("/") + "/health").raise_for_status()
        assert_configuration(out / "tts_configuration.json", config)
        existing = {row["seed_id"]: row for row in read_rows(out / "tts.jsonl")}
        (out / "audio").mkdir(exist_ok=True)
        for seed_row in rows:
            sid = seed_row["seed_id"]
            if sid in existing:
                old = existing[sid]
                if old["seed_sha256"] != seed_row["seed_sha256"] or file_hash(Path(old["audio_path"])) != old["audio_sha256"]:
                    raise RuntimeError(f"Cached synthesis changed: {sid}")
                continue
            original = out / "audio" / f"{sid}.tts.wav"
            canonical = out / "audio" / f"{sid}.wav"
            started = time.perf_counter()
            try:
                payload = dict(text=seed_row["spoken_text"], output_path=str(original),
                               mode="direct", language="Chinese", speaker="Vivian", max_new_tokens=1024)
                result = client.post(args.tts_url.rstrip("/") + "/v1/tts", json=payload).raise_for_status().json()
                produced = Path(result.get("audio_path") or original)
                if not produced.is_file():
                    raise FileNotFoundError("TTS must share a local filesystem with this runner: " + str(produced))
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(produced),
                                "-ac", "1", "-ar", "16000", str(canonical)], check=True)
                quality = audio_quality(canonical)
                append_row(out / "tts.jsonl", dict(**seed_row, audio_path=str(canonical),
                    audio_sha256=file_hash(canonical), quality=quality, tts_response=result,
                    tts_configuration_sha256=digest(config), elapsed_seconds=time.perf_counter() - started))
                print(f"TTS {sid}: {quality['duration_seconds']:.2f}s", flush=True)
            except Exception as error:
                append_row(out / "failures.jsonl", dict(stage="synthesize", seed_id=sid,
                    error_type=type(error).__name__, error=str(error), fallback_used=False))
                raise


def decode_wave(client, url: str, pcm: bytes, chunk_samples: int = 16000) -> tuple[list, dict]:
    sid = client.post(url + "/stream/start", params=dict(language="Chinese", chunk_seconds=0.5)).raise_for_status().json()["session_id"]
    chunks = []
    try:
        for offset in range(0, len(pcm), chunk_samples * 4):
            chunks.append(client.post(url + "/stream/chunk", params=dict(session_id=sid),
                                      content=pcm[offset:offset + chunk_samples * 4]).raise_for_status().json())
        final = client.post(url + "/stream/finish", params=dict(session_id=sid)).raise_for_status().json()
        if not isinstance(final.get("text"), str) or not final["text"].strip():
            raise ValueError("Empty ASR result cannot be treated as a successful sample")
        return chunks, final
    except BaseException:
        try:
            client.post(url + "/stream/cancel", params=dict(session_id=sid))
        except Exception:
            pass
        raise


def decode(args) -> None:
    import httpx
    import numpy as np
    import soundfile as sf
    out = args.output.resolve()
    allowed = {row["seed_id"]: row for row in seeds(out)}
    rows = read_rows(out / "tts.jsonl")
    if getattr(args, 'split', None):
        rows = [row for row in rows if row["synthetic_split"] == args.split]
    if not rows:
        raise RuntimeError("No successful TTS audio yet; run synthesize first")
    rows = rows[:args.limit] if args.limit else rows
    config = dict(asr_url=args.asr_url, transport="float32_mono_16000",
                  chunk_seconds=1.0, variants=args.variants, noise_snr_db=args.noise_snr,
                  seed=json.loads((out / "manifest.json").read_text())["seed"], asr_context="none")
    assert_configuration(out / "asr_configuration.json", config)
    existing = {(row["seed_id"], row["variant"]): row for row in read_rows(out / "asr.jsonl")}
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(args.asr_url.rstrip("/") + "/health").raise_for_status().json()
        append_row(out / "asr_health_history.jsonl", dict(health=health, observed_unix=time.time()))
        for tts in rows:
            sid = tts["seed_id"]
            if sid not in allowed or tts["seed_sha256"] != allowed[sid]["seed_sha256"]:
                raise RuntimeError("TTS seed metadata mismatch")
            source = Path(tts["audio_path"])
            if file_hash(source) != tts["audio_sha256"]:
                raise RuntimeError("Synthesized audio changed")
            audio_quality(source)
            wave, _rate = sf.read(source, dtype="float32")
            for variant in args.variants:
                cached = existing.get((sid, variant))
                if cached:
                    if cached["seed_sha256"] != allowed[sid]["seed_sha256"] or file_hash(Path(cached["audio_path"])) != cached["audio_sha256"]:
                        raise RuntimeError("Cached ASR input changed")
                    continue
                audio = wave.copy()
                if variant == "noise":
                    rng = np.random.default_rng(int(digest([config["seed"], sid, variant])[:16], 16))
                    noise = rng.standard_normal(audio.shape).astype("float32")
                    audio += noise * np.sqrt(np.mean(audio ** 2)) / (10 ** (args.noise_snr / 20)) / np.sqrt(np.mean(noise ** 2))
                peak = float(np.max(np.abs(audio)))
                if peak > 0.99:
                    audio *= 0.99 / peak
                path = out / "audio" / f"{sid}.{variant}.wav"
                sf.write(path, audio, 16000, subtype="FLOAT")
                pcm = audio.astype("<f4").tobytes()
                started = time.perf_counter()
                try:
                    chunks, final = decode_wave(client, args.asr_url.rstrip("/"), pcm)
                    append_row(out / "asr.jsonl", dict(**allowed[sid], id=sid, data_kind="synthetic_tts",
                        recording_group=f"synthetic_text:{sid}",
                        speaker=f"synthetic_qwen_tts:{tts.get('tts_response', {}).get('speaker', 'unknown')}",
                        dataset="Controlled synthetic TTS gate development", variant=variant,
                        audio_path=str(path), audio_sha256=file_hash(path), pcm_sha256=digest(pcm),
                        original_audio_sha256=tts["audio_sha256"], reference=tts["spoken_text"],
                        raw_text=final["text"], chunk_payloads=chunks, final_payload=final,
                        asr_configuration_sha256=digest(config), tts_configuration_sha256=tts["tts_configuration_sha256"],
                        acoustic_quality=tts["quality"], noise_snr_db=args.noise_snr if variant == "noise" else None,
                        asr_elapsed_seconds=time.perf_counter() - started))
                    print(f"ASR {sid} {variant}: {final['text']}", flush=True)
                except Exception as error:
                    append_row(out / "failures.jsonl", dict(stage="decode", seed_id=sid, variant=variant,
                        error_type=type(error).__name__, error=str(error), fallback_used=False))
                    raise
                if args.pause:
                    time.sleep(args.pause)


def prepare_pairs(args) -> None:
    from experiments.scripts import collect_gate_pairs as collector
    from types import SimpleNamespace
    out = args.output.resolve()
    seeds(out)
    if not read_rows(out / "asr.jsonl"):
        raise RuntimeError("No ASR results; run synthesize and decode first")
    paired = args.pairs_dir.resolve() if getattr(args, "pairs_dir", None) else out / "pairs"
    collector.prepare(SimpleNamespace(output=paired, sources=[out / "asr.jsonl"],
        long_results=None, no_demo_sources=True, entity_db=args.entity_db, library=args.library,
        window_count=args.window_count, review_count=args.review_count, seed=args.seed,
        max_window_chars=200, max_review_chars=360))


def lexical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] not in {"P", "Z", "S"} and not c.isspace())


def weak_label(baseline: str, refined: str, target: str, normalize) -> tuple[str, str]:
    if baseline == refined:
        return "neutral", "identical_final_output"
    before, after, expected = [lexical(normalize(text)) for text in (baseline, refined, target)]
    if before == after:
        return "uncertain", "format_or_surface_change_not_semantic_gold"
    if after == expected and before != expected:
        return "benefit", "exact_intended_script_recovery_unverified_audio"
    if before == expected and after != expected:
        return "harm", "departure_from_intended_script_unverified_audio"
    return "uncertain", "neither_branch_exactly_matches_intended_script"


def label_pairs(args) -> None:
    from experiments.scripts import collect_gate_pairs as collector
    out = args.output.resolve()
    by_seed = {row["seed_id"]: row for row in seeds(out)}
    paired = args.pairs_dir.resolve() if getattr(args, "pairs_dir", None) else out / "pairs"
    protocol = collector.verify_frozen(paired)
    cases = {row["case_id"]: row for row in read_rows(paired / "cases.jsonl")}
    pairs = read_rows(paired / "pairs.jsonl")
    if not pairs or len({row["case_id"] for row in pairs}) != len(pairs):
        raise ValueError("No completed pairs or duplicate pair IDs")
    if set(cases) != {row["case_id"] for row in pairs}:
        raise ValueError("Complete the selected pairs before labeling; no partial training export")
    web = collector.load_system(paired)
    norm = web.ZhITNNormalizer(paired / "native/libzh_itn_bridge.so")
    records = []
    for pair in pairs:
        case = cases[pair["case_id"]]
        meta = case.get("synthetic_metadata")
        if case["data_kind"] != "synthetic_tts" or not meta:
            raise ValueError("Refuse to label real recordings with synthesis-script supervision")
        seed_row = by_seed[meta["seed_id"]]
        if meta["seed_sha256"] != seed_row["seed_sha256"] or any(meta[key] != seed_row[key] for key in
                ("intended_text", "spoken_text", "template_group", "synthetic_split", "audio_script_verified")):
            raise ValueError("Frozen synthesis metadata mismatch")
        for arm in pair["arms"].values():
            if arm.get("session_stats", {}).get("failed_call_count", 0) or any(call.get("error_type") for call in
                    arm.get("logical_calls", []) + arm.get("plausibility_calls", [])):
                raise ValueError("Failed operation is not valid supervision")
        baseline = pair["arms"]["rules"]["comparison_text"]
        refined = pair["arms"]["always"]["comparison_text"]
        label, basis = weak_label(baseline, refined, seed_row["intended_text"], lambda text: norm.normalize(text).text)
        feature_input = dict(task_kind=case["task_kind"], domain=case["domain"], data_kind=case["data_kind"])
        if case["task_kind"] == "window_refine":
            feature_input.update(raw_text=case["raw_text"], baseline_text=case["baseline_text"],
                available_context=case["context_available_before"], window_chunk_count=case["window_chunk_count"])
        else:
            feature_input.update(baseline_text=case["candidate"]["context"],
                available_context=case["candidate"]["context"], candidate_local_start=case["candidate_local_start"],
                candidate={key: case["candidate"][key] for key in ("source", "target", "kind")})
        records.append(dict(case_id=case["case_id"], seed_id=meta["seed_id"],
            recording_group=case["recording_group"], split_group=meta["template_group"], split=meta["synthetic_split"],
            input=feature_input, label=label, label_basis=basis,
            supervision_status="operation_effect_exact" if basis == "identical_final_output" else "synthetic_weak",
            eligible_for_weak_pretraining=label != "uncertain", human_gold=False,
            eligible_for_real_risk_calibration=False, audio_script_verified=False,
            baseline_sha256=digest(baseline), always_sha256=digest(refined),
            intended_script_sha256=digest(seed_row["intended_text"])))
    label_output = out if paired == out / "pairs" else paired
    destination = label_output / "weak_gate_labels.jsonl"
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    report = dict(total=len(records), labels=dict(Counter(row["label"] for row in records)),
                  weak_pretraining_eligible=sum(row["eligible_for_weak_pretraining"] for row in records),
                  script_verified=0, human_gold=0, real_risk_calibration_eligible=0,
                  protocol_sha256=digest(protocol), pairs_sha256=file_hash(paired / "pairs.jsonl"),
                  scope="synthetic_weak_pretraining_only_not_real_speech_acceptance")
    write_json(label_output / "weak_label_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "preflight", "synthesize", "decode", "prepare-pairs", "label"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/gate_synthetic_20261009")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--pairs-dir", type=Path, help="Use a new pair directory when expanding beyond the first frozen trial")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--tts-url", default="http://127.0.0.1:8011")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8766")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        help="Restrict TTS synthesis/ASR decode to one frozen source split.")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--variants", nargs="+", choices=["clean", "noise"], default=["clean", "noise"])
    parser.add_argument("--noise-snr", type=float, default=10)
    parser.add_argument("--window-count", type=int, default=10)
    parser.add_argument("--review-count", type=int, default=0)
    parser.add_argument("--entity-db", type=Path, default=ROOT.parent / "AgenticASR_entity_e5bedb3/data/entities.db")
    parser.add_argument("--library", type=Path, default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.timeout <= 0 or args.pause < 0 or not math.isfinite(args.noise_snr):
        parser.error("Limit, timeout, pause and noise SNR must be valid")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Audio variants must be unique")
    if args.window_count < 0 or args.review_count < 0 or args.window_count + args.review_count == 0:
        parser.error("Pair counts must select at least one case")
    {"prepare": prepare, "preflight": preflight, "synthesize": synthesize,
     "decode": decode, "prepare-pairs": prepare_pairs, "label": label_pairs}[args.stage](args)


if __name__ == "__main__":
    main()
