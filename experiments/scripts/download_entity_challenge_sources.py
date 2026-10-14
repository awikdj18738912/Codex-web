#!/usr/bin/env python3
"""Fetch public SpeechIO subsets and export a transcript-only catalog.

Audio is kept in source parquet files. The JSONL catalog deliberately excludes
the nested ``audio`` column and any embedded bytes.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/aim0/data/datasets/ASR/entity_challenge_20261014/speechio")
REGISTRY_PATH = ROOT / "results/entity_challenge_20261014/source_registry.json"
CATALOG_PATH = ROOT / "results/entity_challenge_20261014/source_catalog.jsonl"
EXISTING_CATALOG_PATH = ROOT / "results/entity_challenge_20261014/source_catalog_existing_8.jsonl"
REPO = "yuekai/speechio_test"
REVISION = "bdba6aa3daeddb2996eaa31c44bbdf633a07a31f"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
MAX_WORKERS = 3
RETRIES = 5

# Topic labels transcribed from the SpeechColab official README table.
SUBSETS: dict[str, dict[str, str]] = {
    "SPEECHIO_ASR_ZH00000": {"name": "调试集", "scenario": "视频会议、论坛演讲", "topic": "经济、货币、金融"},
    "SPEECHIO_ASR_ZH00001": {"name": "新闻联播", "scenario": "新闻播报", "topic": "时政"},
    "SPEECHIO_ASR_ZH00002": {"name": "鲁豫有约", "scenario": "访谈电视节目", "topic": "名人工作/生活"},
    "SPEECHIO_ASR_ZH00003": {"name": "天下足球", "scenario": "专题电视节目", "topic": "足球"},
    "SPEECHIO_ASR_ZH00004": {"name": "罗振宇跨年演讲", "scenario": "会场演讲", "topic": "社会、人文、商业"},
    "SPEECHIO_ASR_ZH00005": {"name": "李永乐讲堂", "scenario": "在线教育", "topic": "科普"},
    "SPEECHIO_ASR_ZH00006": {"name": "王者荣耀；张大仙、骚白", "scenario": "直播", "topic": "游戏"},
    "SPEECHIO_ASR_ZH00007": {"name": "直播带货；李佳琪、薇娅", "scenario": "直播", "topic": "电商、美妆"},
    "SPEECHIO_ASR_ZH00008": {"name": "老罗语录", "scenario": "线下培训", "topic": "段子、做人"},
    "SPEECHIO_ASR_ZH00009": {"name": "故事FM", "scenario": "播客", "topic": "人生故事、见闻"},
    "SPEECHIO_ASR_ZH00010": {"name": "创业内幕", "scenario": "播客", "topic": "创业、产品、投资"},
    "SPEECHIO_ASR_ZH00011": {"name": "罗翔刑法法考", "scenario": "在线教育", "topic": "法律、法考"},
    "SPEECHIO_ASR_ZH00012": {"name": "张雪峰考研", "scenario": "在线教育", "topic": "考研、高校报考"},
    "SPEECHIO_ASR_ZH00013": {"name": "谷阿莫；牛叔说电影", "scenario": "短视频", "topic": "电影剪辑"},
    "SPEECHIO_ASR_ZH00014": {"name": "贫穷料理；琼斯爱生活", "scenario": "短视频", "topic": "美食、烹饪"},
    "SPEECHIO_ASR_ZH00015": {"name": "单田芳《白眉大侠》", "scenario": "评书", "topic": "江湖、武侠"},
    "SPEECHIO_ASR_ZH00016": {"name": "德云社演出", "scenario": "剧场相声", "topic": "包袱段子"},
    "SPEECHIO_ASR_ZH00017": {"name": "吐槽大会", "scenario": "脱口秀电视节目", "topic": "明星糗事"},
    "SPEECHIO_ASR_ZH00018": {"name": "小猪佩奇；熊出没", "scenario": "少儿动画", "topic": "童话故事、日常"},
    "SPEECHIO_ASR_ZH00019": {"name": "CCTV5 NBA 转播", "scenario": "体育赛事解说", "topic": "篮球、NBA"},
    "SPEECHIO_ASR_ZH00020": {"name": "篮球人物", "scenario": "纪录片", "topic": "篮球明星、成长"},
    "SPEECHIO_ASR_ZH00021": {"name": "汽车之家评测", "scenario": "短视频", "topic": "汽车测评"},
    "SPEECHIO_ASR_ZH00022": {"name": "小艾大叔豪宅带看", "scenario": "短视频", "topic": "房地产、豪宅"},
    "SPEECHIO_ASR_ZH00023": {"name": "无聊开箱；Zealer 评测", "scenario": "短视频", "topic": "产品开箱评测"},
    "SPEECHIO_ASR_ZH00024": {"name": "付老师种植技术", "scenario": "短视频", "topic": "农业、种植"},
    "SPEECHIO_ASR_ZH00025": {"name": "石国鹏讲历史", "scenario": "线下培训", "topic": "历史、古希腊哲学"},
    "SPEECHIO_ASR_ZH00026": {"name": "张震鬼故事", "scenario": "广播节目", "topic": "鬼故事"},
}

OFFICIAL_README = "https://github.com/SpeechColab/Leaderboard/blob/master/README.md"
HF_DATASET = f"https://huggingface.co/datasets/{REPO}"
LICENSE_NOTE = (
    "Hugging Face dataset card declares Apache-2.0 for the repackaging. The original "
    "SpeechIO README says audio is crawled from public YouTube/TV/podcast sources, but "
    "does not establish a blanket redistribution license for those underlying works. "
    "Treat the audio as evaluation-only and preserve each source's applicable terms; "
    "do not infer that Apache-2.0 clears underlying programme copyrights."
)


def safe_request_metadata(subset: str) -> dict[str, Any]:
    response = requests.get(f"{API_ROOT}/{subset}", params={"expand": "true"}, timeout=40)
    response.raise_for_status()
    entries = response.json()
    parquet_entries = [x for x in entries if x.get("path", "").endswith(".parquet")]
    if len(parquet_entries) != 1:
        raise RuntimeError(f"Expected one parquet shard for {subset}; found {len(parquet_entries)}")
    entry = parquet_entries[0]
    lfs = entry.get("lfs") or {}
    sha256 = lfs.get("oid")
    size = lfs.get("size") or entry.get("size")
    if not sha256 or not isinstance(size, int):
        raise RuntimeError(f"Missing LFS SHA256 or size metadata for {subset}")
    if len(sha256) != 64:
        raise RuntimeError(f"Unexpected LFS SHA256 format for {subset}")
    return {"remote_path": entry["path"], "sha256": sha256, "size_bytes": size}


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def resolve_source_path(subset: str, destination: Path) -> Path | None:
    candidates = (
        ROOT / "results/speechio_professional_entity_eval_20261010/source" / f"{subset}.parquet",
        ROOT / "results/entity_reliable_data_20261012/source/speechio" / f"{subset}.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if destination.is_file():
        return destination
    return None


def download_one(subset: str, metadata: dict[str, Any]) -> tuple[str, Path, str]:
    destination = DATA_ROOT / f"{subset}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = resolve_source_path(subset, destination)
    if source is not None:
        actual_sha, actual_size = digest_file(source)
        if actual_size == metadata["size_bytes"] and actual_sha == metadata["sha256"]:
            return subset, source, "reused" if source != destination else "already_present"
        if source != destination:
            raise RuntimeError(f"Existing parquet failed integrity check for {subset}")
        # A corrupted destination can be fetched again; preserve it until a valid replacement exists.

    partial = destination.with_suffix(".parquet.partial")
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            response = requests.get(
                f"{RESOLVE_ROOT}/{metadata['remote_path']}",
                params={"download": "true"},
                headers=headers,
                stream=True,
                timeout=(20, 90),
            )
            try:
                if offset and response.status_code == 206:
                    mode = "ab"
                elif offset and response.status_code == 200:
                    # Server ignored Range; restart cleanly to avoid appending a duplicate full body.
                    mode = "wb"
                else:
                    response.raise_for_status()
                    mode = "wb"
                response.raise_for_status()
                with partial.open(mode) as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
            finally:
                response.close()

            actual_sha, actual_size = digest_file(partial)
            if actual_size == metadata["size_bytes"] and actual_sha == metadata["sha256"]:
                os.replace(partial, destination)
                return subset, destination, "downloaded"
            if actual_size > metadata["size_bytes"]:
                partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Integrity mismatch for {subset}: got {actual_size} bytes; expected {metadata['size_bytes']}"
            )
        except Exception as error:  # retry network and partial-transfer failures
            last_error = error
            if attempt < RETRIES:
                time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Download failed for {subset} after {RETRIES} attempts ({type(last_error).__name__})")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_catalog(subsets: list[str], assets: dict[str, dict[str, Any]], path: Path, label: str) -> int:
    temporary_catalog = path.with_suffix(path.suffix + ".tmp")
    row_count = 0
    try:
        with temporary_catalog.open("w", encoding="utf-8") as output:
            for subset in subsets:
                asset = assets[subset]
                parquet_path = Path(asset["local_parquet_path"])
                parquet_file = pq.ParquetFile(parquet_path)
                required = {"segment_id", "text", "duration", "speaker"}
                missing = required - set(parquet_file.schema_arrow.names)
                if missing:
                    raise RuntimeError(f"Missing catalog columns in {subset}: {sorted(missing)}")
                row_index = 0
                for batch in parquet_file.iter_batches(
                    batch_size=2048,
                    columns=["segment_id", "text", "duration", "speaker"],
                    use_threads=True,
                ):
                    columns = batch.to_pydict()
                    for index, segment_id in enumerate(columns["segment_id"]):
                        record = {
                            "subset": subset,
                            "row_index": row_index,
                            "segment_id": segment_id,
                            "text": columns["text"][index],
                            "duration": columns["duration"][index],
                            "speaker": columns["speaker"][index],
                            "parquet_path": str(parquet_path),
                            "domain_labels": SUBSETS[subset],
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        row_count += 1
                        row_index += 1
                print(f"{label} {subset} {row_index}", flush=True)
        os.replace(temporary_catalog, path)
    except Exception:
        temporary_catalog.unlink(missing_ok=True)
        raise
    return row_count


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checking HF LFS metadata for {len(SUBSETS)} public subsets at fixed revision.", flush=True)
    metadata: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(safe_request_metadata, subset): subset for subset in SUBSETS}
        for future in concurrent.futures.as_completed(futures):
            subset = futures[future]
            metadata[subset] = future.result()
            print(f"METADATA {subset} size={metadata[subset]['size_bytes']}", flush=True)

    assets: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for subset in SUBSETS:
        source = resolve_source_path(subset, DATA_ROOT / f"{subset}.parquet")
        if source is None:
            continue
        actual_sha, actual_size = digest_file(source)
        if actual_size != metadata[subset]["size_bytes"] or actual_sha != metadata[subset]["sha256"]:
            if source in {
                ROOT / "results/speechio_professional_entity_eval_20261010/source" / f"{subset}.parquet",
                ROOT / "results/entity_reliable_data_20261012/source/speechio" / f"{subset}.parquet",
            }:
                errors.append({"subset": subset, "error_type": "IntegrityError", "message": "Existing source failed HF LFS verification"})
                continue
        else:
            assets[subset] = {
                "subset": subset,
                "domain_labels": SUBSETS[subset],
                "local_parquet_path": str(source),
                "remote_path": metadata[subset]["remote_path"],
                "size_bytes": metadata[subset]["size_bytes"],
                "sha256_lfs_oid": metadata[subset]["sha256"],
                "verified": True,
                "provisioning": "reused" if source != DATA_ROOT / f"{subset}.parquet" else "already_present",
            }
    existing_subsets = [subset for subset in SUBSETS if subset in assets]
    if existing_subsets:
        early_rows = write_catalog(existing_subsets, assets, EXISTING_CATALOG_PATH, "EARLY_CATALOG_ROWS")
        # Publish a valid partial catalog immediately. The canonical file is replaced
        # with the complete 27-subset catalog when all downloads finish.
        write_catalog(existing_subsets, assets, CATALOG_PATH, "EARLY_SOURCE_CATALOG_ROWS")
        early_registry = {
            "dataset": REPO,
            "revision": REVISION,
            "dataset_page": HF_DATASET,
            "official_source": OFFICIAL_README,
            "purpose": "Partial early transcript catalog; contains only pre-existing verified parquet assets",
            "subset_count_expected": len(SUBSETS),
            "subset_count_verified": len(assets),
            "catalog_path": str(EXISTING_CATALOG_PATH),
            "catalog_rows": early_rows,
            "catalog_audio_bytes_included": False,
            "license": {"mirror_declared_license": "Apache-2.0", "underlying_audio_note": LICENSE_NOTE,
                        "hf_dataset_card": HF_DATASET, "official_source_readme": OFFICIAL_README},
            "assets": [assets[key] for key in SUBSETS if key in assets],
            "errors": errors,
        }
        write_json(REGISTRY_PATH, early_registry)
        print(f"EARLY_CATALOG_READY subsets={len(existing_subsets)} rows={early_rows} path={EXISTING_CATALOG_PATH}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_one, subset, metadata[subset]): subset
            for subset in SUBSETS if subset not in assets
        }
        for future in concurrent.futures.as_completed(futures):
            subset = futures[future]
            try:
                _, local_path, state = future.result()
                asset = {
                    "subset": subset,
                    "domain_labels": SUBSETS[subset],
                    "local_parquet_path": str(local_path),
                    "remote_path": metadata[subset]["remote_path"],
                    "size_bytes": metadata[subset]["size_bytes"],
                    "sha256_lfs_oid": metadata[subset]["sha256"],
                    "verified": True,
                    "provisioning": state,
                }
                assets[subset] = asset
                print(f"ASSET_READY {subset} ({state})", flush=True)
            except Exception as error:
                # Exception messages from HTTP clients can include redirected, signed URLs.
                errors.append({"subset": subset, "error_type": type(error).__name__, "message": "details omitted to avoid leaking request credentials"})
                print(f"ASSET_FAILED {subset} ({type(error).__name__})", flush=True)

    registry = {
        "dataset": REPO,
        "revision": REVISION,
        "dataset_page": HF_DATASET,
        "official_source": OFFICIAL_README,
        "purpose": "Chinese multi-domain entity challenge source and reference catalog",
        "subset_count_expected": len(SUBSETS),
        "subset_count_verified": len(assets),
        "license": {
            "mirror_declared_license": "Apache-2.0",
            "underlying_audio_note": LICENSE_NOTE,
            "hf_dataset_card": HF_DATASET,
            "official_source_readme": OFFICIAL_README,
        },
        "assets": [assets[key] for key in SUBSETS if key in assets],
        "errors": errors,
    }
    write_json(REGISTRY_PATH, registry)
    if errors:
        print(f"Registry written with {len(errors)} failed assets; catalog not emitted.", flush=True)
        return 2

    row_count = write_catalog(list(SUBSETS), assets, CATALOG_PATH, "CATALOG_ROWS")

    registry["catalog_path"] = str(CATALOG_PATH)
    registry["catalog_rows"] = row_count
    registry["catalog_fields"] = ["subset", "row_index", "segment_id", "text", "duration", "speaker", "parquet_path", "domain_labels"]
    registry["catalog_audio_bytes_included"] = False
    write_json(REGISTRY_PATH, registry)
    print(f"COMPLETE verified_subsets={len(assets)} catalog_rows={row_count}", flush=True)
    print(f"REGISTRY {REGISTRY_PATH}", flush=True)
    print(f"CATALOG {CATALOG_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Avoid printing exception URLs, which can include signed query strings.
        print(f"FATAL {type(error).__name__}; details omitted to avoid leaking request credentials", file=sys.stderr, flush=True)
        raise SystemExit(1)
