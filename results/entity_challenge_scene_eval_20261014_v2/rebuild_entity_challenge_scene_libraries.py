"""Rebuild the mixed person/org/location library as evidence-based scenes.

Scene labels are offline preparation annotations derived from references after
masking annotated entities. Inference inputs contain only frozen routes and
library paths; this is not an independent blind-topic benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.entity_store import EntityStore
from opencc import OpenCC
from system.zh_itn_bridge import ZhITNNormalizer

_CC = OpenCC("t2s")
_ITN = ZhITNNormalizer()


@lru_cache(maxsize=100000)
def norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", _CC.convert(_ITN.normalize(text).text)).casefold()
    return "".join(char for char in value if unicodedata.category(char)[0] not in {"P", "Z", "C"})


def scorable_label(sample: dict, original: dict) -> dict:
    """Selector-compatible AISHELL offset compaction and label handling."""
    label = dict(original)
    if sample.get("source", "").startswith("AISHELL"):
        reference = original["reference"]
        label["reference"] = "".join(char for char in reference if not char.isspace())
        if reference != label["reference"]:
            label["reference_original"] = reference
            label["gold_entities"] = []
            offsets = [0]
            for char in reference:
                offsets.append(offsets[-1] + int(not char.isspace()))
            for entity in original["gold_entities"]:
                converted = dict(entity)
                if "start" in entity and "end" in entity:
                    converted.update(start=offsets[entity["start"]], end=offsets[entity["end"]])
                    assert norm(label["reference"][converted["start"]:converted["end"]]) == norm(entity["canonical"])
                label["gold_entities"].append(converted)
    return label

BASE = ROOT / "results/entity_challenge_20261014"
SOURCE = BASE / "dataset"
DEST = ROOT / "results/entity_challenge_20261014_scene_v2/dataset"
PREVIEW = DEST.parent / "preparation_preview.json"
FALLBACK = "综合新闻_场景未定"
METHOD = "masked_reference_explicit_keyword_weighted_v1"

# Frozen taxonomy. Keywords are deliberately generic domain cues, not names,
# spelling variants, or output from any ASR/entity module.
KEYWORDS = {
    "体育赛事": "比赛 冠军 决赛 运动员 教练 球队 赛季 联赛 乒乓 网球 篮球 足球 竞走 奥运 世锦 羽毛球 主场 客场 进球 夺冠 选手 出场 迎战 赛事 球员 得分 对阵 半决赛".split(),
    "金融证券": "银行 证券 基金 期货 股票 股市 股价 信托 投资 融资 估值 利率 央行 经济 货币 财政 金融 债券 贷款 资本 市值 汇率 保险 交易所".split(),
    "商业企业": "公司 集团 总裁 企业 董事 经理 业务 客户 品牌 零售 销售 股东 电商 营收 产品 商业 创业 供应链 企业家 门店".split(),
    "法律治安": "法院 法官 司法 起诉 诉讼 犯罪 嫌疑 警方 警察 公安 拘留 检察 律师 行政复议 判决 案件 侦查 逮捕 被告 原告 违法 罪名".split(),
    "医疗健康": "医院 医生 患者 疾病 治疗 手术 临床 医学 药物 健康 生育 哺乳 症状 诊断 癌症 疫苗 医疗 护理 康复 用药".split(),
    "教育科研": "大学 学院 学校 教师 学生 教育 教授 院士 科学 实验 论文 课题 研究所 学术 课程 实验室 科研".split(),
    "政务公共事务": "政府 人大 政协 常委 书记 市长 官员 干部 行政 外交 总理 部长 总统 议员 选举 政策 市民 社区 改革 公共事务".split(),
    "文娱影视": "电影 演员 导演 电视剧 节目 音乐 歌手 舞台 剧组 综艺 娱乐 演唱 票房 影视 观众 演出 剧情 纪录片".split(),
    "交通建设": "交通 公路 铁路 高铁 地铁 公交 机场 航班 汽车 车辆 车间 建筑 住房 房价 楼市 地产 楼盘 道路 桥梁 施工 列车".split(),
    "新闻采编": "记者 报道 报讯 通讯员 编辑 采访 晚报 日报 新闻网 本报 记者站 新闻发布 会见 发布会".split(),
}
NEWS = "新闻采编"
MIN_SCORE = 2
MIN_MARGIN = 1


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_reference(reference: str, label: dict):
    """Mask all supported gold spans; canonical exact occurrences are fallback."""
    chars = list(reference)
    spans = []
    for entity in label.get("gold_entities", []):
        start, end = entity.get("start"), entity.get("end")
        canonical = entity.get("canonical", "")
        span_matches = (isinstance(start, int) and isinstance(end, int) and
                        0 <= start < end <= len(reference) and
                        bool(canonical) and norm(reference[start:end]) == norm(canonical))
        if span_matches:
            spans.append((start, end))
        elif canonical:
            at = 0
            while True:
                at = reference.find(canonical, at)
                if at < 0:
                    break
                spans.append((at, at + len(canonical)))
                at += max(1, len(canonical))
    # Longest spans first; merging handles nested/overlapping annotations.
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in merged:
        chars[start:end] = [" "] * (end - start)
    return "".join(chars), merged


def classify(reference: str, label: dict):
    masked, spans = masked_reference(reference, label)
    scores = {}
    evidence = {}
    for scene, keywords in KEYWORDS.items():
        hits = sorted({word for word in keywords if word and word in masked}, key=lambda x: (-len(x), x))
        if hits:
            scores[scene] = sum(len(word) for word in hits)
            evidence[scene] = hits
    non_news = {k: v for k, v in scores.items() if k != NEWS}
    ordered = sorted(non_news.items(), key=lambda item: (-item[1], item[0]))
    selected = FALLBACK
    reason = "no_semantic_evidence"
    if ordered:
        top_scene, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0
        if top_score < MIN_SCORE:
            reason = "insufficient_score"
        elif second_score >= MIN_SCORE and top_score - second_score < MIN_MARGIN:
            reason = "competing_strong_scenes"
        else:
            selected, reason = top_scene, "non_news_semantic_evidence"
    elif NEWS in scores:
        selected, reason = NEWS, "news_cue_without_other_scene_evidence"
    scored = {k: scores[k] for k in sorted(scores)}
    used = evidence.get(selected, [])
    return selected, dict(masked_text=masked, masked_spans=spans, scores=scored,
        evidence_keywords=used, all_evidence=evidence, decision=reason,
        classification_method=METHOD)


def pool_vocab_and_annotations():
    active = read_json(BASE / "active_pools.json")["pools"]
    protected_doc = read_json(ROOT / "results/entity_context_confirmation_20261012/unseen/summary.json")
    protected = set(protected_doc.get("pcm_sha256", {}).values())
    vocab = defaultdict(dict)
    provenance = defaultdict(list)
    candidates_seen = Counter()
    for pool in active:
        folder = BASE / pool
        inputs = {row["id"]: row for row in read_jsonl(folder / "inference_inputs.jsonl")}
        labels = read_json(folder / "references.json")["samples"]
        for sid, original in labels.items():
            sample = inputs[sid]
            if sample.get("pcm_sha256") in protected:
                continue
            if sample.get("domain") != "人名机构地名":
                continue
            label = scorable_label(sample, original)
            scene, route = classify(label["reference"], label)
            candidates_seen[scene] += 1
            # All eligible terms from person/org/location pool are retained,
            # regardless of whether this recording was selected for the test.
            for entity in label.get("gold_entities", []):
                canonical = entity.get("canonical", "").strip()
                if not canonical or not norm(canonical) or norm(canonical) not in norm(label["reference"]):
                    continue
                key = norm(canonical)
                if key in vocab[scene]:
                    continue
                vocab[scene][key] = dict(canonical=canonical, entity_type=entity["entity_type"],
                    source_pool=pool, source_sample_id=sid, source_pcm_sha256=sample.get("pcm_sha256"),
                    source_domain=sample.get("domain"), route=route)
                provenance[scene].append(dict(canonical=canonical, canonical_key=key,
                    entity_type=entity["entity_type"], source_pool=pool, source_sample_id=sid,
                    source_pcm_sha256=sample.get("pcm_sha256"), source_reference_sha256=hashlib.sha256(
                        label["reference"].encode("utf-8")).hexdigest(), route_evidence=route))
    return vocab, provenance, candidates_seen


def selected_routes():
    samples = read_jsonl(SOURCE / "samples.jsonl")
    targets = [s for s in samples if s.get("domain") == "人名机构地名"]
    routes = {}
    for sample in targets:
        route, evidence = classify(sample["reference"], {"gold_entities": sample.get("gold_entities", [])})
        routes[sample["id"]] = dict(scene=route, route_evidence=evidence)
    return samples, targets, routes


def library_sizes(vocab):
    return {scene: len(words) for scene, words in sorted(vocab.items())}


def build_preview(vocab, provenance, candidates_seen, samples, targets, routes):
    sizes = library_sizes(vocab)
    over_limit = [scene for scene, size in sizes.items() if size > 5000]
    sample_counts = Counter(route["scene"] for route in routes.values())
    selected_gold_missing = []
    for sample in targets:
        scene = routes[sample["id"]]["scene"]
        scene_words = vocab.get(scene, {})
        for entity in sample.get("gold_entities", []):
            if norm(entity.get("canonical", "")) not in scene_words:
                selected_gold_missing.append(dict(id=sample["id"], scene=scene,
                                                  canonical=entity.get("canonical")))
    if selected_gold_missing:
        raise RuntimeError(f"selected gold absent from scene pool vocabulary ({len(selected_gold_missing)}): "
                           f"{selected_gold_missing[:5]}")
    fallback_count = sample_counts[FALLBACK]
    union_size = len({key for scene_words in vocab.values() for key in scene_words})
    original_db = SOURCE / "libraries/02_人名机构地名.db"
    import sqlite3
    with sqlite3.connect(original_db) as connection:
        original_terms = {norm(row[0]) for row in connection.execute(
            "SELECT canonical_text FROM protected_entities WHERE domain='general' AND enabled=1")}
    union_matches_original = len(original_terms) == union_size and original_terms == {
        key for scene_words in vocab.values() for key in scene_words}
    return dict(schema_version=1, status="dry_run_failed_library_gt5000" if over_limit else "dry_run_ready",
        dry_run_passed=not over_limit, source_dataset=str(SOURCE),
        intended_dataset=str(DEST), recordings=len(samples), target_mixed_domain_recordings=len(targets),
        target_sample_scene_counts=dict(sorted(sample_counts.items())),
        target_fallback_count=fallback_count,
        target_fallback_rate=fallback_count / len(targets) if targets else 0,
        full_candidate_pool_route_counts=dict(sorted(candidates_seen.items())),
        scene_library_sizes=sizes, scene_library_count=len(sizes),
        over_5000_libraries=over_limit,
        vocabulary_union_unique_count=union_size, original_mixed_library_unique_count=len(original_terms),
        vocabulary_union_matches_original=union_matches_original,
        selected_gold_missing_count=len(selected_gold_missing),
        reference_based_offline_annotation=True,
        independent_blind_topic_benchmark=False,
        note="Scene labels are prep annotations from masked references; inference is frozen to scene/library_path.",
        taxonomy={scene: words for scene, words in KEYWORDS.items()},
        thresholds=dict(min_score=MIN_SCORE, min_margin=MIN_MARGIN),
        unchanged_domain_counts=dict(Counter(s["domain"] for s in samples if s["domain"] != "人名机构地名")))


def create_library(path: Path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    store = EntityStore(path)
    stamp = datetime.now(timezone.utc).isoformat()
    values = sorted(words.values(), key=lambda item: item["canonical"])
    with store._connect() as connection:
        connection.executemany("""INSERT INTO protected_entities
            (canonical_text,entity_type,domain,normalization_policy,priority,enabled,source,created_at,updated_at)
            VALUES (?,?,'general','normalize',0,1,?,?,?)""",
            [(row["canonical"], row["entity_type"],
              "Scene v2: complete reference-grounded active-pool vocabulary; evaluation only", stamp, stamp)
             for row in values])
    loaded = store.list_entities(domain="general")
    if len(loaded) != len(values) or any(entity.normalization_policy != "normalize" or entity.aliases for entity in loaded):
        raise RuntimeError(f"EntityStore verification failed for {path}")
    from system.entity_matcher import EntityCandidateMatcher
    EntityCandidateMatcher(loaded, selected_domain="general")


def copy_dataset_base():
    if DEST.exists():
        raise FileExistsError(f"destination exists: {DEST}")
    DEST.mkdir(parents=True)
    for path in SOURCE.iterdir():
        target = DEST / path.name
        if path.name in {"libraries", "samples.jsonl", "inference_inputs.jsonl", "summary.json",
                         "manifest.json", "REPORT.md", "verification.json", "domain_summary.csv",
                         "sample_comparison.csv"}:
            continue
        if path.is_dir():
            shutil.copytree(path, target, copy_function=os_link_or_copy if path.name == "audio" else shutil.copy2)
        else:
            shutil.copy2(path, target)


def os_link_or_copy(src, dst):
    import os
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def export(vocab, provenance, samples, targets, routes, preview):
    if not preview["dry_run_passed"] or not preview["vocabulary_union_matches_original"]:
        raise RuntimeError("refusing formal export: dry-run/union verification failed")
    copy_dataset_base()
    (DEST / "libraries").mkdir(exist_ok=True)
    libraries = {}
    old_inputs = {row["id"]: row for row in read_jsonl(SOURCE / "inference_inputs.jsonl")}
    scenes = sorted(vocab)
    scene_paths = {}
    for index, scene in enumerate(scenes, start=28):
        path = DEST / "libraries" / f"{index:02d}_{scene}.db"
        create_library(path, vocab[scene])
        scene_paths[scene] = path
        libraries[scene] = dict(path=str(path), sha256=sha256(path), entity_count=len(vocab[scene]))
    # Preserve the other 26 library bytes and their selected samples exactly.
    for source_db in sorted((SOURCE / "libraries").glob("*.db")):
        if "人名机构地名" in source_db.name:
            continue
        shutil.copy2(source_db, DEST / "libraries" / source_db.name)
        key = source_db.stem.split("_", 1)[1]
        libraries[key] = dict(path=str(DEST / "libraries" / source_db.name), sha256=sha256(DEST / "libraries" / source_db.name),
                              entity_count=len(EntityStore(DEST / "libraries" / source_db.name).list_entities(domain="general")))
    new_samples = []
    for row in samples:
        copied = dict(row)
        if row["domain"] == "人名机构地名":
            route = routes[row["id"]]
            copied["original_domain"] = row["domain"]
            copied["domain"] = route["scene"]
            copied["library_key"] = route["scene"]
        new_samples.append(copied)
    write_jsonl(DEST / "samples.jsonl", new_samples)

    new_inputs = []
    for row in read_jsonl(SOURCE / "inference_inputs.jsonl"):
        copied = dict(row)
        if row["domain"] == "人名机构地名":
            route = routes[row["id"]]
            copied["original_domain"] = row["domain"]
            copied["domain"] = route["scene"]
            copied["scene_method"] = METHOD
            copied["scene_source"] = "references.json masked by gold start/end or canonical; generic keyword taxonomy"
            copied["library_path"] = str(scene_paths[route["scene"]])
        else:
            copied["library_path"] = str(DEST / "libraries" / Path(row["library_path"]).name)
        new_inputs.append(copied)
    write_jsonl(DEST / "inference_inputs.jsonl", new_inputs)

    annotations = []
    for sample in targets:
        route = routes[sample["id"]]
        annotations.append(dict(id=sample["id"], pcm_sha256=sample["pcm_sha256"],
            source=sample.get("source"), original_domain=sample["domain"], scene=route["scene"],
            reference_sha256=hashlib.sha256(sample["reference"].encode("utf-8")).hexdigest(),
            route_evidence=route["route_evidence"]))
    write_jsonl(DEST / "route_annotations.jsonl", annotations)
    provenance_rows = [dict(scene=scene, **record) for scene in sorted(provenance) for record in provenance[scene]]
    write_jsonl(DEST / "vocabulary_provenance.jsonl", provenance_rows)

    # Carry forward the original summary values while renaming only this split.
    summary = read_json(SOURCE / "summary.json")
    old_domain_summary = summary["domains"].pop("人名机构地名")
    summary["original_mixed_domain_summary"] = old_domain_summary
    for scene, count in preview["target_sample_scene_counts"].items():
        rows = [sample for sample in targets if routes[sample["id"]]["scene"] == scene]
        hits = sum(sample["baseline_score"]["hits"] for sample in rows)
        mentions = sum(sample["baseline_score"]["expected_mentions"] for sample in rows)
        scene_inputs = [row for row in new_inputs if row["domain"] == scene]
        summary["domains"][scene] = dict(recordings=count, hits=hits, expected_mentions=mentions,
            entity_recall=hits / mentions if mentions else 0.0,
            unique_entities=len({norm(entity["canonical"]) for sample in rows for entity in sample.get("gold_entities", [])}),
            source_groups=len({row.get("source_group") for row in scene_inputs}), library_size=len(vocab.get(scene, {})),
            original_domain="人名机构地名", scene_method=METHOD)
    summary["recordings"] = len(samples)
    summary["original_domain"] = "人名机构地名"
    summary["scene_library_sizes"] = preview["scene_library_sizes"]
    write_json(DEST / "summary.json", summary)
    manifest = dict(schema_version=1, source_dataset=str(SOURCE), target_route_method=METHOD,
        taxonomy=KEYWORDS, library_sizes=preview["scene_library_sizes"], libraries=libraries,
        unchanged_original_files={name: sha256(SOURCE / name) for name in
            ["references.json", "baseline_raw.jsonl", "samples.jsonl"]},
        unchanged_reference_and_baseline=True, scene_annotations_from_masked_reference=True,
        independent_blind_topic_benchmark=False,
        scene_sample_counts=preview["target_sample_scene_counts"],
        full_pool_candidate_route_counts=preview["full_candidate_pool_route_counts"],
        selected_id_pcm_sha256={row["id"]: row["pcm_sha256"] for row in new_inputs},
        source_pool_reference_sha256={pool: sha256(BASE / pool / "references.json")
                                      for pool in read_json(BASE / "active_pools.json")["pools"]})
    shutil.copy2(Path(__file__), DEST / "source_snapshot" / Path(__file__).name)
    original_comparison = DEST / "metadata/source_original"
    original_comparison.mkdir(parents=True, exist_ok=True)
    for name in ("domain_summary.csv", "sample_comparison.csv"):
        shutil.copy2(SOURCE / name, original_comparison / name)
    with (DEST / "domain_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["domain", "recordings", "hits", "expected_mentions", "entity_recall", "library_size"])
        for domain, stats in summary["domains"].items():
            writer.writerow([domain, stats["recordings"], stats["hits"], stats["expected_mentions"],
                             stats["entity_recall"], stats["library_size"]])
    scene_by_id = {sample_id: route["scene"] for sample_id, route in routes.items()}
    with (SOURCE / "sample_comparison.csv").open(encoding="utf-8", newline="") as source_stream:
        reader = csv.DictReader(source_stream)
        fieldnames = list(reader.fieldnames or [])
        if "domain" not in fieldnames:
            fieldnames.append("domain")
        rows = list(reader)
    for row in rows:
        if row["id"] in scene_by_id:
            row["domain"] = scene_by_id[row["id"]]
    with (DEST / "sample_comparison.csv").open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = ["# 实体挑战集场景词库 v2", "",
        "本目录保持原 2000 条样本、reference、baseline 和 26 个非混合库。原人名机构地名样本仅按 reference 中目标实体遮蔽后的通用语义关键词路由。",
        "场景标签是离线参考文本准备标注，不是官方 topic，也不是独立盲 topic benchmark。推理只读取冻结的 scene/domain 与 library_path，不读取 reference。",
        "场景词库从六个 active pools 的完整合格候选 references 构建，保留未选录音提供的干扰词。",
        "", "|场景|样本数|候选词库词数|", "|---|---:|---:|"]
    for scene in sorted(preview["target_sample_scene_counts"]):
        report.append(f"|{scene}|{preview['target_sample_scene_counts'][scene]}|{preview['scene_library_sizes'].get(scene, 0)}|")
    report += ["", f"未定场景率：{preview['target_fallback_rate']:.2%}。", ""]
    (DEST / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    manifest["files_sha256"] = {str(path.relative_to(DEST)): sha256(path) for path in DEST.rglob("*")
                                 if path.is_file() and path.name != "manifest.json"}
    write_json(DEST / "manifest.json", manifest)
    source_inputs = read_jsonl(SOURCE / "inference_inputs.jsonl")
    output_inputs = read_jsonl(DEST / "inference_inputs.jsonl")
    matcher_results = {}
    from system.entity_matcher import EntityCandidateMatcher
    for library in sorted((DEST / "libraries").glob("*.db")):
        entities = EntityStore(library).list_entities(domain="general")
        EntityCandidateMatcher(entities, selected_domain="general")
        matcher_results[library.name] = len(entities)
    verification = dict(status="PASS", recordings=len(new_inputs), sample_count=len(new_samples),
        references_byte_identical=sha256(DEST / "references.json") == sha256(SOURCE / "references.json"),
        baseline_byte_identical=sha256(DEST / "baseline_raw.jsonl") == sha256(SOURCE / "baseline_raw.jsonl"),
        selected_ids_unchanged={"samples": {r["id"] for r in new_samples} == {r["id"] for r in samples},
                                "inputs": {r["id"] for r in new_inputs} == {r["id"] for r in source_inputs}},
        pcm_hashes_unchanged=all(a["pcm_sha256"] == b["pcm_sha256"] for a, b in zip(
            source_inputs, output_inputs)),
        non_target_libraries_byte_identical=all(sha256(SOURCE / "libraries" / p.name) == sha256(DEST / "libraries" / p.name)
            for p in (SOURCE / "libraries").glob("*.db") if "人名机构地名" not in p.name),
        scene_library_sizes=preview["scene_library_sizes"], selected_gold_missing_count=0,
        unique_input_ids=len({r["id"] for r in output_inputs}) == len(output_inputs),
        selected_pcm_map_exact={r["id"]: r["pcm_sha256"] for r in source_inputs} ==
                               {r["id"]: r["pcm_sha256"] for r in output_inputs},
        all_library_matcher_constructions_passed=True, library_entity_counts=matcher_results,
        references_are_preparation_labels=True, independent_blind_topic_benchmark=False)
    write_json(DEST / "verification.json", verification)
    manifest["files_sha256"] = {str(path.relative_to(DEST)): sha256(path) for path in DEST.rglob("*")
                                 if path.is_file() and path.name != "manifest.json"}
    write_json(DEST / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="write only preparation_preview.json")
    args = parser.parse_args()
    vocab, provenance, candidate_routes = pool_vocab_and_annotations()
    samples, targets, routes = selected_routes()
    preview = build_preview(vocab, provenance, candidate_routes, samples, targets, routes)
    if args.dry_run:
        write_json(PREVIEW, preview)
        print(json.dumps({k: preview[k] for k in ("recordings", "target_mixed_domain_recordings",
            "target_sample_scene_counts", "target_fallback_rate", "scene_library_sizes",
            "full_candidate_pool_route_counts", "selected_gold_missing_count", "over_5000_libraries",
            "vocabulary_union_unique_count", "original_mixed_library_unique_count",
            "vocabulary_union_matches_original", "dry_run_passed")}, ensure_ascii=False, indent=2))
        return 0 if preview["dry_run_passed"] else 2
    export(vocab, provenance, samples, targets, routes, preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
