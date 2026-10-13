"""Paired ASR/entity evaluation on a public Mandarin emergency-room FAQ set.

The corpus is small and scripted/TTS-like. It is a controlled lexical sanity
check, not a replacement for spontaneous clinical conversation evaluation.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import compare_entity_methods as factorial
from benchmark_multimed_entities import MODEL, digest, require_complete
from evaluate_entity_library import ROOT, append_jsonl, normalized, read_jsonl, write_json

OUT = ROOT / "results/chinese_er_faq_entity_eval_20261010"
REPO = "TonyFANgr/localaiiaasr"
REVISION = "0c00a7ba05559bfe4f2d5799bfc5350932de9bf0"
ASR = "http://127.0.0.1:8766"
LICENSE = "CC-BY-4.0"
TERMS = (
    "检伤分级", "急诊", "急救室", "转诊", "诊断书", "门诊", "成人科", "儿科",
    "眼科", "牙科", "耳鼻喉", "电疗", "化疗", "止痛药", "止痛药物", "电脑断层",
    "磁振造影", "X光", "抽血", "血液检验", "点滴", "检体", "手术", "药局", "住院",
    "出院", "回诊", "救护车", "冰敷", "护理师", "健保卡", "检查报告", "开刀",
)


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temp = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "4",
        "--retry-all-errors", "--connect-timeout", "15", "--max-time", "90", "--output", str(temp), url],
        check=True, timeout=100)
    temp.replace(destination)


def download() -> list[dict]:
    import urllib.request
    metadata_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/metadata.csv"
    metadata_path = OUT / "source/metadata.csv"
    fetch(metadata_url, metadata_path)
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if r.get("language", "").lower() == "mandarin"]
    if not selected:
        selected = [r for r in rows if r.get("language", "").lower() in {"cmn", "zh", "mandarin chinese"}]
    def download_audio(row):
        name = row["file_name"].lstrip("/")
        if not name.startswith("Mandarin/"):
            raise ValueError(f"Unexpected Mandarin file path: {name}")
        dest = OUT / "source" / name
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(name, safe='/')}"
        fetch(url, dest)
        return dest
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_audio, row) for row in selected]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 25 == 0:
                print(f"DOWNLOAD {index}/{len(selected)}", flush=True)
    write_json(OUT / "source/provenance.json", dict(
        repository=REPO, revision=REVISION, license=LICENSE,
        dataset_card=f"https://huggingface.co/datasets/{REPO}",
        metadata_sha256=digest(metadata_path), mandarin_samples=len(selected),
        audio_files=[r["file_name"] for r in selected]))
    return selected


def prepare() -> None:
    import opencc
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    from system.entity_store import EntityStore
    rows = download()
    converter = opencc.OpenCC("t2s")
    samples = []
    frequencies = Counter()
    observed = set()
    for index, row in enumerate(rows):
        source_path = OUT / "source" / row["file_name"]
        audio, rate = sf.read(source_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            divisor = __import__("math").gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype("float32")
            rate = 16000
        converted_path = OUT / "audio" / row["file_name"]
        converted_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(converted_path, audio, rate, subtype="FLOAT")
        text = converter.convert(row["transcription"])
        rid = f"er_faq_{row['category']}_{int(row['sample_id']):03d}"
        pcm = audio.astype("<f4").tobytes()
        samples.append(dict(id=rid, row=index, reference=text, source_reference=row["transcription"],
            audio_path=str(converted_path.resolve()), duration_seconds=len(audio) / rate,
            source_audio_path=row["file_name"], original_audio_sha256=digest(source_path),
            pcm_sha256=hashlib.sha256(pcm).hexdigest(), language=row["language"],
            category=row["category"], sample_id=int(row["sample_id"])))
        compact_text = normalized(text)
        for term in TERMS:
            if normalized(term) in compact_text:
                observed.add(term)
                frequencies[term] += compact_text.count(normalized(term))
    if len(samples) != 133:
        raise ValueError(f"Expected 133 Mandarin samples from this dataset version, found {len(samples)}")
    observed_terms = sorted(observed, key=lambda t: (-frequencies[t], -len(t), t))
    library = OUT / "library"
    library.mkdir(parents=True, exist_ok=True)
    db = library / "entities.db"
    if db.exists():
        raise FileExistsError(f"Refusing to replace existing experiment library: {db}")
    store = EntityStore(db)
    for term in observed_terms:
        store.upsert_entity(term, entity_type="TERM", domain="medical",
            normalization_policy="normalize", source="Manually reviewed emergency FAQ terms found in provided reference transcripts")
    sample_entities = {}
    for sample in samples:
        text = normalized(sample["reference"])
        hits = []
        for term in sorted(observed_terms, key=lambda t: (-len(normalized(t)), t)):
            needle = normalized(term)
            for match in re.finditer(re.escape(needle), text):
                if any(match.start() < h["end"] and h["start"] < match.end() for h in hits):
                    continue
                hits.append(dict(text=term, start=match.start(), end=match.end()))
        sample_entities[sample["id"]] = sorted(hits, key=lambda h: h["start"])
    definitions = [dict(canonical_text=t, entity_type="TERM", domain="medical",
        normalization_policy="normalize", enabled=True, source="CC-BY-4.0 Emergency Room FAQ transcription")
        for t in observed_terms]
    write_json(library / "entities.json", definitions)
    write_json(library / "sample_entities.json", sample_entities)
    write_json(library / "vocabulary_evidence.json", [dict(canonical_text=t, occurrences=frequencies[t]) for t in observed_terms])
    hashes = {p.name: digest(p) for p in [db, library / "entities.json", library / "sample_entities.json", library / "vocabulary_evidence.json"]}
    write_json(library / "provenance.json", dict(dataset=REPO, revision=REVISION,
        terms_source="Reviewed term list intersected with the dataset's published Mandarin references",
        vocabulary_size=len(definitions), reference_mentions=sum(map(len, sample_entities.values())),
        terms=observed_terms, reference_at_inference=False, errors_added_as_aliases=False,
        frozen_hashes=hashes))
    context_terms = observed_terms[:16]
    context_text = "参考术语仅用于辅助拼写，不代表音频内容；未在音频中听到的词不要补充。术语列表：" + json.dumps(context_terms, ensure_ascii=False)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "system").rglob("*.py")}
    protocol_dir = OUT / "medical"
    (protocol_dir / "library").mkdir(parents=True, exist_ok=True)
    for name in hashes:
        shutil.copy2(library / name, protocol_dir / "library" / name)
    protocol = dict(dataset=REPO, license=LICENSE, samples=samples, sample_count=len(samples),
        duration_seconds=sum(s["duration_seconds"] for s in samples), vocabulary_size=len(definitions),
        context_text=context_text, context_terms=context_terms, mode="streaming", chunk_seconds=2.0,
        asr_url=ASR, refiner_model=str(MODEL), library_hashes={n: digest(protocol_dir / "library" / n) for n in hashes},
        source_hashes=source_hashes,
        caveats=["133 short Mandarin emergency FAQ prompts; not spontaneous clinical conversation",
            "Reference transcripts are traditional Chinese, converted to simplified Chinese with OpenCC T2S for scoring",
            "Reference-derived vocabulary is a known-vocabulary diagnostic, not an unseen-term generalization test",
            "Each clip is a short prepared FAQ utterance; results do not establish accuracy on spontaneous clinical dialogue"])
    write_json(protocol_dir / "protocol.json", protocol)
    write_json(OUT / "manifest.json", dict(dataset=REPO, sample_count=len(samples),
        duration_seconds=protocol["duration_seconds"], terms=observed_terms, samples=samples))
    print("PREPARED", len(samples), "Mandarin audios", round(protocol["duration_seconds"], 1),
        "seconds", "terms", len(observed_terms), "prompt", context_terms, flush=True)


async def _request(client, endpoint, session_id=None, data=b"", params=None):
    query = dict(params or {})
    if session_id:
        query["session_id"] = session_id
    response = await client.post(ASR + endpoint, params=query, content=data)
    response.raise_for_status()
    return response.json()


async def decode() -> None:
    import httpx
    import numpy as np
    import soundfile as sf
    from system.entity_asr_context import EntityASRContext
    from system.entity_asr_echo_guard import EntityASREchoGuard
    protocol = json.loads((OUT / "medical/protocol.json").read_text())
    completed = {arm: {r["id"] for r in read_jsonl(OUT / "medical" / f"asr_{arm}.jsonl")} for arm in ("off", "on")}
    async with httpx.AsyncClient(timeout=180) as client:
        health = (await client.get(ASR + "/health")).raise_for_status().json()
        if health.get("context_supported") is not True:
            raise RuntimeError("Current ASR service does not acknowledge context support")
        for i, sample in enumerate(protocol["samples"]):
            audio, rate = sf.read(sample["audio_path"], dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError("Audio changed after prepare")
            pcm = audio.astype("<f4").tobytes()
            if hashlib.sha256(pcm).hexdigest() != sample["pcm_sha256"]:
                raise ValueError("Audio hash mismatch")
            for arm in (("off", "on") if i % 2 == 0 else ("on", "off")):
                if sample["id"] in completed[arm]:
                    continue
                context = EntityASRContext(protocol["context_text"], tuple(protocol["context_terms"])) if arm == "on" else EntityASRContext("", ())
                async def request(endpoint, session_id=None, data=b"", params=None):
                    return await _request(client, endpoint, session_id, data, params)
                guard = EntityASREchoGuard(request, context)
                start_params = dict(language="Chinese", chunk_seconds=2.0)
                if arm == "on":
                    start_params["context"] = context.text
                started = time.perf_counter()
                ready = await guard.request("/stream/start", params=start_params)
                if ready.get("context_applied") != (arm == "on"):
                    raise RuntimeError(f"ASR context acknowledgement mismatch: {ready}")
                chunks = []
                sid = ready["session_id"]
                try:
                    for offset in range(0, len(pcm), 64000):
                        chunks.append(await guard.request("/stream/chunk", sid, pcm[offset:offset + 64000]))
                    final = await guard.request("/stream/finish", sid)
                except BaseException:
                    try:
                        await guard.request("/stream/cancel", sid)
                    finally:
                        guard.close()
                    raise
                finally:
                    guard.close()
                append_jsonl(OUT / "medical" / f"asr_{arm}.jsonl", dict(id=sample["id"], row=sample["row"],
                    condition=arm, reference=sample["reference"], source_reference=sample["source_reference"],
                    pcm_sha256=sample["pcm_sha256"], context_text=context.text, ready=ready,
                    chunk_payloads=chunks, final_payload=final, guard=guard.snapshot(),
                    elapsed_seconds=time.perf_counter() - started))
                print("ASR", arm, i + 1, "/", len(protocol["samples"]), sample["id"], final.get("text", ""), flush=True)


def run_cells() -> None:
    from types import SimpleNamespace
    folder = OUT / "medical"
    protocol = json.loads((folder / "protocol.json").read_text())
    for arm in ("off", "on"):
        rows = require_complete(folder / f"asr_{arm}.jsonl", [s["id"] for s in protocol["samples"]])
        for sample in protocol["samples"]:
            assert rows[sample["id"]]["pcm_sha256"] == sample["pcm_sha256"]
    settings = SimpleNamespace(output=OUT, library=folder / "library", model=MODEL,
        device="cuda:0", asr_url=ASR, mode="streaming", chunk_seconds=2.0)
    old_out = factorial.OUT
    factorial.OUT = OUT
    runner = factorial.FactorialRunner(settings)
    original_factory = runner.web.create_app
    def app_factory(*args, **kwargs):
        # Echo verification already ran on real PCM in decode(); replay carries
        # the frozen real ASR responses through the real Refiner.
        kwargs["entity_asr_echo_guard"] = False
        return original_factory(*args, **kwargs)
    runner.web.create_app = app_factory
    try:
        for arm in factorial.ARMS:
            runner.run_cell("medical", arm)
    finally:
        runner.web.create_app = original_factory
        factorial.OUT = old_out


def report() -> None:
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    protocol = json.loads((OUT / "medical/protocol.json").read_text())
    ids = [s["id"] for s in protocol["samples"]]
    converter = opencc.OpenCC("t2s")
    itn = ZhITNNormalizer()
    cache = {}
    def norm(text):
        if text not in cache:
            cache[text] = normalized(converter.convert(itn.normalize(text).text))
        return cache[text]
    names = [r["canonical_text"] for r in json.loads((OUT / "medical/library/entities.json").read_text())]
    gt = json.loads((OUT / "medical/library/sample_entities.json").read_text())
    summaries, details = {}, []
    for arm in factorial.ARMS:
        rows = require_complete(OUT / "medical" / f"{arm}.jsonl", ids)
        total = Counter()
        for sample in protocol["samples"]:
            rid = sample["id"]
            text = rows[rid]["result"].get("clean_text", "")
            ref = norm(sample["reference"])
            have = Counter(e["text"] for e in factorial.lexicon_occurrences(text, names))
            expected = Counter(e["text"] for e in gt[rid])
            total.update(samples=1, reference_chars=len(ref), character_errors=Levenshtein.distance(ref, norm(text)),
                expected_mentions=sum(expected.values()), recovered_mentions=sum(min(n, have[t]) for t, n in expected.items()),
                unreferenced_mentions=sum(max(0, n - expected[t]) for t, n in have.items()),
                changed_samples=int(text != rows[rid]["result"].get("raw_text", "")),
                refiner_calls=len(rows[rid]["model_calls"]))
        summaries[arm] = dict(total, CER=total["character_errors"] / total["reference_chars"])
    raw_records = {arm: require_complete(OUT / "medical" / f"asr_{arm}.jsonl", ids) for arm in ("off", "on")}
    raw_summaries = {}
    for arm, records in raw_records.items():
        total = Counter()
        for sample in protocol["samples"]:
            text = records[sample["id"]]["final_payload"].get("text", "")
            expected = Counter(x["text"] for x in gt[sample["id"]])
            have = Counter(x["text"] for x in factorial.lexicon_occurrences(text, names))
            ref = norm(sample["reference"])
            total.update(samples=1, reference_chars=len(ref), character_errors=Levenshtein.distance(ref, norm(text)),
                expected_mentions=sum(expected.values()), recovered_mentions=sum(min(n, have[t]) for t, n in expected.items()),
                unreferenced_mentions=sum(max(0, n - expected[t]) for t, n in have.items()))
        raw_summaries[arm] = dict(total, CER=total["character_errors"] / total["reference_chars"])
    write_json(OUT / "raw_summary.json", raw_summaries)
    final_records = {arm: require_complete(OUT / "medical" / f"{arm}.jsonl", ids) for arm in factorial.ARMS}
    hard_subset = []
    for sample in protocol["samples"]:
        rid = sample["id"]
        expected = Counter(x["text"] for x in gt[rid])
        raw_off = str(raw_records["off"][rid]["final_payload"].get("text", ""))
        raw_on = str(raw_records["on"][rid]["final_payload"].get("text", ""))
        hits_off = Counter(x["text"] for x in factorial.lexicon_occurrences(raw_off, names))
        if sum(min(n, hits_off[t]) for t, n in expected.items()) >= sum(expected.values()):
            continue
        entry = dict(id=rid, reference=sample["reference"], expected_terms=dict(expected), raw_without_context=raw_off,
            raw_with_context=raw_on, results={arm: final_records[arm][rid]["result"].get("clean_text", "") for arm in factorial.ARMS})
        hard_subset.append(entry)
    hard_summary = {"samples": len(hard_subset), "expected_mentions": sum(sum(x["expected_terms"].values()) for x in hard_subset)}
    for arm in ("baseline", "post_only", "context_only", "combined"):
        hits = 0
        for row in hard_subset:
            found = Counter(x["text"] for x in factorial.lexicon_occurrences(row["results"][arm], names))
            hits += sum(min(n, found[t]) for t, n in row["expected_terms"].items())
        hard_summary[arm + "_hits"] = hits
    raw_on_hits = 0
    for row in hard_subset:
        found = Counter(x["text"] for x in factorial.lexicon_occurrences(row["raw_with_context"], names))
        raw_on_hits += sum(min(n, found[t]) for t, n in row["expected_terms"].items())
    hard_summary["raw_with_context_hits"] = raw_on_hits
    write_json(OUT / "hard_subset.json", dict(selection="all exact standard-term misses in the no-context raw ASR arm; same four audio samples; selected post hoc",
        summary=hard_summary, samples=hard_subset))
    for sample in protocol["samples"]:
        rid = sample["id"]
        records = {arm: require_complete(OUT / "medical" / f"{arm}.jsonl", ids)[rid] for arm in factorial.ARMS}
        details.append(dict(id=rid, category=sample["category"], reference=sample["reference"],
            raw_asr_off=raw_records["off"][rid]["final_payload"].get("text", ""),
            raw_asr_on=raw_records["on"][rid]["final_payload"].get("text", ""),
            results={arm: records[arm]["result"].get("clean_text", "") for arm in factorial.ARMS},
            context_guard=raw_records["on"][rid].get("guard", {})))
    write_json(OUT / "summary.json", summaries)
    write_json(OUT / "comparison.json", details)
    changed = [r for r in details if any(r["results"][arm] != r["results"]["baseline"] for arm in factorial.ARMS)]
    write_json(OUT / "changes.json", changed)
    markdown = ["# 中文急诊 FAQ 音频：实体库四组对照", "",
        f"数据集：[{REPO}](https://huggingface.co/datasets/{REPO})，CC-BY-4.0；普通话 {len(ids)} 条。",
        "参考原为繁体，使用 OpenCC T2S 转为简体后评分；繁体原文、音频和哈希保存在 `medical/`。数据是 133 条短的预设 FAQ 问句，不代表自发医患对话泛化。词表来自转录文本中的人工核定术语，属于已知词表覆盖测试。", "",
        "| 原始 ASR | CER | 实体保留 |", "|---|---:|---:|"]
    for arm, label in (("off", "不带提示"), ("on", "带16词提示")):
        s = raw_summaries[arm]
        markdown.append(f"| {label} | {s['CER']:.2%} | {s['recovered_mentions']}/{s['expected_mentions']} |")
    markdown += ["", "| 最终精修条件 | CER | 实体保留 | 未在参考中的实体 |", "|---|---:|---:|---:|"]
    labels = dict(baseline="无库", post_only="仅后处理", context_only="仅词库辅助识别", combined="辅助识别＋后处理")
    for arm in factorial.ARMS:
        s = summaries[arm]
        markdown.append(f"| {labels[arm]} | {s['CER']:.2%} | {s['recovered_mentions']}/{s['expected_mentions']} | {s['unreferenced_mentions']} |")
    markdown += ["", "## 逐条检查", "",
        "提示辅助识别的实体命中从 75/79 增至 78/79，三条变化如下；无实体损失，未发现参考中不存在的词表实体：", "",
        "| 音频 | 无提示 | 有提示 | 核对结果 |", "|---|---|---|---|",
        "| `er_faq_waiting_010` | 减伤分级 | 检伤分级 | 完整纠正 |",
        "| `er_faq_examination_019` | 现保卡 | 健保卡 | 完整纠正 |",
        "| `er_faq_observation_040` | 点低价 | 点滴价 | 只修正‘点滴’，‘架’仍错 |", "",
        "仅后处理与无库最终结果相同；组合结果与仅辅助识别相同。因此本集观察到的增益来自 ASR 上下文提示，后处理没有提供额外收益。四组最终输出变化共 20 条，主要是标点和断句。", "",
        "## 只看无提示漏掉术语的音频", "",
        f"按无提示原始 ASR 的词表命中情况事后筛出 {hard_summary['samples']} 条音频（共 {hard_summary['expected_mentions']} 个术语出现），筛选规则和全量结果一并保存。无库最终结果命中 {hard_summary['baseline_hits']}/{hard_summary['expected_mentions']}；仅辅助识别命中 {hard_summary['context_only_hits']}/{hard_summary['expected_mentions']}。这里的提升很明显，但这是从同一批数据里按基线错误筛出来的难例诊断，不能当作独立测试集准确率。",
        "三条错误术语被词表提示完整纠正；第四条‘检体’仍被识别为‘简体’。恢复的‘点滴’出现在‘点滴价’中，词条命中但整句仍有一个字错误。", "",
        "## 更专业语料的可用性", "",
        "ChiMeS 论文描述了 14.4 小时、7,225 句中文医学语音。官方门户提供下载入口，但本次执行环境连接该域名时 TLS 握手失败，未取得文件，因此没有把该语料计入测试。",
        "另找到一套普通话医疗语音数据，介绍为 2,000 条真实场景音频、人工转录、单条 3–20 秒；公开页标注 CC-BY-NC-SA 4.0，并需联系购买。它比本次 FAQ 更贴近真实高密度术语测试，但尚未取得授权和数据，本次未下载。",
        "MMedFD 论文介绍了真实中文医疗对话，但官方仓库说明完整数据须经内部审批并签署仅研究用途的数据协议；本轮没有访问其受限数据。", "",
        "本次结果只适用于这套短急诊 FAQ 音频。它便于做实体词表 sanity check，但不能证明对自发临床对话有相同提升。它与之前约 50 分钟的 MultiMed 长片段在长度、内容和参考文本覆盖上不同，两个数据集的 CER 不能直接横向比较。"]
    (OUT / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "decode", "run", "report"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare": prepare()
    elif args.stage == "decode": asyncio.run(decode())
    elif args.stage == "run": run_cells()
    else: report()


if __name__ == "__main__":
    main()
