"""Type-independent entity recall followed by bounded prompted audio replay.

This is an isolated experiment, not a production replacement rule. Candidate
selection uses only the cached no-context ASR and the prebuilt glossary.
Reference fields are consulted only in report(), never during inference.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
import sqlite3
import time

import httpx
import soundfile as sf

from evaluate_entity_library import ROOT, normalized, read_jsonl, write_json, append_jsonl
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard
from system.entity_retrieval import EntityRetriever
from system.entity_store import EntityDefinition

SOURCE = ROOT / "results/speechio_professional_entity_eval_20261010"
OUT = ROOT / "results/generic_entity_audio_20261010"
DOMAINS = {name: SOURCE / name for name in (
    "finance", "law", "technology", "science", "startup_investment",
    "agriculture", "people",
)}
DOMAINS["medical_faq"] = ROOT / "results/chinese_er_faq_entity_eval_20261010/medical"
ASR = "http://127.0.0.1:8766"
PREFIX = "参考术语仅用于辅助拼写，不代表音频内容；未在音频中听到的词不要补充。术语列表："
MAX_TERMS = 8


def load_definitions(folder):
    # Read-only: experiment preparation cannot alter the frozen glossary.
    db = folder / "library/entities.db"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM protected_entities ORDER BY id").fetchall()
        aliases = {}
        for row in connection.execute("SELECT entity_id,alias FROM entity_aliases"):
            aliases.setdefault(row["entity_id"], []).append(row["alias"])
    return tuple(EntityDefinition(
        row["id"], row["canonical_text"], row["entity_type"], row["domain"],
        row["normalization_policy"], row["priority"], tuple(aliases.get(row["id"], [])),
        enabled=bool(row["enabled"]),
    ) for row in rows)


class GenericRecall:
    """The same lexical/sound retrieval settings for every entity type."""
    def __init__(self, definitions):
        self.definitions = tuple(d for d in definitions if d.enabled)
        surfaces, self.owner = [], {}
        for definition in self.definitions:
            for surface in (definition.canonical_text, *definition.aliases):
                if re.fullmatch(r"[\u3400-\u9fff]{2,24}", surface) is None:
                    continue
                identifier = len(surfaces) + 1
                # Type is deliberately not a retrieval feature. Registered
                # aliases supply cross-script pronunciations, never guesses.
                surfaces.append(EntityDefinition(identifier, surface, "TERM", "general", "normalize", 0, ()))
                self.owner[identifier] = definition
        self.retriever = EntityRetriever(surfaces, minimum_score=0.52, max_candidates=5)

    def select(self, text):
        blocked = []
        for definition in self.definitions:
            for surface in (definition.canonical_text, *definition.aliases):
                blocked.extend((m.start(), m.end()) for m in re.finditer(re.escape(surface), text))
        matches = self.retriever.retrieve(text, blocked_spans=blocked)
        dedup = {}
        for match in matches:
            if match.boundary_issue not in (None, "short_chinese_span_requires_alias"):
                continue
            definition = self.owner[match.entity_id]
            key = match.start, match.end, definition.entity_id
            candidate = dict(match.public_dict(), canonical=definition.canonical_text,
                             entity_id=definition.entity_id, evidence_surface=match.canonical)
            candidate.pop("entity_type")
            if key not in dedup or candidate["retrieval_score"] > dedup[key]["retrieval_score"]:
                dedup[key] = candidate
        ordered = sorted(dedup.values(), key=lambda c: (-c["retrieval_score"], -len(c["observed"]), c["start"], c["canonical"]))
        terms = list(dict.fromkeys(c["canonical"] for c in ordered))[:MAX_TERMS]
        selected = [c for c in ordered if c["canonical"] in terms]
        return selected, terms


def bounded_changes(original, replay, candidates):
    """Copy only registered names supported at their original text position.

    Prompted recognition is support, not an independent acoustic likelihood.
    Exact surrounding anchors reduce accidental matches in another clause.
    No punctuation or unrelated replay text is copied to the result.
    """
    replay_compact = normalized(replay)
    confirmed = []
    for candidate in candidates:
        start, end = candidate["start"], candidate["end"]
        left = normalized(original[:start])[-8:]
        right = normalized(original[end:])[:8]
        anchor = left + normalized(candidate["canonical"]) + right
        if (left or right) and anchor in replay_compact:
            confirmed.append(candidate)
    accepted = []
    for candidate in confirmed:
        if any(other["canonical"] != candidate["canonical"] and
               candidate["start"] < other["end"] and other["start"] < candidate["end"]
               for other in confirmed):
            continue
        if any(candidate["start"] < old["end"] and old["start"] < candidate["end"] for old in accepted):
            continue
        accepted.append(candidate)
    result = original
    for candidate in sorted(accepted, key=lambda c: c["start"], reverse=True):
        result = result[:candidate["start"]] + candidate["canonical"] + result[candidate["end"]:]
    return result, accepted


def neighbor_anchored_changes(original, replay, candidates):
    """Allow adjacent registered corrections in the surrounding anchors.

    Each side still consumes the same eight source characters. Its only
    alternatives are original characters or retrieved glossary spellings;
    arbitrary changes to surrounding wording cannot provide support.
    """
    source, target = normalized(original), normalized(replay)
    spans = [(len(normalized(original[:c["start"]])),
              len(normalized(original[:c["end"]])), normalized(c["canonical"]), c)
             for c in candidates]

    def side_matches(a, b, replay_side, query_start, query_end, reverse=False):
        value = source[a:b]
        if reverse:
            value = value[::-1]
        edges = {}
        for start, end, canonical, _ in spans:
            if not a <= start < end <= b or start < query_end and query_start < end:
                continue
            x,y,word = (b-end,b-start,canonical[::-1]) if reverse else (start-a,end-a,canonical)
            edges.setdefault(x, []).append((y,word))

        @lru_cache(maxsize=None)
        def match(i,j):
            if i == len(value):
                return True
            if j >= len(replay_side):
                return False
            if value[i] == replay_side[j] and match(i+1,j+1):
                return True
            return any(replay_side.startswith(word,j) and match(end,j+len(word))
                       for end,word in edges.get(i,()))
        return match(0,0)

    confirmed = []
    for start,end,canonical,candidate in spans:
        if not source[:start] and not source[end:]:
            continue
        for occurrence in re.finditer(re.escape(canonical),target):
            if (side_matches(max(0,start-8),start,target[:occurrence.start()][::-1],start,end,True)
                    and side_matches(end,min(len(source),end+8),target[occurrence.end():],start,end)):
                confirmed.append(candidate)
                break
    accepted = []
    for candidate in confirmed:
        if any(other["canonical"] != candidate["canonical"] and
               candidate["start"] < other["end"] and other["start"] < candidate["end"]
               for other in confirmed):
            continue
        if any(candidate["start"] < old["end"] and old["start"] < candidate["end"] for old in accepted):
            continue
        accepted.append(candidate)
    result = original
    for candidate in sorted(accepted, key=lambda c:c["start"], reverse=True):
        result = result[:candidate["start"]] + candidate["canonical"] + result[candidate["end"]:]
    return result,accepted


def recheck():
    plan = {s["id"]:s for s in json.loads((OUT / "plan.json").read_text())["samples"]}
    rows = read_jsonl(OUT / "outputs.jsonl")
    destination = OUT / "outputs_neighbor.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    for row in rows:
        sample = plan[row["id"]]
        if row["replayed"]:
            corrected,changes = neighbor_anchored_changes(sample["original"],row["replay"],sample["candidates"])
            row = dict(row, corrected=corrected, changes=changes)
        append_jsonl(destination,row)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "plan.json").exists():
        raise FileExistsError("Refusing to overwrite a prepared inference plan")
    plan = []
    for domain, folder in DOMAINS.items():
        protocol = json.loads((folder / "protocol.json").read_text())
        raw = {r["id"]: r for r in read_jsonl(folder / "asr_off.jsonl")}
        definitions = load_definitions(folder)
        recall = GenericRecall(definitions)
        # Check type invariance on actual data, not just a mocked example.
        alternative = GenericRecall([replace(d, entity_type="ORG") for d in definitions])
        for sample in protocol["samples"]:
            source = raw[sample["id"]]["final_payload"]["text"]
            candidates, terms = recall.select(source)
            if (candidates, terms) != alternative.select(source):
                raise AssertionError("Entity type affected generic candidate selection")
            plan.append(dict(id=sample["id"], domain=domain, original=source,
                             audio_path=sample["audio_path"], pcm_sha256=sample["pcm_sha256"],
                             candidates=candidates, prompt_terms=terms))
        print(domain, len(protocol["samples"]), "prepared", flush=True)
    write_json(OUT / "plan.json", dict(
        method="Chinese sound/lexical recall + approved aliases + prompted audio replay + bounded anchored changes",
        candidate_threshold=0.52, minimum_phonetic=0.55, max_terms=MAX_TERMS,
        entity_type_used=False, inference_reference_used=False,
        candidate_settings="Frozen once before the experiment; same values in every domain",
        samples=plan,
    ))
    print("samples", len(plan), "replays", sum(bool(r["prompt_terms"]) for r in plan), flush=True)


async def run():
    plan = json.loads((OUT / "plan.json").read_text())["samples"]
    destination = OUT / "outputs.jsonl"
    done = {r["id"] for r in read_jsonl(destination)}
    async with httpx.AsyncClient(timeout=180) as client:
        health = (await client.get(ASR + "/health")).raise_for_status().json()
        if health.get("context_supported") is not True:
            raise RuntimeError("ASR backend does not support glossary context")
        async def request(endpoint, session_id=None, data=b"", params=None):
            query = dict(params or {})
            if session_id:
                query["session_id"] = session_id
            response = await client.post(ASR + endpoint, params=query, content=data)
            return response.raise_for_status().json()
        for index, sample in enumerate(plan):
            if sample["id"] in done:
                continue
            start = time.perf_counter()
            terms = sample["prompt_terms"]
            if not terms:
                append_jsonl(destination, dict(id=sample["id"], replay=None, corrected=sample["original"], changes=[], replayed=False))
                continue
            audio, rate = sf.read(sample["audio_path"], dtype="float32")
            pcm = audio.astype("<f4").tobytes()
            if rate != 16000 or audio.ndim != 1 or hashlib.sha256(pcm).hexdigest() != sample["pcm_sha256"]:
                raise ValueError(f"PCM verification failed: {sample['id']}")
            context = EntityASRContext(PREFIX + json.dumps(terms, ensure_ascii=False), tuple(terms))
            guard = EntityASREchoGuard(request, context)
            ready = await guard.request("/stream/start", params=dict(language="Chinese", chunk_seconds=2.0, context=context.text))
            if ready.get("context_applied") is not True:
                raise RuntimeError("ASR did not acknowledge entity context")
            try:
                chunks = []
                for offset in range(0, len(pcm), 64000):
                    chunks.append(await guard.request("/stream/chunk", ready["session_id"], pcm[offset:offset+64000]))
                final = await guard.request("/stream/finish", ready["session_id"])
            except BaseException:
                await guard.request("/stream/cancel", ready["session_id"])
                raise
            finally:
                guard.close()
            replay = final.get("text", "")
            corrected, changes = bounded_changes(sample["original"], replay, sample["candidates"])
            append_jsonl(destination, dict(id=sample["id"], replay=replay, corrected=corrected, changes=changes,
                replayed=True, elapsed_seconds=time.perf_counter()-start,
                ready=ready, final_payload=final, chunk_payloads=chunks, guard=guard.snapshot()))
            print(index+1, "/", len(plan), sample["domain"], "changes", [(c["observed"], c["canonical"]) for c in changes], flush=True)


def report(variant="base"):
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from evaluate_speechio_professional import term_counts
    converter, itn = opencc.OpenCC("t2s"), ZhITNNormalizer()
    def norm(value):
        return normalized(converter.convert(itn.normalize(value).text))
    plan = json.loads((OUT / "plan.json").read_text())["samples"]
    suffix = "_neighbor" if variant == "neighbor" else ""
    outputs = {r["id"]: r for r in read_jsonl(OUT / f"outputs{suffix}.jsonl")}
    if set(outputs) != {r["id"] for r in plan}:
        raise ValueError("Evaluation is incomplete")
    summary, changes = {}, []
    for domain, folder in DOMAINS.items():
        definitions = load_definitions(folder)
        names = [d.canonical_text for d in definitions]
        aliases = {d.canonical_text: list(d.aliases) for d in definitions}
        protocol = json.loads((folder / "protocol.json").read_text())
        references = {s["id"]: s["reference"] for s in protocol["samples"]}
        annotations = json.loads((folder / "library/sample_entities.json").read_text())
        counts = {arm: Counter() for arm in ("baseline", "generic")}
        improved = worsened = 0
        for sample in (s for s in plan if s["domain"] == domain):
            rid = sample["id"]
            reference = references[rid]
            expected = Counter(item["text"] for item in annotations[rid])
            texts = dict(baseline=sample["original"], generic=outputs[rid]["corrected"])
            errors = {}
            for arm, text in texts.items():
                found = term_counts(text, names, aliases)
                ref_terms = term_counts(reference, names, aliases)
                errors[arm] = Levenshtein.distance(norm(reference), norm(text))
                counts[arm].update(samples=1, mentions=sum(expected.values()), hits=sum(min(n, found[t]) for t,n in expected.items()),
                    reference_chars=len(norm(reference)), character_errors=errors[arm],
                    unreferenced=sum(max(0,n-ref_terms[t]) for t,n in found.items()))
            improved += errors["generic"] < errors["baseline"]
            worsened += errors["generic"] > errors["baseline"]
            if texts["baseline"] != texts["generic"]:
                changes.append(dict(id=rid, domain=domain, reference=reference, **texts,
                    character_errors=errors, accepted_changes=outputs[rid]["changes"]))
        summary[domain] = dict(
            **{arm: dict(c, CER=c["character_errors"]/c["reference_chars"]) for arm,c in counts.items()},
            improved_samples=improved, worsened_samples=worsened,
            replays=sum(outputs[s["id"]]["replayed"] for s in plan if s["domain"]==domain),
        )
    write_json(OUT / f"summary{suffix}.json", summary)
    write_json(OUT / f"changes{suffix}.json", changes)
    lines = ["# 不依赖实体类型的候选检索与原音重识别实验", "",
        "实验只读既有词库与无提示 ASR 初稿，不读取参考转录参与候选检索、提示或替换判定。实体类型不参与推理，在实际样本上把所有类型换为 ORG 得到相同候选。", "",
        "| 领域 | 样本 | 基线实体命中 | 通用实体命中 | 基线 CER | 通用 CER | 改善/退化句 | 原音重识别 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|" ]
    for domain,s in summary.items():
        a,b = s["baseline"],s["generic"]
        lines.append(f"| {domain} | {a['samples']} | {a['hits']}/{a['mentions']} | {b['hits']}/{b['mentions']} | {a['CER']:.2%} | {b['CER']:.2%} | {s['improved_samples']}/{s['worsened_samples']} | {s['replays']} |")
    lines += ["", "## 结果边界", "",
        "这是隔离实验，未接入生产替换规则。重识别带候选提示，不能当作独立、无偏的声学置信度；两字完整同音普通词尤其仍可能被提示误导。", 
        "候选召回支持 2–24 字的中文标准名及已登记中文别名；没有中文别名的纯拉丁错读未在本原型覆盖。原始全文省略和完全丢失的实体无法靠邻近字音找回。",
        "SpeechIO 词库来自训练划分；切分按语句而非说话人。急诊 FAQ 为已知标准词表覆盖测试。部分领域 ASR 基线已经全命中，主要用于观察误改。不同领域可能复用相同原音，表格按音频/词库场景计数。",
        "实体命中按标准名或已登记别名计数。例如 Hans 的登记别名‘汉斯’仍算命中；这些数字不等于全部输出都已改成标准名写法。",
        "只复制原稿中已定位的实体跨度，要求重识别输出中目标名与左右最多八个归一化字符锚点完全对应。无法对应、跨词边界或候选相互冲突时保留初稿。CER 统一使用项目 ITN、简体转换及去标点归一化。", "",
        f"逐条修改及退化见 changes{suffix}.json，候选、提示与完整 ASR 响应分别见 plan.json 与 outputs{suffix}.jsonl。"]
    if variant == "neighbor":
        lines += ["", "本版本允许锚点内的其他已检索实体也使用原写法或标准写法；其他文字仍须完全一致。只复用本轮重识别响应，没有新增模型请求。该调整属于开发集规则迭代。"]
    (OUT / f"REPORT{suffix}.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "report", "recheck"))
    parser.add_argument("--variant", choices=("base", "neighbor"), default="base")
    args = parser.parse_args()
    stage = args.stage
    if stage == "prepare":
        prepare()
    elif stage == "run":
        asyncio.run(run())
    elif stage == "recheck":
        recheck()
    else:
        report(args.variant)
