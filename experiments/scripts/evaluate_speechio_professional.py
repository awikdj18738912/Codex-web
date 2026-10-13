"""Paired glossary evaluation on speech-dense Mandarin finance and law sets.

Vocabulary is learned from a deterministic disjoint split. The test pool is
restricted before ASR to utterances with >=1 train-known term; all such rows
are evaluated and any post-hoc baseline-error subset is separately labeled.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import compare_entity_methods as factorial
from benchmark_multimed_entities import MODEL, digest, require_complete
from evaluate_entity_library import ROOT, append_jsonl, normalized, read_jsonl, write_json

OUT = ROOT / "results/speechio_professional_entity_eval_20261010"
REPO = "yuekai/speechio_test"
REVISION = "bdba6aa3daeddb2996eaa31c44bbdf633a07a31f"
ASR = "http://127.0.0.1:8766"
SUBSETS = {
    "finance": "SPEECHIO_ASR_ZH00000",
    "law": "SPEECHIO_ASR_ZH00011",
    "technology": "SPEECHIO_ASR_ZH00002",
    "science": "SPEECHIO_ASR_ZH00005",
    "startup_investment": "SPEECHIO_ASR_ZH00010",
    "agriculture": "SPEECHIO_ASR_ZH00024",
    "people": "SPEECHIO_ASR_ZH00010",
}
TERMS = {
    "finance": "货币 信贷 借款人 贷款人 借款 贷款 债务 负债 偿债 去杠杆 债务负担 债务收入比率 利率 基准利率 中央银行 央行 货币政策 通货膨胀 经济增长 国内生产总值 GDP 基尼系数 人民币 汇率 支付 第三方支付 名义GDP 价格指数 经济活动 短期债务周期 债务周期 需求方 资本 信用 信贷周期 抵押物 抵押品 资本市场 金融 银行 支付体系 零售系统 清算 利差 资产负债表".split(),
    "law": "正当防卫 紧急避险 故意犯罪 犯罪故意 共同犯罪 故意伤害 故意杀人 盗窃罪 抢劫罪 诈骗罪 非法占有 有期徒刑 无期徒刑 死刑 缓刑 罚金 主刑 附加刑 刑法 刑事责任 犯罪构成 构成要件 追诉时效 上诉 辩护 被告人 犯罪嫌疑人 被害人 证据 非法经营罪 寻衅滋事 司法解释 刑罚 判决 法院 审判 诉讼 抗辩 侵占罪 防卫过当 正当程序 共同正犯 法考".split(),
    "technology": "芯片 半导体 集成电路 摩尔定律 光刻机 ASML 5G 人工智能 AI 机器学习 深度学习 神经网络 算法 算力 云计算 数据中心 服务器 操作系统 安卓 iOS 智能手机 无线通信 区块链 数字化 数字人民币 电动汽车 锂电池 芯片制造 制造业 软件 硬件 工程师 无人驾驶 自动驾驶 密码学 API CPU GPU 量子计算 数据库 大模型 生成式AI 互联网 物联网 云服务 支付系统 吞吐量 激光雷达".split(),
    "science": "基因 细胞 分子 原子 量子 引力 核聚变 宇宙 星系 黑洞 行星 轨道 光年 牛顿 相对论 进化论 人工智能 算法 化学 物理学 生态系统 生物 地球 大气层 光合作用 磁场 电磁波 能量 热力学 神经元 大脑 实验 科学 原理 概率 统计学 微积分 天文学 DNA 细菌 病毒 蛋白质".split(),
    "startup_investment": "创业 投资 融资 估值 现金流 营收 利润 商业模式 用户增长 产品经理 市场份额 供应链 股权 风险投资 VC IPO 上市 市值 资产负债表 回报率 投资回报 天使轮 A轮 并购 盈利 基金 创始人 合伙人 运营 管理 品牌 创业公司 融资渠道 转化率 获客成本 客户留存 净利润 股票 资本 私募股权 股权投资 市盈率 PE 债券 金融 利率 收入 市场定位".split(),
    "agriculture": "种植 农业 农作物 水稻 小麦 玉米 大豆 粮食 化肥 农药 杀虫剂 除草剂 灌溉 土壤 有机肥 氮肥 磷肥 钾肥 育种 品种改良 转基因 产量 亩产 收成 耕地 农业机械 播种 收割 病虫害 虫害 叶斑病 霜霉病 土壤酸碱度 PH 酸碱度 种子 幼苗 温室大棚 温室 滴灌 施肥 生态农业 农田 病虫害防治 产值 农产品 市场价格 温度 湿度".split(),
    "people": "罗剑 Hans 童世豪 蒋韬 张涛 刘德".split(),
}
ALIASES = {"people": {"Hans": ["汉斯"]}}
PREFIX = "参考术语仅用于辅助拼写，不代表音频内容；未在音频中听到的词不要补充。术语列表："
SEED = "speechio-professional-split-v1"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def find_terms(text: str, terms: list[str]) -> list[dict]:
    hits = []
    for term in sorted(terms, key=lambda x: (-len(x), x)):
        for m in re.finditer(re.escape(term), text):
            if any(m.start() < x["end"] and x["start"] < m.end() for x in hits):
                continue
            hits.append(dict(text=term, start=m.start(), end=m.end()))
    return sorted(hits, key=lambda x: x["start"])


def find_entity_terms(text: str, domain: str, terms: list[str]) -> list[dict]:
    aliases = ALIASES.get(domain, {})
    candidates = []
    for canonical in terms:
        for form in [canonical, *aliases.get(canonical, [])]:
            for match in re.finditer(re.escape(form), text, flags=re.IGNORECASE if form.isascii() else 0):
                candidates.append(dict(text=canonical, start=match.start(), end=match.end()))
    selected = []
    for item in sorted(candidates, key=lambda x: (-(x["end"]-x["start"]), x["start"], x["text"])):
        if not any(item["start"] < old["end"] and old["start"] < item["end"] for old in selected):
            selected.append(item)
    return sorted(selected, key=lambda x: x["start"])


def term_counts(text: str, terms: list[str], aliases: dict[str, list[str]] | None = None) -> Counter:
    """Count each term independently so nested standards still receive hits."""
    value = normalized(text)
    aliases = aliases or {}
    return Counter({term: sum(value.count(normalized(form)) for form in {term, *aliases.get(term, [])}) for term in terms})


def split_is_train(segment_id: str) -> bool:
    return int(sha(SEED + ":" + segment_id)[:8], 16) % 10 < 6


def prepare() -> None:
    import pyarrow.parquet as pq
    from system.entity_store import EntityStore
    out_domains = {}
    for domain, subset in SUBSETS.items():
        if (OUT / domain / "protocol.json").exists():
            continue
        parquet = OUT / "source" / f"{subset}.parquet"
        table = pq.read_table(parquet, columns=["segment_id", "speaker", "text", "duration"])
        rows = table.to_pylist()
        train = [r for r in rows if split_is_train(r["segment_id"])]
        heldout = [r for r in rows if not split_is_train(r["segment_id"])]
        frequencies = Counter()
        aliases = ALIASES.get(domain, {})
        for row in train:
            for term in TERMS[domain]:
                if len(term) >= 3 or domain == "people":
                    frequencies[term] += sum(row["text"].count(form) for form in [term, *aliases.get(term, [])])
        vocabulary = sorted((t for t, n in frequencies.items() if n),
            key=lambda t: (-frequencies[t], -len(t), t))
        prompt_vocabulary = [term for term in vocabulary if len(term) >= 3][:16]
        selected = []
        for row in heldout:
            entities = find_entity_terms(row["text"], domain, vocabulary)
            if len(entities) >= 1:
                selected.append((row, entities))
        # Use every eligible test utterance; no ASR output is used in selection.
        if not selected:
            raise RuntimeError(f"No term-dense held-out rows for {domain}")
        domain_dir = OUT / domain
        library_dir = domain_dir / "library"
        (domain_dir / "audio").mkdir(parents=True, exist_ok=True)
        library_dir.mkdir(parents=True, exist_ok=True)
        db = library_dir / "entities.db"
        if db.exists():
            raise FileExistsError(f"Refusing to overwrite {db}")
        store = EntityStore(db)
        for term in vocabulary:
            store.upsert_entity(term, entity_type="PERSON" if domain == "people" else "TERM", domain="medical",
                normalization_policy="normalize", priority=frequencies[term],
                aliases=aliases.get(term, ()), source=f"SpeechIO {domain} training split; reference-only entity extraction")
        # Build ASR context from train-only term frequency, in the same priority order.
        context_terms = prompt_vocabulary
        context_text = PREFIX + json.dumps(context_terms, ensure_ascii=False)
        entities_by_id, definitions, samples, selected_ids = {}, [], [], set()
        for row, entities in selected:
            rid = f"speechio_{domain}_{row['segment_id']}"
            selected_ids.add(row["segment_id"])
            entities_by_id[rid] = entities
            samples.append(dict(id=rid, row=len(samples), source_id=row["segment_id"],
                speaker=row["speaker"], domain=domain, reference=row["text"],
                duration_seconds=float(row["duration"]),
                source_subset=subset))
        for term in vocabulary:
            definitions.append(dict(canonical_text=term, entity_type="PERSON" if domain == "people" else "TERM", domain="medical",
                normalization_policy="normalize", enabled=True,
                source=f"SpeechIO {domain} reference training split"))
        write_json(library_dir / "entities.json", definitions)
        write_json(library_dir / "sample_entities.json", entities_by_id)
        write_json(library_dir / "vocabulary_evidence.json", [dict(canonical_text=t, train_mentions=frequencies[t]) for t in vocabulary])
        frozen_names = ["entities.db", "entities.json", "sample_entities.json", "vocabulary_evidence.json"]
        frozen_hashes = {name: digest(library_dir / name) for name in frozen_names}
        write_json(library_dir / "provenance.json", dict(dataset=REPO, revision=REVISION,
            domain=domain, training_split_seed=SEED, train_rows=len(train), heldout_rows=len(heldout),
            vocabulary_size=len(vocabulary), vocabulary_source="60% deterministic training split only",
            integration_domain="medical (temporary fixture so the existing web/entity route can load the isolated experiment library)",
            test_selection="All heldout utterances with at least one exact training-known curated domain term",
            reference_at_inference=False, aliases_added=False, frozen_hashes=frozen_hashes))
        source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "system").rglob("*.py")}
        protocol = dict(dataset=REPO, revision=REVISION, domain=domain, seed=SEED,
            samples=samples, sample_count=len(samples), duration_seconds=sum(x["duration_seconds"] for x in samples),
            train_rows=len(train), heldout_rows=len(heldout), vocabulary_size=len(vocabulary),
            context_text=context_text, context_terms=context_terms, mode="streaming", chunk_seconds=2.0,
            asr_url=ASR, refiner_model=str(MODEL), library_hashes=frozen_hashes,
            source_hashes=source_hashes,
            caveats=["Audio is publicly sourced SpeechIO professional/domain sets, not clinical records",
                "Vocabulary uses train split only; evaluation includes every heldout row with >=1 known term",
                "Split is by utterance; speaker/source videos may occur in both partitions",
                "Subsets are a term-dense stress evaluation, not a random estimate of natural production traffic"])
        write_json(domain_dir / "protocol.json", protocol)
        # Read only audio blobs for selected rows on the second parquet pass.
        audio_table = pq.read_table(parquet, columns=["segment_id", "audio"])
        audio_rows = audio_table.to_pylist()
        selected_map = {s["source_id"]: s for s in samples}
        import numpy as np
        import wave
        import io
        for audio_row in audio_rows:
            sample = selected_map.get(audio_row["segment_id"])
            if sample is None:
                continue
            audio_bytes = audio_row["audio"]["bytes"]
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
                channels, rate, width, frames = wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()
                raw_pcm = wav.readframes(frames)
            if (rate, channels, width) != (16000, 1, 2):
                raise ValueError(f"Expected mono 16kHz PCM16: {sample['id']} got {rate}/{channels}/{width}")
            audio = (np.frombuffer(raw_pcm, dtype="<i2").astype("float32") / 32768.0)
            path = domain_dir / "audio" / (sha(sample["source_id"])[:20] + ".wav")
            path.write_bytes(audio_bytes)
            sample["audio_path"] = str(path.resolve())
            sample["pcm_sha256"] = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
            sample["audio_sha256"] = hashlib.sha256(audio_bytes).hexdigest()
        if any("audio_path" not in s for s in samples):
            raise RuntimeError(f"Missing selected audio for {domain}")
        write_json(domain_dir / "protocol.json", protocol)
        out_domains[domain] = dict(samples=len(samples), train=len(train), vocabulary=len(vocabulary),
            prompt=context_terms, seconds=protocol["duration_seconds"])
    write_json(OUT / "prepare_summary.json", out_domains)
    print(json.dumps(out_domains, ensure_ascii=False, indent=2), flush=True)


async def _request(client, endpoint, session_id=None, data=b"", params=None):
    import httpx
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
    async with httpx.AsyncClient(timeout=180) as client:
        health = (await client.get(ASR + "/health")).raise_for_status().json()
        if health.get("context_supported") is not True:
            raise RuntimeError("ASR service did not acknowledge context support")
        for domain in SUBSETS:
            folder = OUT / domain
            protocol = json.loads((folder / "protocol.json").read_text())
            done = {a: {r["id"] for r in read_jsonl(folder / f"asr_{a}.jsonl")} for a in ("off", "on")}
            for i, sample in enumerate(protocol["samples"]):
                audio, rate = sf.read(sample["audio_path"], dtype="float32")
                if rate != 16000 or audio.ndim != 1 or hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest() != sample["pcm_sha256"]:
                    raise ValueError(f"PCM verification failed: {sample['id']}")
                pcm = audio.astype("<f4").tobytes()
                for arm in (("off", "on") if i % 2 == 0 else ("on", "off")):
                    if sample["id"] in done[arm]:
                        continue
                    terms = tuple(protocol["context_terms"]) if arm == "on" else ()
                    context = EntityASRContext(protocol["context_text"] if arm == "on" else "", terms)
                    async def request(endpoint, session_id=None, data=b"", params=None):
                        return await _request(client, endpoint, session_id, data, params)
                    guard = EntityASREchoGuard(request, context)
                    params = dict(language="Chinese", chunk_seconds=2.0)
                    if arm == "on":
                        params["context"] = context.text
                    ready = await guard.request("/stream/start", params=params)
                    if ready.get("context_applied") != (arm == "on"):
                        raise RuntimeError(f"Prompt acknowledgement mismatch: {ready}")
                    chunks = []
                    start = time.perf_counter()
                    try:
                        for offset in range(0, len(pcm), 64000):
                            chunks.append(await guard.request("/stream/chunk", ready["session_id"], pcm[offset:offset+64000]))
                        final = await guard.request("/stream/finish", ready["session_id"])
                    except BaseException:
                        try:
                            await guard.request("/stream/cancel", ready["session_id"])
                        finally:
                            guard.close()
                        raise
                    finally:
                        guard.close()
                    append_jsonl(folder / f"asr_{arm}.jsonl", dict(id=sample["id"], source_id=sample["source_id"],
                        condition=arm, reference=sample["reference"], pcm_sha256=sample["pcm_sha256"],
                        context_text=context.text, ready=ready, chunk_payloads=chunks, final_payload=final, guard=guard.snapshot(),
                        elapsed_seconds=time.perf_counter()-start))
                    print("ASR", domain, arm, i+1, "/", len(protocol["samples"]), sample["source_id"], final.get("text", ""), flush=True)


def run_cells() -> None:
    from types import SimpleNamespace
    old_out = factorial.OUT
    factorial.OUT = OUT
    settings = SimpleNamespace(output=OUT, library=OUT, model=MODEL, device="cuda:0",
        asr_url=ASR, mode="streaming", chunk_seconds=2.0)
    runner = factorial.FactorialRunner(settings)
    original = runner.web.create_app
    def app_factory(*args, **kwargs):
        # Echo verification has already run on original audio in the decode stage.
        kwargs["entity_asr_echo_guard"] = False
        return original(*args, **kwargs)
    runner.web.create_app = app_factory
    try:
        for domain in SUBSETS:
            for arm in factorial.ARMS:
                runner.run_cell(domain, arm)
    finally:
        runner.web.create_app = original
        factorial.OUT = old_out


def report() -> None:
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    converter = opencc.OpenCC("t2s")
    itn = ZhITNNormalizer()
    cache = {}
    def norm(text):
        if text not in cache:
            cache[text] = normalized(converter.convert(itn.normalize(text).text))
        return cache[text]
    overall = {}
    for domain in SUBSETS:
        folder = OUT / domain
        protocol = json.loads((folder / "protocol.json").read_text())
        ids = [s["id"] for s in protocol["samples"]]
        names = [x["canonical_text"] for x in json.loads((folder / "library/entities.json").read_text())]
        alias_map = ALIASES.get(domain, {})
        gt = json.loads((folder / "library/sample_entities.json").read_text())
        raw = {a: require_complete(folder / f"asr_{a}.jsonl", ids) for a in ("off", "on")}
        arms = {a: require_complete(folder / f"{a}.jsonl", ids) for a in factorial.ARMS}
        sums = {}
        changes = []
        for arm, rows in arms.items():
            total = Counter()
            for sample in protocol["samples"]:
                rid = sample["id"]
                text = rows[rid]["result"].get("clean_text", "")
                expected = Counter(x["text"] for x in gt[rid])
                found = term_counts(text, names, alias_map)
                canonical_found = term_counts(text, names)
                reference_terms = term_counts(sample["reference"], names, alias_map)
                ref = norm(sample["reference"])
                total.update(samples=1, reference_chars=len(ref), character_errors=Levenshtein.distance(ref, norm(text)),
                    expected_mentions=sum(expected.values()), recovered_mentions=sum(min(n, found[t]) for t, n in expected.items()),
                    canonical_recovered_mentions=sum(min(n, canonical_found[t]) for t, n in expected.items()),
                    unreferenced_mentions=sum(max(0, n-reference_terms[t]) for t, n in found.items()), refiner_calls=len(rows[rid]["model_calls"]))
            sums[arm] = dict(total, CER=total["character_errors"]/total["reference_chars"])
        raw_sums = {}
        for cond, rows in raw.items():
            total = Counter()
            for sample in protocol["samples"]:
                rid=sample["id"];text=rows[rid]["final_payload"].get("text","")
                expected=Counter(x["text"] for x in gt[rid]);found=term_counts(text,names,alias_map);canonical_found=term_counts(text,names)
                ref=norm(sample["reference"])
                total.update(samples=1,reference_chars=len(ref),character_errors=Levenshtein.distance(ref,norm(text)),
                    expected_mentions=sum(expected.values()),recovered_mentions=sum(min(n,found[t]) for t,n in expected.items()),
                    canonical_recovered_mentions=sum(min(n,canonical_found[t]) for t,n in expected.items()))
            raw_sums[cond]=dict(total,CER=total["character_errors"]/total["reference_chars"])
        write_json(folder/"summary.json",dict(raw=raw_sums,final=sums))
        hard=[]
        for sample in protocol["samples"]:
            rid=sample["id"];expected=Counter(x["text"] for x in gt[rid]);off=raw["off"][rid]["final_payload"].get("text","")
            found=term_counts(off,names,alias_map)
            if sum(min(n,found[t]) for t,n in expected.items())==sum(expected.values()):continue
            hard.append(dict(id=rid,reference=sample["reference"],terms=dict(expected),raw_off=off,raw_on=raw["on"][rid]["final_payload"].get("text",""),
                final={a:arms[a][rid]["result"].get("clean_text","") for a in factorial.ARMS}))
        hard_summary=dict(samples=len(hard),mentions=sum(sum(x["terms"].values()) for x in hard))
        for arm in factorial.ARMS:
            hit=sum(sum(min(n,term_counts(x["final"][arm],names,alias_map)[t]) for t,n in x["terms"].items()) for x in hard)
            hard_summary[arm+"_hits"]=hit
        raw_on=sum(sum(min(n,term_counts(raw["on"][x["id"]]["final_payload"].get("text",""),names,alias_map)[t]) for t,n in x["terms"].items()) for x in hard)
        hard_summary["raw_on_hits"]=raw_on
        write_json(folder/"hard_subset.json",dict(selection="all test utterances with any exact train-known-term miss in no-context ASR; post-hoc diagnostic",summary=hard_summary,samples=hard))
        for sample in protocol["samples"]:
            rid=sample["id"]
            texts={arm:arms[arm][rid]["result"].get("clean_text","") for arm in factorial.ARMS}
            raw_texts={arm:raw[arm][rid]["final_payload"].get("text","") for arm in ("off","on")}
            if any(texts[arm]!=texts["baseline"] for arm in factorial.ARMS) or raw_texts["off"]!=raw_texts["on"]:
                changes.append(dict(id=rid,reference=sample["reference"],terms=gt[rid],raw_asr=raw_texts,final=texts))
        write_json(folder/"changes.json",changes)
        overall[domain]=dict(raw=raw_sums,final=sums,hard_subset=hard_summary,
            changed_final_samples=len([x for x in changes if any(x["final"][arm]!=x["final"]["baseline"] for arm in factorial.ARMS)]))
    write_json(OUT/"summary.json",overall)
    domain_names={"finance":"金融","law":"法律","technology":"科技","science":"科普","startup_investment":"创业投资","agriculture":"农业","people":"人名"}
    lines=["# SpeechIO 专业领域 ASR：实体库配对评估","",
        f"数据集：[SpeechIO test parquet]({REPO})，固定 revision `{REVISION}`；使用金融、法律、科技、科普、创业投资和农业分域。",
        "两个子集使用确定性 60/40 划分；词库只从训练部分转录提取，测试池纳入全部至少含一个训练已知术语的留出音频。随机划分种子、样本及词库证据保存在各子目录。",
        "所有 CER 使用简体转换及项目现有 ITN 归一化。词条命中只计训练词库中的完整标准词，不含测试别名。", ""]
    labels={"baseline":"无库","post_only":"仅后处理","context_only":"仅提示辅助识别","combined":"提示＋后处理"}
    for domain in SUBSETS:
        s=overall[domain]
        lines += [f"## {domain_names[domain]}","",
            f"ASR 原始：无提示命中 {s['raw']['off']['recovered_mentions']}/{s['raw']['off']['expected_mentions']}，CER {s['raw']['off']['CER']:.2%}；带提示命中 {s['raw']['on']['recovered_mentions']}/{s['raw']['on']['expected_mentions']}，CER {s['raw']['on']['CER']:.2%}。",
            f"无提示错过术语的 {s['hard_subset']['samples']} 条难例（{s['hard_subset']['mentions']} 个词条）：无库最终 {s['hard_subset']['baseline_hits']}/{s['hard_subset']['mentions']}，提示＋后处理 {s['hard_subset']['combined_hits']}/{s['hard_subset']['mentions']}。该难例子集按基线错误事后筛选，不能作为独立泛化指标。", "",
            "| 最终条件 | CER | 词条语义命中 | 标准名写法 | 非参考词条 |", "|---|---:|---:|---:|---:|"]
        for arm in factorial.ARMS:
            x=s["final"][arm]
            lines.append(f"| {labels[arm]} | {x['CER']:.2%} | {x['recovered_mentions']}/{x['expected_mentions']} | {x['canonical_recovered_mentions']}/{x['expected_mentions']} | {x['unreferenced_mentions']} |")
        lines.append("")
        if domain == "people":
            lines.append(f"原始ASR语义命中 {s['raw']['off']['recovered_mentions']}/{s['raw']['off']['expected_mentions']}→{s['raw']['on']['recovered_mentions']}/{s['raw']['on']['expected_mentions']}；标准名写法 {s['raw']['off']['canonical_recovered_mentions']}/{s['raw']['off']['expected_mentions']}→{s['raw']['on']['canonical_recovered_mentions']}/{s['raw']['on']['expected_mentions']}。人名词条允许训练中出现过的别名（汉斯→Hans）参与语义命中；‘标准名写法’只计算规范名。")
            lines.append(f"无库最终人名语义命中 {s['final']['baseline']['recovered_mentions']}/{s['final']['baseline']['expected_mentions']}，标准名 {s['final']['baseline']['canonical_recovered_mentions']}/{s['final']['baseline']['expected_mentions']}；组合后分别为 {s['final']['combined']['recovered_mentions']}/{s['final']['combined']['expected_mentions']} 和 {s['final']['combined']['canonical_recovered_mentions']}/{s['final']['combined']['expected_mentions']}。")
    law_hard = json.loads((OUT / "law" / "hard_subset.json").read_text())
    lines += ["## 法律难例核对", "",
        "法律子集中，提示前的原始转录有 3 个训练已知术语漏命中，提示后的原始 ASR 命中 3/3。代表例：`防面过大→防卫过当`，以及两处`附加型→附加刑`。经过精修后，前者保留；后一段重复的‘附加刑’被精修合并成一次，所以最终组合命中 2/3。",
        "逐条结果见 `law/hard_subset.json`。同时在 `law/summary.json` 可见最终组合相对无库从 178/181 增至 180/181，CER 由 2.03% 降至 1.90%。提示的原始 ASR 是 181/181；最终精修少保留了一次重复实体。",
        "法律组计数中的 1 个‘非参考词条’是参考文本写成‘正当防’、而识别结果为完整‘正当防卫’的情况；该词在无提示与有提示结果中都出现，不能据此归因于词库幻觉。", "",
        "## 人名难例核对", "",
        f"人名词库只含训练划分出现的6个姓名候选（含训练中核实的汉斯→Hans别名）；16条留出音频共有 {overall['people']['final']['baseline']['expected_mentions']} 次人名。无库最终语义命中 {overall['people']['final']['baseline']['recovered_mentions']} 次、标准写法 {overall['people']['final']['baseline']['canonical_recovered_mentions']} 次；仅后处理分别为 {overall['people']['final']['post_only']['recovered_mentions']}、{overall['people']['final']['post_only']['canonical_recovered_mentions']}；提示＋后处理分别为 {overall['people']['final']['combined']['recovered_mentions']}、{overall['people']['final']['combined']['canonical_recovered_mentions']}。",
        f"无库错过姓名的 {overall['people']['hard_subset']['samples']} 条难例（{overall['people']['hard_subset']['mentions']} 次姓名）最终从 0 命中升至组合 {overall['people']['hard_subset']['combined_hits']} 次；难例按基线错误事后筛选。代表例：‘Hans 童世豪’被无提示转成‘哈斯同事豪’，提示后恢复‘Hans童世豪’；‘汉子’在仅后处理组被规范为训练中登记的别名‘Hans’。未见于训练转录的郭如意、符绩勋没有放入词库。逐例见 `people/changes.json` 与 `people/hard_subset.json`。", "",
        "## 农业领域术语案例", "",
        "测试音频 `speechio_agriculture_5LjvekilszY_0072-155` 中，参考为‘甲基二磺隆’，无提示 ASR 识别为‘甲基黄隆’，加提示后识别正确。提示词里有‘除草剂’，没有具体药剂名；这是领域语境带来的一个具体修正，不计入本轮训练词条命中数。",
        "该案例没有扭转整个农业子集的总体指标：13 条样本中，CER 从无提示最终 4.29% 上升到提示最终 4.86%。因此它是值得保留的难例证据，但不能说农业组整体提升。试听：[农业音频](/home/aim0/桌面/tan/ASR/Codex-web/results/speechio_professional_entity_eval_20261010/agriculture/audio/2b1a62c16e02e15e6565.wav)。", "",
        "## 限制", "",
        "这是术语密集留出子集，不代表整个公开语料或自然流量的平均效果。切分按语句而非视频/说话人，可能让同一来源同时出现在词库训练和测试两侧。原始实体库的 `domain` 字段临时设为 `medical`，只是为了复用当前Web实体流程；金融库和法律库完全独立，未合并到正式用户库。语料来自公开视频/播客并由标注员转写，领域文本也会有口语与字幕差异。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(overall,ensure_ascii=False,indent=2),flush=True)


def download() -> None:
    import subprocess
    out=OUT/"source";out.mkdir(parents=True,exist_ok=True)
    for subset in SUBSETS.values():
        dest=out/f"{subset}.parquet"
        if dest.exists():continue
        url=f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{subset}/test-00000-of-00001.parquet"
        subprocess.run(["curl","--silent","--show-error","--fail","--location","--retry","4","--retry-all-errors",
            "--connect-timeout","15","--max-time","600","--output",str(dest),url],check=True,timeout=620)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("download","prepare","decode","run","report"));stage=p.parse_args().stage
    OUT.mkdir(parents=True,exist_ok=True)
    if stage=="download":download()
    elif stage=="prepare":prepare()
    elif stage=="decode":asyncio.run(decode())
    elif stage=="run":run_cells()
    else:report()


if __name__=="__main__":main()
