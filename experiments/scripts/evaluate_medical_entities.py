"""Medical-domain entity evaluation with independently sourced terminology.

MultiMed Chinese eval supplies development errors; the test split remains
held out. CMeEE training annotations and MultiMed train supply canonical terms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3

from evaluate_entity_library import ROOT, decode, normalized, read_jsonl, write_json
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector

APPROVED_SHORT_TERMS = frozenset("""
脊髓 脊椎 颈椎 腰椎 胸椎 椎管 肌腱 韧带 瓣膜 肺门 胸膜 纵隔 胸骨 腹腔
胃镜 活检 输液 哮喘 肾炎 鼻炎 肺炎 皮疹 骨髓 抗体 抗原 肿瘤 血栓 心肌
心房 心室 动脉 静脉 冠脉 神经 钙化 血糖 血压 放疗 化疗 乳癌 乳腺 腋窝
尿酸 结节 气管 疝气 贫血 骨折 感染 麻醉 癌症 细胞 肝癌 肺癌 胃癌 肾癌
肠癌 骨骼 麻痹 免疫
""".split())
# Incomplete/generic annotation fragments are not useful verified glossary names.
EXCLUDED_FRAGMENTS = frozenset("""
变化大 神经系 原发性 功能亢进 非手术 四肢肌 玻璃样 检查CT 泌尿系 瘤组织
腹腔内 右下腹 头颈部 器官组织 病情危重
""".split())
CURATED_TRAIN_TERMS = ("TAVI", "ENB", "TEE", "CT", "经导管主动脉瓣植入术", "椎间盘突出", "肺结节", "鳞状细胞癌")


def training_occurrences(word, texts):
    compact_word = normalized(word)
    if word.isascii():
        pattern = re.compile(r"(?<![a-z])"+re.escape(compact_word)+r"(?![a-z])")
        return sum(len(pattern.findall(text)) for text in texts)
    return sum(text.count(compact_word) for text in texts)


def lexicon_occurrences(text, vocabulary):
    """Longest non-overlapping glossary occurrences, not independent NER gold."""
    surface = normalized(text)
    candidates = []
    for canonical in vocabulary:
        word = normalized(canonical)
        for match in re.finditer(re.escape(word), surface):
            candidates.append((match.start(),match.end(),canonical))
    selected = []
    for start,end,word in sorted(candidates,key=lambda item:(-(item[1]-item[0]),item[0],item[2])):
        if not any(start<b and a<end for a,b,_ in selected):
            selected.append((start,end,word))
    return [dict(text=word,entity_type="TERM",start=start,end=end,covered=True)
            for start,end,word in sorted(selected)]


def prepare(args):
    import pyarrow.parquet as pq
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    source = args.output/"source"
    train = pq.read_table(source/"chinese_train.parquet",columns=["text"]).to_pylist()
    train_text = [normalized(row["text"]) for row in train]
    cmeee = json.loads((source/"CMeEE_train.json").read_text())
    frequencies = Counter()
    labels = defaultdict(Counter)
    for row in cmeee:
        for entity in row["entities"]:
            word = entity["entity"]
            if ((3<=len(word)<=24 or word in APPROVED_SHORT_TERMS)
                and re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9-]+",word)
                and word not in EXCLUDED_FRAGMENTS):
                frequencies[word] += 1
                labels[word][entity["type"]] += 1
    vocabulary = []
    for word,count in sorted(frequencies.items(),key=lambda item:(-item[1],item[0])):
        train_count = training_occurrences(word,train_text)
        if train_count or word in APPROVED_SHORT_TERMS:
            vocabulary.append(dict(canonical_text=word,cmeee_train_count=count,
                multimed_train_count=train_count,cmeee_type=labels[word].most_common(1)[0][0],
                source="CMeEE train; MultiMed train coverage, plus independently reviewed short anatomy/disease terms"))
    known = {row["canonical_text"] for row in vocabulary}
    for word in CURATED_TRAIN_TERMS:
        count = training_occurrences(word,train_text)
        if count and word not in known:
            vocabulary.append(dict(canonical_text=word,cmeee_train_count=0,
                multimed_train_count=count,cmeee_type="reviewed_medical_term",source="MultiMed train; curated terminology"))
            known.add(word)
    database = args.output/"medical.db"
    if database.exists():
        raise FileExistsError("Use a fresh output directory; refusing to overwrite a frozen medical dictionary")
    store = EntityStore(database)
    for row in vocabulary:
        store.upsert_entity(row["canonical_text"],entity_type="TERM",domain="medical",
            normalization_policy="normalize",source=row["source"])
    write_json(args.output/"vocabulary.json",vocabulary)
    for split in ("eval","test"):
        destination = args.output/"development" if split == "eval" else args.output
        destination.mkdir(exist_ok=True)
        audio_dir = destination/"audio"
        audio_dir.mkdir(exist_ok=True)
        rows = pq.read_table(source/f"chinese_{split}.parquet").to_pylist()
        selected = []
        excluded = []
        for index,row in enumerate(rows):
            if row["duration"]>30:
                excluded.append(dict(index=index,duration=row["duration"],reason="predeclared duration limit >30 seconds"))
                continue
            record_id = f"MultiMed_Chinese_{split}_{index:04d}"
            samples,rate = sf.read(io.BytesIO(row["audio"]["bytes"]),dtype="float32")
            if samples.ndim>1:
                samples=samples.mean(axis=1)
            if rate != 16000:
                from math import gcd
                divisor=gcd(rate,16000)
                samples=resample_poly(samples,16000//divisor,rate//divisor).astype("float32")
            path=audio_dir/f"{record_id}.wav"
            sf.write(path,samples,16000,subtype="PCM_16")
            entities=lexicon_occurrences(row["text"],known)
            selected.append(dict(id=record_id,speaker="MultiMed_unknown_speaker",audio_path=str(path.resolve()),
                reference=row["text"],entities=entities,
                group="covered_entity" if entities else "no_entity",dataset="MultiMed Chinese",
                split=split,parquet_row=index,original_audio_path=row["audio"]["path"],
                original_audio_sha256=hashlib.sha256(row["audio"]["bytes"]).hexdigest(),
                actual_duration_seconds=len(samples)/16000))
        write_json(destination/"selection.json",dict(seed=20261003,noise_snr_db=10.0,
            sample_count=len(selected),split=split,samples=selected,excluded=excluded,
            vocabulary_source="CMeEE train + MultiMed train only; no test ASR or test-derived aliases",
            lexicon_occurrence_definition="Longest non-overlapping canonical occurrences in supplied transcripts; not manually annotated NER"))
        print(split,"recordings",len(selected),"glossary-bearing",sum(bool(r['entities']) for r in selected),flush=True)
    source_db=ROOT.parent/"AgenticASR_entity_e5bedb3/data/entities.db"
    with sqlite3.connect(f"file:{source_db}?mode=ro",uri=True) as original,sqlite3.connect(args.output/"current_snapshot.db") as destination:
        original.backup(destination)
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT/"system").rglob("*") if path.is_file() and path.suffix in {".py",".json",".html"}}
    write_json(args.output/"provenance.json",dict(source_urls={
        "audio":"https://huggingface.co/datasets/leduckhai/MultiMed",
        "terminology":"https://huggingface.co/datasets/Aunderline/CMeEE",
        "terminology_original":"https://github.com/CBLUEbenchmark/CBLUE"},
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*") if p.is_file()},
        runtime_source_hashes=hashes,canonical_terms=len(vocabulary)))
    print("FROZEN medical canonical terms",len(vocabulary),flush=True)


def transcribe(args, development=False):
    if development:
        args.output=args.output/"development"
    decode(args)


def proposals(args):
    """Propose ASR aliases from development only for an explicit review file."""
    from rapidfuzz.distance import Levenshtein
    known={r['canonical_text'] for r in json.loads((args.output/'vocabulary.json').read_text())}
    evidence=defaultdict(list)
    for row in read_jsonl(args.output/"development/asr.jsonl"):
        reference,hypothesis=normalized(row["reference"]),normalized(row["raw_text"])
        opcodes=Levenshtein.opcodes(reference,hypothesis)
        def boundary(position):
            for op in opcodes:
                if op.src_start<=position<=op.src_end and op.src_end>op.src_start:
                    return round(op.dest_start+(position-op.src_start)*(op.dest_end-op.dest_start)/(op.src_end-op.src_start))
            return len(hypothesis)
        for entity in lexicon_occurrences(row["reference"],known):
            canonical=entity["text"]
            observed=hypothesis[boundary(entity["start"]):boundary(entity["end"])]
            if (observed and normalized(canonical)!=observed and
                2<=len(observed)<=24 and len(observed)>=len(canonical)*0.6 and
                len(observed)<=len(canonical)*1.5 and
                not any(c.isdigit() for c in observed)):
                evidence[(canonical,observed)].append(dict(id=row["id"],variant=row["variant"],
                    reference=row["reference"],raw=row["raw_text"]))
    result=[dict(canonical=word,alias=alias,count=len(rows),evidence=rows,approved=False)
        for (word,alias),rows in sorted(evidence.items())]
    write_json(args.output/"development_alias_proposals.json",result)
    for row in result:
        print(row["canonical"],"<=",row["alias"],"observations",row["count"],row["evidence"][0],flush=True)


def build_aliases(args):
    reviewed=json.loads((args.output/"reviewed_development_aliases.json").read_text())
    with sqlite3.connect(args.output/"medical.db") as source,sqlite3.connect(args.output/"medical_alias.db") as destination:
        source.backup(destination)
    store=EntityStore(args.output/"medical_alias.db")
    for row in reviewed:
        entity=store.get_entity(row["canonical"],"medical")
        store.upsert_entity(entity.canonical_text,entity_type=entity.entity_type,domain="medical",
            normalization_policy="normalize",aliases=row["aliases"],source="Reviewed MultiMed eval ASR errors; canonical from frozen terminology")
    print("BUILT development aliases",sum(len(r['aliases']) for r in reviewed),flush=True)


def match(args):
    configs={}
    for name in ["current_snapshot","medical","medical_alias"]:
        path=args.output/f"{name}.db"
        if not path.exists():continue
        definitions=EntityStore(path).list_entities(domain="medical")
        for mode in ("off","shadow","hint","auto"):
            configs[f"{name}_{mode}"]=(EntityProtector(definitions,protect_numeric_spans=False),
                EntityCandidateMatcher(definitions,selected_domain="medical",mode=mode) if mode!="off" else None)
    results=[]
    for index,row in enumerate(read_jsonl(args.output/"asr.jsonl")):
        conditions={"no_dictionary":dict(text=row["raw_text"],normalizations=[],hints=[],candidates=[],matcher_latency_ms=0.0)}
        for name,(protector,matcher) in configs.items():
            prepared=prepare_entity_segment(row["raw_text"],protector,matcher,allow_auto=True)
            conditions[name]=dict(text=prepared.baseline_text,normalizations=list(prepared.normalizations),
                hints=list(prepared.hints),candidates=[m.public_dict() for m in prepared.report.matches],
                matcher_latency_ms=prepared.report.matcher_latency_ms)
        results.append(dict(id=row["id"],variant=row["variant"],conditions=conditions))
        if (index+1)%50==0:print("MATCH",index+1,flush=True)
    (args.output/"entity_only.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["prepare","asr-eval","proposals","aliases","asr-test","match"])
    parser.add_argument("--output",type=Path,default=ROOT/"results/medical_entity_20261003")
    parser.add_argument("--asr-url",default="http://127.0.0.1:8766")
    args=parser.parse_args()
    {"prepare":prepare,"asr-eval":lambda a:transcribe(a,True),"asr-test":transcribe,
        "proposals":proposals,"aliases":build_aliases,"match":match}[args.phase](args)


if __name__=="__main__":main()
