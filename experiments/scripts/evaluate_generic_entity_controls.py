"""Fresh lexical-negative audio controls for the generic entity prototype."""
import argparse
import asyncio
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import wave

import httpx
import soundfile as sf

from evaluate_entity_library import normalized, read_jsonl, write_json, append_jsonl
from evaluate_generic_entity_audio import (
    SOURCE, OUT as PARENT, DOMAINS, ASR, PREFIX, GenericRecall,
    load_definitions, neighbor_anchored_changes,
)
from evaluate_speechio_professional import split_is_train, term_counts
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard

OUT = PARENT / "negative_controls"
PER_DOMAIN = 8


def prepare():
    import pyarrow.parquet as pq
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Refusing to overwrite frozen negative controls")
    plan = []
    for domain, folder in DOMAINS.items():
        if domain == "medical_faq":
            continue
        old = json.loads((folder / "protocol.json").read_text())
        parquet = SOURCE / "source" / f"{old['samples'][0]['source_subset']}.parquet"
        words = [normalized(word) for d in load_definitions(folder) for word in (d.canonical_text,*d.aliases)]
        rows = pq.read_table(parquet, columns=["segment_id","text","duration"]).to_pylist()
        pool = [r for r in rows if not split_is_train(r["segment_id"]) and 3<=r["duration"]<=20
                and not any(w in normalized(r["text"]) for w in words)]
        chosen = sorted(pool, key=lambda r:hashlib.sha256(("generic-negative-v1:"+r["segment_id"]).encode()).hexdigest())[:PER_DOMAIN]
        if len(chosen) != PER_DOMAIN:
            raise ValueError(f"Insufficient lexical-negative audio for {domain}")
        selected = {r["segment_id"]:r for r in chosen}
        for row in pq.read_table(parquet,columns=["segment_id","audio"]).to_pylist():
            if row["segment_id"] not in selected:
                continue
            metadata = selected[row["segment_id"]]
            content = row["audio"]["bytes"]
            with wave.open(io.BytesIO(content),"rb") as audio:
                if (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())!=(16000,1,2):
                    raise ValueError("Unexpected audio format")
                pcm=(np.frombuffer(audio.readframes(audio.getnframes()),dtype="<i2").astype("float32")/32768).astype("<f4").tobytes()
            audio_dir=OUT/domain/"audio";audio_dir.mkdir(parents=True,exist_ok=True)
            path=audio_dir/(hashlib.sha256(metadata["segment_id"].encode()).hexdigest()[:20]+".wav")
            path.write_bytes(content)
            plan.append(dict(id=f"negative_{domain}_{metadata['segment_id']}",domain=domain,
                source_id=metadata["segment_id"],reference=metadata["text"],audio_path=str(path.resolve()),
                pcm_sha256=hashlib.sha256(pcm).hexdigest(),audio_sha256=hashlib.sha256(content).hexdigest()))
        print(domain,len(chosen),"negative audio prepared",flush=True)
    write_json(OUT/"protocol.json",dict(selection="Eight deterministic heldout 3–20-second utterances per domain with no known canonical/alias in reference, selected before ASR; no type-dependent settings",samples=plan))


async def run():
    plan=json.loads((OUT/"protocol.json").read_text())["samples"]
    destination=OUT/"outputs.jsonl";done={r["id"] for r in read_jsonl(destination)}
    recalls={domain:GenericRecall(load_definitions(folder)) for domain,folder in DOMAINS.items()}
    async with httpx.AsyncClient(timeout=180) as client:
        async def request(endpoint,session_id=None,data=b"",params=None):
            query=dict(params or {})
            if session_id:query["session_id"]=session_id
            return (await client.post(ASR+endpoint,params=query,content=data)).raise_for_status().json()
        async def decode(pcm,terms):
            context=EntityASRContext(PREFIX+json.dumps(terms,ensure_ascii=False) if terms else "",tuple(terms))
            guard=EntityASREchoGuard(request,context)
            params=dict(language="Chinese",chunk_seconds=2.0)
            if terms:params["context"]=context.text
            ready=await guard.request("/stream/start",params=params)
            if ready.get("context_applied")!=bool(terms):raise RuntimeError("Context acknowledgement mismatch")
            try:
                for offset in range(0,len(pcm),64000):
                    await guard.request("/stream/chunk",ready["session_id"],pcm[offset:offset+64000])
                final=await guard.request("/stream/finish",ready["session_id"])
                return final
            except BaseException:
                await guard.request("/stream/cancel",ready["session_id"])
                raise
            finally:guard.close()
        for index,sample in enumerate(plan):
            if sample["id"] in done:continue
            audio,rate=sf.read(sample["audio_path"],dtype="float32");pcm=audio.astype("<f4").tobytes()
            if rate!=16000 or audio.ndim!=1 or hashlib.sha256(pcm).hexdigest()!=sample["pcm_sha256"]:
                raise ValueError("PCM mismatch")
            off=await decode(pcm,[]);original=off["text"]
            candidates,terms=recalls[sample["domain"]].select(original)
            on=await decode(pcm,terms) if terms else None
            corrected,changes=neighbor_anchored_changes(original,on["text"],candidates) if terms else (original,[])
            append_jsonl(destination,dict(id=sample["id"],original=original,corrected=corrected,
                candidates=candidates,prompt_terms=terms,changes=changes,baseline_payload=off,replay_payload=on))
            print(index+1,"/",len(plan),sample["domain"],"changes",[(c["observed"],c["canonical"]) for c in changes],flush=True)


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    converter,itn=opencc.OpenCC("t2s"),ZhITNNormalizer()
    def norm(text):return normalized(converter.convert(itn.normalize(text).text))
    plan=json.loads((OUT/"protocol.json").read_text())["samples"]
    rows={r["id"]:r for r in read_jsonl(OUT/"outputs.jsonl")}
    if set(rows)!={s["id"] for s in plan}:raise ValueError("Incomplete evaluation")
    summary={};changed=[]
    for domain in {s["domain"] for s in plan}:
        definitions=load_definitions(DOMAINS[domain]);names=[d.canonical_text for d in definitions];aliases={d.canonical_text:list(d.aliases) for d in definitions}
        counts=Counter()
        for sample in (s for s in plan if s["domain"]==domain):
            row=rows[sample["id"]];ref=norm(sample["reference"])
            before=Levenshtein.distance(ref,norm(row["original"]));after=Levenshtein.distance(ref,norm(row["corrected"]))
            counts.update(samples=1,replays=int(bool(row["prompt_terms"])),reference_chars=len(ref),
                original_errors=before,corrected_errors=after,improved=int(after<before),worsened=int(after>before),
                baseline_dictionary_mentions=sum(term_counts(row["original"],names,aliases).values()),
                final_dictionary_mentions=sum(term_counts(row["corrected"],names,aliases).values()))
            if row["changes"]:changed.append(dict(reference=sample["reference"],**row))
        summary[domain]=dict(counts)
    write_json(OUT/"summary.json",dict(domains=summary,unique_audio=len({s["audio_sha256"] for s in plan}),changed=changed))
    lines=["# 通用实体原音方案：新增词库负例", "", "在重识别前，按固定哈希选取每域八条不含该库标准名或登记别名的留出音频（3–20 秒）。沿用同一候选阈值和相邻实体锚点规则。", "", "|领域|样本|重识别|改善/退化|无提示/处理后词库提及|", "|---|---:|---:|---:|---:|"]
    for domain,c in sorted(summary.items()):lines.append(f"|{domain}|{c['samples']}|{c['replays']}|{c['improved']}/{c['worsened']}|{c['baseline_dictionary_mentions']}/{c['final_dictionary_mentions']}|")
    lines += ["", "来源仍是既有 SpeechIO 子集，因此是新的音频负例，而非新的数据来源或独立说话人评测。无参考词库词的出现都保留为诊断，不凭参考标签直接认定原音内容。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","run","report"));stage=p.parse_args().stage
    if stage=="prepare":prepare()
    elif stage=="run":asyncio.run(run())
    else:report()
