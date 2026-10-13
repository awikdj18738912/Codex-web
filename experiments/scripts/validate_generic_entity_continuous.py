"""One continuous podcast audio check; no reference-based accuracy claim."""
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx
import numpy as np
import soundfile as sf

from evaluate_entity_library import ROOT,write_json
from evaluate_generic_entity_audio import ASR,PREFIX,GenericRecall,load_definitions,neighbor_anchored_changes
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard

OUT=ROOT/"results/generic_entity_validation_20261010/continuous_podcast"


async def main():
    if (OUT/"result.json").exists():raise FileExistsError("Continuous audio check is already complete")
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/"results/oncology_podcast_20261008/manifest.json"
    manifest=json.loads(source.read_text())
    chosen=[s for s in manifest["samples"] if 19<=s["row"]<=23]
    previous=chosen[0]["start_seconds"];waves=[]
    for sample in chosen:
        if sample["start_seconds"]!=previous:raise ValueError("Podcast pieces are not consecutive")
        audio,rate=sf.read(sample["audio_path"],dtype="float32")
        if rate!=16000 or hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()!=sample["pcm_sha256"]:
            raise ValueError("Original podcast PCM differs")
        waves.append(audio);previous=sample["end_seconds"]
    wave=np.concatenate(waves).astype("<f4");pcm=wave.tobytes()
    path=OUT/"continuous_19m_24m.wav";sf.write(path,wave,16000,subtype="FLOAT")
    library=ROOT/"data/entity_benchmarks/oncology_repaired_20261009"
    if not (OUT/"library").exists():(OUT/"library").symlink_to(library.resolve(),target_is_directory=True)
    definitions=load_definitions(OUT);recall=GenericRecall(definitions)
    write_json(OUT/"protocol.json",dict(original_source=manifest["dataset"],source_sha256=manifest["dataset_sha256"],
        start_seconds=chosen[0]["start_seconds"],end_seconds=chosen[-1]["end_seconds"],audio_seconds=len(wave)/16000,
        source_ids=[s["id"] for s in chosen],pcm_sha256=hashlib.sha256(pcm).hexdigest(),
        dictionary_sha256=hashlib.sha256((library/"entities.db").read_bytes()).hexdigest(),
        reference=None,selection="Original continuous 19–24 minute oncology discussion; exact consecutive PCM, no inserted silence"))
    async with httpx.AsyncClient(timeout=180) as client:
        async def request(endpoint,session_id=None,data=b"",params=None):
            query=dict(params or {})
            if session_id:query["session_id"]=session_id
            return (await client.post(ASR+endpoint,params=query,content=data)).raise_for_status().json()
        async def decode(terms):
            context=EntityASRContext(PREFIX+json.dumps(terms,ensure_ascii=False) if terms else "",tuple(terms))
            guard=EntityASREchoGuard(request,context)
            params=dict(language="Chinese",chunk_seconds=2.0)
            if terms:params["context"]=context.text
            started=time.perf_counter();ready=await guard.request("/stream/start",params=params)
            if ready.get("context_applied")!=bool(terms):raise ValueError("Context acknowledgement differs")
            chunks=[]
            try:
                for offset in range(0,len(pcm),64000):
                    chunks.append(await guard.request("/stream/chunk",ready["session_id"],pcm[offset:offset+64000]))
                final=await guard.request("/stream/finish",ready["session_id"])
            except BaseException:
                await guard.request("/stream/cancel",ready["session_id"]);raise
            finally:guard.close()
            return dict(ready=ready,chunk_payloads=chunks,final_payload=final,seconds=time.perf_counter()-started,guard=guard.snapshot())
        baseline=await decode([]);original=baseline["final_payload"]["text"]
        print("BASELINE",round(baseline["seconds"],2),"s",flush=True)
        started=time.perf_counter();candidates,terms=recall.select(original);retrieval=time.perf_counter()-started
        replay=await decode(terms) if terms else None
        started=time.perf_counter()
        corrected,changes=neighbor_anchored_changes(original,replay["final_payload"]["text"],candidates) if replay else (original,[])
        verification=time.perf_counter()-started
        write_json(OUT/"result.json",dict(baseline=baseline,replay=replay,candidates=candidates,prompt_terms=terms,
            corrected=corrected,changes=changes,retrieval_seconds=retrieval,verification_seconds=verification))
        extra=replay["seconds"] if replay else 0.0
        lines=["# 连续五分钟医疗播客：通用实体原音复核", "",
            "原视频播客 19:00–24:00 的连续原音，直接拼回五个相邻存档块，没有插入静音、换说话人或重排内容。无标准字幕，不计算 CER 或实体准确率。", "",
            f"音频 {len(wave)/16000:.1f} 秒；无提示 ASR {baseline['seconds']:.2f} 秒；候选检索 {retrieval:.3f} 秒；额外原音复核 {extra:.2f} 秒；跨度核对 {verification:.3f} 秒。", "",
            f"提示候选 {len(terms)} 个，实际修改 {len(changes)} 个实体跨度。完整初稿、候选提示、重识别响应与修改记录见 result.json。", "",
            "| 初稿跨度 | 词库标准名 |", "|---|---|" ]
        for change in changes:lines.append(f"|{change['observed']}|{change['canonical']}|")
        lines += ["", "这些修改只是规则和原音提示重识别的决定，没有人工标准转录核准，不能直接计为正确恢复。连续播客检查没有运行额外精修；完整真实精修保留效果见上一级的 403 场景验证。"]
        (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
        print("COMPLETE",len(changes),"changes; replay",round(extra,2),"s",flush=True)


if __name__=="__main__":asyncio.run(main())
