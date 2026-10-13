"""Real Web/Refiner validation of the frozen generic audio entity prototype.

Cached ASR responses come from actual audio decoding. Only accepted entity
spans are revised at finish; all Refiner calls in the paired replay are real.
No production function is changed by this experimental adapter.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import httpx
import numpy as np
import soundfile as sf

from benchmark_multimed_entities import Runner, MODEL, digest, require_complete
from evaluate_entity_library import ROOT, read_jsonl, append_jsonl, write_json, normalized
from evaluate_generic_entity_audio import (
    OUT as SOURCE, DOMAINS, ASR, PREFIX, GenericRecall, load_definitions,
    neighbor_anchored_changes,
)
from evaluate_speechio_professional import term_counts
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard

OUT = ROOT / "results/generic_entity_validation_20261010"


def revise_record(record, corrected, changes, *, force=False):
    """Keep original stream hypotheses; publish all revised segments at finish."""
    result = deepcopy(record)
    if not changes and not force:
        return result
    original = record["final_payload"]["text"]
    segments = {}
    for payload in [*record["chunk_payloads"], record["final_payload"]]:
        for segment in payload.get("completed_segments", []):
            identifier = segment["segment_id"]
            if identifier in segments and segments[identifier]["text"] != segment["text"]:
                raise ValueError("An already completed ASR segment changed")
            segments[identifier] = deepcopy(segment)
    cursor, applied = 0, 0
    for segment in segments.values():
        text = segment["text"].strip()
        start = original.find(text, cursor)
        if start < 0:
            raise ValueError("Cannot map completed segment to authoritative ASR text")
        end = start + len(text)
        local = [c for c in changes if start <= c["start"] < c["end"] <= end]
        for change in sorted(local, key=lambda c:c["start"], reverse=True):
            a,b = change["start"]-start, change["end"]-start
            if text[a:b] != change["observed"]:
                raise ValueError("Entity offsets no longer refer to the baseline")
            text = text[:a] + change["canonical"] + text[b:]
            applied += 1
        segment["text"] = text
        if local:
            segment["confidence"] = None
            segment["confidence_metadata"] = dict(
                confidence=None, confidence_calibrated=False,
                confidence_covers_full_state=False, confidence_covers_full_text=False,
                confidence_scope="unscored_entity_revision", confidence_source="entity_audio_repair",
            )
        cursor = end
    if applied != len(changes):
        raise ValueError("An accepted entity change crosses/misses recorded segments")
    final = result["final_payload"]
    final["text"] = corrected
    final["completed_segments"] = list(segments.values())
    final["entity_asr_context_guard_revision"] = 1
    final["entity_audio_repair"] = dict(changes=changes, experimental=True)
    # Original token probabilities do not score newly corrected spellings.
    final.update(confidence=None, confidence_calibrated=False,
                 confidence_covers_full_state=False, confidence_covers_full_text=False,
                 confidence_scope="unscored_entity_revision", confidence_source="entity_audio_repair",
                 confidence_mean_logprob=None, confidence_min_probability=None, confidence_token_count=0)
    return result


def make_entities(text, definitions):
    hits = []
    for definition in definitions:
        for surface in (definition.canonical_text, *definition.aliases):
            import re
            for match in re.finditer(re.escape(surface), text, flags=re.IGNORECASE if surface.isascii() else 0):
                hits.append(dict(text=definition.canonical_text, start=match.start(), end=match.end()))
    chosen = []
    for hit in sorted(hits, key=lambda e:(-(e["end"]-e["start"]),e["start"],e["text"])):
        if not any(hit["start"]<other["end"] and other["start"]<hit["end"] for other in chosen):
            chosen.append(hit)
    return sorted(chosen,key=lambda e:e["start"])


def prepare():
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Validation inputs are already frozen")
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((SOURCE/"plan.json").read_text())["samples"]
    treatments={r["id"]:r for r in read_jsonl(SOURCE/"outputs_neighbor.jsonl")}
    samples,annotations,baseline=[],{},[]
    for domain,folder in DOMAINS.items():
        library=OUT/"libraries"/domain
        shutil.copytree(folder/"library",library)
        protocol=json.loads((folder/"protocol.json").read_text())
        gt=json.loads((folder/"library/sample_entities.json").read_text())
        raw={r["id"]:r for r in read_jsonl(folder/"asr_off.jsonl")}
        for sample in protocol["samples"]:
            samples.append(dict(sample,domain=domain,cohort="short_positive",library_key=domain))
            annotations[sample["id"]]=gt[sample["id"]]
            baseline.append(raw[sample["id"]])
    for record in baseline:
        append_jsonl(OUT/"asr_baseline.jsonl",record)
        treatment=treatments[record["id"]]
        append_jsonl(OUT/"asr_generic.jsonl",revise_record(record,treatment["corrected"],treatment["changes"]))
        if treatment["changes"]:
            append_jsonl(OUT/"asr_revision_control.jsonl",revise_record(record,record["final_payload"]["text"],[],force=True))
    for sample in json.loads((SOURCE/"negative_controls/protocol.json").read_text())["samples"]:
        samples.append(dict(sample,cohort="negative",library_key=sample["domain"]))
        annotations[sample["id"]]=[]
    # Library is fixed before selecting or decoding long stress audio.
    from system.entity_store import EntityStore
    union=OUT/"libraries"/"mixed_long";union.mkdir(parents=True,exist_ok=True)
    store=EntityStore(union/"entities.db")
    by_name={}
    for folder in DOMAINS.values():
        for d in load_definitions(folder):
            if d.canonical_text in by_name:
                old=by_name[d.canonical_text]
                aliases=set(old.aliases)|set(d.aliases)
                d=type(d)(d.entity_id,d.canonical_text,old.entity_type,"medical","normalize",0,tuple(sorted(aliases)))
            by_name[d.canonical_text]=d
    for d in by_name.values():
        store.upsert_entity(d.canonical_text,entity_type=d.entity_type,domain="medical",normalization_policy="normalize",aliases=d.aliases,source="Union of previously frozen evaluation dictionaries; long-audio stress fixture")
    write_json(union/"entities.json",[dict(canonical_text=d.canonical_text,entity_type=d.entity_type,domain="medical",normalization_policy="normalize") for d in by_name.values()])
    # This is a known-error development stress test, explicitly not fresh
    # accuracy evidence. Whole real utterances retain complete references.
    changed=[s for s in samples if s["cohort"]=="short_positive" and treatments[s["id"]]["changes"]]
    healthy=[s for s in samples if s["cohort"]=="short_positive" and not treatments[s["id"]]["changes"]]
    healthy.sort(key=lambda s:hashlib.sha256(("long-generic-stress:"+s["id"]).encode()).hexdigest())
    sequence=changed+healthy
    definitions=store.list_entities(domain="medical")
    for target_seconds in (60,180,300):
        audio_parts,references,ids,total=[],[],[],0.0
        for source in sequence:
            audio,rate=sf.read(source["audio_path"],dtype="float32")
            if rate!=16000 or audio.ndim!=1:raise ValueError("Unexpected stress input")
            audio_parts.extend([audio,np.zeros(4000,dtype="float32")])
            references.append(source["reference"]);ids.append(source["id"])
            total+=(len(audio)+4000)/16000
            if total>=target_seconds:break
        wave=np.concatenate(audio_parts).astype("<f4")
        rid=f"generic_long_{target_seconds:03d}"
        path=OUT/"long_audio"/(rid+".wav");path.parent.mkdir(exist_ok=True)
        sf.write(path,wave,16000,subtype="FLOAT")
        sample=dict(id=rid,row=len(samples),reference="。".join(references),audio_path=str(path.resolve()),
            pcm_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),duration_seconds=len(wave)/16000,
            domain="mixed_long",cohort="long_stress",library_key="mixed_long",source_ids=ids,
            caveat="Concatenated real known-development utterances with 250ms gaps, complete references; not continuous natural long speech")
        samples.append(sample);annotations[rid]=make_entities(sample["reference"],definitions)
    frozen={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"system").rglob("*.py")}
    libraries={str(p.relative_to(OUT)):digest(p) for p in (OUT/"libraries").rglob("entities.db")}
    write_json(OUT/"manifest.json",dict(samples=samples,sample_count=len(samples),source_hashes=frozen,library_hashes=libraries,
        model=str(MODEL),conditions=["baseline","generic","unchanged_final_revision_on_changed_cases"],
        changed_positive_samples=len(changed),asr_replay="Real recorded stream responses, with generic modifications only at finish",
        refiner="Real TransformersRefiner in all cells; tri_state, final off, zh-itn; same dictionary in both arms"))
    write_json(OUT/"sample_entities.json",annotations)
    print("PREPARED",len(samples),"samples;",len(changed),"known corrected utterances",flush=True)


async def collect():
    manifest=json.loads((OUT/"manifest.json").read_text())
    done={r["id"] for r in read_jsonl(OUT/"asr_generic.jsonl")}
    recalls={}
    async with httpx.AsyncClient(timeout=180) as client:
        health=(await client.get(ASR+"/health")).raise_for_status().json()
        if not health.get("context_supported"):raise RuntimeError("ASR context unavailable")
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
            started=time.perf_counter();chunks=[]
            try:
                for offset in range(0,len(pcm),64000):
                    chunks.append(await guard.request("/stream/chunk",ready["session_id"],pcm[offset:offset+64000]))
                final=await guard.request("/stream/finish",ready["session_id"])
            except BaseException:
                await guard.request("/stream/cancel",ready["session_id"]);raise
            finally:guard.close()
            return dict(ready=ready,chunk_payloads=chunks,final_payload=final,context_text=context.text,
                context_terms=context.terms,elapsed_seconds=time.perf_counter()-started,guard=guard.snapshot())
        for sample in manifest["samples"]:
            if sample["id"] in done:continue
            folder=OUT/"libraries"/sample["library_key"]
            # load_definitions expects a library child, so its parent is used.
            if sample["library_key"] not in recalls:
                fixture=OUT/"fixtures"/sample["library_key"];fixture.mkdir(parents=True,exist_ok=True)
                if not (fixture/"library").exists():
                    (fixture/"library").symlink_to(folder.resolve(),target_is_directory=True)
                recalls[sample["library_key"]]=GenericRecall(load_definitions(fixture))
            audio,rate=sf.read(sample["audio_path"],dtype="float32");pcm=audio.astype("<f4").tobytes()
            if rate!=16000 or hashlib.sha256(pcm).hexdigest()!=sample["pcm_sha256"]:raise ValueError("Audio differs")
            off=dict(await decode(pcm,[]),id=sample["id"],reference=sample["reference"],pcm_sha256=sample["pcm_sha256"])
            candidates,terms=recalls[sample["library_key"]].select(off["final_payload"]["text"])
            on=await decode(pcm,terms) if terms else None
            corrected,changes=neighbor_anchored_changes(off["final_payload"]["text"],on["final_payload"]["text"],candidates) if on else (off["final_payload"]["text"],[])
            append_jsonl(OUT/"asr_baseline.jsonl",off)
            append_jsonl(OUT/"asr_generic.jsonl",revise_record(off,corrected,changes))
            if changes:append_jsonl(OUT/"asr_revision_control.jsonl",revise_record(off,off["final_payload"]["text"],[],force=True))
            append_jsonl(OUT/"new_audio_requests.jsonl",dict(id=sample["id"],candidates=candidates,prompt_terms=terms,
                replay=on,corrected=corrected,changes=changes,baseline_seconds=off["elapsed_seconds"],
                replay_seconds=on["elapsed_seconds"] if on else 0.0,audio_seconds=len(audio)/16000))
            print("COLLECT",sample["id"],"seconds",len(audio)/16000,"changes",[(c["observed"],c["canonical"]) for c in changes],flush=True)


class ValidationRunner(Runner):
    def run_validation(self):
        from fastapi.testclient import TestClient
        from system.zh_itn_bridge import DEFAULT_LIBRARY
        manifest=json.loads((OUT/"manifest.json").read_text())
        for name,expected in manifest["source_hashes"].items():
            if digest(ROOT/name)!=expected:raise ValueError("Production code changed during validation: "+name)
        ids=[s["id"] for s in manifest["samples"]]
        sources={arm:require_complete(OUT/f"asr_{arm}.jsonl",ids) for arm in ("baseline","generic")}
        sources["revision_control"]={r["id"]:r for r in read_jsonl(OUT/"asr_revision_control.jsonl")}
        done={arm:{r["id"] for r in read_jsonl(OUT/f"{arm}.jsonl")} for arm in sources}
        owner=self
        async def replay(url,endpoint,session_id=None,data=b"",params=None,**kwargs):
            if endpoint=="/stream/start":
                if (params or {}).get("context"):raise ValueError("Unexpected initial ASR prompt")
                owner.replay_index=0;return deepcopy(owner.active["ready"])
            if endpoint=="/stream/chunk":
                result=owner.active["chunk_payloads"][owner.replay_index];owner.replay_index+=1
                return deepcopy(result)
            if endpoint=="/stream/finish":return deepcopy(owner.active["final_payload"])
            if endpoint=="/stream/cancel":return dict(cancelled=True)
            raise ValueError(endpoint)
        self.web._stream_request_async=replay
        query=urlencode(dict(mode="streaming",domain="medical",language="Chinese",asr_chunk_seconds=2.0,entity_asr_context=0))
        for library_key in dict.fromkeys(s["library_key"] for s in manifest["samples"]):
            samples=[s for s in manifest["samples"] if s["library_key"]==library_key]
            with ExitStack() as stack:
                clients={}
                for arm in sources:
                    log=OUT/"service_logs"/library_key/(arm+".jsonl");log.parent.mkdir(parents=True,exist_ok=True)
                    app=self.web.create_app(MODEL,"cuda:0",ASR,"Chinese",256,log,
                        entity_db=OUT/"libraries"/library_key/"entities.db",entity_fuzzy_mode="auto",
                        refinement_gate_mode="tri_state",final_refinement_mode="off",zh_itn_library=DEFAULT_LIBRARY,
                        entity_asr_context=False,entity_asr_echo_guard=False)
                    clients[arm]=stack.enter_context(TestClient(app))
                for index,sample in enumerate(samples):
                    arms=["baseline","generic"] if index%2==0 else ["generic","baseline"]
                    if sample["id"] in sources["revision_control"]:arms.append("revision_control")
                    for arm in arms:
                        if sample["id"] in done[arm]:continue
                        self.active=sources[arm][sample["id"]];self.calls.clear();events=[];start=time.perf_counter()
                        with clients[arm].websocket_connect("/ws/stream?"+query) as socket:
                            ready=socket.receive_json()
                            if ready.get("event")!="ready" or ready.get("entity_asr_context_applied"):raise ValueError(ready)
                            for _ in self.active["chunk_payloads"]:
                                socket.send_bytes(b"\0\0\0\0")
                                while True:
                                    event=socket.receive_json();events.append(event)
                                    if event.get("event")=="error":raise RuntimeError(event)
                                    if event.get("event")=="chunk_ack":break
                            socket.send_json(dict(event="finish"))
                            while True:
                                event=socket.receive_json();events.append(event)
                                if event.get("event")=="error":raise RuntimeError(event)
                                if event.get("event")=="final":break
                        if self.replay_index!=len(self.active["chunk_payloads"]):raise ValueError("Incomplete ASR stream replay")
                        if event.get("raw_text","")!=self.active["final_payload"].get("text","").strip():raise ValueError("Web final source differs from assigned ASR input")
                        append_jsonl(OUT/f"{arm}.jsonl",dict(id=sample["id"],condition=arm,result=event,
                            model_calls=list(self.calls),elapsed_seconds=time.perf_counter()-start,
                            event_counts=dict(Counter(e.get("event") for e in events))))
                        print("WEB",arm,sample["id"],event.get("clean_text",""),flush=True)


def run():
    args=SimpleNamespace(output=OUT,library=OUT/"libraries",model=MODEL,device="cuda:0",asr_url=ASR,mode="streaming",chunk_seconds=2.0)
    ValidationRunner(args).run_validation()


def report():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.entity_store import EntityStore
    converter,itn=opencc.OpenCC("t2s"),ZhITNNormalizer();cache={}
    def norm(text):
        if text not in cache:cache[text]=normalized(converter.convert(itn.normalize(text).text))
        return cache[text]
    manifest=json.loads((OUT/"manifest.json").read_text());ids=[s["id"] for s in manifest["samples"]]
    for name,expected in manifest["source_hashes"].items():
        if digest(ROOT/name)!=expected:raise ValueError("Production code changed: "+name)
    for name,expected in manifest["library_hashes"].items():
        if digest(OUT/name)!=expected:raise ValueError("Experiment library changed: "+name)
    rows={arm:require_complete(OUT/f"{arm}.jsonl",ids) for arm in ("baseline","generic")}
    revisions={r["id"]:r for r in read_jsonl(OUT/"revision_control.jsonl")}
    gt=json.loads((OUT/"sample_entities.json").read_text())
    raw={arm:require_complete(OUT/f"asr_{arm}.jsonl",ids) for arm in rows}
    requests={r["id"]:r for r in read_jsonl(OUT/"new_audio_requests.jsonl")}
    summary,details,definitions={},[],{}
    for sample in manifest["samples"]:
        rid=sample["id"];key=sample["library_key"]
        if key not in definitions:definitions[key]=EntityStore(OUT/"libraries"/key/"entities.db").list_entities(domain="medical")
        names=[d.canonical_text for d in definitions[key]];aliases={d.canonical_text:list(d.aliases) for d in definitions[key]}
        expected=Counter(e["text"] for e in gt[rid]);reference=norm(sample["reference"])
        cohort=sample["cohort"]+":"+sample["domain"]
        group=summary.setdefault(cohort,{arm:Counter() for arm in rows})
        results={};stage_hits={}
        for arm in rows:
            text=rows[arm][rid]["result"].get("clean_text","");found=term_counts(text,names,aliases)
            canonical=term_counts(text,names);ref_found=term_counts(sample["reference"],names,aliases)
            hit=sum(min(n,found[t]) for t,n in expected.items());error=Levenshtein.distance(reference,norm(text))
            raw_found=term_counts(raw[arm][rid]["final_payload"]["text"],names,aliases)
            raw_hit=sum(min(n,raw_found[t]) for t,n in expected.items())
            group[arm].update(samples=1,mentions=sum(expected.values()),hits=hit,canonical_hits=sum(min(n,canonical[t]) for t,n in expected.items()),
                reference_chars=len(reference),errors=error,model_calls=len(rows[arm][rid]["model_calls"]),raw_hits=raw_hit,
                unreferenced=sum(max(0,n-ref_found[t]) for t,n in found.items()))
            results[arm]=dict(text=text,hits=hit,raw_hits=raw_hit,errors=error,calls=len(rows[arm][rid]["model_calls"]))
            stage_hits[arm]=dict(raw=raw_hit,final=hit)
        delta=results["generic"]["errors"]-results["baseline"]["errors"]
        group["generic"].update(improved=int(delta<0),worsened=int(delta>0),gains=max(0,results["generic"]["hits"]-results["baseline"]["hits"]),
            losses=max(0,results["baseline"]["hits"]-results["generic"]["hits"]))
        if rid in revisions:results["revision_control"]=dict(text=revisions[rid]["result"].get("clean_text",""),
            errors=Levenshtein.distance(reference,norm(revisions[rid]["result"].get("clean_text",""))))
        if results["generic"]["text"]!=results["baseline"]["text"] or raw["generic"][rid]["final_payload"]["text"]!=raw["baseline"][rid]["final_payload"]["text"]:
            details.append(dict(id=rid,cohort=sample["cohort"],reference=sample["reference"],results=results,
                raw={arm:raw[arm][rid]["final_payload"]["text"] for arm in rows},changes=raw["generic"][rid]["final_payload"].get("entity_audio_repair",{}).get("changes",[])))
    final={key:{arm:dict(c,CER=c["errors"]/c["reference_chars"]) for arm,c in group.items()} for key,group in summary.items()}
    write_json(OUT/"summary.json",final);write_json(OUT/"changes.json",details)
    short={arm:Counter() for arm in rows}
    for sample_group,arms in summary.items():
        if sample_group.startswith("long_stress:"):continue
        for arm,count in arms.items():short[arm].update(count)
    short={arm:dict(c,CER=c["errors"]/c["reference_chars"]) for arm,c in short.items()}
    write_json(OUT/"short_summary.json",short)
    cpu={r["id"]:r for r in json.loads((OUT/"long_cpu_performance.json").read_text())} if (OUT/"long_cpu_performance.json").exists() else {}
    long=[]
    for sample in manifest["samples"]:
        if sample["cohort"]!="long_stress":continue
        request=requests[sample["id"]]
        expected=Counter(e["text"] for e in gt[sample["id"]]);defs=definitions[sample["library_key"]]
        names=[d.canonical_text for d in defs];aliases={d.canonical_text:list(d.aliases) for d in defs}
        extra=cpu.get(sample["id"],{})
        stage_metrics={}
        for arm in rows:
            text=rows[arm][sample["id"]]["result"].get("clean_text","");found=term_counts(text,names,aliases)
            ref=norm(sample["reference"])
            stage_metrics[arm]=dict(final_hits=sum(min(n,found[t]) for t,n in expected.items()),mentions=sum(expected.values()),CER=Levenshtein.distance(ref,norm(text))/len(ref))
        long.append(dict(id=sample["id"],audio_seconds=request["audio_seconds"],baseline_asr_seconds=request["baseline_seconds"],
            extra_asr_seconds=request["replay_seconds"],extra_asr_rtf=request["replay_seconds"]/request["audio_seconds"],
            prompt_terms=request["prompt_terms"],changes=request["changes"],
            baseline_web_seconds=rows["baseline"][sample["id"]]["elapsed_seconds"],generic_web_seconds=rows["generic"][sample["id"]]["elapsed_seconds"],
            cpu_retrieval_seconds=extra.get("retrieval_seconds"),cpu_patch_verification_seconds=extra.get("patch_verification_seconds"),stage_metrics=stage_metrics))
    write_json(OUT/"long_performance.json",long)
    lines=["# 通用实体方案：真实精修与长音频验证", "", "真实 ASR 已保存的响应进入实际 WebSocket、实体保护、tri_state 精修和数字归一化流程；Refiner 在每个条件中真实推理。ASR 复核结果只在结束时更新实体跨度与全部分段事件，没有改生产代码。", "",
        "|场景|样本|无通用修正最终命中|通用修正最终命中|原始/修正最终 CER|改善/退化句|", "|---|---:|---:|---:|---:|---:|"]
    for key,s in final.items():
        a,b=s["baseline"],s["generic"]
        lines.append(f"|{key}|{a['samples']}|{a['hits']}/{a['mentions']}|{b['hits']}/{b['mentions']}|{a['CER']:.2%}/{b['CER']:.2%}|{b['improved']}/{b['worsened']}|")
    lines += ["", f"400 个短音频场景最终命中 {short['baseline']['hits']}/{short['baseline']['mentions']}→{short['generic']['hits']}/{short['generic']['mentions']}；改善 {short['generic']['improved']} 个场景，退化 {short['generic']['worsened']} 个场景。", "",
        "## 长音频压力测试", "", "使用带完整参考的真实短片段拼接，包含此前开发错例；是长度/流程压力测试，不是自然连续语音或独立泛化准确率。", "", "|音频秒数|最终命中 基线/修正|最终 CER 基线/修正|额外 ASR 复核秒数|额外复核 RTF|基线/修正 Web 精修秒数|", "|---|---:|---:|---:|---:|---:|"]
    for row in long:
        a,b=row['stage_metrics']['baseline'],row['stage_metrics']['generic']
        lines.append(f"|{row['audio_seconds']:.1f}|{a['final_hits']}/{a['mentions']}→{b['final_hits']}/{b['mentions']}|{a['CER']:.2%}/{b['CER']:.2%}|{row['extra_asr_seconds']:.2f}|{row['extra_asr_rtf']:.3f}|{row['baseline_web_seconds']:.2f}/{row['generic_web_seconds']:.2f}|")
    lines += ["", "长音频的 CPU 候选检索与跨度核对另在相同缓存响应上复测，见 long_cpu_performance.json；不是额外 ASR 调用，且已断言候选和修改与真实复核结果一致。"]
    lines += ["", "## 核对口径", "", "两组都加载相同的已有实体库；基线不使用新的通用修正，保留已有实体后处理。命中包括标准名或登记别名。初稿与修正后文本的原始命中和最终命中都保存在 summary.json。", 
        "正例使用上一轮真实 ASR 流式响应；负例重新实际解码以保存完整流式事件。最终修订后原 token logprob 被标记不覆盖修订文本，不假造纠正置信度。", 
        "每个发生修正的场景增加‘不改文字、只重发分段并取消最终置信度’控制组，结果保存在 revision_control.jsonl，用于识别重算的影响。逐条损失、错误增加与模型调用见 changes.json。", 
        "全量生产源文件哈希和实验数据库哈希保存在 manifest.json。正式实体库未写入；此实验没有接入当前服务。"]
    if (OUT/"control_audit.json").exists():
        audit=json.loads((OUT/"control_audit.json").read_text())
        lines += ["", f"执行 {len(rows['baseline'])+len(rows['generic'])+len(revisions)} 次 Web 会话、{audit['actual_refiner_calls']} 次实际精修调用。{audit['revision_control_samples']} 个只重发分段控制组与基线完全一致；{audit['unmodified_input_samples']} 个没有实体修改的场景在两组中最终文字完全一致。详细审查见 control_audit.json、final_text_diff_audit.json 与 scope_verification.json。",
            "原始音频实体修改恢复了 13 次已知短音频实体；最终只增加 12 次命中。连续两次‘附加刑’被既有精修合并成一次，未改字形控制组也合并‘附加型’，这项限制保留在统计中。"]
    if (OUT/"continuous_podcast/result.json").exists():
        continuous=json.loads((OUT/"continuous_podcast/result.json").read_text())
        lines += ["", "## 真实连续五分钟播客补测", "",
            f"原视频 19:00–24:00 的相邻 PCM 原样连接，无插入静音或重排。无提示 ASR {continuous['baseline']['seconds']:.2f} 秒，额外原音复核 {continuous['replay']['seconds']:.2f} 秒，候选检索 {continuous['retrieval_seconds']:.2f} 秒。产生 {len(continuous['changes'])} 处规范到‘黑色素瘤’的决定。",
            "缺标准字幕，未计算 CER 或实体准确率，也没有把这两处计入上表恢复次数。连续播客补测未额外运行精修。完整记录见 continuous_podcast/REPORT.md 与 result.json。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","collect","run","report"));stage=parser.parse_args().stage
    if stage=="prepare":prepare()
    elif stage=="collect":asyncio.run(collect())
    elif stage=="run":run()
    else:report()
