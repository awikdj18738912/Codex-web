"""Exercise the production audio wrapper through Web, with real Refiner.

Full cohort reuses recorded real ASR responses and verifies identical PCM.
A subsequent five-case check uses the actual ASR service as well.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import soundfile as sf

from benchmark_multimed_entities import Runner,MODEL,digest,require_complete
from evaluate_entity_library import ROOT,read_jsonl,append_jsonl,write_json
from system.entity_asr_context import context_digest

SOURCE=ROOT/"results/generic_entity_validation_20261010"
OUT=ROOT/"results/generic_entity_integration_20261010"
ASR="http://127.0.0.1:8766"


def prepare():
    if (OUT/"manifest.json").exists():raise FileExistsError("Integration cohort already frozen")
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE/"libraries",OUT/"libraries")
    shutil.copy2(SOURCE/"sample_entities.json",OUT/"sample_entities.json")
    shutil.copy2(SOURCE/"long_cpu_performance.json",OUT/"long_cpu_performance.json")
    shutil.copy2(SOURCE/"new_audio_requests.jsonl",OUT/"new_audio_requests.jsonl")
    manifest=json.loads((SOURCE/"manifest.json").read_text())
    manifest.update(source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"system").rglob("*.py")},
        library_hashes={str(p.relative_to(OUT)):digest(p) for p in (OUT/"libraries").rglob("entities.db")},
        integration="Production EntityAudioRepair invoked by the actual Web endpoint; no precomputed corrected text injected",
        conditions=["baseline","generic"],refiner="Actual GPU Refiner in both conditions")
    write_json(OUT/"manifest.json",manifest)


class IntegrationRunner(Runner):
    def __init__(self):
        args=SimpleNamespace(output=OUT,library=OUT/"libraries",model=MODEL,device="cuda:0",asr_url=ASR,mode="streaming",chunk_seconds=2.0)
        super().__init__(args)
        self.initial_payload=None
        self.repaired_payload=None
        import system.entity_audio_repair as repair
        original=repair.EntityAudioRepair
        owner=self
        class Observed(original):
            async def request(self,endpoint,*args,**kwargs):
                result=await super().request(endpoint,*args,**kwargs)
                if endpoint=="/stream/finish":owner.repaired_payload=deepcopy(result)
                return result
        repair.EntityAudioRepair=Observed

    def app(self,key,arm):
        from system.zh_itn_bridge import DEFAULT_LIBRARY
        log=OUT/"service_logs"/key/(arm+".jsonl");log.parent.mkdir(parents=True,exist_ok=True)
        return self.web.create_app(MODEL,"cuda:0",ASR,"Chinese",256,log,
            entity_db=OUT/"libraries"/key/"entities.db",entity_fuzzy_mode="auto",refinement_gate_mode="tri_state",
            final_refinement_mode="off",zh_itn_library=DEFAULT_LIBRARY,entity_asr_context=False,entity_audio_repair=True)

    def exercise(self,client,sample,arm):
        query=urlencode(dict(mode="streaming",domain="medical",language="Chinese",asr_chunk_seconds=2.0,
                             entity_asr_context=0,entity_audio_repair=int(arm=="generic")))
        audio,rate=sf.read(sample["audio_path"],dtype="float32");pcm=audio.astype("<f4").tobytes()
        if rate!=16000 or hashlib.sha256(pcm).hexdigest()!=sample["pcm_sha256"]:raise ValueError("Audio changed")
        self.calls.clear();self.initial_payload=None;self.repaired_payload=None
        events=[];started=time.perf_counter()
        with client.websocket_connect("/ws/stream?"+query) as socket:
            ready=socket.receive_json()
            if ready.get("event")!="ready" or ready.get("entity_asr_context_applied"):raise ValueError(ready)
            if ready.get("entity_audio_repair_requested")!=(arm=="generic"):raise ValueError("Repair flag not applied")
            for offset in range(0,len(pcm),64000):
                socket.send_bytes(pcm[offset:offset+64000])
                while True:
                    event=socket.receive_json();events.append(event)
                    if event.get("event")=="error":raise RuntimeError(event)
                    if event.get("event")=="chunk_ack":break
            socket.send_json(dict(event="finish"))
            while True:
                event=socket.receive_json();events.append(event)
                if event.get("event")=="error":raise RuntimeError(event)
                if event.get("event")=="final":break
        final=self.repaired_payload if arm=="generic" else self.initial_payload
        if final is None or event.get("raw_text","")!=final.get("text","").strip():raise ValueError("Final Web source mismatch")
        return dict(id=sample["id"],condition=arm,result=event,model_calls=list(self.calls),
            ready=ready,elapsed_seconds=time.perf_counter()-started,event_counts=dict(Counter(e.get("event") for e in events))), final

    def recorded(self):
        from fastapi.testclient import TestClient
        manifest=json.loads((OUT/"manifest.json").read_text())
        ids=[s["id"] for s in manifest["samples"]]
        raw=require_complete(SOURCE/"asr_baseline.jsonl",ids)
        gold=require_complete(SOURCE/"asr_generic.jsonl",ids)
        replay={r["id"]:r for r in read_jsonl(ROOT/"results/generic_entity_audio_20261010/outputs_neighbor.jsonl") if r["replayed"]}
        replay.update({r["id"]:r["replay"] for r in read_jsonl(SOURCE/"new_audio_requests.jsonl") if r["replay"]})
        done={arm:{r["id"] for r in read_jsonl(OUT/f"{arm}.jsonl")} for arm in ("baseline","generic")}
        sessions={};owner=self
        async def request(url,endpoint,session_id=None,data=b"",params=None,**kwargs):
            rid=owner.sample["id"]
            if endpoint=="/stream/start":
                prompted=bool((params or {}).get("context"))
                if prompted and rid not in replay:raise ValueError("Unexpected replay candidate")
                record=replay[rid] if prompted else raw[rid]
                ready=deepcopy(record["ready"])
                if prompted and ready.get("context_digest")!=context_digest(params["context"]):raise ValueError("Production prompt differs from verified prompt")
                sessions[ready["session_id"]]=dict(record=record,index=0,pcm=bytearray(),prompted=prompted)
                return ready
            if endpoint=="/stream/cancel":sessions.pop(session_id,None);return dict(cancelled=True)
            state=sessions[session_id];record=state["record"]
            if endpoint=="/stream/chunk":
                state["pcm"].extend(data)
                result=record["chunk_payloads"][state["index"]];state["index"]+=1
                return deepcopy(result)
            if endpoint=="/stream/finish":
                if state["index"]!=len(record["chunk_payloads"]):raise ValueError("Incomplete actual-audio stream")
                if hashlib.sha256(state["pcm"]).hexdigest()!=owner.sample["pcm_sha256"]:raise ValueError("Main/replayed PCM differs")
                result=deepcopy(record["final_payload"])
                if not state["prompted"]:owner.initial_payload=deepcopy(result)
                sessions.pop(session_id);return result
            raise ValueError(endpoint)
        self.web._stream_request_async=request
        for key in dict.fromkeys(s["library_key"] for s in manifest["samples"]):
            samples=[s for s in manifest["samples"] if s["library_key"]==key]
            with ExitStack() as stack:
                clients={arm:stack.enter_context(TestClient(self.app(key,arm))) for arm in ("baseline","generic")}
                for index,sample in enumerate(samples):
                    self.sample=sample
                    for arm in (("baseline","generic") if index%2==0 else ("generic","baseline")):
                        if sample["id"] in done[arm]:continue
                        row,final=self.exercise(clients[arm],sample,arm)
                        expected=gold[sample["id"]] if arm=="generic" else raw[sample["id"]]
                        if final["text"]!=expected["final_payload"]["text"]:raise ValueError("Production audio repair differs from validated decision: "+sample["id"])
                        append_jsonl(OUT/f"{arm}.jsonl",row)
                        append_jsonl(OUT/f"asr_{arm}.jsonl",dict(id=sample["id"],final_payload=final))
                        print("INTEGRATED",arm,sample["id"],row["result"].get("clean_text",""),flush=True)
        if sessions:raise ValueError("ASR replay sessions leaked")

    def live(self):
        from fastapi.testclient import TestClient
        manifest=json.loads((OUT/"manifest.json").read_text())
        selected={"speechio_law_rIfA-JnK798_0106-861","speechio_people_startup_insider_tongdunkeji_0241-1458",
                  "er_faq_waiting_010","er_faq_examination_019","negative_people_startup_insider_rongzi_0260-983"}
        owner=self;roles={}
        async def real(url,endpoint,session_id=None,data=b"",params=None,**kwargs):
            result=await owner.original_http(url,endpoint,session_id=session_id,data=data,params=params,**kwargs)
            if endpoint=="/stream/start":roles[result["session_id"]]=bool((params or {}).get("context"))
            if endpoint=="/stream/finish" and not roles.get(session_id):owner.initial_payload=deepcopy(result)
            if endpoint in {"/stream/finish","/stream/cancel"}:roles.pop(session_id,None)
            return result
        self.web._stream_request_async=real
        done={(r["id"],r["condition"]) for r in read_jsonl(OUT/"live.jsonl")}
        for sample in manifest["samples"]:
            if sample["id"] not in selected:continue
            with TestClient(self.app(sample["library_key"],"live")) as client:
                for arm in ("baseline","generic"):
                    if (sample["id"],arm) in done:continue
                    row,final=self.exercise(client,sample,arm)
                    append_jsonl(OUT/"live.jsonl",dict(row,final_payload=final,initial_payload=self.initial_payload,
                                                        reference=sample["reference"]))
                    print("LIVE",arm,sample["id"],row["result"].get("clean_text",""),flush=True)
        if roles:raise ValueError("Actual ASR sessions leaked")


def report():
    import validate_generic_entity_pipeline as validation
    validation.OUT=OUT
    validation.report()
    report_path=OUT/"REPORT.md"
    text=report_path.read_text().replace("没有改生产代码。", "本轮使用已接入的正式实体模块与实体会话连接，其它业务逻辑保持原样。")
    text=text.replace("每个发生修正的场景增加‘不改文字、只重发分段并取消最终置信度’控制组，结果保存在 revision_control.jsonl，用于识别重算的影响。", "分段重发控制组已在上一轮验证；本轮比较正式原音复核开启与关闭两组。")
    text=text.replace("此实验没有接入当前服务。", "本次通过正式模块的独立测试实例验证；服务部署状态见 deployment.json。")
    text += "\n本轮长音频的 ASR 复核秒数引用上一轮实际音频请求；本轮完整场景使用录制响应检查正式模块与精修的集成，另有五场景使用实际 ASR 服务复测，见 INTEGRATION.md。\n"
    report_path.write_text(text)
    manifest=json.loads((OUT/"manifest.json").read_text())
    for path,value in manifest["source_hashes"].items():
        if digest(ROOT/path)!=value:raise ValueError("Production source changed during integration test")
    rows=read_jsonl(OUT/"live.jsonl")
    lines=["# 正式实体原音复核：接入结果", "", "完整 403 场景通过正式 EntityAudioRepair 和实际 Web 路由，使用已录制的真实 ASR 响应及真实 GPU Refiner。主识别和原音复核都验证 PCM 哈希，正式模块自己检索候选和修改文本，没有注入实验预计算的修正文。", "", "## 实际 ASR 服务＋实际精修的五场景复测", "", "| 音频 | 原有流程 | 正式原音复核流程 | 原音修改数 |", "|---|---|---|---:|" ]
    pairs={}
    for row in rows:pairs.setdefault(row["id"],{})[row["condition"]]=row
    for rid,pair in pairs.items():
        if set(pair)!={"baseline","generic"}:raise ValueError("Incomplete live pair")
        a,b=pair["baseline"]["result"],pair["generic"]["result"]
        lines.append(f"|{rid}|{a.get('clean_text','')}|{b.get('clean_text','')}|{b.get('entity_audio_repair',{}).get('change_count',0)}|")
    lines += ["", "全量命中、CER、退化和长度压力数据见 REPORT.md 与 summary.json。实际 ASR 复测是新请求，初稿可能存在解码波动，保留原始响应供复核。此运行只写独立实验词库，没有导入正式用户库。"]
    (OUT/"INTEGRATION.md").write_text("\n".join(lines)+"\n")


def approve_deployment():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.zh_itn_bridge import ZhITNNormalizer
    from evaluate_entity_library import normalized
    summary=json.loads((OUT/"summary.json").read_text())
    for key,group in summary.items():
        if group["generic"]["worsened"] or group["generic"]["hits"]<group["baseline"]["hits"]:
            raise ValueError("Integration regressed: "+key)
    converter,itn=opencc.OpenCC("t2s"),ZhITNNormalizer()
    def norm(text):return normalized(converter.convert(itn.normalize(text).text))
    for row in read_jsonl(OUT/"live.jsonl"):
        if row["condition"]!="generic":continue
        ref=norm(row["reference"])
        before=norm(row["initial_payload"]["text"]);after=norm(row["final_payload"]["text"])
        if Levenshtein.distance(ref,after)>Levenshtein.distance(ref,before):raise ValueError("Live audio correction regressed")
    (OUT/"DEPLOY_ALLOWED").write_text("Full production-module integration and live ASR checks passed\n")
    print("QUALITY CHECKS PASSED",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","run","live","report"));stage=p.parse_args().stage
    if stage=="prepare":prepare()
    elif stage=="report":report()
    else:
        runner=IntegrationRunner()
        if stage=="run":
            runner.recorded();runner.live();report();approve_deployment()
        else:runner.live()
