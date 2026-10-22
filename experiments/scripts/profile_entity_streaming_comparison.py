#!/usr/bin/env python3
"""Paced, no-reference three-arm streaming profile on eight fixed test inputs.

The only output is the independent paced_profile.jsonl trace. Test IDs are
selected by an input-only SHA-256 min-hash rank; references are never opened.
"""
from __future__ import annotations
import argparse, hashlib, json, math, queue, sqlite3, sys, threading, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN_DEFAULT = ROOT / "results/entity_backend_comparison_20261015"
NATIVE_MODEL = Path("/home/aim0/data/models/ASR/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30")
CTC_MODEL = RUN_DEFAULT / "models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30"
SAMPLE_RATE=16000; CHUNK_SAMPLES=2560; CHUNK_BYTES=CHUNK_SAMPLES*4; TAIL_SAMPLES=round(.66*SAMPLE_RATE); THREAD_TIMEOUT=100.
QWEN_URL="http://127.0.0.1:8766"
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from benchmark_native_entity_streaming import load_canonicals, hotword_payload, token_inventory, make_recognizer, _result_snapshot
from entity_ctc_frontend import StreamingCtcModel
from entity_ctc_word_spotter import StreamingWordSpotter
from evaluate_entity_ctc_fusion import definitions, fuse
from benchmark_native_broad_vocabulary import MODEL_DIR as BROAD_MODEL_DIR, make_recognizer as make_broad_recognizer


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []
def sha256(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def join_thread(thread:threading.Thread,label:str,timeout:float=THREAD_TIMEOUT):
    thread.join(timeout)
    if thread.is_alive():raise TimeoutError(f"{label} worker did not stop within {timeout:.0f}s")
def load_inputs(run:Path)->list[dict]:
    rows=read_jsonl(run/"inputs.jsonl"); seen=set()
    for r in rows:
        if r["id"] in seen:raise ValueError(f"duplicate input id {r['id']}")
        seen.add(r["id"])
    return rows

def broad_token_inventory()->set[str]:
    result=set()
    with (BROAD_MODEL_DIR/"tokens.txt").open(encoding="utf-8") as f:
        for line in f:
            parts=line.rstrip("\n").rsplit(" ",1)
            if len(parts)==2:result.add(parts[0])
    return result

def pick_eight(inputs:list[dict])->list[dict]:
    pool=[r for r in inputs if r.get("split")=="test"]
    # Min-hash over immutable input IDs; no references or model outcomes are read.
    return sorted(pool,key=lambda r:hashlib.sha256(("paced-stream-profile-v1:"+str(r["id"])).encode()).hexdigest())[:8]

def library_for(row:dict)->tuple[list[str],str]:
    p=Path(row["library_path"])
    words=load_canonicals(p,str(row.get("domain","")))
    if not words:raise ValueError(f"empty entity library for {row['id']}: {p}")
    return words,str(p)

class CtcWorker:
    def __init__(self,model,template,audio_start):
        self.model=model;self.stream=model.create_stream();self.spotter=template.create_stream();self.audio_start=audio_start
        self.q=queue.Queue();self.lock=threading.Lock();self.events=[];self.error=None;self.compute_start=time.perf_counter();self.frontend_compute_wall=0.
        self.thread=threading.Thread(target=self.run,name="ctc-stream-worker",daemon=True)
    def start(self):self.thread.start()
    def put(self,item):self.q.put(item)
    def _consume(self,blocks,after=False):
        for b in blocks:
            hyp=self.spotter.accept(b["log_probs"],b["frame_offset"],b["available_audio_seconds"])
            ev={"audio_seconds":b["available_audio_seconds"],"after_input_finished":after or bool(b.get("after_input_finished")),
                "ctc_text":self.stream.text,"ctc_tokens":list(self.stream.token_ids),"ctc_frame_times":list(self.stream.token_frames),"hypotheses":hyp,
                "compute_elapsed":time.perf_counter()-self.compute_start,"wall_seconds":time.perf_counter()-self.audio_start}
            with self.lock:self.events.append(ev)
    def run(self):
        try:
            while True:
                item=self.q.get()
                if item is None:break
                index,pos,raw,submit_rel=item
                import numpy as np
                samples=np.frombuffer(raw,dtype="<f4"); began=time.perf_counter();blocks=self.stream.accept_audio(samples);self.frontend_compute_wall+=time.perf_counter()-began
                self._consume(blocks)
            before=time.perf_counter(); blocks=self.stream.finish(); self.frontend_compute_wall+=time.perf_counter()-before; self._consume(blocks,after=True)
            self.final_hypotheses=self.spotter.finish(); self.finish_wall=time.perf_counter()-before
        except BaseException as e:self.error=e
    def snapshot(self,available:float|None=None):
        with self.lock:
            es=[dict(e) for e in self.events if available is None or (not e["after_input_finished"] and e["audio_seconds"]<=available+1e-6)]
        if es:
            last=es[-1]; ids=last["ctc_tokens"]; frames=last["ctc_frame_times"]; text=last["ctc_text"]
        else:ids=[];frames=[];text=""
        return {"audio_seconds":(available if available is not None else self.stream.audio_samples/SAMPLE_RATE),
            "ctc_tokens":ids,"ctc_token_frames":frames,"ctc_text":text,
            "hypotheses":[h for e in es for h in e["hypotheses"]],"events":es}
    def result(self):
        if self.error:raise self.error
        return {"events":[dict(e) for e in self.events],"ctc_text":self.stream.text,"ctc_tokens":list(self.stream.token_ids),
            "ctc_token_frames":list(self.stream.token_frames),"audio_seconds":self.stream.audio_samples/SAMPLE_RATE,
            "hypotheses":self.final_hypotheses,"compute_seconds":self.frontend_compute_wall+self.spotter.stats["compute_seconds"],
            "frontend_compute_seconds":self.frontend_compute_wall,"onnx_compute_seconds":self.stream.compute_seconds,"spotter_compute_seconds":self.spotter.stats["compute_seconds"],
            "finish_wall_seconds":self.finish_wall,"worker_wall_seconds":time.perf_counter()-self.audio_start}

class QwenWorker:
    def __init__(self,sidecar=False,ctc=None,defs=(),tokens=(),gap=-2.,margin=.1):
        self.q=queue.Queue();self.ready=threading.Event();self.abort=threading.Event();self.start_info={};self.final=None;self.error=None
        self.sidecar=sidecar;self.ctc=ctc;self.defs=defs;self.tokens=tokens;self.gap=gap;self.margin=margin
        self.audio_start=None;self.events=[];self.first_text_wall=None;self.finish_tail=None
        self.thread=threading.Thread(target=self.run,name="qwen-http-worker",daemon=True)
    def start(self):
        self.thread.start()
        if not self.ready.wait(THREAD_TIMEOUT):
            self.abort.set();self.q.put(None);self.thread.join(5.)
            raise TimeoutError("Qwen worker did not become ready within 100s; abort requested and session cleanup will run when HTTP returns")
        if self.error:raise self.error
    def put(self,item):self.q.put(item)
    def run(self):
        sid=None;client=None
        try:
            import httpx
            with httpx.Client(timeout=90,trust_env=False) as client:
                t=time.perf_counter();health=client.get(QWEN_URL+"/health");health.raise_for_status();self.health=health.json()
                if self.abort.is_set():raise TimeoutError("Qwen startup aborted before stream creation")
                start=client.post(QWEN_URL+"/stream/start",params={"language":"Chinese","chunk_seconds":2.});start.raise_for_status();ready=start.json()
                sid=ready["session_id"]
                if self.abort.is_set():raise TimeoutError("Qwen startup aborted after stream creation")
                self.audio_start=time.perf_counter();self.start_info={"health":self.health,"start_wall_seconds":time.perf_counter()-t,"context_applied":ready.get("context_applied")};self.ready.set()
                last_text=None
                while True:
                    if self.abort.is_set():raise TimeoutError("Qwen worker aborted after a bounded wait")
                    item=self.q.get()
                    if item is None:break
                    index,position,raw,submit_rel=item
                    began=time.perf_counter();resp=client.post(QWEN_URL+"/stream/chunk",params={"session_id":sid},content=raw);resp.raise_for_status();payload=resp.json();arrived=time.perf_counter()
                    if self.abort.is_set():raise TimeoutError("Qwen worker aborted while awaiting response")
                    text=str(payload.get("text", "")); event={"chunk_index":index,"audio_available_seconds":position,"text":text,
                        "request_wall_seconds":arrived-began,"event_wall_seconds":arrived-self.audio_start,"chunk_submit_wall_seconds":submit_rel}
                    if text and self.first_text_wall is None:self.first_text_wall=event["event_wall_seconds"]
                    if self.sidecar:
                        # Re-evaluate on each Qwen response (~160ms minimum poll),
                        # so newly arrived CTC evidence can act even if Qwen text is unchanged.
                        net=self.ctc.snapshot(position);fusion_started=time.perf_counter();fused,changes=fuse(text,net,self.defs,self.tokens,self.gap,self.margin,available=position,final=False)
                        event["fusion_trigger"]="qwen_response";event["fused_text"]=fused;event["changes"]=changes
                        event["fusion_compute_seconds"]=time.perf_counter()-fusion_started;event["fusion_completed_wall_seconds"]=time.perf_counter()-self.audio_start
                        event["ctc_evidence_count"]=len(net["hypotheses"]);event["ctc_snapshot_event_count"]=len(net["events"])
                    last_text=text;self.events.append(event)
                fin=time.perf_counter();response=client.post(QWEN_URL+"/stream/finish",params={"session_id":sid});response.raise_for_status();self.finish_tail=time.perf_counter()-fin;self.final=response.json();sid=None
                self.finish_wall_seconds=time.perf_counter()-self.audio_start
        except BaseException as e:self.error=e;self.ready.set()
        finally:
            if sid:
                try:
                    import httpx
                    with httpx.Client(timeout=3,trust_env=False) as cleanup:
                        cleanup.post(QWEN_URL+"/stream/cancel",params={"session_id":sid})
                except Exception:pass
    def result(self):
        self.thread.join(THREAD_TIMEOUT)
        if self.thread.is_alive():
            self.abort.set();self.q.put(None);self.thread.join(5.)
            raise TimeoutError("Qwen HTTP worker did not finish within 100s; abort requested and session cleanup will run when HTTP returns")
        if self.error:raise self.error
        return {"events":self.events,"final_text":str((self.final or {}).get("text","")),"first_text_wall_seconds":self.first_text_wall,
            "finish_http_seconds":self.finish_tail,"finish_wall_seconds":self.finish_wall_seconds,"total_wall_seconds":self.finish_wall_seconds,"health":self.start_info.get("health"),
            "start_context_applied":self.start_info.get("context_applied"),"http_seconds":sum(e["request_wall_seconds"] for e in self.events)+(self.finish_tail or 0)}

def paced_send(pcm:bytes,workers:list,accelerated:bool,audio_start:float):
    actual=[];last=0.
    for index,offset in enumerate(range(0,len(pcm),CHUNK_BYTES)):
        raw=pcm[offset:offset+CHUNK_BYTES];position=(offset+len(raw))/64000
        if not accelerated:
            target=audio_start+position;delay=target-time.perf_counter()
            if delay>0:time.sleep(delay)
        submit=time.perf_counter()-audio_start
        for worker in workers:worker.put((index,position,raw,submit))
        actual.append({"chunk_index":index,"audio_available_seconds":position,"submit_wall_seconds":submit,"chunk_bytes":len(raw)})
        last=submit
    return actual,last

def run_native(row,pcm,recognizer,hotword,accelerated,selection_hash):
    import numpy as np
    stream=recognizer.create_stream(hotwords=hotword);partials=[];started=time.perf_counter();audio_start=started;submitted=0;chunk_events=[];compute_total=0.
    for i,offset in enumerate(range(0,len(pcm),CHUNK_BYTES)):
        raw=pcm[offset:offset+CHUNK_BYTES];position=(offset+len(raw))/64000
        if not accelerated:
            delay=audio_start+position-time.perf_counter()
            if delay>0:time.sleep(delay)
        input_submit=time.perf_counter();samples=np.frombuffer(raw,dtype="<f4");compute_before=time.perf_counter();stream.accept_waveform(SAMPLE_RATE,samples);submitted+=len(samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream);snap=_result_snapshot(recognizer,stream)
            partials.append({"audio_available_seconds":position,"wall_seconds":time.perf_counter()-audio_start,"text":snap.get("text","")})
        compute_total+=time.perf_counter()-compute_before
        chunk_events.append({"chunk_index":i,"audio_available_seconds":position,"submit_wall_seconds":input_submit-audio_start})
    last_audio_submit=chunk_events[-1]["submit_wall_seconds"] if chunk_events else 0.;before=time.perf_counter();stream.accept_waveform(SAMPLE_RATE,np.zeros(TAIL_SAMPLES,dtype=np.float32));stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream);snap=_result_snapshot(recognizer,stream);partials.append({"audio_available_seconds":len(pcm)/64000,"wall_seconds":time.perf_counter()-audio_start,"is_flush_padding":True,"text":snap.get("text","")})
    final=_result_snapshot(recognizer,stream);finish_tail=time.perf_counter()-before;compute_total+=finish_tail
    return {"text":final.get("text",""),"partials":partials,"chunk_schedule":chunk_events,"first_text_wall_seconds":next((p["wall_seconds"] for p in partials if p.get("text")),None),
        "last_audio_submit_wall_seconds":last_audio_submit,"finish_tail_seconds":time.perf_counter()-audio_start-last_audio_submit,"flush_compute_seconds":finish_tail,"finish_wall_seconds":time.perf_counter()-audio_start,
        "total_wall_seconds":time.perf_counter()-started,"audio_seconds":len(pcm)/64000,"compute_wall_seconds":compute_total,
        "flush_includes_padding_seconds":TAIL_SAMPLES/SAMPLE_RATE,"timing_scope":"native streaming CPU, 2 threads; chunks wallclock-paced at 160ms; ASR compute wall excludes deliberate sleeps approximately"}

def run_qwen_arm(row,pcm,sidecar,ctc_model,template,defs,tokens,gap,margin,accelerated):
    # Start Qwen first, then give CTC an independent ordered queue; producer fans out identical bytes.
    qworker=QwenWorker(sidecar=sidecar,defs=defs,tokens=tokens,gap=gap,margin=margin)
    qworker.start()
    audio_start=qworker.audio_start
    cworker=None
    if sidecar:
        cworker=CtcWorker(ctc_model,template,audio_start);cworker.start()
        # Qwen worker references ctc dynamically for live snapshots.
        qworker.ctc=cworker
    workers=[qworker]+([cworker] if cworker else [])
    try:
        schedule,last= paced_send(pcm,workers,accelerated,audio_start)
    finally:
        for w in workers:w.put(None)
    qresult=qworker.result()
    qresult["finish_tail_seconds"]=max(0.,qresult["finish_wall_seconds"]-last)
    result={"audio_seconds":len(pcm)/64000,"chunk_schedule":schedule,"last_audio_submit_wall_seconds":last,**qresult}
    if cworker:
        join_thread(cworker.thread,"CTC")
        ctc=cworker.result();fusion_started=time.perf_counter()
        finaltext,changes=fuse(qresult["final_text"],ctc,defs,tokens,gap,margin,available=ctc["audio_seconds"],final=True)
        fusion_compute=time.perf_counter()-fusion_started;final_completed=time.perf_counter()-audio_start
        workers_finish=max(qresult["finish_wall_seconds"],ctc["worker_wall_seconds"])
        result.update({"ctc":ctc,"workers_finish_wall_seconds":workers_finish,"final_fused_text":finaltext,"final_changes":changes,
            "final_fusion_compute_seconds":fusion_compute,"final_fusion_completed_wall_seconds":final_completed,
            "qwen_finish_http_seconds":qresult["finish_http_seconds"],"qwen_finish_wall_seconds":qresult["finish_wall_seconds"],"ctc_finish_tail_seconds":ctc["finish_wall_seconds"],
            "sidecar_finish_wall_seconds":final_completed,"finish_tail_seconds":max(0.,final_completed-last),"total_wall_seconds":final_completed,
            "ctc_compute_seconds":ctc["compute_seconds"],"timing_scope":"same PCM chunks fanned out in order to actual Qwen HTTP worker and CPU2threads incremental CTC worker; CTC and Qwen flush run concurrently; fusion is attempted on every Qwen response (minimum 160ms polling) using only CTC events already available and no later than that audio position; final fusion is included in total completion wall"})
    else:result.update({"timing_scope":"actual Qwen HTTP stream; same 160ms PCM blocks, wallclock paced"})
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run-dir",type=Path,default=RUN_DEFAULT);ap.add_argument("--limit",type=int,default=8)
    ap.add_argument("--accelerated",action="store_true",help="smoke only; submit chunks without real-time waits")
    ap.add_argument("--include-broad",action="store_true",help="add broad 2023 native hotword as a fourth arm using broad_vocab/selection.json")
    ap.add_argument("--output",type=Path,help="JSONL output (default RUN_DIR/paced_profile.jsonl)")
    args=ap.parse_args();run=args.run_dir.resolve();out=(args.output or run/"paced_profile.jsonl").resolve()
    selection=json.loads((run/"selection.json").read_text(encoding="utf-8"));score=float(selection["native_hotword_score"]);gap=float(selection["hybrid_minimum_gap"]);margin=float(selection["hybrid_minimum_margin"])
    broad_selection=None;broad_score=None;broad_tokens=None
    if args.include_broad:
        bp=run/"broad_vocab/selection.json"
        if not bp.is_file():raise FileNotFoundError(f"broad profiler requires frozen dev selection: {bp}")
        broad_selection=json.loads(bp.read_text(encoding="utf-8"));broad_score=float(broad_selection["native_hotword_score"])
        broad_tokens=broad_token_inventory()
    if score!=2.0 or gap!=-2.0 or margin!=.1:raise ValueError(f"Frozen profile requires native hotword 2.0 and hybrid (-2.0,.1); got {score},{gap},{margin}")
    fixed_eight=pick_eight(load_inputs(run));chosen=fixed_eight[:args.limit]
    if not chosen:raise ValueError("no test inputs")
    tokens=token_inventory();ctc_model=None;templates={};defs_by_library={};hotwords={}
    broad_hotwords={}
    for r in chosen:
        words,library=library_for(r);hotwords[r["id"]]=hotword_payload(words,tokens)[0]
        if args.include_broad:broad_hotwords[r["id"]]=hotword_payload(words,broad_tokens)[0]
        if library not in templates:
            if ctc_model is None:
                ctc_path=run/"models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30"
                ctc_model=StreamingCtcModel(ctc_path/"model.int8.onnx",ctc_path/"tokens.txt",num_threads=2)
            all_words=[]
            with sqlite3.connect(f"file:{Path(library).resolve()}?mode=ro",uri=True) as db:all_words=[str(x[0]) for x in db.execute("SELECT DISTINCT canonical_text FROM protected_entities WHERE enabled=1 ORDER BY id")]
            templates[library]=StreamingWordSpotter(ctc_model.tokens,all_words,context_weight=2.,beam=12.,minimum_gap=-3.,blank_id=ctc_model.blank_id,frame_seconds=ctc_model.frame_shift_seconds)
            defs_by_library[library]=definitions(library)
    native=make_recognizer("hotword",score)
    broad_native=make_broad_recognizer("hotword",broad_score) if args.include_broad else None
    token_table=ctc_model.tokens
    order=("qwen_raw","qwen_ctc_sidecar","native_hotword")+( ("native_broad_hotword",) if args.include_broad else ())
    chosen_meta=[{"id":x["id"],"pcm_sha256":x["pcm_sha256"],"split":x["split"],"domain":x["domain"],"duration_seconds":x["duration_seconds"]} for x in fixed_eight]
    mode="accelerated_smoke" if args.accelerated else "realtime_160ms_wallclock_paced"
    existing=read_jsonl(out);done={(x.get("id"),x.get("arm"),x.get("profile_mode")) for x in existing}
    out.parent.mkdir(parents=True,exist_ok=True)
    for idx,row in enumerate(chosen):
        pcm_path=Path(row["pcm_path"]);pcm=pcm_path.read_bytes()
        if sha256(pcm)!=row["pcm_sha256"]:raise ValueError(f"PCM hash mismatch: {row['id']}")
        if len(pcm)%4:raise ValueError(f"invalid float32 pcm: {row['id']}")
        lib=str(Path(row["library_path"]).resolve());defs=defs_by_library[lib];template=templates[lib]
        arms=list(order[idx%3:]+order[:idx%3])
        for arm in arms:
            if (row["id"],arm,mode) in done:continue
            setup=time.perf_counter()
            if arm=="native_hotword": trace=run_native(row,pcm,native,hotwords[row["id"]],args.accelerated,hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest())
            elif arm=="native_broad_hotword": trace=run_native(row,pcm,broad_native,broad_hotwords[row["id"]],args.accelerated,hashlib.sha256(json.dumps(broad_selection,sort_keys=True).encode()).hexdigest())
            else:trace=run_qwen_arm(row,pcm,arm=="qwen_ctc_sidecar",ctc_model,template,defs,token_table,gap,margin,args.accelerated)
            record={"id":row["id"],"arm":arm,"split":row["split"],"domain":row["domain"],"source":row.get("source"),"source_group":row.get("source_group"),
                "pcm_sha256":row["pcm_sha256"],"audio_seconds":row["duration_seconds"],"selection_method":"min SHA256(paced-stream-profile-v1:id) over test inputs only; references not read",
                "fixed_eight_inputs":chosen_meta,"order_rotation_index":idx,"arm_order":arms,"arm_order_position":arms.index(arm),
                "selection":{"native_hotword_score":score,"hybrid_minimum_gap":gap,"hybrid_minimum_margin":margin,"selection_sha256":hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
                    "native_broad_hotword_score":broad_score,"broad_selection_sha256":hashlib.sha256(json.dumps(broad_selection,sort_keys=True).encode()).hexdigest() if broad_selection else None},
                "profile_mode":mode,"broad_arm_enabled":args.include_broad,"setup_seconds_including_arm_setup":time.perf_counter()-setup,"trace":trace}
            with out.open("a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
            done.add((row["id"],arm,mode));print("PACED_PROFILE",row["id"],arm,"complete",flush=True)
    print(json.dumps({"output":str(out),"profile_mode":mode,"fixed_eight_ids":[x["id"] for x in fixed_eight],"selected_ids":[x["id"] for x in chosen],"completed_records":len(read_jsonl(out))},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
