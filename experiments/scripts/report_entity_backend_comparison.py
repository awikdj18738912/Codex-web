#!/usr/bin/env python3
"""Offline scorer/report generator for the frozen entity backend comparison."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, sys, unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opencc import OpenCC
from system.zh_itn_bridge import ZhITNNormalizer

ITN = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
OPENCC = OpenCC("t2s")
RUN_DEFAULT = ROOT / "results/entity_backend_comparison_20261015"
ARMS = ("qwen_raw", "ctc_greedy", "native_greedy", "native_beam", "native_hotword", "qwen_ctc_sidecar")
LABELS = {"qwen_raw":"Qwen 原始主流", "ctc_greedy":"CTC greedy", "native_greedy":"Native greedy",
          "native_beam":"Native beam", "native_hotword":"Native hotword（dev 选参）", "qwen_ctc_sidecar":"Qwen + CTC sidecar（dev 选参）"}

@lru_cache(maxsize=200000)
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", OPENCC.convert(ITN.normalize(s).text)).casefold()
    return "".join(c for c in s if unicodedata.category(c)[0] not in {"P", "Z", "C"})

def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b): a, b = b, a
    row = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1): nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def entity_counts(text: str, gold: list[dict]) -> tuple[Counter, Counter]:
    gc=Counter(norm(str(e.get("canonical") or e.get("surface") or "")) for e in gold)
    gc.pop("",None); h=norm(text)
    pred=Counter({k:h.count(k) for k in gc})
    return gc,pred

def score(ref: str, hyp: str, gold: list[dict]) -> dict:
    rn,hn=norm(ref),norm(hyp); err=edit_distance(list(rn),list(hn)); gc,pc=entity_counts(hyp,gold)
    hits=sum(min(n,pc[k]) for k,n in gc.items()); exp=sum(gc.values())
    return {"entity_hits":hits,"entity_mentions":exp,"entity_recall":hits/exp if exp else 0.0,
            "cer_errors":err,"reference_chars":len(rn),"cer":err/len(rn) if rn else (0.0 if err==0 else 1.0),
            "gold_counts":dict(gc),"predicted_gold_mentions":sum(pc.values())}

def read_jsonl(p: Path) -> list[dict]:
    out=[]
    with p.open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                x=json.loads(line)
                if not isinstance(x,dict): raise ValueError(f"{p}:{i}: expected object")
                out.append(x)
    return out

def index_rows(rows: list[dict], label: str, key=lambda x:str(x.get("id",""))) -> tuple[dict,list[str]]:
    d={}; dup=[]
    for x in rows:
        k=key(x)
        if k in d: dup.append(k)
        d[k]=x
    if dup: raise ValueError(f"{label} duplicate keys: {sorted(set(dup))[:20]}")
    return d,dup

def finite_num(v):
    try:
        f=float(v)
        return f if math.isfinite(f) and f>=0 else None
    except (TypeError,ValueError): return None

def finite_signed(v):
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (TypeError,ValueError): return None

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,default=RUN_DEFAULT)
    ap.add_argument("--selection",type=Path,help="dev-selected parameters JSON; defaults to RUN_DIR/selection.json")
    ap.add_argument("--allow-partial",action="store_true",help="score the common available IDs within each split")
    args=ap.parse_args(); run=args.run_dir.resolve()
    protocol=json.loads((run/"protocol.json").read_text(encoding="utf-8"))
    refsdoc=json.loads((run/"references.json").read_text(encoding="utf-8")); refs=refsdoc.get("samples",{})
    inputs_rows=read_jsonl(run/"inputs.jsonl"); inputs, _=index_rows(inputs_rows,"inputs")
    if len(inputs)!=len(inputs_rows): raise ValueError("input IDs are not unique")
    ref_ids=set(refs); input_ids=set(inputs)
    for filename in ("inputs.jsonl","references.json"):
        expected_hash=protocol.get("frozen_sha256",{}).get(filename)
        if expected_hash:
            actual=hashlib.sha256((run/filename).read_bytes()).hexdigest()
            if actual!=expected_hash: raise ValueError(f"frozen file hash mismatch: {filename}")
    if ref_ids!=input_ids: raise ValueError(f"input/reference ID mismatch: missing_inputs={sorted(ref_ids-input_ids)[:10]} extra_inputs={sorted(input_ids-ref_ids)[:10]}")
    split_ids=defaultdict(set)
    for sid,row in inputs.items(): split_ids[str(row.get("split","unknown"))].add(sid)
    expected_counts={"dev":171,"test":1829,"control":100}
    for split,n in expected_counts.items():
        if len(split_ids[split])!=n: raise ValueError(f"frozen {split} expected {n}, found {len(split_ids[split])}")

    selection_path=(args.selection or run/"selection.json").resolve()
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    # Accept nested or flat selection, but require all values to be explicit.
    sel=selection.get("selected_parameters",selection)
    required=("native_hotword_score","hybrid_minimum_gap","hybrid_minimum_margin")
    missing=[k for k in required if k not in sel or sel[k] is None]
    if missing: raise ValueError(f"selection.json missing dev-selected parameters: {missing}")

    qrows=read_jsonl(run/"qwen_primary_streams.jsonl"); qwen,_=index_rows(qrows,"qwen primary")
    nrows=read_jsonl(run/"native_results.jsonl")
    native={}
    chosen_hotword=float(sel["native_hotword_score"])
    for r in nrows:
        cond=str(r.get("condition")); sid=str(r.get("id")); rawscore=r.get("score")
        scorekey=None if rawscore is None else float(rawscore)
        k=(sid,cond,scorekey)
        if k in native: raise ValueError(f"native duplicate key: {k}")
        native[k]=r
    crows=read_jsonl(run/"ctc_results.jsonl"); ctc,_=index_rows(crows,"ctc")
    hrows=read_jsonl(run/"hybrid_results.jsonl")
    selected_gap=float(sel["hybrid_minimum_gap"]); selected_margin=float(sel["hybrid_minimum_margin"])
    hybrid={}
    for r in hrows:
        sid=str(r.get("id","")); inp=inputs.get(sid,{})
        gap=finite_signed(r.get("minimum_gap")); margin=finite_signed(r.get("minimum_margin"))
        if str(inp.get("split"))=="test" and (gap!=selected_gap or margin!=selected_margin):
            raise ValueError(f"test hybrid row for {sid} does not use selected dev parameters")
        if gap==selected_gap and margin==selected_margin:
            if sid in hybrid: raise ValueError(f"hybrid duplicate selected key: {sid}")
            hybrid[sid]=r
    expected=set(inputs)
    unexpected={"qwen":set(qwen)-expected,"ctc":set(ctc)-expected,"hybrid":set(hybrid)-expected,
                "native":{sid for sid,_,_ in native}-expected}
    if any(unexpected.values()): raise ValueError(f"extra IDs are always rejected: { {k:sorted(v)[:20] for k,v in unexpected.items() if v} }")
    # Test ID coverage is checked independently for every produced backend. Partial mode uses intersection.
    arm_ids={"qwen_raw":set(qwen),"ctc_greedy":set(ctc),"qwen_ctc_sidecar":set(hybrid)}
    for cond,arm in (("greedy","native_greedy"),("beam","native_beam"),("hotword","native_hotword")):
        arm_ids[arm]={sid for sid,c,sc in native if c==cond and (arm!="native_hotword" or sc==chosen_hotword)}
    if any(c not in {"greedy","beam","hotword"} for _,c,_ in native): raise ValueError("native has unknown condition")
    dev_groups={str(inputs[sid].get("source_group")) for sid in split_ids["dev"]}
    test_groups={str(inputs[sid].get("source_group")) for sid in split_ids["test"]}
    if dev_groups & test_groups: raise ValueError(f"dev/test source-group leakage: {sorted(dev_groups & test_groups)[:10]}")
    test_expected=split_ids["test"]
    test_coverage={a:{"expected":len(test_expected),"actual":len(ids&test_expected),"missing":sorted(test_expected-ids),"extra":sorted(ids-test_expected)} for a,ids in arm_ids.items()}
    if not args.allow_partial:
        bad={a:v["missing"][:10] for a,v in test_coverage.items() if v["missing"] or v["actual"]!=v["expected"]}
        if bad: raise ValueError(f"test backend IDs must exactly cover all 1829 test inputs; use --allow-partial for paired common IDs: {bad}")
    per_split_ids={}
    for split,ids in split_ids.items():
        common=set(ids)
        for aset in arm_ids.values(): common &= aset
        per_split_ids[split]=common if args.allow_partial else set(ids)
        if not args.allow_partial:
            # Every backend, including the newly completed Qwen control run, must cover each split.
            for arm,aset in arm_ids.items():
                if not ids<=aset:
                    raise ValueError(f"{split} {arm} missing {len(ids-aset)} IDs; use --allow-partial")

    sample_out=[]; domain_acc=defaultdict(lambda:defaultdict(list)); scores_by_split={}
    raw_texts={}; library_cache={}
    def get_texts(sid):
        qr=qwen.get(sid); cr=ctc.get(sid); hr=hybrid.get(sid)
        nr={"greedy":native.get((sid,"greedy",None)),"beam":native.get((sid,"beam",None)),"hotword":native.get((sid,"hotword",chosen_hotword))}
        texts={"qwen_raw":str((qr or {}).get("final_payload",{}).get("text", "")) if qr else None,
               "ctc_greedy":str((cr or {}).get("ctc_text", "")) if cr else None,
               "native_greedy":str(nr["greedy"].get("final",{}).get("text","")) if nr["greedy"] else None,
               "native_beam":str(nr["beam"].get("final",{}).get("text","")) if nr["beam"] else None,
               "native_hotword":str(nr["hotword"].get("final",{}).get("text","")) if nr["hotword"] else None,
               "qwen_ctc_sidecar":str((hr or {}).get("text", "")) if hr else None}
        return texts,qr,cr,hr,nr
    for split,ids in per_split_ids.items():
        for sid in sorted(ids):
            inp=inputs[sid]; refobj=refs[sid]; ref=str(refobj.get("reference","")); gold=refobj.get("gold_entities",[])
            if not isinstance(gold,list): raise ValueError(f"{sid}: gold_entities must be list")
            texts,qr,cr,hr,nr=get_texts(sid); row={"id":sid,"split":split,"domain":inp.get("domain","unknown"),"source":inp.get("source"),"source_group":inp.get("source_group"),"reference":ref}
            scored={}
            for arm in ARMS:
                text=texts[arm]
                if text is not None:
                    scored[arm]=score(ref,text,gold); domain_acc[(split,str(inp.get("domain") or "unknown"))][arm].append(scored[arm])
                row[arm+"_text"]=text
                if arm in scored:
                    for key in ("entity_hits","entity_mentions","entity_recall","cer_errors","reference_chars","cer"):
                        row[f"{arm}_{key}"]=scored[arm][key]
                else:
                    for key in ("entity_hits","entity_mentions","entity_recall","cer_errors","reference_chars","cer"): row[f"{arm}_{key}"]=None
            # explicit paired correction deltas vs Qwen and Native beam, clipped by gold mention count
            for arm in ("ctc_greedy","native_greedy","native_hotword","qwen_ctc_sidecar"):
                for base in ("qwen_raw","native_beam"):
                    prefix=f"{arm}_vs_{base}"
                    if arm not in scored or base not in scored:
                        row[prefix+"_entity_recovered"]=None; row[prefix+"_entity_lost"]=None; row[prefix+"_cer_improved"]=None; row[prefix+"_cer_degraded"]=None
                        continue
                    gc=Counter(scored[arm]["gold_counts"]); old=norm(texts[base]); new=norm(texts[arm]); rec=lost=0
                    for canon,n in gc.items():
                        d=min(n,new.count(canon))-min(n,old.count(canon)); rec+=max(0,d); lost+=max(0,-d)
                    row[prefix+"_entity_recovered"]=rec; row[prefix+"_entity_lost"]=lost
                    row[prefix+"_cer_improved"]=scored[arm]["cer_errors"]<scored[base]["cer_errors"]
                    row[prefix+"_cer_degraded"]=scored[arm]["cer_errors"]>scored[base]["cer_errors"]
            # Lexical library audit: report raw unsupported occurrences and incremental additions.
            lib=Path(str(inp.get("library_path", "")))
            if not lib.is_file(): lib=run/"libraries"/lib.name
            names=[]
            if lib.is_file():
                import sqlite3
                lp=str(lib.resolve())
                if lp not in library_cache:
                    with sqlite3.connect(f"file:{lp}?mode=ro",uri=True) as con:
                        library_cache[lp]=[str(x[0]) for x in con.execute("SELECT DISTINCT canonical_text FROM protected_entities WHERE enabled=1")]
                names=library_cache[lp]
            refn=norm(ref)
            unsupported_names={}
            for aud_arm in ("ctc_greedy","native_greedy","native_beam","native_hotword","qwen_ctc_sidecar"):
                text=texts[aud_arm]
                counts={}
                if text is not None:
                    h=norm(text)
                    for name in names:
                        needle=norm(name)
                        if needle and needle not in refn and h.count(needle): counts[name]=h.count(needle)
                unsupported_names[aud_arm]=counts
                row["unsupported_library_mentions_"+aud_arm]=sum(counts.values()) if text is not None and names else None
                row["unsupported_library_canonicals_"+aud_arm]="|".join(sorted(counts))
            for aud_arm,base_arm in (("native_hotword","native_beam"),("qwen_ctc_sidecar","qwen_raw")):
                newcounts=unsupported_names[aud_arm]; oldtext=texts[base_arm]
                additions=[]
                if oldtext is not None and texts[aud_arm] is not None:
                    old=norm(oldtext); new=norm(texts[aud_arm])
                    for name in names:
                        needle=norm(name)
                        if needle and needle not in refn and new.count(needle)>old.count(needle): additions.append(name)
                row[f"added_unsupported_library_canonicals_{aud_arm}_vs_{base_arm}"]="|".join(sorted(set(additions)))
                row[f"added_unsupported_library_mention_count_{aud_arm}_vs_{base_arm}"]=sum(max(0,newcounts.get(name,0)- (norm(oldtext).count(norm(name)) if oldtext is not None and norm(name) else 0)) for name in additions) if oldtext is not None else None
            # wall/compute measurements, keeping estimates distinctly labeled
            timing={}
            for arm in ("native_greedy","native_beam","native_hotword"):
                cond={"native_greedy":"greedy","native_beam":"beam","native_hotword":"hotword"}[arm]; x=nr[cond]
                if x:
                    timing[arm+"_compute_seconds"]=finite_num(x.get("compute_seconds_total")); timing[arm+"_finish_wall_seconds"]=finite_num(x.get("finish_seconds_including_flush")); timing[arm+"_audio_seconds"]=finite_num(x.get("audio_seconds"))
            if cr:
                timing["ctc_compute_seconds"]=finite_num(cr.get("compute_seconds")); timing["ctc_finish_wall_seconds"]=finite_num(cr.get("finish_seconds")); timing["ctc_audio_seconds"]=finite_num(cr.get("audio_seconds"))
            if hr:
                timing["hybrid_fusion_compute_seconds"]=finite_num(hr.get("fusion_compute_seconds")); timing["hybrid_ctc_compute_seconds"]=finite_num(hr.get("ctc_compute_seconds")); timing["hybrid_ctc_finish_wall_seconds"]=finite_num(hr.get("ctc_finish_seconds")); timing["hybrid_audio_seconds"]=finite_num(inp.get("duration_seconds"))
                # sequential compute estimate only; qwen saved HTTP wall is reported separately
                timing["hybrid_sequential_compute_estimate_seconds"]=(sum(finite_num(hr.get(k)) or 0 for k in ("fusion_compute_seconds","ctc_compute_seconds")) + (sum(finite_num(z.get("http_seconds")) or 0 for z in qr.get("chunk_payloads",[])) if qr else 0))
            if qr:
                timing["qwen_saved_http_seconds"]=sum(finite_num(x.get("http_seconds")) or 0 for x in qr.get("chunk_payloads",[]))
            row["timing"]=timing
            # First of two consecutive canonical matches, at audio availability; never speech-end latency.
            def first_two_consecutive(events, text_key, time_key):
                seq=[]
                for ev in events or []:
                    t=ev.get(text_key)
                    if t is None: continue
                    seq.append((finite_num(ev.get(time_key)), norm(str(t))))
                found=[]
                for ent in gold:
                    canonical=str(ent.get("canonical") or ent.get("surface") or "")
                    needle=norm(canonical)
                    if not needle: continue
                    for i in range(len(seq)-1):
                        if seq[i][0] is not None and needle in seq[i][1] and needle in seq[i+1][1]:
                            found.append((seq[i][0],canonical)); break
                return min(found)[0] if found else None
            for arm, events, textkey, timekey in (
                ("native_greedy", [e for e in (nr["greedy"] or {}).get("partials",[]) if not e.get("is_flush_padding")], "text", "input_audio_seconds"),
                ("native_beam", [e for e in (nr["beam"] or {}).get("partials",[]) if not e.get("is_flush_padding")], "text", "input_audio_seconds"),
                ("native_hotword", [e for e in (nr["hotword"] or {}).get("partials",[]) if not e.get("is_flush_padding")], "text", "input_audio_seconds"),
                ("ctc_greedy", [e for e in (cr or {}).get("events",[]) if not e.get("after_input_finished")], "ctc_text", "audio_seconds"),
                ("qwen_ctc_sidecar", (hr or {}).get("events",[]), "text", "audio_seconds")):
                row[arm+"_first_two_consecutive_gold_match_audio_available_seconds"]=first_two_consecutive(events,textkey,timekey)
            sample_out.append(row)

    def aggregate(items):
        if not items:return {"samples":0,"entity_hits":None,"entity_mentions":None,"entity_recall":None,"cer_errors":None,"reference_chars":None,"cer":None}
        h=sum(x["entity_hits"] for x in items); m=sum(x["entity_mentions"] for x in items); e=sum(x["cer_errors"] for x in items); n=sum(x["reference_chars"] for x in items)
        return {"samples":len(items),"entity_hits":h,"entity_mentions":m,"entity_recall":h/m if m else 0.0,"cer_errors":e,"reference_chars":n,"cer":e/n if n else (0.0 if e==0 else 1.0)}
    summaries={}
    for split in ("dev","test","control"):
        summaries[split]={}
        ids=per_split_ids.get(split,set())
        for arm in ARMS:
            items=[score(str(refs[sid].get("reference","")),get_texts(sid)[0][arm],refs[sid].get("gold_entities",[])) for sid in ids if get_texts(sid)[0][arm] is not None]
            summaries[split][arm]=aggregate(items)
    # performance from records with finite values: sum compute / sum audio; wall percentiles
    perf=defaultdict(dict)
    for split in ("dev","test","control"):
        ids=per_split_ids.get(split,set())
        for arm in ARMS:
            observations=[]
            for sid in ids:
                _,qr,cr,hr,nr=get_texts(sid)
                if arm.startswith("native_"):
                    cond={"native_greedy":"greedy","native_beam":"beam","native_hotword":"hotword"}[arm]; x=nr[cond]
                    if x: observations.append((finite_num(x.get("compute_seconds_total")),finite_num(x.get("finish_seconds_including_flush")),finite_num(x.get("audio_seconds"))))
                elif arm=="ctc_greedy" and cr: observations.append((finite_num(cr.get("compute_seconds")),finite_num(cr.get("finish_seconds")),finite_num(cr.get("audio_seconds"))))
                elif arm=="qwen_raw" and qr: observations.append((None,None,finite_num(inputs[sid].get("duration_seconds"))))
                elif arm=="qwen_ctc_sidecar" and hr:
                    seq=(finite_num(hr.get("fusion_compute_seconds")) or 0)+(finite_num(hr.get("ctc_compute_seconds")) or 0)+(sum(finite_num(z.get("http_seconds")) or 0 for z in qr.get("chunk_payloads",[])) if qr else 0)
                    observations.append((seq,None,finite_num(inputs[sid].get("duration_seconds"))))
            comp_audio=[(x[0],x[2]) for x in observations if x[0] is not None and x[2] is not None]; comp=[x[0] for x in comp_audio]; wall=[x[1] for x in observations if x[1] is not None]; audio=[x[1] for x in comp_audio]
            perf[split][arm]={"samples_with_compute":len(comp),"samples_with_finish_wall":len(wall),"sum_compute_seconds":sum(comp) if comp else None,
              "sum_audio_seconds_for_compute":sum(audio) if comp_audio else None,
              "compute_rtf":(sum(comp)/sum(audio)) if comp_audio and sum(audio) else None,
              "finish_wall_p50_seconds":statistics.median(wall) if wall else None,
              "finish_wall_p95_seconds":sorted(wall)[max(0,math.ceil(.95*len(wall))-1)] if wall else None}
    # domain CSV rows from scored sample table
    domains=defaultdict(lambda:defaultdict(list))
    for r in sample_out:
        for arm in ARMS:
            if r[arm+"_text"] is not None:
                domains[(r["split"],r["domain"])][arm].append({"entity_hits":r[arm+"_entity_hits"],"entity_mentions":r[arm+"_entity_mentions"],"cer_errors":r[arm+"_cer_errors"],"reference_chars":r[arm+"_reference_chars"]})
    with (run/"domain_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["split","domain","arm","samples","entity_hits","entity_mentions","entity_recall","cer_errors","reference_chars","cer"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for (split,domain),arms in sorted(domains.items()):
            for arm,vals in sorted(arms.items()): w.writerow({"split":split,"domain":domain,"arm":arm,**aggregate(vals)})
    with (run/"sample_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=list(sample_out[0]) if sample_out else ["id","split","domain"]; w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sample_out)
    # reports and machine-readable verification
    verification={"status":"partial" if args.allow_partial and any(len(per_split_ids[s])<len(split_ids[s]) for s in split_ids) else "complete",
      "allow_partial":args.allow_partial,"input_count":len(inputs),"reference_count":len(refs),"frozen_protocol_hashes_verified":True,"dev_test_source_group_disjoint":True,"split_expected":{k:len(v) for k,v in split_ids.items()},
      "split_scored_common_ids":{k:len(v) for k,v in per_split_ids.items()},"test_backend_coverage":test_coverage,
      "extra_ids_rejected":True,"duplicate_ids_rejected":True,"selection_file":str(selection_path),"selected_parameters":{k:sel[k] for k in required},
      "normalization":"ZhITNNormalizer then OpenCC t2s, NFKC, casefold, remove Unicode P/Z/C; clipped canonical mention counts use non-overlapping str.count; CER character Levenshtein.",
      "test_tuning":"test split only scored with dev selection values; no test parameter selection performed.","control_qwen":"Qwen primary is scored when present; any genuinely missing hypothesis remains NA and is excluded from that arm sample denominator.",
      "metric_denominators":"sample counts are scored non-missing hypotheses; entity recall denominator is summed gold mentions; CER denominator is summed normalized reference characters.",
      "limitations":["Dataset is source-group separated dev/test; reports are specific to these model backends and this corpus.","Native CJK character-only dictionary encoding can cover only part of the all-canonical library; encoded and requested coverage are reported from native outputs.","Non-reference library canonical occurrences and incremental additions are lexical audit candidates only, not human-reviewed precision.","First match means the canonical appears in two consecutive available partials; the field records the first partial audio-available position. There is no gold acoustic entity-end time, so this is not entity-speech-end-to-caption latency.","Hybrid sequential compute is a compute estimate from CTC plus fusion plus saved Qwen primary HTTP wall where available; it is not measured concurrent end-to-end latency and is not comparable to native CPU wall time."],
      "fresh_latency":"Actual wallclock-paced four-arm profile is reported separately in PACED_PROFILE.md and paced_profile_summary.json; only that file establishes profile completeness."}
    (run/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    native_coverage={}
    for split in ("dev","test","control"):
        for arm,cond in (("native_greedy","greedy"),("native_beam","beam"),("native_hotword","hotword")):
            rows=[native.get((sid,cond,None if cond!="hotword" else chosen_hotword)) for sid in per_split_ids.get(split,set())]
            rows=[x for x in rows if x]
            native_coverage.setdefault(split,{})[arm]={"samples":len(rows),
                "canonical_requested_sum":sum(int(x.get("hotword_canonical_count") or 0) for x in rows),
                "canonical_encoded_sum":sum(int(x.get("hotword_encoded_count") or 0) for x in rows),
                "canonical_encoded_fraction":(sum(int(x.get("hotword_encoded_count") or 0) for x in rows)/sum(int(x.get("hotword_canonical_count") or 0) for x in rows)) if sum(int(x.get("hotword_canonical_count") or 0) for x in rows) else None}
    summary={"splits":summaries,"performance":perf,"native_dictionary_coverage":native_coverage,"selection":{k:sel[k] for k in required},"samples_with_unsupported_library_additions":sum(bool(x.get("added_unsupported_library_canonicals_native_hotword_vs_native_beam") or x.get("added_unsupported_library_canonicals_qwen_ctc_sidecar_vs_qwen_raw")) for x in sample_out),"limitations":verification["limitations"]}
    (run/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# 实体后端对比报告","","| Split | 评分臂 | 样本数 | 实体命中 / mentions | 实体召回率 | CER |","|---|---|---:|---:|---:|---:|"]
    for split in ("dev","test","control"):
        for arm in ARMS:
            s=summaries[split][arm]; h="NA" if s["entity_hits"] is None else f'{s["entity_hits"]} / {s["entity_mentions"]}'; recall="NA" if s["entity_recall"] is None else f'{s["entity_recall"]:.2%}'; cerv="NA" if s["cer"] is None else f'{s["cer"]:.2%}'
            lines.append(f'| {split} | {LABELS[arm]} | {s["samples"]} | {h} | {recall} | {cerv} |')
    lines += ["","实体命中数按同一规范化 canonical 的 gold mention 数截顶；召回率分母为该 split 的 gold mention 总数，CER 分母为规范化 reference 字符数。每个表中样本数是该臂存在 hypothesis 的样本数。dev、test、control 分开统计；控制集 Qwen 有实际主流记录时正常评分，真实缺失值显示为 NA。test 仅套用 dev 选择的参数。","","## 配对变化","","sample_comparison.csv 按样本给出 CTC、native hotword、hybrid 相对 Qwen raw 和 native beam 的 clipped 实体 recovered/lost 数及 CER 改善/退化标记；聚合指标见 summary.json。词库审计列出 CTC、native 与 hybrid 输出里标准 reference 不支持的本样本词库 canonical 命中，并比较 native hotword 对 native beam、hybrid 对 Qwen 的新增项；这是自动词法候选，不是人工 precision。","","## 性能与限制","","Native/CTC compute RTF = sum(compute seconds) / sum(audio seconds)；finish wall 的 p50/p95 按样本 finish 字段计算。Hybrid 的顺序 compute estimate 包含 CTC 与 fusion compute，以及可用的保存 Qwen HTTP wall，不能当作真实并发端到端 latency，也不与 native CPU wall time等同。真实按时钟送音的四臂延迟另见 PACED_PROFILE.md；完整性及计时校验见 paced_profile_summary.json。首次匹配要求 canonical 在连续两个 partial 中出现，记录第一个 partial 的音频可用位置；没有 gold 声学结束时间，因此不称为实体说完到字幕延迟。Native CJK 字符编码只覆盖全部 canonical 的一部分，需结合 hotword_encoded_count / canonical_count 及 coverage 字段解释。","","这些结果只代表本次冻结语料和具体 Qwen/native/CTC/hybrid 后端，不能外推到全部 CTC 模型或所有替换 ASR。",""]
    (run/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
