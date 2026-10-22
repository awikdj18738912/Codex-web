#!/usr/bin/env python3
"""Select and report a broad-vocabulary native backend using dev-only tuning."""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from report_entity_backend_comparison import score as score_text, norm

COUNTS={"dev":171,"test":1829,"control":100}
MAIN_ARMS=("qwen_raw","qwen_ctc_sidecar","native_beam","native_hotword")
BROAD_ARMS=("broad2023_beam","broad2023_hotword")

def read_jsonl(path:Path)->list[dict]:
    if not path.exists():return []
    rows=[]
    with path.open(encoding="utf-8") as f:
        for lineno,line in enumerate(f,1):
            if line.strip():
                x=json.loads(line)
                if not isinstance(x,dict):raise ValueError(f"{path}:{lineno}: expected object")
                rows.append(x)
    return rows

def read_csv(path:Path)->list[dict]:
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def metric_add(dst:dict,x:dict):
    for k in ("samples","entity_hits","entity_mentions","cer_errors","reference_chars"):
        dst[k]=dst.get(k,0)+int(x.get(k,1) if k=="samples" else x[k])
def finish_metric(x:dict)->dict:
    n=x.get("samples",0);hits=x.get("entity_hits",0);mentions=x.get("entity_mentions",0);errors=x.get("cer_errors",0);chars=x.get("reference_chars",0)
    return {"samples":n,"entity_hits":hits,"entity_mentions":mentions,"entity_recall":hits/mentions if mentions else 0.0,
            "cer_errors":errors,"reference_chars":chars,"cer":errors/chars if chars else (0.0 if errors==0 else 1.0)}
def text_of(row:dict)->str:
    return str((row.get("final") or {}).get("text",""))
def ref_gold(refs:dict,sid:str)->tuple[str,list]:
    r=refs[sid];return str(r.get("reference","")),r.get("gold_entities",[])
def make_scored(rows:dict,ids:set[str],condition:str,score_value:float|None,refs:dict)->dict[str,dict]:
    out={}
    for sid in ids:
        row=rows.get((sid,condition,score_value))
        if row is None:continue
        ref,gold=ref_gold(refs,sid);out[sid]=score_text(ref,text_of(row),gold);out[sid]["hypothesis"]=text_of(row)
    return out
def aggregate_scored(scored:dict[str,dict])->dict:
    sums={"samples":0,"entity_hits":0,"entity_mentions":0,"cer_errors":0,"reference_chars":0}
    for x in scored.values():metric_add(sums,x)
    return finish_metric(sums)
def load_broad_rows(base:Path)->tuple[list[dict],list[str]]:
    broad=base/"broad_vocab"; candidates=[broad/"native_results.jsonl",*sorted(broad.glob("native_results_shard*.jsonl")),
        broad/"native_broad_vocab_development.jsonl",base/"native_broad_vocab_development.jsonl"]
    files=[]; seenfiles=set()
    for p in candidates:
        if p.exists() and p.resolve() not in seenfiles:files.append(p);seenfiles.add(p.resolve())
    rows=[]; index={}
    for p in files:
        for r in read_jsonl(p):
            condition=str(r.get("condition"));sid=str(r.get("id",""));sv=r.get("score")
            score_value=None if sv is None else float(sv);key=(sid,condition,score_value)
            if key in index:
                old=index[key]
                if old.get("pcm_sha256")!=r.get("pcm_sha256") or text_of(old)!=text_of(r):
                    raise ValueError(f"conflicting duplicate broad result {key} in {p}")
                continue
            index[key]=r;rows.append(r)
    if not files:raise FileNotFoundError(f"No broad native output found under {broad} or {base}")
    return rows,[str(p) for p in files]
def load_main_csvs(base:Path,inputs:dict)->tuple[dict,dict,list[str]]:
    domains_path=base/"domain_metrics.csv";samples_path=base/"sample_comparison.csv"
    for p in (domains_path,samples_path):
        if not p.is_file():raise FileNotFoundError(f"Main comparison CSV not ready: {p}")
    domain_rows=read_csv(domains_path);sample_rows=read_csv(samples_path)
    sample_map={}
    for r in sample_rows:
        sid=str(r.get("id",""))
        if sid in sample_map:raise ValueError(f"duplicate id in main sample_comparison.csv: {sid}")
        sample_map[sid]=r
    if set(sample_map)!=set(inputs):raise ValueError(f"main sample CSV ID mismatch: missing={len(set(inputs)-set(sample_map))} extra={len(set(sample_map)-set(inputs))}")
    metrics=defaultdict(dict)
    sums=defaultdict(lambda:{"samples":0,"entity_hits":0,"entity_mentions":0,"cer_errors":0,"reference_chars":0})
    for r in domain_rows:
        split,arm=str(r.get("split")),str(r.get("arm"))
        if arm not in MAIN_ARMS:continue
        key=(split,arm);item={"samples":int(r["samples"]),"entity_hits":int(r["entity_hits"]),"entity_mentions":int(r["entity_mentions"]),
            "cer_errors":int(r["cer_errors"]),"reference_chars":int(r["reference_chars"])}
        metric_add(sums[key],item)
    for (split,arm),x in sums.items():metrics[split][arm]=finish_metric(x)
    for split,count in COUNTS.items():
        ids={sid for sid,r in inputs.items() if r.get("split")==split}
        if len(ids)!=count:raise ValueError(f"expected {count} {split} inputs, found {len(ids)}")
        for arm in MAIN_ARMS:
            metric=metrics.get(split,{}).get(arm)
            if metric is None or metric["samples"]!=count:raise ValueError(f"main CSV arm {arm}/{split} coverage is {None if metric is None else metric['samples']}/{count}")
    return metrics,sample_map,[str(domains_path),str(samples_path)]
def field_int(row:dict,key:str)->int|None:
    v=row.get(key)
    if v in (None,""):return None
    try:
        f=float(v)
        return int(f) if math.isfinite(f) else None
    except (TypeError,ValueError):return None
def paired_compare(broad:dict[str,dict],main_samples:dict[str,dict],split_ids:set[str],base_arm:str)->dict:
    improved=degraded=equal=rec=lost=paired=0
    for sid in split_ids:
        b=broad.get(sid);m=main_samples.get(sid)
        if not b or not m:continue
        be=b["cer_errors"];me=field_int(m,f"{base_arm}_cer_errors");mh=field_int(m,f"{base_arm}_entity_hits")
        if me is None or mh is None:continue
        paired+=1;improved+=be<me;degraded+=be>me;equal+=be==me
        
        bn=norm(b["hypothesis"]);mn=norm(m.get(base_arm+"_text", ""))
        for canonical,n in b["gold_counts"].items():
            delta=min(n,bn.count(canonical))-min(n,mn.count(canonical));rec+=max(0,delta);lost+=max(0,-delta)
    return {"paired_samples":paired,"cer_improved_samples":improved,"cer_degraded_samples":degraded,"cer_equal_samples":equal,
            "entity_hit_delta_vs_main_arm":rec-lost,"broad_entity_recovered_mentions":rec,"broad_entity_lost_mentions":lost}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run-dir",type=Path,default=ROOT/"results/entity_backend_comparison_20261015")
    ap.add_argument("--select-only",action="store_true",help="select dev hotword score and write selection.json; do not require completed test/control or main CSVs")
    a=ap.parse_args();base=a.run_dir.resolve();broad=base/"broad_vocab"
    protocol=json.loads((base/"protocol.json").read_text(encoding="utf-8"));refs=json.loads((base/"references.json").read_text(encoding="utf-8"))["samples"]
    input_rows=read_jsonl(base/"inputs.jsonl");inputs={str(x["id"]):x for x in input_rows}
    if len(inputs)!=len(input_rows) or set(inputs)!=set(refs):raise ValueError("frozen input/reference IDs are not a unique exact match")
    split_ids={s:{sid for sid,r in inputs.items() if r.get("split")==s} for s in COUNTS}
    for split,n in COUNTS.items():
        if len(split_ids[split])!=n:raise ValueError(f"expected {n} {split} IDs, found {len(split_ids[split])}")
    broad_rows,source_files=load_broad_rows(base);broad_idx={}
    for r in broad_rows:
        sid=str(r.get("id",""));condition=str(r.get("condition"));sv=r.get("score");sc=None if sv is None else float(sv);key=(sid,condition,sc)
        if key in broad_idx:raise ValueError(f"duplicate broad key {key}")
        if sid not in inputs:raise ValueError(f"extra broad ID: {sid}")
        if str(r.get("split"))!=str(inputs[sid].get("split")):raise ValueError(f"split mismatch for broad row {sid}")
        if r.get("pcm_sha256") and r["pcm_sha256"]!=inputs[sid].get("pcm_sha256"):raise ValueError(f"PCM hash mismatch for broad row {sid}")
        broad_idx[key]=r
    dev=split_ids["dev"]
    beams=make_scored(broad_idx,dev,"beam",None,refs)
    if set(beams)!=dev:raise ValueError(f"broad dev beam must cover all 171 unique IDs; missing={len(dev-set(beams))}")
    beam_metric=aggregate_scored(beams)
    hot_scores=sorted({sc for sid,cond,sc in broad_idx if sid in dev and cond=="hotword" and sc is not None})
    candidates=[]
    for weight in hot_scores:
        scored=make_scored(broad_idx,dev,"hotword",weight,refs)
        if set(scored)!=dev:continue
        metric=aggregate_scored(scored)
        candidates.append({"score":weight,"metrics":metric,"complete":True})
    if {c["score"] for c in candidates}!={.5,1.,1.5,2.,3.}:raise ValueError("All five preregistered hotword weights need complete 171-sample dev coverage")
    safe=[c for c in candidates if c["metrics"]["cer_errors"]<=beam_metric["cer_errors"]]
    if safe:chosen=max(safe,key=lambda c:(c["metrics"]["entity_hits"],-c["metrics"]["cer_errors"],-c["score"]))
    else:chosen=min(candidates,key=lambda c:(c["metrics"]["cer_errors"],-c["metrics"]["entity_hits"],c["score"]))
    selection={"model":"small-bilingual-zh-en-2023-02-16","development_samples":len(dev),"selection_uses_test_labels":False,
      "criterion":"Among hotword weights with complete dev coverage, maximize entity hits while not increasing normalized CER errors over broad-vocabulary beam; tie-break fewer errors then lower score. If none is safe, choose minimum CER errors, then more hits, then lower score.",
      "beam_dev":beam_metric,"tested_hotword_scores":candidates,"native_hotword_score":chosen["score"],"selected_hotword_dev":chosen["metrics"]}
    selection_path=broad/"selection.json";broad.mkdir(parents=True,exist_ok=True)
    if selection_path.exists():
        if json.loads(selection_path.read_text())!=selection:raise ValueError("Broad selection already frozen and differs")
    else:selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if a.select_only:
        print(json.dumps({"selection_file":str(selection_path),"native_hotword_score":chosen["score"],"dev_candidates":len(candidates)},ensure_ascii=False));return
    # Held-out broad-vocabulary output must contain only the chosen hotword score and beam.
    broad_scores={}
    for split in ("test","control"):
        ids=split_ids[split]
        for sid,cond,sc in broad_idx:
            if sid in ids and cond not in {"beam","hotword"}:raise ValueError(f"unexpected {cond} row on held-out {split}: {sid}")
            if sid in ids and cond=="hotword" and sc!=chosen["score"]:raise ValueError(f"held-out {split} contains unselected hotword score {sc} for {sid}")
        for arm,cond,sc in (("broad2023_beam","beam",None),("broad2023_hotword","hotword",chosen["score"])):
            scored=make_scored(broad_idx,ids,cond,sc,refs)
            if set(scored)!=ids:raise ValueError(f"broad {split} {arm} must exactly cover {len(ids)} unique IDs; missing={len(ids-set(scored))}")
            broad_scores[(split,arm)]=scored
    main_metrics,main_samples,main_csvs=load_main_csvs(base,inputs)
    main_selection=json.loads((base/"selection.json").read_text(encoding="utf-8")) if (base/"selection.json").is_file() else {}
    comparisons={}
    for split in ("dev","test","control"):
        comparisons[split]={arm:main_metrics[split][arm] for arm in MAIN_ARMS}
        if split in ("test","control"):
            for arm in BROAD_ARMS:comparisons[split][arm]=aggregate_scored(broad_scores[(split,arm)])
        else:
            # Dev broad metrics are used only for selection and are not presented as held-out estimates.
            comparisons[split]["broad2023_beam_dev_selection_only"]=beam_metric
            comparisons[split]["broad2023_hotword_dev_selection_only"]=chosen["metrics"]
    paired={}
    for split in ("test","control"):
        paired[split]={};ids=split_ids[split]
        for broad_arm in BROAD_ARMS:
            scored=broad_scores[(split,broad_arm)]
            paired[split][broad_arm]={base_arm:paired_compare(scored,main_samples,ids,base_arm) for base_arm in ("qwen_raw","qwen_ctc_sidecar","native_beam","native_hotword")}
    summary={"run_dir":str(base),"input_counts":{s:len(v) for s,v in split_ids.items()},"selected_hotword_score":chosen["score"],
      "selection_file":str(selection_path),"selection_method":"dev only","native_2025_selected_hotword_score":main_selection.get("native_hotword_score"),"normalization_source":"experiments/scripts/score_current_entity_challenge.py via report_entity_backend_comparison.score; ZhITN, OpenCC t2s, NFKC, casefold, remove Unicode P/Z/C; clipped canonical mention counts; normalized-character Levenshtein CER.",
      "metrics_denominators":"samples are exact scored hypotheses; entity recall denominator is summed gold mentions; CER denominator is summed normalized reference characters.",
      "comparison":{"main_run_csv_arms":MAIN_ARMS,"broad_arms":BROAD_ARMS,"splits":comparisons,"paired_broad_vs_main":paired},
      "broad_vocabulary_coverage":{s:{arm:{"samples":len(scored),"requested_words_sum":sum(int(broad_idx[(sid,"hotword",chosen["score"])].get("requested_words",broad_idx[(sid,"hotword",chosen["score"])].get("hotword_canonical_count",0)) or 0) for sid in scored if (sid,"hotword",chosen["score"]) in broad_idx),
        "encoded_words_sum":sum(int(broad_idx[(sid,"hotword",chosen["score"])].get("encoded_words",broad_idx[(sid,"hotword",chosen["score"])].get("hotword_encoded_count",0)) or 0) for sid in scored if (sid,"hotword",chosen["score"]) in broad_idx)} for arm,scored in (("broad2023_hotword",broad_scores.get((s,"broad2023_hotword"),{})),)} for s in ("test","control")},
      "source_csvs":main_csvs,"broad_result_sources":source_files,
      "integrity":{"test_exact_unique_ids":True,"test_expected":1829,"control_exact_unique_ids":True,"control_expected":100,"dev_test_controls_scored_separately":True,"test_parameters_from_dev":True},
      "limitations":["Results describe this specific broad-vocabulary 2023 bilingual model and current 2025 native/Qwen/CTC backends; they do not generalize to all native ASR models.","Hotword selection uses only dev; test/control are evaluated once at the selected score.","Main-model metrics are read from the completed main-run CSVs; broad-vocabulary metrics are rescored offline against frozen references."]}
    (broad/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Broad-vocabulary Native 对比报告","","Broad model: bilingual zh-en 2023 small, int8 encoder/joiner and FP32 decoder. Hotword score is selected from dev only; test/control use this single selected score.","","| Split | Arm | Samples | Entity hits / mentions | Entity recall | CER |","|---|---|---:|---:|---:|---:|"]
    labels={"qwen_raw":"Qwen raw","qwen_ctc_sidecar":"Qwen + CTC sidecar","native_beam":"Native 2025 beam","native_hotword":"Native 2025 hotword","broad2023_beam":"Broad 2023 beam","broad2023_hotword":"Broad 2023 hotword"}
    for split in ("dev","test","control"):
        arms=MAIN_ARMS if split=="dev" else MAIN_ARMS+BROAD_ARMS
        for arm in arms:
            m=comparisons[split][arm];lines.append(f"| {split} | {labels[arm]} | {m['samples']} | {m['entity_hits']} / {m['entity_mentions']} | {m['entity_recall']:.2%} | {m['cer']:.2%} |")
    lines += ["","分母：实体召回率使用 summed gold mentions；CER 使用 normalized reference characters。所有 dev/test/control 独立统计。测试集和控制集要求分别完整覆盖 1829/100 个唯一 ID，且只评估 dev 选出的 broad hotword 权重。","",f"Dev 选择的 broad hotword score：**{chosen['score']}**。选择基准为 broad beam；优先在 CER errors 不增加的候选中最大化实体命中，完全依据 dev。","","成对样本改变量、主 run CSV 来源、broad 词表请求/编码覆盖及限制见 `summary.json`。","", "本报告只支持对这些具体模型、词库和数据分区的比较，不推断其他 native 替换模型的效果。",""]
    (broad/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"report":str(broad/"REPORT.md"),"summary":str(broad/"summary.json"),"selection_score":chosen["score"]},ensure_ascii=False))
if __name__=="__main__":main()
