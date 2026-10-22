"""Create editable review rows from actual fresh ASR and matched Refiner arms."""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.asr_window_confidence import locate_chunks, project_window_scores
from system.confidence_supplement import window_evidence
from experiments.scripts.benchmark_fixed_gate_pipeline import sha, write


def edits(before, after):
    return [dict(kind=tag, before=before[a:b], after=after[c:d], start=a, end=b)
            for tag,a,b,c,d in SequenceMatcher(None,before,after,autojunk=False).get_opcodes()
            if tag != "equal"]


def lexical(value):
    return "".join(char for char in value if unicodedata.category(char)[0] in {"L", "N", "M"})


def terminal_windows(raw, final, profiles):
    """Partition final text for display; flag cuts inside replacement blocks.

    This is an audit alignment, not a gold edit alignment. Whole-transcript
    local review can change text after the per-window trace was committed.
    """
    opcodes = SequenceMatcher(None, raw, final, autojunk=False).get_opcodes()
    cuts, ambiguous = [0], []
    for profile in profiles[:-1]:
        boundary = profile["end"]
        for tag,a,b,c,d in opcodes:
            if a <= boundary <= b:
                uncertain = tag != "equal" and a < boundary < b
                mapped = c + boundary-a if tag == "equal" else c if boundary == a else d
                cuts.append(mapped)
                ambiguous.append(uncertain)
                break
        else:
            raise ValueError("Final text cannot be aligned for display")
    cuts.append(len(final))
    parts = [final[a:b] for a,b in zip(cuts,cuts[1:])]
    if "".join(parts) != final or cuts != sorted(cuts):
        raise ValueError("Terminal display partition does not reconstruct final text")
    return parts, ambiguous


def prepare(out):
    asr = json.loads((out/"asr_final.json").read_text())
    events = asr["all_completed_segments"]
    projection = asr["confidence_projection"]
    projected = project_window_scores(projection,final=True,source_chunks=[item["text"] for item in events])
    profiles = projected["windows"]
    policies = ("all_refine","native","supplement","deployed_learned")
    arms = {policy:json.loads((out/f"{policy}.windows.json").read_text()) for policy in policies}
    finals = {policy:json.loads((out/f"{policy}.json").read_text())["clean_text"] for policy in policies}
    terminal = {policy:terminal_windows(asr["text"],value,profiles) for policy,value in finals.items()}
    for policy, rows in arms.items():
        if [row["source"] for row in rows] != [profile["text"] for profile in profiles]:
            raise ValueError("Actual replay partition mismatch: " + policy)
    offsets = locate_chunks(asr["text"], [event["text"] for event in events])
    reviews = []
    for index, profile in enumerate(profiles):
        full, native, supplement, learned = [arms[policy][index] for policy in policies]
        baseline, output = full["baseline"], full["output"]
        proposed = edits(baseline,output)
        change = lexical(baseline) != lexical(output)
        evidence = window_evidence(profile["text"],projection)
        relevant = [event for event,(a,b) in zip(events,offsets)
                    if a < profile["end"] and b > profile["start"]]
        recommendation = "partial" if proposed else "uncertain" if full["outcome"] != "accepted" else "keep"
        reason = ("全量管线产生词语改动；逐项核对口吃清理是否保留原意。" if change else
                  "全量管线仅调整标点/空白；必要性待核对，可接受原文时不必调用。" if proposed else
                  "本次全量精修未产生可接受改动；不能据此认定原文无问题。")
        row = dict(id=f"W{index+1:03d}", window_index=index+1,
            raw_text=profile["text"], baseline_text=baseline,
            output_all_refine=output, output_native=native["output"], output_supplement=supplement["output"],
            output_learned=learned["output"], model_calls=full.get("model_calls",[]),
            full_refine_outcome=full["outcome"], full_refine_reject_reasons=full.get("reject_reasons",[]),
            native_action=native["gate"]["action"], supplement_action=supplement["gate"]["action"],
            learned_action=learned["gate"]["action"], native_gate=native["gate"],
            supplement_gate=supplement["gate"], confidence_supplement=supplement.get("confidence_supplement"),
            score_min_fragment=min((f["score"] for f in evidence["fragments"] if f.get("score") is not None),default=None),
            score_status=evidence["status"], confidence_evidence=evidence,
            assistant_recommendation=recommendation, assistant_reason=reason, proposed_edits=proposed,
            lexical_change=change, miss_candidate=bool(proposed) and supplement["gate"]["action"] != "refine",
            left_context="".join(w["text"] for w in profiles[max(0,index-2):index]),
            right_context="".join(w["text"] for w in profiles[index+1:index+3]),
            audio_start=min(event["audio_start_bound"] for event in relevant),
            audio_end=max(event["audio_end_bound"] for event in relevant), audio_url="audio_16k.wav",
            raw_start=profile["start"], raw_end=profile["end"],
            human_decision=None, human_target=None, human_audio_checked=False,
            independent_acceptance=False, training_eligible=False)
        row["window_basis"] = hashlib.sha256(json.dumps(dict(raw_asr_sha256=sha(out/"asr_final.json"),
            raw_start=profile["start"],raw_end=profile["end"],baseline=baseline),ensure_ascii=False,
            sort_keys=True).encode()).hexdigest()
        row["terminal_outputs"] = {policy:terminal[policy][0][index] for policy in policies}
        row["terminal_alignment_ambiguous"] = any(
            (index > 0 and terminal[policy][1][index-1])
            or (index < len(profiles)-1 and terminal[policy][1][index]) for policy in policies)
        row["terminal_alignment_method"] = "whole-transcript SequenceMatcher display alignment; not quality gold"
        reviews.append(row)
    override = out/"assistant_review_notes.json"
    if override.exists():
        notes = json.loads(override.read_text())
        for row in reviews:
            row.update(notes.get(row["id"],{}))
    with (out/"review_queue.jsonl").open("w") as stream:
        for row in reviews:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/"review.html").write_bytes((ROOT/"experiments/fixtures/audio5_gate_review.html").read_bytes())
    counts = dict(windows=len(reviews), native_refine=sum(r["native_action"]=="refine" for r in reviews),
        supplement_refine=sum(r["supplement_action"]=="refine" for r in reviews),
        learned_refine=sum(r["learned_action"]=="refine" for r in reviews),
        supplement_extra_review=sum(r["native_action"]!="refine" and r["supplement_action"]=="refine" for r in reviews),
        supplement_kept_native=sum(r["native_action"]=="refine" and r["supplement_action"]=="refine" for r in reviews),
        accepted_pipeline_changes=sum(bool(r["proposed_edits"]) for r in reviews),
        accepted_lexical_changes=sum(r["lexical_change"] for r in reviews),
        supplement_skipped_change_candidates=sum(r["miss_candidate"] for r in reviews),
        skipped_lexical_change_candidates=sum(r["miss_candidate"] and r["lexical_change"] for r in reviews),
        suggestions=dict(Counter(r["assistant_recommendation"] for r in reviews)), human_labels=0,
        true_missing_refinement_rate=None, matched_windows=True)
    if counts["supplement_kept_native"] != counts["native_refine"]:
        raise AssertionError("Supplement cancelled native review")
    write(out/"review_summary.json", counts)
    lines = ["# 测试音频5：待用户核对的精修窗口", "",
        "整段新 ASR；相同源窗口经过真实当前 Refiner 管线对照。下面的建议为文本层面的待核验意见，未代替听音金标。", "",
        f"共 {len(reviews)} 个窗口，原规则送审 {counts['native_refine']}，补充门控送审 {counts['supplement_refine']}。",
        "音频区间是 VAD 请求边界的上下文范围，不是逐字时间戳。", "",
        "[浏览器核验页](http://127.0.0.1:8084/review.html)支持整窗精修/部分精修/应跳过/待定、目标文字和备注。", "",
        "## 建议优先检查的窗口", ""]
    label = dict(whole="建议整窗送审",partial="建议仅处理指定部分",uncertain="需要确认",keep="暂建议跳过")
    priority = sorted(reviews,key=lambda r:(not r["miss_candidate"],not r["lexical_change"],r["window_index"]))
    for row in priority:
        if row["assistant_recommendation"] == "keep" and not row["miss_candidate"]:
            continue
        lines += [f"### {row['id']} · {label[row['assistant_recommendation']]} · {row['audio_start']:.0f}–{row['audio_end']:.0f} 秒", "",
            f"原始 ASR：{row['raw_text']}", "", f"规则准备结果：{row['baseline_text']}", "",
            f"全量精修最终结果：{row['output_all_refine']}", "",
            f"整条终态补充策略结果（显示对齐）：{row['terminal_outputs']['supplement']}", "",
            f"原规则 **{row['native_action']}**；补充门控 **{row['supplement_action']}**；既有学习门控 **{row['learned_action']}**。", "",
            row["assistant_reason"], ""]
        for edit in row["proposed_edits"]:
            lines += [f"- `{edit['before'] or '∅'}` → `{edit['after'] or '∅'}`"]
        if row.get("assistant_target"):
            lines += ["", "仅供核查的局部改动目标："+row["assistant_target"]]
        lines += ["", "你的结论：□ 整窗精修　□ 仅部分精修　□ 应跳过　□ 待定", "",
                  "可接受的改动/目标文字：", ""]
    lines += ["## 全部窗口索引", "", "| ID | 原 ASR | 原规则 | 补充门控 | 建议 |", "|---|---|---|---|---|"]
    for row in reviews:
        lines.append(f"| {row['id']} | {row['raw_text'].replace('|','\\|')} | {row['native_action']} | {row['supplement_action']} | {label[row['assistant_recommendation']]} |")
    (out/"REVIEW.md").write_text("\n".join(lines)+"\n")
    write(out/"verification.json",dict(counts=counts,output_sha256={path.name:sha(path) for path in
        [out/"asr_final.json",out/"raw_asr.txt",out/"audio_provenance.json",out/"audio_16k.wav",
         out/"review_queue.jsonl",out/"review.html",out/"REVIEW.md",*[out/f"{p}.json" for p in policies]]},
        no_human_gold_created=True))
    print(json.dumps(counts,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
