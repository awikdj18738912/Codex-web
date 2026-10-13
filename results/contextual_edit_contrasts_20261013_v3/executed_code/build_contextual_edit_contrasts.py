"""Authored context contrasts, not real audio supervision or public-data gold."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import align_edits, apply_edits, digest, review_basis
from system.edit_location_detector import input_words
from system.refinement_guard import restore_rule_numeric_surfaces

CONCEPTS = {
    "train": [("核验", "封条"), ("递交", "附件"), ("检查", "接头"), ("更新", "索引"),
              ("记录", "温度"), ("确认", "工位"), ("关闭", "阀门"), ("扫描", "条码"),
              ("复核", "账单"), ("整理", "档案"), ("填写", "清单"), ("校验", "时钟"),
              ("清理", "缓存"), ("核算", "费用"), ("传送", "资料"), ("登记", "访客"),
              ("封装", "样品"), ("替换", "滤芯"), ("测试", "线路"), ("备份", "配置")],
    "validation": [("签收", "包裹"), ("归还", "钥匙"), ("重启", "终端"),
                   ("核对", "页码"), ("装订", "凭证"), ("补充", "备注")],
    "test": [("检测", "电源"), ("清点", "螺帽"), ("打印", "标签"),
             ("收回", "证件"), ("同步", "日志"), ("调整", "支架")],
}
FRAMES = {
    "train": dict(repeat=["下一项工作是{r}。", "我们随后需要{r}再完成登记。", "接下来先{r}。"],
                  filler=["{f}，今天需要{v}{n}。", "正在安排{v}{n}，{f}，其余事项稍后处理。"],
                  positive_context="同事询问接下来的操作安排。", quote_context="发言记录中的引文如下：",
                  noun_drop="请把{n}{n}交给值班员。", noun_keep="{n}，{n}交给值班员。",
                  emphasis="这份{n}很重要，很重要，关系到后续工作。",
                  distributive="请将有关{n}的凭条一张一张地核对。",
                  demonstrative="请把那个{n}和这个{n}分别登记。",
                  normal="完成{v}{n}后，请通知同事。",
                  quotation="原句写着“{v}{v}{n}”，出现在会议记录中。",
                  answer="嗯，关于{n}的资料已经齐备了。",
                  numeric="请核验{n}编号A17，预计需要十二件备用品。",
                  degree="有关{n}的步骤需要慢慢核对，不能急着跳过。"),
    "validation": dict(repeat=["先{r}，随后登记结果。", "应该{r}之后再安排交接。", "操作开始前，需要{r}。"],
                       filler=["{f}，稍后要{v}{n}。", "准备{v}{n}时，{f}，先核实负责人。"],
                       positive_context="主管询问本次工作的执行顺序。", quote_context="档案上的原句写作：",
                       noun_drop="把{n}{n}送到复核窗口。", noun_keep="{n}，{n}送到复核窗口。",
                       emphasis="请重视这份{n}，很重要，很重要。",
                       distributive="与{n}有关的表单应一份一份地检查。",
                       demonstrative="那个{n}在左侧，这个{n}在右侧。",
                       normal="本次需要{v}{n}并通知负责人。",
                       quotation="档案引用了“{v}{v}{n}”这段文字。",
                       answer="嗯，{n}的相关手续已经办好了。",
                       numeric="涉及{n}的编号B26已经登记，总共有十四件备用品。",
                       degree="处理{n}时要仔仔细细地检查每处接口。"),
    "test": dict(repeat=["汇报进度后，请{r}。", "待负责人到场，再{r}并签字。", "正式交接之前要{r}。"],
                 filler=["{f}，后续将{v}{n}。", "需要{v}{n}的时候，{f}，提前告知现场人员。"],
                 positive_context="现场人员询问后续流程。", quote_context="手稿逐字记载的内容是：",
                 noun_drop="待会儿把{n}{n}送去登记。", noun_keep="{n}，{n}待会儿送去登记。",
                 emphasis="关于{n}的这一项很关键，很关键，务必认真核实。",
                 distributive="请把涉及{n}的文件逐份归档，一份一份核实。",
                 demonstrative="这边那个{n}已经核实，旁边这个还没有。",
                 normal="负责人已经同意{v}{n}，可以继续办理。",
                 quotation="手稿上的文字是“{v}{v}{n}”。",
                 answer="嗯，{n}确实已经按计划安排了。",
                 numeric="这份{n}对应编号C38，备用品数量是十八件。",
                 degree="有关{n}的流程要踏踏实实地完成。"),
}
FILLERS = ("嗯", "呃", "那个", "这个")
UNSEEN_FILLERS = ("额", "就是说", "呃呃")


def model_signature(row):
    return digest(input_words(row)[0])


def validate_partitions(partitions):
    for key in ("id", "baseline_text", "pair_id", "concept_group", "delexicalized_template"):
        groups = [{row[key] for row in rows} for rows in partitions.values()]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Contrast split leakage: " + key)
    for name, rows in partitions.items():
        signatures = {}
        for row in rows:
            if row["split"] != name:
                raise ValueError("Misassigned split")
            if apply_edits(row["baseline_text"], row["edits"]) != row["desired_text"]:
                raise ValueError("Target cannot be reconstructed")
            signature = model_signature(row)
            old = signatures.setdefault(signature, row["desired_text"])
            if old != row["desired_text"]:
                raise ValueError("Identical visible model input has conflicting targets")
            if row["training_eligible"] != (name == "train") or row["human_audio_gold"]:
                raise ValueError("Invalid supervision eligibility")


def build_rows(normalizer):
    result = {key: [] for key in CONCEPTS}
    for split, concepts in CONCEPTS.items():
        frame = FRAMES[split]
        for verb, noun in concepts:
            concept = "contrast:" + split + ":" + verb + ":" + noun
            def add(raw, target, family, role, pair, *, left="", lexical_scope="seen_surfaces_new_context", template=None):
                baseline = normalizer.normalize(raw).text
                desired = normalizer.normalize(target).text
                if family != "numeric" and (baseline != raw or desired != target):
                    raise ValueError("Unexpected upstream rule transformation in semantic contrast")
                if restore_rule_numeric_surfaces(baseline, desired) != desired:
                    raise ValueError("Construction changes rule-owned numbers")
                task = family if role == "edit" else None
                row = dict(schema_version="edit_locations_v1", id=digest([concept, family, role, pair, raw, left])[:24],
                    baseline_text=baseline, raw_asr=raw, desired_text=desired, left_context=left, right_context="",
                    available_context="", entity_candidates=[], entity_hints=[],
                    category=family, split=split, recording_group=concept, split_group=concept,
                    concept_group=concept, pair_id=digest([concept, family, pair])[:24],
                    contrast_role=role, lexical_scope=lexical_scope,
                    delexicalized_template=template or family + ":" + str(pair),
                    label_status="authored_context_contrast", label_origin="authored_controlled_text",
                    allowed_usage="text_pretraining_only", training_eligible=split == "train",
                    human_audio_gold=False, audio_path=None, audio_sha256=None,
                    independent_acceptance_eligible=False, context_is_authored_not_ASR_history=True,
                    source_intent="known_injected_noise_or_explicit_semantic_control_not_audio_truth",
                    upstream_numeric_changes=baseline != raw)
                row.update(align_edits(baseline, desired, task))
                row["review_basis_sha256"] = review_basis(row)
                result[split].append(row)
            for index, template in enumerate(frame["repeat"]):
                literal = (verb[0] * 2 + verb + noun if index == 0 else (verb + noun) * 2 if index == 1 else verb[0] + verb + noun)
                raw = template.format(r=literal)
                desired = template.format(r=verb + noun)
                pair = "quote_repeat:" + str(index)
                add(raw, desired, "repetition", "edit", pair, left=frame["positive_context"], template=template)
                add(raw, raw, "repetition", "quoted_keep", pair, left=frame["quote_context"], template=template)
            tokens = (*FILLERS, *UNSEEN_FILLERS) if split == "test" else FILLERS
            for token in tokens:
                for index, template in enumerate(frame["filler"]):
                    raw = template.format(f=token, v=verb, n=noun)
                    desired = raw.replace(token + "，", "", 1)
                    pair = "quote_filler:" + str(index) + ":" + token
                    scope = "unseen_surface_stress" if token in UNSEEN_FILLERS else "seen_surfaces_new_context"
                    add(raw, desired, "disfluency", "edit", pair, left=frame["positive_context"], lexical_scope=scope, template=template + ":" + token)
                    add(raw, raw, "disfluency", "quoted_keep", pair, left=frame["quote_context"], lexical_scope=scope, template=template + ":" + token)
            pair = "noun_link"
            raw = frame["noun_drop"].format(n=noun)
            add(raw, raw.replace(noun + noun, noun, 1), "repetition", "edit", pair, template=frame["noun_drop"])
            raw = frame["noun_keep"].format(n=noun)
            add(raw, raw, "repetition", "topic_link_keep", pair, template=frame["noun_keep"])
            for family in ("emphasis", "distributive", "demonstrative", "normal", "quotation", "answer", "numeric", "degree"):
                raw = frame[family].format(v=verb, n=noun)
                left = "这些资料都已经处理了吗？" if family == "answer" else ""
                add(raw, raw, family, "semantic_keep", family, left=left, template=frame[family])
            # Normal verb reduplication is a light/suggestive action, not
            # automatically an ASR stutter. Keep it as an explicit countercase.
            template = {"train": "不着急，先{v}{v}{n}吧。",
                        "validation": "可以先{v}{v}{n}，熟悉一下流程。",
                        "test": "先试着{v}{v}{n}吧，慢慢来。"}[split]
            raw = template.format(v=verb, n=noun)
            add(raw, raw, "normal_verb_reduplication", "semantic_keep", "normal_verb_reduplication", template=template)
    validate_partitions(result)
    return result


def prepare(args):
    from system.zh_itn_bridge import ZhITNNormalizer
    from experiments.scripts.train_edit_location_detector import sha, write, write_rows

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    partitions = build_rows(ZhITNNormalizer(args.zh_itn))
    old_train = [json.loads(line) for line in args.anchor_train.open()]
    eval_baselines = {row["baseline_text"] for name in ("validation", "test") for row in partitions[name]}
    old_kept = [row for row in old_train if row["baseline_text"] not in eval_baselines]
    if any(row["split"] != "train" or not row["training_eligible"] for row in old_kept):
        raise ValueError("Only the frozen previous train partition may anchor fitting")
    # Old data is not resplit; consumed validation/test never become fit data.
    for name, rows in partitions.items():
        write_rows(output / f"contrast_{name}.jsonl", rows)
    write_rows(output / "train.jsonl", old_kept + partitions["train"])
    write_rows(output / "validation.jsonl", partitions["validation"])
    write_rows(output / "test.jsonl", partitions["test"])
    manifest = dict(scope="authored_context_contrasts_not_natural_audio_validation",
        counts={name: dict(rows=len(rows), edits=sum(bool(row["edits"]) for row in rows),
                          keeps=sum(not row["edits"] for row in rows),
                          concept_groups=len({row["concept_group"] for row in rows}),
                          surface_strata=dict(Counter(row["lexical_scope"] for row in rows))) for name, rows in partitions.items()},
        anchor_train_rows=len(old_kept), anchor_overlap_excluded=len(old_train) - len(old_kept),
        combined_training_rows=len(old_kept) + len(partitions["train"]),
        anchor_train_sha256=sha(args.anchor_train), original_consumed_dev_test_not_fitted=True,
        quote_pairs_same_baseline_different_visible_context=True, annotation_metadata_not_model_input=True,
        train_validation_test_concept_and_baseline_disjoint=True,
        template_claim="Authored frame banks differ; grammatical primitives and seen filler vocabulary shared deliberately",
        unseen_test_fillers=list(UNSEEN_FILLERS), human_audio_gold=0,
        code_sha256=sha(Path(__file__)), output_sha256={path.name: sha(path) for path in output.glob("*.jsonl")})
    write(output / "manifest.json", manifest)
    (output / "executed_code").mkdir()
    (output / "executed_code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-train", type=Path, default=ROOT / "results/edit_detector_traceable_encoder_20261013/train.jsonl")
    parser.add_argument("--zh-itn", type=Path, default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    prepare(parser.parse_args())
