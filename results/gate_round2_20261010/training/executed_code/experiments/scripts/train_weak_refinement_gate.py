"""Train and assess an offline synthetic weak gate without touching ASR services.

The original source dev/test splits are untouched. This experiment subdivides
source train by template/recording/input components, freezes thresholds on an
internal validation set, and only then scores its internal synthetic holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.weak_refinement_gate import (INPUT_VERSION, LABELS, TASKS, WeakRefinementGate,
    build_network, collate_inputs, encode_input, input_text, make_tokenizer, select_action)
from experiments.scripts import collect_gate_pairs as collector
from experiments.scripts import synthetic_gate_data as synthetic

PARTITIONS = ("train", "validation", "test")
RATIOS = (0.6, 0.2, 0.2)
PAIR_DIRS = {"window_full": "pairs_full", "review_full": "pairs_review_full"}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def input_signature(row):
    # Exactly the strings consumed by the models; labels/provenance are excluded.
    return synthetic.digest(input_text(row["input"]))


def assign_partitions(rows, seed=20261010, attempts=12000):
    """Union template, recording and identical model-input links before splitting.

    Search uses only source labels for stratification, never model predictions,
    reference distances or policy outcomes. The assignment is frozen pre-fit.
    """
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def join(left, right):
        parent[find(right)] = find(left)

    for row in rows:
        template = "t:" + row["split_group"]
        join(template, "r:" + row["recording_group"])
        join(template, "i:" + input_signature(row))
    components = defaultdict(list)
    for row in rows:
        components[find("t:" + row["split_group"])].append(row)
    units = [components[k] for k in sorted(components)]
    if len(units) < 6:
        raise ValueError("Too few independent groups for a three-way prototype split")
    totals = Counter(r["label"] for r in rows if r["eligible_for_weak_pretraining"])
    families = sorted({r["split_group"].split(":")[0] for r in rows})
    families_total = Counter(r["split_group"].split(":")[0] for r in rows)
    vectors = []
    for unit in units:
        vectors.append(dict(size=len(unit), labels=Counter(r["label"] for r in unit if r["eligible_for_weak_pretraining"]),
                            families=Counter(r["split_group"].split(":")[0] for r in unit),
                            tasks=Counter(r["input"]["task_kind"] for r in unit)))
    n_train = round(len(units) * RATIOS[0])
    n_val = max(1, round(len(units) * RATIOS[1]))
    rng, best = random.Random(seed), None
    for _ in range(attempts):
        order = list(range(len(units)))
        rng.shuffle(order)
        cuts = [order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]]
        score, missing = 0.0, 0
        for indices, ratio in zip(cuts, RATIOS):
            labels, family_counts, tasks = Counter(), Counter(), Counter()
            count = 0
            for index in indices:
                v = vectors[index]
                count += v["size"]
                labels.update(v["labels"])
                family_counts.update(v["families"])
                tasks.update(v["tasks"])
            missing += sum(not labels[k] for k in totals) + sum(not tasks[t] for t in TASKS)
            missing += sum(not family_counts[f] for f in families)
            score += 3 * (count / len(rows) - ratio) ** 2
            score += sum((labels[k] / totals[k] - ratio) ** 2 for k in totals)
            score += sum((family_counts[f] / families_total[f] - ratio) ** 2 for f in families)
        score += 100 * missing
        if best is None or score < best[0]:
            best = score, cuts, missing
    if best[2]:
        raise ValueError("Cannot obtain class/task/family coverage without splitting independent groups")
    assigned = {}
    for partition, indices in zip(PARTITIONS, best[1]):
        for index in indices:
            for row in units[index]:
                assigned[row["case_id"]] = partition
    for key in ("split_group", "recording_group"):
        seen = defaultdict(set)
        for row in rows:
            seen[row[key]].add(assigned[row["case_id"]])
        assert all(len(v) == 1 for v in seen.values())
    seen = defaultdict(set)
    for row in rows:
        seen[input_signature(row)].add(assigned[row["case_id"]])
    assert all(len(v) == 1 for v in seen.values())
    return assigned, dict(seed=seed, ratios=RATIOS, independent_components=len(units),
                          attempts=attempts, stratification_objective=best[0],
                          constraints=["template", "recording", "identical_model_input"],
                          selection_uses="labels_and_group_sizes_only_before_training")


def prepare(source, output, seed):
    from system.zh_itn_bridge import ZhITNNormalizer
    labels_path = source / "weak_gate_labels_train_all.jsonl"
    rows = synthetic.read_rows(labels_path)
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate weak-label cases")
    if any(r["split"] != "train" or r["human_gold"] or r["audio_script_verified"] for r in rows):
        raise ValueError("This prototype accepts only unverified synthetic source-train rows")
    seeds = {r["seed_id"]: r for r in synthetic.seeds(source)}
    sources, outcomes, normalized_cache = {}, {}, {}
    for key, directory in PAIR_DIRS.items():
        path = source / directory
        collector.verify_frozen(path)
        sources[key] = dict(path=path, cases={r["case_id"]: r for r in synthetic.read_rows(path / "cases.jsonl")},
                            pairs={r["case_id"]: r for r in synthetic.read_rows(path / "pairs.jsonl")},
                            normalizer=ZhITNNormalizer(path / "native/libzh_itn_bridge.so"))
    for row in rows:
        src = sources[row["pair_set"]]
        case, pair = src["cases"][row["case_id"]], src["pairs"][row["case_id"]]
        seed_row = seeds[row["seed_id"]]
        if case["synthetic_metadata"]["seed_sha256"] != seed_row["seed_sha256"]:
            raise ValueError("Seed provenance mismatch")
        texts = {arm: pair["arms"][arm]["comparison_text"] for arm in ("rules", "current", "always")}
        label, _ = synthetic.weak_label(texts["rules"], texts["always"], seed_row["intended_text"],
                                       lambda text: src["normalizer"].normalize(text).text)
        if label != row["label"]:
            raise ValueError("Weak label cannot be reproduced: " + row["case_id"])
        def normalized(text):
            key = (row["pair_set"], text)
            if key not in normalized_cache:
                normalized_cache[key] = synthetic.lexical(src["normalizer"].normalize(text).text)
            return normalized_cache[key]
        target, spoken = normalized(seed_row["intended_text"]), normalized(seed_row["spoken_text"])
        arms = {}
        for arm, text in texts.items():
            a = pair["arms"][arm]
            calls = a["logical_calls"] + a["plausibility_calls"]
            if any(c.get("error_type") for c in calls) or a["session_stats"].get("failed_call_count"):
                raise ValueError("Failed refinement is not valid supervision")
            normalized_text = normalized(text)
            realized, _ = synthetic.weak_label(texts["rules"], text, seed_row["intended_text"],
                                             lambda x: src["normalizer"].normalize(x).text)
            arms[arm] = dict(intended_distance=collector.edit_distance(target, normalized_text),
                             spoken_distance=collector.edit_distance(spoken, normalized_text),
                             exact_intended=normalized_text == target, effect=realized,
                             logical_refiner_calls=len(a["logical_calls"]),
                             logical_plausibility_calls=len(a["plausibility_calls"]),
                             operation_cost_ms=sum(c["wall_ms"] for c in calls),
                             executed=bool(calls), changed=text != texts["rules"])
        outcomes[row["case_id"]] = dict(case_id=row["case_id"], intended_chars=len(target), spoken_chars=len(spoken),
                                        arms=arms, reference_status="unverified_synthesis_script")
    assigned, split_info = assign_partitions(rows, seed)
    for row in rows:
        row["partition"] = assigned[row["case_id"]]
    for partition in PARTITIONS:
        write_rows(output / f"{partition}.jsonl", [r for r in rows if r["partition"] == partition])
    write_rows(output / "paired_outcomes.jsonl", list(outcomes.values()))
    counts = {}
    for partition in PARTITIONS:
        subset = [r for r in rows if r["partition"] == partition]
        counts[partition] = dict(cases=len(subset), eligible=sum(r["eligible_for_weak_pretraining"] for r in subset),
                                labels=dict(Counter(r["label"] for r in subset)),
                                by_task={t: dict(Counter(r["label"] for r in subset if r["input"]["task_kind"] == t)) for t in TASKS},
                                template_groups=sorted({r["split_group"] for r in subset}),
                                recordings=len({r["recording_group"] for r in subset}))
    manifest = dict(scope="synthetic_weak_exploratory_group_holdout_not_real_risk_calibration", human_gold=0,
                    source=str(source), labels_sha256=synthetic.file_hash(labels_path), input_version=INPUT_VERSION,
                    source_split_used="train_only", source_dev_test_touched=False, split=split_info, counts=counts,
                    pair_hashes={k: synthetic.file_hash(v["path"] / "pairs.jsonl") for k, v in sources.items()},
                    split_file_hashes={p: synthetic.file_hash(output / f"{p}.jsonl") for p in PARTITIONS})
    write_json(output / "dataset_manifest.json", manifest)
    print("SPLIT " + json.dumps(counts, ensure_ascii=False), flush=True)
    return rows, outcomes, manifest


def choose_thresholds(rows, probabilities):
    """Explore only internal validation; zero observed weak harm is not a guarantee."""
    thresholds, evidence = {}, {}
    grid_b = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    grid_h = [0.02, 0.05, 0.1, 0.2, 0.4]
    for task in TASKS:
        subset = [(r, p) for r, p in zip(rows, probabilities) if r["input"]["task_kind"] == task]
        best = None
        for benefit in grid_b:
            for harm in grid_h:
                limit = dict(benefit=benefit, harm=harm)
                selected = [r for r, p in subset if select_action(p, limit) == "refine"]
                n_b = sum(r["label"] == "benefit" for r in selected)
                n_h = sum(r["label"] == "harm" for r in selected)
                if n_h:
                    continue
                utility = n_b - 0.05 * len(selected)
                rank = (utility, -len(selected), benefit, -harm)
                if best is None or rank > best[0]:
                    best = rank, limit, dict(selected=len(selected), captured_benefit=n_b, selected_harm=n_h,
                                             utility=utility, validation_harm_cases=sum(r["label"] == "harm" for r, _ in subset))
        thresholds[task], evidence[task] = best[1], best[2]
    return thresholds, dict(by_task=evidence, utility=sum(v["utility"] for v in evidence.values()),
                           grid=dict(benefit=grid_b, harm=grid_h),
                           criterion="max_benefit_minus_0.05_calls_with_zero_observed_validation_weak_harm",
                           calibrated=False, real_risk_guarantee=False)


def classification(rows, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    pairs = [(LABELS.index(r["label"]), p) for r, p in zip(rows, probabilities) if r["eligible_for_weak_pretraining"]]
    truth = [a for a, _ in pairs]
    predicted = [int(np.argmax(p)) for _, p in pairs]
    return dict(cases=len(truth), labels=list(LABELS), accuracy=float(accuracy_score(truth, predicted)),
                macro_f1=float(f1_score(truth, predicted, labels=list(range(3)), average="macro", zero_division=0)),
                confusion_matrix=confusion_matrix(truth, predicted, labels=list(range(3))).tolist())


def model_probabilities(model, tokenizer, rows, device, max_tokens, batch_size=32):
    import torch
    results = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            values = [encode_input(tokenizer, r["input"], max_tokens) for r in rows[start:start + batch_size]]
            batch = {k: v.to(device) for k, v in collate_inputs(tokenizer, values).items()}
            results.extend(model(**batch).softmax(-1).cpu().tolist())
    return results


def fit_encoder(train, validation, output, args, weighted):
    import numpy as np
    import torch
    variant = "encoder_weighted" if weighted else "encoder_unweighted"
    directory = output / variant
    directory.mkdir()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = make_tokenizer(args.encoder)
    model = build_network(args.encoder).to(args.device)
    encoded = [encode_input(tokenizer, r["input"], args.max_tokens) for r in train]
    labels = torch.tensor([LABELS.index(r["label"]) for r in train], dtype=torch.long)
    counts = Counter(r["label"] for r in train)
    weights = torch.tensor([(len(train) / (3 * counts[k])) ** 0.5 for k in LABELS], device=args.device)
    weights /= weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=weights if weighted else None)
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=2e-5),
        dict(params=list(model.head.parameters()) + list(model.task_embedding.parameters()), lr=5e-4)], weight_decay=0.01)
    rng, history, best = random.Random(args.seed), [], None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train))); rng.shuffle(order)
        model.train(); total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = {k: v.to(args.device) for k, v in collate_inputs(tokenizer, [encoded[i] for i in indices]).items()}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**batch), labels[indices].to(args.device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        probs = model_probabilities(model, tokenizer, validation, args.device, args.max_tokens)
        val_loss = sum(-np.log(max(p[LABELS.index(r["label"])], 1e-12)) for r, p in zip(validation, probs)) / len(validation)
        record = dict(epoch=epoch, train_loss=total_loss / len(train), validation_unweighted_loss=float(val_loss))
        history.append(record)
        if best is None or val_loss < best[0]:
            best = val_loss, epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / "gate.pt")
        print(variant + " " + json.dumps(record), flush=True)
    model.load_state_dict(torch.load(directory / "gate.pt", map_location=args.device, weights_only=True))
    validation_probs = model_probabilities(model, tokenizer, validation, args.device, args.max_tokens)
    (directory / "encoder").mkdir()
    model.encoder.config.save_pretrained(directory / "encoder")
    tokenizer.save_pretrained(directory / "encoder")
    info = dict(input_version=INPUT_VERSION, labels=list(LABELS), tasks=list(TASKS), max_tokens=args.max_tokens,
                variant=variant, parameters=sum(p.numel() for p in model.parameters()), best_epoch=best[1],
                validation_loss=float(best[0]), history=history, device=args.device, torch=torch.__version__,
                encoder_source=str(args.encoder), train_class_weights=weights.cpu().tolist() if weighted else None,
                encoder_lr=2e-5, head_lr=5e-4, batch_size=args.batch_size, train_seconds=time.perf_counter() - started,
                calibrated=False, human_gold=0, scope="synthetic_weak_prototype_offline_only", online_enabled=False)
    write_json(directory / "training.json", info)
    del model, optimizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return variant, validation_probs, info


def fit_linear(train, validation, output):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    def text(row):
        first, target, context = input_text(row["input"])
        return first + context + "[SEP]" + target
    pipeline = Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                                                   max_features=10000, sublinear_tf=True)),
                         ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=20261010))])
    pipeline.fit([text(r) for r in train], [r["label"] for r in train])
    path = output / "linear_tfidf"; path.mkdir()
    joblib.dump(pipeline, path / "model.joblib")
    classes = list(pipeline.named_steps["classifier"].classes_)
    values = pipeline.predict_proba([text(r) for r in validation])[:, [classes.index(k) for k in LABELS]].tolist()
    write_json(path / "training.json", dict(variant="linear_tfidf", labels=LABELS, input_version=INPUT_VERSION,
                                            class_weight=None, feature_fit_partition="train_only", calibrated=False))
    return pipeline, values, text


def policy_metrics(rows, outcomes, arm_selector, overhead_ms=0.0):
    chosen = [(r, outcomes[r["case_id"]], arm_selector(r)) for r in rows]
    arms = [o["arms"][a] for _, o, a in chosen]
    effects = Counter(a["effect"] for a in arms)
    available = Counter(r["label"] for r in rows)
    selected = sum(a["executed"] for a in arms)
    weak_harm = effects["harm"]
    return dict(cases=len(rows), weak_eligible_cases=sum(r["eligible_for_weak_pretraining"] for r in rows),
                selected_decisions=selected, selected_fraction=selected / len(rows),
                available_benefit=available["benefit"], captured_benefit=effects["benefit"],
                benefit_recall=effects["benefit"] / max(1, available["benefit"]),
                weak_harm=weak_harm, weak_harm_fraction_all_cases=weak_harm / len(rows),
                weak_harm_fraction_selected=weak_harm / max(1, selected),
                selected_uncertain=sum(r["label"] == "uncertain" and o["arms"][a]["executed"] for r, o, a in chosen),
                exact_intended=sum(a["exact_intended"] for a in arms),
                exact_intended_fraction=sum(a["exact_intended"] for a in arms) / len(rows),
                intended_micro_cer=sum(a["intended_distance"] for a in arms) / max(1, sum(o["intended_chars"] for _, o, _ in chosen)),
                spoken_micro_cer=sum(a["spoken_distance"] for a in arms) / max(1, sum(o["spoken_chars"] for _, o, _ in chosen)),
                improved_intended=sum(o["arms"][a]["intended_distance"] < o["arms"]["rules"]["intended_distance"] for _, o, a in chosen),
                degraded_intended=sum(o["arms"][a]["intended_distance"] > o["arms"]["rules"]["intended_distance"] for _, o, a in chosen),
                logical_refiner_calls=sum(a["logical_refiner_calls"] for a in arms),
                logical_plausibility_calls=sum(a["logical_plausibility_calls"] for a in arms),
                recorded_operation_cost_ms=sum(a["operation_cost_ms"] for a in arms),
                gate_cpu_estimated_ms=overhead_ms,
                estimated_serial_cost_ms=sum(a["operation_cost_ms"] for a in arms) + overhead_ms)


def benchmark(score, rows):
    import numpy as np
    for _ in range(5):
        score(rows[0])
    samples = []
    for row in rows:
        started = time.perf_counter(); score(row)
        samples.append(dict(task=row["input"]["task_kind"], ms=(time.perf_counter() - started) * 1000))
    def stats(values):
        return dict(n=len(values), mean_ms=float(np.mean(values)), p50_ms=float(np.percentile(values, 50)),
                    p95_ms=float(np.percentile(values, 95)))
    return dict(overall=stats([s["ms"] for s in samples]),
                by_task={t: stats([s["ms"] for s in samples if s["task"] == t]) for t in TASKS},
                scope="CPU_2_threads_single_request_includes_tokenization_no_cache_no_ASR_no_Refiner")


def assess_adoption(report):
    primary = report["primary_model"]
    learned = report["test"]["policies"][primary]
    current = report["test"]["policies"]["current"]
    matrix = report["test"]["classification"][primary]["confusion_matrix"]
    harm_cases = sum(matrix[2])
    return dict(status="research_only_not_ready_to_replace_current_gate",
                missed_available_benefit=learned["available_benefit"] - learned["captured_benefit"],
                logical_call_reduction_vs_current=1 - learned["logical_refiner_calls"] / max(1, current["logical_refiner_calls"]),
                exact_intended_fraction_change_vs_current=learned["exact_intended_fraction"] - current["exact_intended_fraction"],
                harm_class_test_cases=harm_cases, harm_class_recall=matrix[2][2] / max(1, harm_cases),
                test_policy_retuned=False, online_enabled=False,
                reasons=["Lower intended-script recovery than current gate despite fewer calls",
                         "Very few heldout harm examples; zero selected harm does not demonstrate harm classification",
                         "No verified audio gold or full sequential session replay"])


def write_report(output, report):
    counts, test = report["dataset"]["counts"], report["test"]
    lines = ["# 合成弱监督门控原型实测", "", "本试验只使用已有合成 train；原始 dev/test 未触碰。内部测试按模板、录音和相同模型输入隔离。所有质量数字基于未核验 TTS 脚本，不代表真实语音验收。", "",
             "## 固定数据划分", "", "| 内部集合 | 案例 | 可训练标签 | benefit | neutral | harm | uncertain |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in PARTITIONS:
        c = counts[name]; labels = c["labels"]
        lines.append(f"| {name} | {c['cases']} | {c['eligible']} | {labels.get('benefit',0)} | {labels.get('neutral',0)} | {labels.get('harm',0)} | {labels.get('uncertain',0)} |")
    lines += ["", "训练时排除 uncertain；它们仍保留在各组的政策评估分母。类别加权只使用训练标签计数。模型输入只包括处理前文本、任务类型、领域和已有候选，排除目标脚本、模板编号、标签和本次精修输出。", "",
              "## 清理目标口径重算", "", "下表是全部 833 个已有源训练案例的描述统计，不是模型的独立测试结果。目标为 intended_text，转录忠实度 spoken_text 单独保留。", "",
              "| 原有政策 | 目标精确案例 | exact benefit | exact harm | 目标距离改善 | 目标距离退化 |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["all_source_train_references"].items():
        lines.append(f"| {name} | {m['exact_intended']} | {m['captured_benefit']} | {m['weak_harm']} | {m['improved_intended']} | {m['degraded_intended']} |")
    lines += ["", "当前门控和始终精修都存在清理收益，也有退化。先前针对朗读脚本的零改善不等于清理任务零改善；两种目标不能混用。", "",
              "## 模型与选择", "", f"主原型：`{report['primary_model']}`。编码器采用本地 4 层、256 维中文 BERT 初始化，GPU 训练；共享三分类头结合 CLS、候选跨度表示和任务嵌入。另训练未加权字符 TF-IDF 线性基线。", "",
              "编码器的未加权/加权版本预先声明，各自按验证损失选 epoch，再根据验证政策收益选择主版本。各任务阈值仅在内部 validation 上搜索，要求该集合观察到的弱 harm 为零；这不是独立风险校准或零风险保证。主模型与所有阈值在读取内部 test 预测之前写入 selection.json。", "",
              "## 内部测试政策比较", "", "| 政策 | 选择操作数 | 捕获 benefit | weak harm | 精修逻辑调用 | 目标精确率 | 目标 CER | 估算串行成本 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in test["policies"].items():
        lines.append(f"| {name} | {m['selected_decisions']} | {m['captured_benefit']} | {m['weak_harm']} | {m['logical_refiner_calls']} | {m['exact_intended_fraction']:.3f} | {m['intended_micro_cer']:.4f} | {m['estimated_serial_cost_ms']:.1f} |")
    lines += ["", "目标精确率与目标 CER 使用清理后的 intended_text；spoken_text 的转录忠实度另存在 report.json。政策模拟针对独立配对决策：不同窗口/候选的修改没有串成完整会话，不能称为端到端转录改善。", "",
              "成本为此前冻结实验记录的逻辑操作耗时加本次 CPU 门控平均耗时估算；存在 GPU 共享、跨实验计时及并行差异，不是在线墙钟延迟。", "",
              "## CPU 单请求耗时", "", "| 模型 | p50 ms | p95 ms | 均值 ms |", "|---|---:|---:|---:|"]
    for name, latency in report["latency"].items():
        s = latency["overall"]
        lines.append(f"| {name} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['mean_ms']:.3f} |")
    a = report["adoption_assessment"]
    lines += ["", "## 是否替换现有门控", "", "结论：保留现有门控，训练产物用于离线研究。",
              f"主原型的逻辑调用减少 {a['logical_call_reduction_vs_current']:.1%}，但漏掉 {a['missed_available_benefit']} 次可用弱 benefit，目标精确率相对当前门控变化 {a['exact_intended_fraction_change_vs_current'] * 100:.2f} 个百分点。",
              f"内部测试只有 {a['harm_class_test_cases']} 条 harm，主分类器的 harm 类召回率为 {a['harm_class_recall']:.1%}。零次选中 weak harm 不能据此解释为学会了可靠退化识别。", "",
              "下一步应增加独立收益/退化模板与可靠音频标注，保留本次划分和阈值，不再根据已读测试结果调整阈值或反复加训。各模型最佳 epoch 和训练/验证损失轨迹见对应目录的 training.json。"]
    lines += ["", "## 产物与限制", "", "train.jsonl / validation.jsonl / test.jsonl 为固定分组；paired_outcomes.jsonl 为独立评估数据；selection.json 记录验证选择；模型目录中 metadata.json 记录阈值，gate.pt 为完整权重，encoder/ 包含 config/tokenizer。system.weak_refinement_gate.WeakRefinementGate 提供显式离线加载接口，未接入 Web 服务。", "",
              "35 条 harm 集中在 3 个模板组，无法充分证明跨模板退化识别。请查看 dataset_manifest.json 的逐任务类别分布，某些任务内部集合没有 harm。预测概率未校准，未使用 ASR 置信度；所有音频与收益标签保持 unverified / synthetic_weak，人工金标为零。正式启用前需要可靠真实语音验证与完整会话回放。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/qwen_tts_asr_gpu_pilot_20261010")
    parser.add_argument("--output", type=Path, default=ROOT / "results/learned_gate_weak_prototype_20261010")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/entity_ranker/encoder")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if args.epochs < 1 or args.batch_size < 1 or not 32 <= args.max_tokens <= 512:
        parser.error("Invalid epochs, batch size or token budget")
    args.source, args.output, args.encoder = args.source.resolve(), args.output.resolve(), args.encoder.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Never overwrite an experiment; choose a fresh output directory")
    if args.device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(torch.device(args.device))
        if free < 1024 ** 3:
            raise RuntimeError("Less than 1 GiB GPU memory free; leave existing services untouched")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "configuration.json", dict(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
               device=args.device, encoder=str(args.encoder), encoder_sha256=synthetic.file_hash(args.encoder / "pytorch_model.bin"),
               script_sha256=synthetic.file_hash(Path(__file__)), input_module_sha256=synthetic.file_hash(ROOT / "system/weak_refinement_gate.py"),
               max_tokens=args.max_tokens, online_enabled=False, models_declared_before_test=["linear_tfidf", "encoder_unweighted", "encoder_weighted"]))
    rows, outcomes, manifest = prepare(args.source, args.output, args.seed)
    partitioned = {p: [r for r in rows if r["partition"] == p] for p in PARTITIONS}
    train = [r for r in partitioned["train"] if r["eligible_for_weak_pretraining"]]
    validation = partitioned["validation"]
    validation_eligible = [r for r in validation if r["eligible_for_weak_pretraining"]]
    pipeline, linear_val, linear_text = fit_linear(train, validation, args.output)
    thresholds, evidence = choose_thresholds(validation, linear_val)
    variants = {"linear_tfidf": dict(thresholds=thresholds, validation_policy=evidence,
                                      validation_classification=classification(validation, linear_val))}
    for weighted in (False, True):
        name, eligible_probs, info = fit_encoder(train, validation_eligible, args.output, args, weighted)
        # Explicit reload verifies saved checkpoint/config/tokenizer portability.
        metadata = dict(**info, thresholds={t: dict(benefit=1.01, harm=0) for t in TASKS})
        write_json(args.output / name / "metadata.json", metadata)
        gate = WeakRefinementGate(args.output / name, device="cpu")
        probs = model_probabilities(gate.model, gate.tokenizer, validation, "cpu", args.max_tokens)
        reloaded = [p for r, p in zip(validation, probs) if r["eligible_for_weak_pretraining"]]
        if not np.allclose(eligible_probs, reloaded, atol=1e-4):
            raise RuntimeError("GPU training checkpoint and CPU inference reload disagree")
        thresholds, evidence = choose_thresholds(validation, probs)
        metadata["thresholds"] = thresholds
        write_json(args.output / name / "metadata.json", metadata)
        variants[name] = dict(thresholds=thresholds, validation_policy=evidence,
                              validation_classification=classification(validation, probs), validation_loss=info["validation_loss"])
        del gate
    encoder_names = ("encoder_unweighted", "encoder_weighted")
    primary = max(encoder_names, key=lambda n: (variants[n]["validation_policy"]["utility"], -variants[n]["validation_loss"]))
    selection = dict(primary_model=primary, variants=variants, selection_partition="validation_only",
                     event="policy_frozen_before_test_scoring",
                     test_predictions_generated_at_selection=False, real_risk_calibrated=False)
    write_json(args.output / "selection.json", selection)
    print("FROZEN_SELECTION " + json.dumps(selection, ensure_ascii=False), flush=True)
    test_rows = partitioned["test"]
    test_policies = {a: policy_metrics(test_rows, outcomes, lambda r, a=a: a) for a in ("rules", "current", "always")}
    overall_references = {a: policy_metrics(rows, outcomes, lambda r, a=a: a) for a in ("rules", "current", "always")}
    test_classification, latencies, predictions = {}, {}, []
    for name in ("linear_tfidf", *encoder_names):
        if name == "linear_tfidf":
            classes = list(pipeline.named_steps["classifier"].classes_)
            order = [classes.index(k) for k in LABELS]
            probs = pipeline.predict_proba([linear_text(r) for r in test_rows])[:, order].tolist()
            score = lambda r: pipeline.predict_proba([linear_text(r)])
        else:
            gate = WeakRefinementGate(args.output / name, device="cpu")
            probs = model_probabilities(gate.model, gate.tokenizer, test_rows, "cpu", args.max_tokens)
            score = lambda r: gate.decide(r["input"])
        latency = benchmark(score, test_rows)
        latencies[name] = latency
        actions = {r["case_id"]: select_action(p, variants[name]["thresholds"][r["input"]["task_kind"]]) for r, p in zip(test_rows, probs)}
        selector = lambda r: "always" if actions[r["case_id"]] == "refine" else "rules"
        overhead = sum(latency["by_task"][r["input"]["task_kind"]]["mean_ms"] for r in test_rows)
        metric = policy_metrics(test_rows, outcomes, selector, overhead)
        metric["by_task"] = {t: policy_metrics([r for r in test_rows if r["input"]["task_kind"] == t], outcomes, selector,
                                    latency["by_task"][t]["mean_ms"] * sum(r["input"]["task_kind"] == t for r in test_rows)) for t in TASKS}
        test_policies[name] = metric
        test_classification[name] = classification(test_rows, probs)
        predictions.extend(dict(case_id=r["case_id"], model=name, probabilities=dict(zip(LABELS, p)),
                                action=actions[r["case_id"]], weak_label=r["label"], partition="test") for r, p in zip(test_rows, probs))
        if name != "linear_tfidf":
            del gate
    write_rows(args.output / "test_predictions.jsonl", predictions)
    report = dict(primary_model=primary, dataset=manifest, validation_selection=selection,
                  all_source_train_references=overall_references,
                  test=dict(policies=test_policies, classification=test_classification), latency=latencies,
                  limits=["Synthetic unverified scripts; no human gold or real risk calibration",
                          "Only three template groups contain weak harm",
                          "Independent decision replay, not sequential full-session replay",
                          "Estimated serial operation cost, not end-to-end streaming latency",
                          "Internal source-train holdout, original dev/test untouched"], online_enabled=False)
    report["adoption_assessment"] = assess_adoption(report)
    write_json(args.output / "report.json", report)
    write_report(args.output, report)
    write_json(args.output / "completed.json", dict(report_sha256=synthetic.file_hash(args.output / "report.json"),
                selection_sha256=synthetic.file_hash(args.output / "selection.json"),
                model_hashes={n: synthetic.file_hash(args.output / n / "gate.pt") for n in encoder_names}))
    print("COMPLETE " + json.dumps(dict(primary_model=primary, test=test_policies), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
