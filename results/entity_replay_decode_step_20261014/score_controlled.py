#!/usr/bin/env python3
"""Offline score for same-source old/new entity replay decode-step control."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUTS = OUT / "controlled_regression_inputs.jsonl"
RESULTS = OUT / "controlled_results.jsonl"
PROTOCOL = OUT / "controlled_protocol.json"
WORKER = OUT / "worker.json"
REFERENCES = ROOT / "results/entity_modes_dataset_20261013/references.json"
ARMS = ("old", "new")
sys.path.insert(0, str(ROOT / ".runtime/functional_test_deps"))
sys.path.insert(0, str(OUT / "source_snapshot"))
import opencc  # noqa: E402
from system.entity_audio_repair import normalized  # noqa: E402
from system.zh_itn_bridge import ZhITNNormalizer  # noqa: E402

cc = opencc.OpenCC("t2s")
itn = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")


def norm(text):
    return normalized(cc.convert(itn.normalize(str(text)).text))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}

    def percentile(q):
        point = (len(values) - 1) * q
        lo, hi = int(point), min(int(point) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (point - lo)

    return {"n": len(values), "mean": sum(values) / len(values),
            "median": percentile(.5), "p90": percentile(.9)}


def read_jsonl_complete(path: Path, partial: bool):
    raw = path.read_bytes() if path.exists() else b""
    rows = []
    skipped_incomplete_tail = False
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith((b"\n", b"\r")):
            if partial and index == len(lines) - 1:
                skipped_incomplete_tail = True
                break
            raise ValueError(f"incomplete JSONL row at end of {path.name}; use --partial")
        rows.append(json.loads(line))
    return rows, skipped_incomplete_tail, raw


class JsonCharReader:
    """One-character buffered UTF-8 reader for selective JSON field decoding."""
    def __init__(self, stream):
        self.stream = stream
        self.buffer = None

    def read(self):
        if self.buffer is not None:
            value, self.buffer = self.buffer, None
            return value
        return self.stream.read(1)

    def unread(self, value):
        if value:
            if self.buffer is not None:
                raise RuntimeError("JSON reader supports one character of lookahead")
            self.buffer = value

    def nonspace(self):
        value = self.read()
        while value and value.isspace():
            value = self.read()
        return value


def read_json_string(reader):
    if reader.nonspace() != '"':
        raise ValueError("expected JSON string key")
    chars = ['"']
    escaped = False
    while True:
        value = reader.read()
        if not value:
            raise ValueError("unterminated JSON string")
        chars.append(value)
        if escaped:
            escaped = False
        elif value == "\\":
            escaped = True
        elif value == '"':
            break
    return json.loads("".join(chars))


def consume_json_value(reader, first, capture=False):
    """Consume one JSON value, optionally returning its source text.

    Unselected reference values are skipped lexically, so their label objects
    are never decoded into Python objects.
    """
    if not first:
        raise ValueError("expected JSON value")
    output = [first] if capture else None
    if first == '"':
        escaped = False
        while True:
            value = reader.read()
            if not value:
                raise ValueError("unterminated JSON string value")
            if capture:
                output.append(value)
            if escaped:
                escaped = False
            elif value == "\\":
                escaped = True
            elif value == '"':
                break
        return "".join(output) if capture else None
    if first in "{[":
        stack = ["}" if first == "{" else "]"]
        in_string = False
        escaped = False
        while stack:
            value = reader.read()
            if not value:
                raise ValueError("unterminated JSON object/array")
            if capture:
                output.append(value)
            if in_string:
                if escaped:
                    escaped = False
                elif value == "\\":
                    escaped = True
                elif value == '"':
                    in_string = False
                continue
            if value == '"':
                in_string = True
            elif value == "{":
                stack.append("}")
            elif value == "[":
                stack.append("]")
            elif value in "}]":
                if not stack or value != stack[-1]:
                    raise ValueError("mismatched JSON object/array delimiter")
                stack.pop()
        return "".join(output) if capture else None
    while True:
        value = reader.read()
        if not value:
            break
        if value.isspace() or value in ",}]":
            reader.unread(value)
            break
        if capture:
            output.append(value)
    return "".join(output) if capture else None


def load_selected_reference_samples(path: Path, wanted_ids):
    """Decode only requested entries from references.json; skip other labels."""
    wanted = set(wanted_ids)
    selected = {}
    found_samples = False
    with path.open(encoding="utf-8") as stream:
        reader = JsonCharReader(stream)
        if reader.nonspace() != "{":
            raise ValueError("references.json must have a top-level object")
        while True:
            value = reader.nonspace()
            if value == "}":
                break
            if value == ",":
                value = reader.nonspace()
            reader.unread(value)
            key = read_json_string(reader)
            if reader.nonspace() != ":":
                raise ValueError("expected colon after top-level references key")
            first = reader.nonspace()
            if key != "samples":
                consume_json_value(reader, first, capture=False)
                continue
            found_samples = True
            if first != "{":
                raise ValueError("references.json samples must be an object")
            while True:
                token = reader.nonspace()
                if token == "}":
                    break
                if token == ",":
                    token = reader.nonspace()
                reader.unread(token)
                sample_id = read_json_string(reader)
                if reader.nonspace() != ":":
                    raise ValueError(f"expected colon after reference sample ID {sample_id}")
                first_value = reader.nonspace()
                raw_value = consume_json_value(reader, first_value, capture=sample_id in wanted)
                if sample_id in wanted:
                    selected[sample_id] = json.loads(raw_value)
            break
    if not found_samples:
        raise ValueError("references.json has no samples object")
    return selected


def validate_protocol(inputs, input_bytes):
    protocol = json.loads(PROTOCOL.read_text())
    worker = json.loads(WORKER.read_text())
    if protocol.get("input_sha256") != sha_bytes(input_bytes):
        raise ValueError("controlled input SHA-256 does not match protocol")
    if protocol.get("sample_order") != [row["id"] for row in inputs]:
        raise ValueError("protocol sample_order does not match controlled input order")
    candidate_path = OUT / "candidate.py"
    candidate_sha = sha_file(candidate_path)
    if candidate_sha != protocol.get("candidate_sha256"):
        raise ValueError("protocol candidate_sha256 does not match candidate.py")
    source_hashes = protocol.get("source_hashes") or {}
    if not source_hashes or source_hashes != worker.get("source_hashes"):
        raise ValueError("protocol source_hashes do not match worker.json")
    for relative, expected in source_hashes.items():
        path = OUT / "source_snapshot" / relative
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f"source snapshot SHA-256 mismatch: {relative}")
    settings = protocol.get("settings") or {}
    expected_steps = {"old": 2.0, "new": 10.0}
    if float(settings.get("old_review_seconds", -1)) != expected_steps["old"]:
        raise ValueError("protocol old review decode step is not 2 seconds")
    if float(settings.get("new_review_seconds", -1)) != expected_steps["new"]:
        raise ValueError("protocol new review decode step is not 10 seconds")
    return protocol, {
        "controlled_input_sha256": sha_bytes(input_bytes),
        "candidate_sha256": candidate_sha,
        "source_hash_count": len(source_hashes),
        "source_hashes": source_hashes,
        "protocol_sha256": sha_file(PROTOCOL),
    }


def validate_inputs(inputs, protocol):
    if len(inputs) != 206 or len({row["id"] for row in inputs}) != 206:
        raise ValueError(f"expected 206 unique controlled input IDs, found {len(inputs)}")
    old_changed = [row for row in inputs if int(row["old_change_count"]) > 0]
    controls = [row for row in inputs if int(row["old_change_count"]) == 0]
    if len(old_changed) != 174 or len(controls) != 32:
        raise ValueError(f"expected 174 old_change_count>0 and 32 old_change_count=0; got {len(old_changed)} and {len(controls)}")
    if sum(int(row["old_change_count"]) for row in old_changed) != 180:
        raise ValueError("historical old_change_count sum does not equal 180")
    seen_pcm = {}
    for row in inputs:
        sid = row["id"]
        if int(row["old_change_count"]) < 0:
            raise ValueError(f"negative old_change_count for {sid}")
        text_hash = sha_bytes(str(row["source_text"]).encode("utf-8"))
        if text_hash != row.get("source_text_sha256"):
            raise ValueError(f"source_text SHA-256 mismatch for {sid}")
        pcm_path = Path(row["pcm_path"])
        if not pcm_path.is_absolute():
            pcm_path = ROOT / pcm_path
        pcm_path = pcm_path.resolve()
        if not pcm_path.is_file():
            raise FileNotFoundError(pcm_path)
        digest = seen_pcm.setdefault(str(pcm_path), sha_file(pcm_path))
        if digest != row["pcm_sha256"]:
            raise ValueError(f"PCM SHA-256 mismatch for {sid}")
    output = protocol.get("input_sha256")
    return {
        "input_count": len(inputs),
        "historically_changed_samples": len(old_changed),
        "historical_change_count_sum": sum(int(row["old_change_count"]) for row in old_changed),
        "old_change_count_zero_controls": len(controls),
        "negative_cohort_with_old_change_count_gt0": sum(
            str(row.get("cohort", "")).casefold() == "negative" and int(row["old_change_count"]) > 0
            for row in inputs),
        "input_file_sha256": output,
        "unique_pcm_files_hashed": len(seen_pcm),
    }


def build_labels(inputs, references):
    labels = {}
    missing = []
    for row in inputs:
        sid = row["id"]
        label = references.get(sid)
        if label is None:
            missing.append(sid)
            continue
        reference = norm(label["reference"])
        gold = Counter(norm(entity["canonical"]) for entity in label.get("gold_entities", []))
        if not reference or any(not word for word in gold):
            raise ValueError(f"invalid fixed reference/gold canonical for {sid}")
        if any(reference.count(word) < count for word, count in gold.items()):
            raise ValueError(f"gold mention Counter exceeds normalized reference for {sid}")
        labels[sid] = {"reference": reference, "gold": gold,
                       "expected_mentions": sum(gold.values()),
                       "reference_characters": len(reference)}
    if missing:
        raise ValueError(f"fixed reference file lacks controlled IDs: {missing[:8]}")
    return labels


def actual_replay_steps(row):
    steps = []
    for replay in row.get("replay_starts") or []:
        value = replay.get("chunk_seconds")
        if value is not None:
            steps.append(float(value))
    return steps


def change_signature(change):
    return (int(change.get("start", -1)), int(change.get("end", -1)),
            norm(change.get("observed", "")), norm(change.get("canonical", "")))


def score_result(row, source, label):
    result = row.get("result") or {}
    repair = result.get("entity_audio_repair") or {}
    source_text = norm(source["source_text"])
    result_text = norm(result.get("raw_text", result.get("text", "")))
    reference = label["reference"]
    gold = label["gold"]
    source_hits = sum(min(count, source_text.count(word)) for word, count in gold.items())
    result_hits = sum(min(count, result_text.count(word)) for word, count in gold.items())
    source_errors = Levenshtein.distance(reference, source_text)
    result_errors = Levenshtein.distance(reference, result_text)
    changes = list(repair.get("changes") or [])
    absent = [change for change in changes
              if norm(change.get("canonical", "")) not in reference]
    declared_change_count = repair.get("change_count")
    if declared_change_count is not None and int(declared_change_count) != len(changes):
        raise ValueError(f"change_count/list mismatch for {row['id']}/{row['arm']}")
    expected_step = 2.0 if row["arm"] == "old" else 10.0
    actual_steps = actual_replay_steps(row)
    if actual_steps and any(step != expected_step for step in actual_steps):
        raise ValueError(f"unexpected replay decode step for {row['id']}/{row['arm']}: {actual_steps}")
    return {
        "id": row["id"], "arm": row["arm"], "cohort": row.get("cohort"),
        "old_change_count": int(row["old_change_count"]),
        "history_group": "old_change_count_gt0" if int(row["old_change_count"]) > 0 else "old_change_count_eq0_control",
        "library_key": row["library_key"], "corpus": row.get("corpus"),
        "source_text_sha256": row["source_text_sha256"], "pcm_sha256": row["pcm_sha256"],
        "source_text_hits": source_hits, "result_raw_hits": result_hits,
        "expected_mentions": label["expected_mentions"],
        "source_text_character_errors": source_errors,
        "result_raw_character_errors": result_errors,
        "reference_characters": label["reference_characters"],
        "source_text_CER": source_errors / label["reference_characters"],
        "result_raw_CER": result_errors / label["reference_characters"],
        "net_hit_gain_vs_same_source_text": result_hits - source_hits,
        "character_error_reduction_vs_same_source_text": source_errors - result_errors,
        "changes": changes, "change_count": len(changes),
        "reference_absent_change_count": len(absent), "reference_absent_changes": absent,
        "entity_review_seconds": float(row.get("entity_review_seconds", 0) or 0),
        "backend_http_calls": int(row.get("backend_http_calls", 0) or 0),
        "reported_model_ms": float(row.get("reported_model_ms", 0) or 0),
        "replay_starts": row.get("replay_starts") or [],
        "actual_replay_decode_steps_seconds": actual_steps,
        "replay_outputs": row.get("replay_outputs") or [],
    }


def summarize(rows):
    output = {arm: {"samples": 0, "expected_mentions": 0,
        "source_text_hits": 0, "result_raw_hits": 0,
        "source_text_character_errors": 0, "result_raw_character_errors": 0,
        "reference_characters": 0, "net_hit_gain_vs_same_source_text": 0,
        "character_error_reduction_vs_same_source_text": 0,
        "changes": 0, "reference_absent_change_count": 0,
        "entity_review_seconds": [], "backend_http_calls": 0,
        "reported_model_ms": 0.0, "replay_start_count": 0,
        "replay_decode_steps_seconds": Counter(),
    } for arm in ARMS}
    for row in rows:
        item = output[row["arm"]]
        item["samples"] += 1
        for key in ("expected_mentions", "source_text_hits", "result_raw_hits",
                    "source_text_character_errors", "result_raw_character_errors",
                    "reference_characters", "net_hit_gain_vs_same_source_text",
                    "character_error_reduction_vs_same_source_text", "change_count",
                    "reference_absent_change_count", "backend_http_calls"):
            if key == "change_count":
                item["changes"] += row[key]
            elif key == "reference_absent_change_count":
                item[key] += row[key]
            else:
                item[key] += row[key]
        item["entity_review_seconds"].append(row["entity_review_seconds"])
        item["reported_model_ms"] += row["reported_model_ms"]
        item["replay_start_count"] += len(row["replay_starts"])
        item["replay_decode_steps_seconds"].update(str(v) for v in row["actual_replay_decode_steps_seconds"])
    for arm, item in output.items():
        mentions = item["expected_mentions"]
        chars = item["reference_characters"]
        item["source_text_entity_recall"] = item["source_text_hits"] / mentions if mentions else None
        item["result_raw_entity_recall"] = item["result_raw_hits"] / mentions if mentions else None
        item["source_text_CER"] = item["source_text_character_errors"] / chars if chars else None
        item["result_raw_CER"] = item["result_raw_character_errors"] / chars if chars else None
        item["mean_entity_review_seconds"] = distribution(item["entity_review_seconds"])
        item["entity_review_seconds"] = sum(item["entity_review_seconds"])
        item["replay_decode_steps_seconds"] = dict(item["replay_decode_steps_seconds"])
    return output


def run(partial=False):
    input_bytes = INPUTS.read_bytes()
    inputs = [json.loads(line) for line in input_bytes.splitlines() if line]
    if len(inputs) != 206 or len({row["id"] for row in inputs}) != 206:
        raise ValueError(f"expected 206 unique controlled input rows, found {len(inputs)}")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("input_sha256") != sha_bytes(input_bytes):
        raise ValueError("input file SHA-256 does not match controlled_protocol.json")
    if protocol.get("sample_order") != [row["id"] for row in inputs]:
        raise ValueError("protocol sample_order does not match controlled input order")
    worker = json.loads(WORKER.read_text())
    if protocol.get("source_hashes") != worker.get("source_hashes"):
        raise ValueError("protocol source_hashes differ from worker.json")
    if sha_file(OUT / "candidate.py") != protocol.get("candidate_sha256"):
        raise ValueError("protocol candidate_sha256 does not match candidate.py")
    for relative, expected_hash in protocol["source_hashes"].items():
        source_path = OUT / "source_snapshot" / relative
        if not source_path.is_file() or sha_file(source_path) != expected_hash:
            raise ValueError(f"source snapshot hash mismatch: {relative}")
    input_validation = validate_inputs(inputs, protocol)

    input_ids = {row["id"] for row in inputs}
    references = load_selected_reference_samples(REFERENCES, input_ids)
    if set(references) != input_ids:
        raise ValueError(f"reference missing controlled ID(s): {sorted(input_ids-set(references))[:8]}")
    labels = {}
    for sid, reference in references.items():
        norm_ref = norm(reference["reference"])
        gold = Counter(norm(entity["canonical"]) for entity in reference.get("gold_entities", []))
        if not norm_ref or any(not word for word in gold):
            raise ValueError(f"invalid fixed reference/gold label for {sid}")
        if any(norm_ref.count(word) < count for word, count in gold.items()):
            raise ValueError(f"gold mention Counter exceeds reference text for {sid}")
        labels[sid] = {"reference": norm_ref, "gold": gold,
                       "expected_mentions": sum(gold.values()),
                       "reference_characters": len(norm_ref)}

    result_rows, skipped_tail, results_bytes = read_jsonl_complete(RESULTS, partial)
    input_by_id = {row["id"]: row for row in inputs}
    seen = {}
    for row in result_rows:
        sid, arm = row.get("id"), row.get("arm")
        key = (sid, arm)
        if sid not in input_by_id or arm not in ARMS or key in seen:
            raise ValueError(f"unexpected or duplicate result row: {key}")
        source = input_by_id[sid]
        for field in ("library_key", "corpus", "cohort", "old_change_count",
                      "pcm_sha256", "source_text_sha256", "source_text"):
            if row.get(field) != source.get(field):
                raise ValueError(f"result/input {field} mismatch for {sid}/{arm}")
        seen[key] = row
    if not partial:
        if len(seen) != 412:
            raise AssertionError(f"full score requires exactly 412 rows, found {len(seen)}")
        for source in inputs:
            if any((source["id"], arm) not in seen for arm in ARMS):
                raise AssertionError(f"old/new pair missing for {source['id']}")

    details = [score_result(row, input_by_id[row["id"]], labels[row["id"]])
               for row in result_rows]
    by_arm = summarize(details)
    by_key = {(item["id"], item["arm"]): item for item in details}
    paired, shared_examples = [], []
    paired_results = defaultdict(lambda: {"count": 0, "old_changes": 0, "new_changes": 0,
                                          "shared_change_count": 0, "old_new_change_same": 0})
    for source in inputs:
        sid = source["id"]
        old, new = by_key.get((sid, "old")), by_key.get((sid, "new"))
        if old is None or new is None:
            continue
        for field in ("library_key", "corpus", "cohort", "old_change_count",
                      "pcm_sha256", "source_text_sha256", "source_text_hits",
                      "expected_mentions"):
            if old.get(field) != new.get(field):
                raise ValueError(f"old/new paired source mismatch for {sid}: {field}")
        old_signatures = {change_signature(change) for change in old["changes"]}
        new_signatures = {change_signature(change) for change in new["changes"]}
        shared = sorted(old_signatures & new_signatures)
        group = "old_change_count_gt0" if int(source["old_change_count"]) > 0 else "old_change_count_eq0_control"
        counter = paired_results[group]
        counter["count"] += 1
        counter["old_changes"] += old["change_count"]
        counter["new_changes"] += new["change_count"]
        counter["shared_change_count"] += len(shared)
        counter["old_new_change_same"] += bool(old_signatures & new_signatures)
        pair = {
            "id": sid, "history_group": group, "cohort": source.get("cohort"),
            "old_change_count": int(source["old_change_count"]),
            "library_key": source["library_key"],
            "source_text_hits": old["source_text_hits"],
            "expected_mentions": old["expected_mentions"],
            "old_result_hits": old["result_raw_hits"], "new_result_hits": new["result_raw_hits"],
            "old_hit_gain_vs_source_text": old["net_hit_gain_vs_same_source_text"],
            "new_hit_gain_vs_source_text": new["net_hit_gain_vs_same_source_text"],
            "old_error_reduction_vs_source_text": old["character_error_reduction_vs_same_source_text"],
            "new_error_reduction_vs_source_text": new["character_error_reduction_vs_same_source_text"],
            "new_minus_old_result_hit_delta": new["result_raw_hits"] - old["result_raw_hits"],
            "new_minus_old_error_delta": new["result_raw_character_errors"] - old["result_raw_character_errors"],
            "old_change_count_current": old["change_count"],
            "new_change_count_current": new["change_count"],
            "shared_current_old_new_changes": [list(signature) for signature in shared],
            "old_reference_absent_change_count": old["reference_absent_change_count"],
            "new_reference_absent_change_count": new["reference_absent_change_count"],
            "old_entity_review_seconds": old["entity_review_seconds"],
            "new_entity_review_seconds": new["entity_review_seconds"],
            "old_reported_model_ms": old["reported_model_ms"],
            "new_reported_model_ms": new["reported_model_ms"],
        }
        paired.append(pair)
        if group == "old_change_count_gt0" and shared and len(shared_examples) < 12:
            shared_examples.append(pair)

    by_history_group = {}
    for group in ("old_change_count_eq0_control", "old_change_count_gt0"):
        group_details = [item for item in details if item["history_group"] == group]
        by_history_group[group] = {
            "input_samples": sum(1 for row in inputs if (int(row["old_change_count"]) == 0)
                                 == (group == "old_change_count_eq0_control")),
            "paired_samples": paired_results[group]["count"],
            "historical_change_count_sum": sum(int(row["old_change_count"]) for row in inputs
                                                if (int(row["old_change_count"]) > 0)
                                                == (group == "old_change_count_gt0")),
            "by_arm": {arm: summarize([item for item in group_details if item["arm"] == arm])[arm]
                       for arm in ARMS},
            "paired_current_change_summary": dict(paired_results[group]),
        }

    cohort_history_cross_tab = Counter(
        (str(row.get("cohort", "<missing>")),
         "old_change_count_gt0" if int(row["old_change_count"]) > 0 else "old_change_count_eq0_control")
        for row in inputs)
    cohort_history_cross_tab = {
        cohort: {group: count for (name, group), count in cohort_history_cross_tab.items() if name == cohort}
        for cohort in sorted({str(row.get("cohort", "<missing>")) for row in inputs})
    }
    absent_changes = {
        arm: sum(item["reference_absent_change_count"] for item in details if item["arm"] == arm)
        for arm in ARMS
    }
    report = {
        "scope": "Controlled same-source-text entity replay only; source_text is reconstructed ASR first-pass text, not gold. It is used as the per-ID same-input baseline.",
        "partial": partial, "complete": len(seen) == 412,
        "result_rows": len(result_rows), "planned_inputs": len(inputs),
        "paired_sample_count": len(paired), "skipped_incomplete_trailing_row": skipped_tail,
        "cohort_is_not_history_group": "Controls/history groups are defined only by old_change_count == 0 versus > 0. cohort is reported separately; the historical wrong-negative row remains in the >0 group.",
        "historical_baseline_policy": "All old/new outputs below are fresh actual replay arms on the same reconstructed source_text and PCM. Archived old results are not used as a cached baseline.",
        "normalization_and_scoring": "Reference and texts use the conservative zh-itn then OpenCC t2s then normalized pipeline. Canonical hits are substring mentions capped by reference gold counts; CER is Levenshtein against the fixed reference.",
        "timing_scope": "entity_review_seconds includes EntityAudioRepair start/chunk/finish work only; it excludes first-pass ASR, refiner, browser and full-Web latency.",
        "reported_model_ms_scope": "ASR endpoint reported model-stage timing, not GPU hardware time or model invocation count.",
        "input_validation": input_validation,
        "protocol_hashes": {
            "protocol_sha256": sha_file(PROTOCOL),
            "candidate_sha256": protocol["candidate_sha256"],
            "source_hash_count": len(protocol["source_hashes"]),
            "source_hashes": protocol["source_hashes"],
            "references_sha256": sha_file(REFERENCES),
        },
        "by_arm": by_arm,
        "by_history_group": by_history_group,
        "cohort_history_cross_tab": cohort_history_cross_tab,
        "paired_change_absence_totals": absent_changes,
        "paired_samples": paired,
        "historical_examples_with_shared_current_changes": shared_examples,
        "reference_policy": "Only IDs in controlled_regression_inputs.jsonl are looked up in the frozen reference file. No reference or gold text is written to inference input or score output.",
    }
    output_path = OUT / ("partial_controlled_score.json" if partial else "controlled_score.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": output_path.name, "rows": len(result_rows),
        "pairs": len(paired), "complete": report["complete"],
        "history_groups": {key: {"inputs": value["input_samples"], "pairs": value["paired_samples"]}
                           for key, value in by_history_group.items()},
        "by_arm": {arm: {
            "samples": values["samples"], "hits": values["result_raw_hits"],
            "expected": values["expected_mentions"], "CER": values["result_raw_CER"],
            "review_seconds_mean": values["mean_entity_review_seconds"]["mean"],
            "reported_model_ms": values["reported_model_ms"],
        } for arm, values in by_arm.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true",
                        help="score complete result rows available now and pair IDs with both arms")
    args = parser.parse_args()
    run(args.partial)
