"""Explicitly loaded synthetic-domain gate prototype; no Web integration.

Only information available before the pending refinement operation is encoded.
Scores and validation-selected thresholds are not certified probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS = ("benefit", "neutral", "harm")
GAIN_LABELS = ("benefit", "not_benefit")
TASKS = ("window_refine", "local_edit_review")
INPUT_VERSION = 1


def input_text(value: dict) -> tuple[str, str, str]:
    task = value.get("task_kind")
    if task not in TASKS:
        raise ValueError("unsupported_task")
    baseline = value.get("baseline_text")
    if not isinstance(baseline, str) or not baseline.strip():
        raise ValueError("empty_baseline")
    domain = value.get("domain", "general")
    if domain not in {"general", "medical"}:
        raise ValueError("unsupported_domain")
    if task == "window_refine":
        raw = value.get("raw_text")
        context = value.get("available_context", "")
        if not isinstance(raw, str) or not raw.strip() or not isinstance(context, str):
            raise ValueError("invalid_window")
        return f"窗口精修。领域：{domain}。左侧上下文：", f"原始文本：{raw}。规则结果：{baseline}", context
    candidate = value.get("candidate") or {}
    source, target, kind = (candidate.get(k) for k in ("source", "target", "kind"))
    offset = value.get("candidate_local_start")
    if (not isinstance(source, str) or not source or not isinstance(target, str)
            or not isinstance(kind, str) or not isinstance(offset, int) or isinstance(offset, bool)
            or offset < 0 or baseline[offset:offset + len(source)] != source):
        raise ValueError("invalid_candidate_offset")
    if "[unused1]" in baseline or "[unused2]" in baseline:
        raise ValueError("reserved_span_marker")
    marked = baseline[:offset] + "[unused1]" + source + "[unused2]" + baseline[offset + len(source):]
    description = f"局部审核。领域：{domain}。类型：{kind}。原文：{source}。拟改为：{target}。"
    return description, marked, ""


def make_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    # Register existing vocabulary tokens so the tokenizer preserves the span.
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
    return tokenizer


def encode_input(tokenizer, value: dict, max_tokens: int = 256) -> dict:
    description, target, context = input_text(value)
    while True:
        encoded = tokenizer(description + context, target, truncation=False)
        if len(encoded["input_ids"]) <= max_tokens:
            break
        if not context:
            raise ValueError("target_exceeds_token_budget")
        # Crop only the oldest left context. Never truncate the pending target.
        context = context[len(context) // 2 + 1:]
    mask = [0.0] * len(encoded["input_ids"])
    if value["task_kind"] == "local_edit_review":
        ids = encoded["input_ids"]
        left, right = tokenizer.convert_tokens_to_ids(["[unused1]", "[unused2]"])
        if ids.count(left) != 1 or ids.count(right) != 1:
            raise ValueError("invalid_span_tokens")
        start, end = ids.index(left) + 1, ids.index(right)
        if start >= end:
            raise ValueError("empty_candidate_span")
        mask[start:end] = [1.0] * (end - start)
    encoded["source_mask"] = mask
    encoded["task_ids"] = TASKS.index(value["task_kind"])
    return encoded


def collate_inputs(tokenizer, items: list[dict]):
    import torch
    token_keys = ("input_ids", "attention_mask", "token_type_ids")
    batch = tokenizer.pad([{k: item[k] for k in token_keys if k in item} for item in items],
                          padding=True, return_tensors="pt")
    width = batch["input_ids"].shape[1]
    batch["source_mask"] = torch.tensor([item["source_mask"] + [0.0] * (width - len(item["source_mask"]))
                                        for item in items], dtype=torch.float32)
    batch["task_ids"] = torch.tensor([item["task_ids"] for item in items], dtype=torch.long)
    return batch


def build_network(encoder_path, *, pretrained: bool = True, num_labels: int = 3):
    import torch
    from transformers import AutoConfig, AutoModel

    class GateNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if pretrained:
                self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True, add_pooling_layer=False)
            else:
                config = AutoConfig.from_pretrained(encoder_path, local_files_only=True)
                self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
            hidden = self.encoder.config.hidden_size
            self.task_embedding = torch.nn.Embedding(len(TASKS), 16)
            self.head = torch.nn.Sequential(torch.nn.Linear(hidden * 2 + 16, 128),
                                            torch.nn.GELU(), torch.nn.Dropout(0.1),
                                            torch.nn.Linear(128, num_labels))

        def forward(self, input_ids, attention_mask, source_mask, task_ids, token_type_ids=None):
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            encoded = self.encoder(**kwargs).last_hidden_state
            mask = source_mask.to(encoded.dtype).unsqueeze(-1)
            span = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1)
            features = torch.cat([encoded[:, 0], span, self.task_embedding(task_ids)], dim=-1)
            return self.head(features)

    return GateNetwork()


def select_action(probabilities, thresholds: dict) -> str:
    import math
    if len(probabilities) != len(LABELS) or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities):
        return "keep"
    if abs(sum(probabilities) - 1) > 1e-4:
        return "keep"
    return "refine" if (probabilities[0] >= thresholds["benefit"]
                        and probabilities[2] <= thresholds["harm"]) else "keep"


def select_veto_action(probabilities, thresholds: dict, native_action: str) -> str:
    """Suppress a native refinement only with strong neutral/harm evidence."""
    import math
    if native_action not in {"keep", "defer", "refine"}:
        raise ValueError("invalid_native_action")
    if native_action != "refine":
        return native_action
    if (probabilities is None or len(probabilities) != len(LABELS)
            or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-4):
        return native_action
    return "keep" if (probabilities[1] >= thresholds["neutral"]
                      or probabilities[2] >= thresholds["harm"]) else native_action


def select_gain_action(probabilities, threshold: float, native_action: str) -> str:
    """High-recall benefit filtering; unsupported scores preserve scheduling."""
    import math
    if native_action not in {"keep", "defer", "refine"}:raise ValueError("invalid_native_action")
    if native_action != "refine":return native_action
    if (probabilities is None or len(probabilities)!=2
        or any(not math.isfinite(float(p)) or not 0<=p<=1 for p in probabilities)
        or abs(sum(probabilities)-1)>1e-4):return native_action
    return "refine" if probabilities[0]>=threshold else "keep"


class WeakRefinementGate:
    """Research inference API. Does not change text or load into online services."""

    def __init__(self, path: str | Path, *, device: str = "cpu"):
        import torch
        self.torch, self.device, self.path = torch, device, Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        self.labels=tuple(self.metadata["labels"])
        if self.metadata["input_version"] != INPUT_VERSION or self.labels not in {LABELS,GAIN_LABELS}:
            raise ValueError("unsupported_checkpoint_schema")
        self.tokenizer = make_tokenizer(self.path / "encoder")
        self.model = build_network(self.path / "encoder", pretrained=False, num_labels=len(self.labels))
        self.model.load_state_dict(torch.load(self.path / "gate.pt", map_location="cpu", weights_only=True))
        self.model.to(device).eval()

    def score(self, value: dict) -> list[float]:
        encoded = encode_input(self.tokenizer, value, self.metadata["max_tokens"])
        batch = {k: v.to(self.device) for k, v in collate_inputs(self.tokenizer, [encoded]).items()}
        with self.torch.inference_mode():
            return self.model(**batch).softmax(-1)[0].cpu().tolist()

    def decide(self, value: dict) -> dict:
        try:
            scores = self.score(value)
        except ValueError as error:
            return dict(action="keep", reason=str(error), probabilities=None, calibrated=False,
                        scope="synthetic_weak_prototype")
        action = (select_gain_action(scores,self.metadata['gain_thresholds'][value['task_kind']],'refine')
                  if getattr(self,'labels',LABELS)==GAIN_LABELS else
                  select_action(scores, self.metadata["thresholds"][value["task_kind"]]))
        return dict(action=action, reason="validation_selected_policy", probabilities=dict(zip(getattr(self,'labels',LABELS), scores)),
                    calibrated=False, scope="synthetic_weak_prototype")

    def decide_with_fallback(self, value: dict, native_action: str) -> dict:
        """Native scheduler owns KEEP/DEFER; uncertain scores preserve its action."""
        if native_action not in {"keep", "defer", "refine"}:
            raise ValueError("invalid_native_action")
        common = dict(native_action=native_action, calibrated=False, scope="synthetic_weak_prototype")
        if native_action != "refine":
            return dict(**common, action=native_action, reason="native_action_preserved", probabilities=None)
        binary=getattr(self,'labels',LABELS)==GAIN_LABELS
        thresholds = self.metadata.get("gain_thresholds" if binary else "veto_thresholds", {}).get(value.get("task_kind"))
        if thresholds is None:
            return dict(**common, action=native_action, reason="veto_policy_unavailable", probabilities=None)
        try:
            scores = self.score(value)
        except ValueError as error:
            return dict(**common, action=native_action, reason="native_fallback:" + str(error), probabilities=None)
        action = select_gain_action(scores,thresholds,native_action) if binary else select_veto_action(scores, thresholds, native_action)
        return dict(**common, action=action, reason="benefit_filter_veto" if binary and action=='keep' else "high_neutral_or_harm_veto" if action == "keep" else "native_fallback",
                    probabilities=dict(zip(getattr(self,'labels',LABELS), scores)))
