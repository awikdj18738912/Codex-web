"""Offline character/gap edit-location detector. Never edits or skips Refiner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LABELS = ("KEEP", "DELETE_CANDIDATE", "CORRECTION_CANDIDATE")
INPUT_VERSION = "contextual_char_gap_v1"


def input_words(row):
    candidates = []
    for candidate in row.get("entity_candidates", [])[:16]:
        observed = str(candidate.get("observed", ""))
        canonical = str(candidate.get("canonical", candidate.get("canonical_text", "")))
        candidates.append(observed + "→" + canonical)
    context = ("上文：" + row.get("left_context", "")[-40:] + "；下文：" + row.get("right_context", "")[:40]
               + "；背景：" + row.get("available_context", "")[:64]
               + "；实体：" + "|".join(candidates)[:128])
    prefix = list(context) + ["[unused1]"]
    text = row["baseline_text"]
    return prefix + list(text) + ["[unused2]"], len(prefix), len(text)


def make_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
    return tokenizer


def encode_input(tokenizer, row, max_tokens=512):
    words, source_start, length = input_words(row)
    if not length:
        raise ValueError("Empty source")
    encoded = tokenizer(words, is_split_into_words=True, truncation=False)
    if len(encoded["input_ids"]) > max_tokens:
        raise ValueError("Input exceeds token budget; do not truncate source or context silently")
    positions = {}
    for token, word in enumerate(encoded.word_ids()):
        if word is not None:
            positions.setdefault(word, token)
    required = [source_start - 1, *range(source_start, source_start + length), source_start + length]
    if any(word not in positions for word in required):
        raise ValueError("Source character has no token position")
    encoded = dict(encoded)
    encoded["source_positions"] = [positions[word] for word in range(source_start, source_start + length)]
    encoded["gap_left"] = [positions[source_start - 1], *encoded["source_positions"]]
    encoded["gap_right"] = [*encoded["source_positions"], positions[source_start + length]]
    return encoded


def collate(tokenizer, items):
    import torch

    batch = dict(tokenizer.pad([{key: value for key, value in item.items()
                                if key in {"input_ids", "attention_mask", "token_type_ids"}}
                               for item in items], return_tensors="pt"))
    width = max(len(item["source_positions"]) for item in items)
    for key, extra in (("source_positions", 0), ("gap_left", 1), ("gap_right", 1)):
        batch[key] = torch.tensor([item[key] + [0] * (width + extra - len(item[key])) for item in items])
    return batch


def build_network(encoder, pretrained=True, num_hidden_layers=None):
    import torch
    from transformers import AutoConfig, AutoModel

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if pretrained and num_hidden_layers is not None:
                raise ValueError("Shortened encoders require explicit verified-teacher initialization")
            config = AutoConfig.from_pretrained(encoder, local_files_only=True)
            if num_hidden_layers is not None:
                config.num_hidden_layers = num_hidden_layers
            self.encoder = (AutoModel.from_pretrained(encoder, local_files_only=True, add_pooling_layer=False)
                            if pretrained else AutoModel.from_config(config, add_pooling_layer=False))
            hidden = self.encoder.config.hidden_size
            self.char_head = torch.nn.Linear(hidden, len(LABELS))
            self.gap_head = torch.nn.Linear(hidden * 2, 2)

        def forward(self, input_ids, attention_mask, source_positions, gap_left, gap_right, token_type_ids=None):
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            states = self.encoder(**kwargs).last_hidden_state
            def gather(indices):
                return states.gather(1, indices.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
            return self.char_head(gather(source_positions)), self.gap_head(torch.cat([gather(gap_left), gather(gap_right)], -1))

    return Network()


def initialize_from_teacher(student, teacher, layer_indices):
    """Copy embeddings, selected blocks and heads; record derivation explicitly.

    This is an initialization for distillation, not a claim that the shortened
    network completed the original upstream language-model pretraining.
    """
    if len(layer_indices) != student.encoder.config.num_hidden_layers or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("Invalid student layer mapping")
    if any(index < 0 or index >= teacher.encoder.config.num_hidden_layers for index in layer_indices):
        raise ValueError("Teacher layer mapping out of range")
    if student.encoder.config.hidden_size != teacher.encoder.config.hidden_size:
        raise ValueError("Teacher and student hidden widths differ")
    student.encoder.embeddings.load_state_dict(teacher.encoder.embeddings.state_dict())
    for index, original in enumerate(layer_indices):
        student.encoder.encoder.layer[index].load_state_dict(teacher.encoder.encoder.layer[original].state_dict())
    student.char_head.load_state_dict(teacher.char_head.state_dict())
    student.gap_head.load_state_dict(teacher.gap_head.state_dict())


def target_options(row):
    """Allow equivalent repeated-copy locations during supervised fitting."""
    text = row["baseline_text"]
    edits = row["edits"]
    alternatives = row.get("offset_alternatives", []) if len(edits) == 1 else []
    possibilities = [[{**edits[0], **alternative}] for alternative in alternatives] if alternatives else [edits]
    result = []
    for option in possibilities:
        chars, gaps = [0] * len(text), [0] * (len(text) + 1)
        for edit in option:
            start, end = edit["start"], edit["end"]
            if start == end:
                gaps[start] = 1
            else:
                chars[start:end] = [LABELS.index(edit["kind"])] * (end - start)
        result.append((chars, gaps))
    return result


def candidate_spans(char_probabilities, gap_probabilities, threshold):
    result, start, current_kind = [], None, None
    for index, probabilities in enumerate([*char_probabilities, [1., 0., 0.]]):
        active = index < len(char_probabilities) and 1. - probabilities[0] >= threshold
        kind = LABELS[1 if probabilities[1] >= probabilities[2] else 2]
        if start is not None and (not active or kind != current_kind):
            result.append(dict(start=start, end=index, kind=current_kind,
                               score=max(1. - item[0] for item in char_probabilities[start:index])))
            start = None
        if active and start is None:
            start, current_kind = index, kind
    result.extend(dict(start=index, end=index, kind="CORRECTION_CANDIDATE", score=probabilities[1])
                  for index, probabilities in enumerate(gap_probabilities) if probabilities[1] >= threshold)
    return sorted(result, key=lambda item: (item["start"], item["end"]))


class EditLocationDetector:
    """Research inference interface; scores are uncalibrated candidate evidence."""

    def __init__(self, directory, device="cpu"):
        import torch

        self.directory, self.device = Path(directory), device
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["input_version"] != INPUT_VERSION or self.metadata["labels"] != list(LABELS):
            raise ValueError("Unsupported detector protocol")
        weights = self.directory / "detector.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != self.metadata["weights_sha256"]:
            raise ValueError("Detector checkpoint hash mismatch")
        self.tokenizer = make_tokenizer(self.directory / "encoder")
        self.model = build_network(self.directory / "encoder", pretrained=False).to(device).eval()
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

    def detect(self, row):
        import torch

        try:
            encoded = encode_input(self.tokenizer, row, self.metadata["max_tokens"])
        except ValueError as error:
            return dict(status="unsupported", reason=str(error), candidates=[], requires_fallback=True,
                        calibrated=False, production_skip_allowed=False)
        with torch.inference_mode():
            chars, gaps = self.model(**{key: value.to(self.device) for key, value in collate(self.tokenizer, [encoded]).items()})
            chars, gaps = chars[0].softmax(-1).cpu().tolist(), gaps[0].softmax(-1).cpu().tolist()
        candidates = candidate_spans(chars, gaps if self.metadata.get("insertion_detection_supported") else [], self.metadata["threshold"])
        return dict(status="candidates" if candidates else "no_candidate", candidates=candidates,
                    char_probabilities=chars, gap_probabilities=gaps, requires_fallback=True,
                    calibrated=False, production_skip_allowed=False)
