from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.scripts.build_contextual_edit_contrasts import build_rows, validate_partitions, model_signature
from experiments.scripts.train_edit_location_detector import load_predefined_partitions
from system.edit_detection_data import apply_edits
from system.edit_location_detector import input_words


class IdentityNormalizer:
    def normalize(self, text):
        return SimpleNamespace(text=text)


class ContextualContrastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partitions = build_rows(IdentityNormalizer())

    def test_targets_roundtrip_numbers_and_keep_roles(self):
        for split, rows in self.partitions.items():
            for row in rows:
                self.assertEqual(apply_edits(row["baseline_text"], row["edits"]), row["desired_text"])
                self.assertEqual(row["training_eligible"], split == "train")
                self.assertFalse(row["human_audio_gold"])
                if row["contrast_role"] != "edit":
                    self.assertEqual(row["edits"], [])
                    self.assertEqual(row["baseline_text"], row["desired_text"])

    def test_identical_source_pairs_have_visible_context_and_opposite_targets(self):
        pairs = defaultdict(list)
        for rows in self.partitions.values():
            for row in rows:
                pairs[row["pair_id"]].append(row)
        seen = 0
        for pair in pairs.values():
            if len(pair) == 2 and pair[0]["baseline_text"] == pair[1]["baseline_text"]:
                seen += 1
                self.assertNotEqual(pair[0]["left_context"], pair[1]["left_context"])
                self.assertNotEqual(model_signature(pair[0]), model_signature(pair[1]))
                self.assertEqual(sum(bool(row["edits"]) for row in pair), 1)
        self.assertGreater(seen, 200)

    def test_annotation_metadata_is_not_visible_model_input(self):
        row = self.partitions["train"][0]
        altered = {**row, "desired_text": "KEEP DELETE SECRET", "contrast_role": "SECRET_ROLE",
                   "pair_id": "SECRET_PAIR", "category": "SECRET_LABEL", "edits": "SECRET_SPAN"}
        self.assertEqual(input_words(row), input_words(altered))
        for r in self.partitions["test"]:
            visible_context = r["left_context"] + r["right_context"] + r["available_context"]
            self.assertFalse(any(word in visible_context for word in ("口吃", "填充", "删除", "保留", "KEEP", "DELETE")))

    def test_concepts_frames_and_sources_do_not_cross_partitions(self):
        validate_partitions(self.partitions)
        contaminated = {name: list(rows) for name, rows in self.partitions.items()}
        contaminated["test"].append(self.partitions["train"][0])
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_partitions(contaminated)

    def test_unseen_surface_stress_is_test_only(self):
        self.assertTrue(all(row["lexical_scope"] != "unseen_surface_stress" for name in ("train", "validation") for row in self.partitions[name]))
        self.assertEqual(sum(row["lexical_scope"] == "unseen_surface_stress" for row in self.partitions["test"]), 72)

    def test_real_numeric_backend_does_not_pollute_semantic_frames(self):
        from system.zh_itn_bridge import ZhITNNormalizer
        root = Path(__file__).resolve().parents[1]
        partitions = build_rows(ZhITNNormalizer(root / "zh-itn/zh-itn/build/libzh_itn_bridge.so"))
        self.assertTrue(all(row["raw_asr"] == row["baseline_text"] for rows in partitions.values()
                            for row in rows if row["category"] != "numeric"))

    def test_predefined_partition_loader_rejects_fit_of_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text("{}")
            for name, rows in self.partitions.items():
                (root / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            loaded, _ = load_predefined_partitions(root)
            self.assertEqual(len(loaded["validation"]), 222)
            bad = loaded["validation"]
            bad[0]["training_eligible"] = True
            (root / "validation.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bad))
            with self.assertRaisesRegex(ValueError, "eligibility"):
                load_predefined_partitions(root)


if __name__ == "__main__":
    unittest.main()
