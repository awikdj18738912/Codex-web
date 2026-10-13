from __future__ import annotations

from pathlib import Path
import unittest

from system.edit_location_detector import build_network, initialize_from_teacher

ROOT = Path(__file__).resolve().parents[1]


class EditDetectorDistillationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.teacher = build_network(ROOT / "models/entity_ranker/encoder", pretrained=False)

    def test_selected_blocks_embeddings_and_heads_copy_exactly(self):
        import torch
        student = build_network(ROOT / "models/entity_ranker/encoder", pretrained=False, num_hidden_layers=2)
        initialize_from_teacher(student, self.teacher, [0, 3])
        self.assertTrue(torch.equal(student.encoder.embeddings.word_embeddings.weight,
                                    self.teacher.encoder.embeddings.word_embeddings.weight))
        for current, original in enumerate([0, 3]):
            actual = student.encoder.encoder.layer[current].state_dict()
            expected = self.teacher.encoder.encoder.layer[original].state_dict()
            self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in actual))
        self.assertTrue(torch.equal(student.char_head.weight, self.teacher.char_head.weight))
        self.assertTrue(torch.equal(student.gap_head.weight, self.teacher.gap_head.weight))

    def test_invalid_layer_maps_are_not_silently_accepted(self):
        student = build_network(ROOT / "models/entity_ranker/encoder", pretrained=False, num_hidden_layers=2)
        for mapping in ([0], [0, 0], [0, 4]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                initialize_from_teacher(student, self.teacher, mapping)

    def test_shortened_encoder_cannot_claim_direct_full_pretrained_loading(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            build_network(ROOT / "models/entity_ranker/encoder", pretrained=True, num_hidden_layers=2)


if __name__ == "__main__":
    unittest.main()
