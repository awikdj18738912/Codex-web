from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.scripts.prepare_audio_edit_review import packet_basis, select_queue, source_windows, verify_clip
from experiments.scripts.prepare_edit_detection_data import Preparer, sha
from experiments.scripts.import_audio_edit_reviews import validate_review
from system.edit_detection_data import review_basis
from tests.test_edit_detection_data import IdentityNormalizer


class AudioEditReviewTest(unittest.TestCase):
    def source(self, target=None):
        return dict(seed_id="recording", raw_text="嗯，请确认。", historical_final_text=target,
                    audio_path="original.wav", audio_sha256="digest", recording_group="same_recording",
                    template_group="same_recording", split="regression", category="existing_podcast", context="",
                    audio_provenance=dict(original_path="source.mp3", original_sha256="source_digest",
                                          clip_start_seconds=0, clip_end_seconds=60))

    def test_historical_edit_proposal_never_becomes_gold(self):
        row = source_windows(Preparer(IdentityNormalizer(), []), self.source("请确认。"))[0]
        self.assertEqual(row["historical_output"]["projected_window_text"], "请确认。")
        for key in ("desired_text", "edits", "character_tags", "insertion_gaps"):
            self.assertIsNone(row[key])
        self.assertFalse(row["human_audio_gold"])
        self.assertFalse(row["training_eligible"])

    def test_unchanged_output_is_unknown_not_keep_gold(self):
        row = source_windows(Preparer(IdentityNormalizer(), []), self.source("嗯，请确认。"))[0]
        self.assertEqual(row["historical_output"]["projected_window_text"], row["baseline_text"])
        self.assertIsNone(row["edits"])
        self.assertIsNone(row["desired_text"])

    def test_missing_output_stays_missing(self):
        row = source_windows(Preparer(IdentityNormalizer(), []), self.source())[0]
        self.assertIsNone(row["historical_output"]["projected_window_text"])

    def test_context_and_historical_output_invalidate_packet(self):
        row = source_windows(Preparer(IdentityNormalizer(), []), self.source("请确认。"))[0]
        row["model_scores"] = {}
        original = packet_basis(row)
        changed = deepcopy(row)
        changed["left_context"] = "请保留原话。"
        self.assertNotEqual(original, packet_basis(changed))
        changed = deepcopy(row)
        changed["historical_output"]["projected_window_text"] = "请查阅。"
        self.assertNotEqual(original, packet_basis(changed))

    def test_selection_preserves_source_quotas_and_clip_bounds(self):
        rows = []
        for kind, count in (("existing_podcast", 48), ("existing_test_recording", 16)):
            for index in range(count):
                rows.append(dict(id=kind+str(index), category=kind, baseline_text=kind+str(index),
                                 left_context="", right_context="", audio_path=kind+str(index//2),
                                 selection_tags=["teacher_native_disagreement"]))
        selected = select_queue(rows)
        self.assertEqual(len(selected), 40)
        self.assertEqual(sum(row["category"] == "existing_podcast" for row in selected), 32)
        self.assertTrue(all(sum(other["audio_path"] == row["audio_path"] for other in selected) <= 2 for row in selected))

    def test_pcm_provenance_is_not_confused_with_wav_file_hash(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.wav"
            sf.write(path, np.zeros(1600), 16000)
            pcm, _ = sf.read(path, dtype="float32")
            expected = hashlib.sha256(pcm.tobytes()).hexdigest()
            info = verify_clip(path, expected)
            self.assertEqual(info["decoded_float32_pcm_sha256"], expected)
            self.assertNotEqual(info["audio_sha256"], expected)
            with self.assertRaises(ValueError):
                verify_clip(path, "wrong_pcm")

    def prepare_packet(self, directory):
        (directory / "audio").mkdir()
        original = directory / "original.wav"
        packaged = directory / "audio/clip.wav"
        original.write_bytes(b"test_audio_identity")
        packaged.write_bytes(original.read_bytes())
        row = source_windows(Preparer(IdentityNormalizer(), []), self.source("请确认。"))[0]
        row.update(audio_path=str(original), audio_sha256=sha(original), packaged_audio="audio/clip.wav", model_scores={})
        row["audio_provenance"].update(original_path=str(original), original_sha256=sha(original))
        row["review_basis_sha256"] = review_basis(row)
        row["packet_basis_sha256"] = packet_basis(row)
        review = dict(id=row["id"], review_basis_sha256=row["review_basis_sha256"],
                      packet_basis_sha256=row["packet_basis_sha256"], reviewer="test_reviewer",
                      review_kind="human_audio_cleaned", audio_checked=True, desired_text="请确认。",
                      edit_tasks=["disfluency"], single_task_for_all_edits=True, historical_pipeline_effect="beneficial")
        return row, review

    def test_explicit_review_stays_diagnostic_and_rejects_audio_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row, review = self.prepare_packet(directory)
            result = validate_review(row, review, directory)
            self.assertTrue(result["human_audio_gold"])
            self.assertFalse(result["training_eligible"])
            self.assertFalse(result["independent_acceptance_eligible"])
            self.assertIsNone(result["isolated_current_refiner_effect"])
            (directory / row["packaged_audio"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Packaged"):
                validate_review(row, review, directory)

    def test_unheard_and_stale_reviews_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row, review = self.prepare_packet(directory)
            with self.assertRaises(ValueError):
                validate_review(row, {**review, "audio_checked": False}, directory)
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate_review(row, {**review, "packet_basis_sha256": "old"}, directory)


if __name__ == "__main__":
    unittest.main()
