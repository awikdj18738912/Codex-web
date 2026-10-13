import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf

from experiments.scripts import synthetic_gate_data as data
from experiments.scripts import collect_gate_pairs as collector


def test_thousand_seeds_are_unique_reproducible_and_grouped_by_template():
    rows = data.make_seeds(1000, 20261009)
    assert rows == data.make_seeds(1000, 20261009)
    assert len({(row["spoken_text"], row["intended_text"]) for row in rows}) == 1000
    assert {row["category"] for row in rows} == set(data.TEMPLATES)
    splits = defaultdict(set)
    for row in rows:
        splits[row["template_group"]].add(row["synthetic_split"])
        assert row["audio_script_verified"] is False
        assert "label" not in row
    assert all(len(value) == 1 for value in splits.values())
    assert {row["template_group"] for row in rows if row["category"] == "clean"} == {
        row["template_group"] for row in rows if row["category"] == "filler"}
    assert {row["template_group"] for row in rows if row["category"] == "clean"} == {
        row["template_group"] for row in rows if row["category"] == "stutter"}
    repeated = [row for row in rows if row["category"] == "meaningful_repeat"]
    assert all(row["spoken_text"] == row["intended_text"] for row in repeated)
    assert any("一级一级" in row["intended_text"] for row in repeated)


def test_prepared_seed_mutation_and_configuration_drift_reject_resume(tmp_path):
    args = argparse.Namespace(output=tmp_path, count=48, seed=20261009)
    data.prepare(args)
    assert len(data.seeds(tmp_path)) == 48
    with pytest.raises(FileExistsError):
        data.prepare(args)
    data.assert_configuration(tmp_path / "config.json", {"tts_url": "service_a"})
    with pytest.raises(RuntimeError, match="configuration changed"):
        data.assert_configuration(tmp_path / "config.json", {"tts_url": "service_b"})
    with (tmp_path / "texts.jsonl").open("a") as stream:
        stream.write('{}\n')
    with pytest.raises(RuntimeError, match="seed texts changed"):
        data.seeds(tmp_path)


@pytest.mark.parametrize("baseline,refined,target,label", [
    ("我们我们开会。", "我们我们开会。", "我们开会。", "neutral"),
    ("我们我们开会。", "我们开会。", "我们开会。", "benefit"),
    ("一级一级地提水。", "一级地提水。", "一级一级地提水。", "harm"),
    ("不是今天。", "是今天。", "不是今天。", "harm"),
    ("费用是20元。", "费用是200元。", "费用是20元。", "harm"),
    ("如果到场，我们开会。", "如果到场。我们开会。", "如果到场，我们开会。", "uncertain"),
    ("甲甲去杭州。", "甲去杭州。", "王明去杭州。", "uncertain"),
])
def test_script_supervision_is_conservative_and_does_not_label_punctuation_as_gold(baseline, refined, target, label):
    actual, _basis = data.weak_label(baseline, refined, target, lambda text: text)
    assert actual == label


def test_synthetic_noise_versions_share_content_identity_and_domain():
    original = dict(id="synth_gate_00001", data_kind="synthetic_tts", domain="general",
                    recording_group="synthetic_text:synth_gate_00001", variant="clean")
    assert collector.recording_identity(original) == collector.recording_identity({**original, "variant": "noise"})
    assert collector.recording_identity(original)[1:] == ("general", "synthetic_tts")
    assert collector.recording_identity({**original, "domain": "medical"})[1] == "medical"
    with pytest.raises(ValueError):
        collector.recording_identity({**original, "recording_group": ""})


def test_asr_failure_cancels_owned_session():
    calls = []
    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/stream/start":
            return httpx.Response(200, json={"session_id": "owned"})
        if request.url.path == "/stream/chunk":
            return httpx.Response(500, json={"error": "failure"})
        return httpx.Response(200, json={"ok": True})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            data.decode_wave(client, "http://asr", b"\0" * 64000)
    assert calls == ["/stream/start", "/stream/chunk", "/stream/cancel"]


def test_audio_sanity_does_not_claim_script_verification(tmp_path):
    path = tmp_path / "test.wav"
    sf.write(path, np.sin(np.arange(16000) * 0.1).astype("float32") * 0.1, 16000)
    quality = data.audio_quality(path)
    assert quality["acoustic_sanity_passed"] is True
    assert quality["audio_script_verified"] is False
    sf.write(path, np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="silent"):
        data.audio_quality(path)


def test_moss_service_accepts_json_body_without_loading_model(monkeypatch):
    from fastapi.testclient import TestClient
    from pipeline.src.speech import moss_tts_service as moss
    seen = []
    def synthesize(self, **kwargs):
        seen.append(kwargs)
        return {"audio_path": str(kwargs["output_path"])}
    monkeypatch.setattr(moss.MossTTSService, "synthesize", synthesize)
    client = TestClient(moss.create_app("/unused", "cpu"))
    response = client.post("/v1/tts", json={"text": "你好", "output_path": "/tmp/tts-test.wav",
                                          "language": "Chinese", "max_new_tokens": 1024})
    assert response.status_code == 200
    assert seen[0]["text"] == "你好"
    assert seen[0]["language"] == "Chinese"
    assert seen[0]["max_new_tokens"] == 1024
    assert client.post("/v1/tts", json={"text": "", "output_path": "/tmp/unused.wav"}).status_code == 422


def test_synthesis_and_decode_resume_without_sending_references_to_asr(tmp_path, monkeypatch):
    # Sine audio is a transport fixture, not speech or a real TTS experiment.
    args = argparse.Namespace(output=tmp_path, count=2, seed=20261009, limit=2,
        tts_url="http://tts", asr_url="http://asr", timeout=10, pause=0,
        variants=["clean", "noise"], noise_snr=10)
    data.prepare(args)
    calls = []
    def respond(request):
        calls.append((request.url.host, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/v1/tts":
            body = json.loads(request.content)
            assert "intended_text" not in body
            sf.write(body["output_path"], 0.1 * np.sin(np.arange(24000) * 0.1), 24000)
            return httpx.Response(200, json={"audio_path": body["output_path"]})
        if request.url.path == "/stream/start":
            assert set(request.url.params) == {"language", "chunk_seconds"}
            return httpx.Response(200, json={"session_id": "fixture"})
        if request.url.path == "/stream/chunk":
            assert request.content and len(request.content) % 4 == 0
            return httpx.Response(200, json={"text": "识别中"})
        if request.url.path == "/stream/finish":
            return httpx.Response(200, json={"text": "测试转录。", "completed_segments": []})
        raise AssertionError(request.url)
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    data.synthesize(args)
    data.decode(args)
    assert len(data.read_rows(tmp_path / "tts.jsonl")) == 2
    records = data.read_rows(tmp_path / "asr.jsonl")
    assert len(records) == 4
    for record in records:
        assert record["data_kind"] == "synthetic_tts"
        assert record["audio_script_verified"] is False
        assert record["raw_text"] == "测试转录。"
        assert Path(record["audio_path"]).is_file()
    expensive_before = [item for item in calls if item[1] != "/health"]
    data.synthesize(args)
    data.decode(args)
    assert [item for item in calls if item[1] != "/health"] == expensive_before


def test_failed_synthesis_has_no_text_simulation_fallback(tmp_path, monkeypatch):
    args = argparse.Namespace(output=tmp_path, count=1, seed=20261009, limit=1,
                              tts_url="http://tts", timeout=5)
    data.prepare(args)
    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, json={"error": "TTS failed"})
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(httpx.HTTPStatusError):
        data.synthesize(args)
    assert not (tmp_path / "asr.jsonl").exists()
    assert not (tmp_path / "tts.jsonl").exists()
    failure = data.read_rows(tmp_path / "failures.jsonl")[0]
    assert failure["fallback_used"] is False


def test_synthetic_pair_export_uses_dedicated_sources(tmp_path, monkeypatch):
    args = argparse.Namespace(output=tmp_path, count=1, seed=20261009,
                              entity_db=Path("entities.db"), library=Path("itn.so"),
                              window_count=1, review_count=0)
    data.prepare(args)
    data.append_row(tmp_path / "asr.jsonl", {"raw_text": "测试转录。"})
    captured = []
    monkeypatch.setattr(collector, "prepare", captured.append)
    data.prepare_pairs(args)
    assert captured[0].no_demo_sources is True
    assert captured[0].sources == [tmp_path / "asr.jsonl"]
    assert captured[0].output == tmp_path / "pairs"


def label_fixture(tmp_path, monkeypatch):
    args = argparse.Namespace(output=tmp_path, count=1, seed=20261009)
    data.prepare(args)
    seed = data.seeds(tmp_path)[0]
    paired = tmp_path / "pairs"
    paired.mkdir()
    case = dict(case_id="case-a", data_kind="synthetic_tts", task_kind="window_refine", domain=seed["domain"],
                recording_group="synthetic_text:" + seed["seed_id"], raw_text=seed["spoken_text"] + "错误",
                baseline_text=seed["spoken_text"] + "错误", context_available_before="", window_chunk_count=1,
                synthetic_metadata={key: seed[key] for key in ("seed_id", "seed_sha256", "intended_text",
                "spoken_text", "template_group", "synthetic_split", "audio_script_verified")})
    pair = dict(case_id="case-a", arms={key: dict(comparison_text=text) for key, text in
                (("rules", case["baseline_text"]), ("current", seed["intended_text"]), ("always", seed["intended_text"]))})
    data.append_row(paired / "cases.jsonl", case)
    data.append_row(paired / "pairs.jsonl", pair)
    data.write_json(paired / "protocol.json", dict(frozen_hashes={"cases.jsonl": data.file_hash(paired / "cases.jsonl")}))
    class Normalizer:
        def __init__(self, path):
            pass
        def normalize(self, text):
            return SimpleNamespace(text=text)
    monkeypatch.setattr(collector, "load_system", lambda _path: SimpleNamespace(ZhITNNormalizer=Normalizer))
    return args, case, pair


@pytest.mark.parametrize("alternate", [False, True])
def test_weak_training_export_keeps_reference_and_future_output_out_of_features(tmp_path, monkeypatch, alternate):
    args, _case, _pair = label_fixture(tmp_path, monkeypatch)
    if alternate:
        args.pairs_dir = tmp_path / "expanded_pairs"
        (tmp_path / "pairs").rename(args.pairs_dir)
    data.label_pairs(args)
    label_output = args.pairs_dir if alternate else tmp_path
    row = data.read_rows(label_output / "weak_gate_labels.jsonl")[0]
    if alternate:
        assert not (tmp_path / "weak_gate_labels.jsonl").exists()
    assert row["label"] == "benefit"
    assert row["supervision_status"] == "synthetic_weak"
    assert row["human_gold"] is False
    assert row["eligible_for_real_risk_calibration"] is False
    assert row["audio_script_verified"] is False
    assert row["eligible_for_weak_pretraining"] is True
    assert set(row["input"]) == {"task_kind", "domain", "data_kind", "raw_text",
        "baseline_text", "available_context", "window_chunk_count"}


@pytest.mark.parametrize("invalid", ["failed_pair", "incomplete", "changed_target"])
def test_invalid_pairs_do_not_create_synthetic_supervision(tmp_path, monkeypatch, invalid):
    args, case, pair = label_fixture(tmp_path, monkeypatch)
    paired = tmp_path / "pairs"
    if invalid == "failed_pair":
        pair["arms"]["always"]["session_stats"] = {"failed_call_count": 1}
        (paired / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    else:
        if invalid == "incomplete":
            case = {**case, "case_id": "not-yet-run"}
            data.append_row(paired / "cases.jsonl", case)
        else:
            case["synthetic_metadata"]["intended_text"] = "被篡改的目标"
            (paired / "cases.jsonl").write_text(json.dumps(case) + "\n")
        data.write_json(paired / "protocol.json", dict(frozen_hashes={"cases.jsonl": data.file_hash(paired / "cases.jsonl")}))
    with pytest.raises(ValueError):
        data.label_pairs(args)
    assert not (tmp_path / "weak_gate_labels.jsonl").exists()
