import argparse
import csv
import json
from pathlib import Path

import pytest

from experiments.scripts import collect_gate_pairs as collector


def test_sampling_keeps_original_keep_windows_and_is_reproducible():
    pool = [dict(case_id=str(i), selection_stratum="keep" if i < 8 else "refine") for i in range(12)]
    selected = collector.balanced_sample(pool, 6, 20261009)
    assert selected == collector.balanced_sample(pool, 6, 20261009)
    assert len({c["case_id"] for c in selected}) == 6
    assert {c["selection_stratum"] for c in selected} == {"keep", "refine"}
    with pytest.raises(ValueError):
        collector.balanced_sample(pool, 13, 1)


def test_noise_versions_share_recording_identity():
    a = collector.recording_identity(dict(id="BAC009S0905W0350", variant="clean"))
    b = collector.recording_identity(dict(id="BAC009S0905W0350", variant="noise"))
    assert a == b
    assert a[2] == "human_news"
    assert collector.recording_identity(dict(id="MMedFD_user_0005"))[2] == "public_tts"


def test_final_segment_events_require_complete_authoritative_text():
    from system.refinement_guard import join_refined_segments
    from types import SimpleNamespace
    web = SimpleNamespace(join_refined_segments=join_refined_segments)
    events = [dict(segment_id=1, text="今天开会。"), dict(segment_id=2, text="明天休息。")]
    row = dict(raw_text="今天开会。明天休息。", chunk_payloads=[dict(completed_segments=events),
               dict(completed_segments=events)], final_payload={})
    assert collector.final_source_segments(row, web) == ("今天开会。", "明天休息。")
    row["raw_text"] += "不要漏字。"
    assert collector.final_source_segments(row, web) is None


def test_force_call_preserves_window_mode_and_native_cleanup_metadata(monkeypatch):
    import system.web_app as web
    original = web.RefinementGate
    monkeypatch.setattr(web, "RefinementGate", original)
    collector.routing_factory(web)
    gate = web.RefinementGate("tri_state", refine_on_unverified_confidence=False)
    gate.policy = "current"
    keep = gate.decide("今天开会。", is_final=True, numeric_rule_mode=True)
    assert not keep.should_refine
    gate.policy = "always"
    force = gate.decide("今天开会。", is_final=True, numeric_rule_mode=True)
    assert force.should_refine
    assert force.mode == keep.mode == "tri_state"
    assert force.cleanup_signals == keep.cleanup_signals
    gate.policy = "rules"
    assert not gate.decide("嗯，我明天回家。", is_final=True).should_refine


def test_closure_adapter_fails_explicitly_when_contract_changes():
    def outer():
        def refine_update():
            return "owned"
        def endpoint():
            return refine_update()
        return endpoint
    assert collector.closure_value(outer(), "refine_update")() == "owned"
    with pytest.raises(RuntimeError, match="contract|exposes"):
        collector.closure_value(outer(), "removed_function")


def test_frozen_input_mutation_prevents_resume(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"a"}\n')
    collector.write_json(tmp_path / "protocol.json", dict(frozen_hashes={"cases.jsonl":collector.file_hash(path)}))
    collector.verify_frozen(tmp_path)
    path.write_text('{"case_id":"modified"}\n')
    with pytest.raises(RuntimeError, match="changed"):
        collector.verify_frozen(tmp_path)


def test_report_keeps_human_annotations_and_never_turns_distance_into_labels(tmp_path):
    case = dict(case_id="a", task_kind="window_refine", recording_group="speaker:1", variant="clean",
                data_kind="human_news", domain="general", audio_path="test.wav", origin_file="asr.jsonl",
                reference_full="明天开会", raw_text="明天开会。", raw_text_full="明天开会。",
                baseline_text="明天开会。",context_available_before="",window_chunk_count=1)
    collector.append_row(tmp_path / "cases.jsonl", case)
    arm = dict(result=dict(clean_text="明天开会。",refiner_executed=False,refiner_accepted=True),
               comparison_text="明天开会。",logical_calls=[],plausibility_calls=[],physical_calls=0)
    collector.append_row(tmp_path / "pairs.jsonl", dict(case_id="a",task_kind="window_refine",recording_group="speaker:1",
               arms={k:arm for k in ['rules','current','always']},reference_distance_proxy=dict(rules=1,current=0,always=0)))
    collector.write_json(tmp_path / "protocol.json", dict(scope="pilot",sampling="exploratory",
              frozen_hashes={"cases.jsonl":collector.file_hash(tmp_path / "cases.jsonl")}))
    args = argparse.Namespace(output=tmp_path)
    collector.report(args)
    with (tmp_path / "annotations.csv").open(encoding='utf-8-sig',newline='') as f:
        row = next(csv.DictReader(f))
    assert row['human_label'] == ''
    assert row['automatic_pair_label'] == 'neutral'
    assert row['automatic_label_basis'] == 'identical_final_output'
    neutral=collector.read_rows(tmp_path/'neutral_pairs.jsonl')
    assert neutral[0]['label']=='neutral'
    assert 'reference_full' not in neutral[0]['input']
    with (tmp_path/'changed_for_review.csv').open(encoding='utf-8-sig',newline='') as f:
        assert list(csv.DictReader(f))==[]
    row['human_label'], row['reviewer'] = 'uncertain', 'human'
    with (tmp_path / "annotations.csv").open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    collector.report(args)
    with (tmp_path / "annotations.csv").open(encoding='utf-8-sig',newline='') as f:
        updated = next(csv.DictReader(f))
    assert updated['human_label'] == 'uncertain' and updated['reviewer'] == 'human'
    assert json.loads((tmp_path/'summary.json').read_text())['human_reviewed'] == 1


def test_failed_operation_is_never_exported_as_neutral_supervision(tmp_path):
    collector.append_row(tmp_path/'cases.jsonl',dict(case_id='bad'))
    arm=dict(session_stats=dict(failed_call_count=1),logical_calls=[],plausibility_calls=[])
    collector.append_row(tmp_path/'pairs.jsonl',dict(case_id='bad',arms=dict(rules=arm,current=arm,always=arm)))
    collector.write_json(tmp_path/'protocol.json',dict(frozen_hashes={'cases.jsonl':collector.file_hash(tmp_path/'cases.jsonl')}))
    with pytest.raises(RuntimeError,match='not valid pairs'):
        collector.report(argparse.Namespace(output=tmp_path))
    assert not (tmp_path/'neutral_pairs.jsonl').exists()


def test_real_web_validator_rejects_wrong_number_in_forced_pair(tmp_path,monkeypatch):
    import system.web_app as web
    from system.zh_itn_bridge import DEFAULT_LIBRARY
    from system.entity_store import EntityStore
    import shutil
    if not DEFAULT_LIBRARY.exists():pytest.skip('Native ITN library unavailable')
    (tmp_path/'native').mkdir()
    shutil.copy2(DEFAULT_LIBRARY,tmp_path/'native/libzh_itn_bridge.so')
    EntityStore(tmp_path/'entities_snapshot.db')
    original_gate=web.RefinementGate
    original_refiner=web.TransformersRefiner
    monkeypatch.setattr(web,'RefinementGate',original_gate)
    monkeypatch.setattr(web,'TransformersRefiner',original_refiner)
    class WrongNumber:
        supports_entity_candidates=True
        def __init__(self):self.attempts=[]
        def refine_with_numeric_rules(self,text,**kwargs):
            self.attempts.append(kwargs.get('strict_placeholders',False))
            return json.dumps(dict(action='replace',source='2',target='20',reason='local_typo')),0.0
    model=WrongNumber()
    refine,_review,gate=collector.create_runtime(web,tmp_path,tmp_path,'cpu',refiner=model)
    protector,matcher=collector.tools_for_domain(web,tmp_path,'general')
    source='医生建议每天服用2片。'
    gate.policy='rules'
    baseline=refine(source,'Chinese',True,protector,None,matcher,single_window=True)
    assert baseline['clean_text']==source and model.attempts==[]
    gate.policy='current'
    skipped=refine(source,'Chinese',True,protector,None,matcher,single_window=True)
    assert not skipped['refiner_executed'] and model.attempts==[]
    gate.policy='always'
    result=refine(source,'Chinese',True,protector,None,matcher,single_window=True)
    assert result['clean_text']==source
    assert result['refiner_accepted'] is False
    assert model.attempts==[False,True]
    assert any('numeric' in reason for reason in result['refiner_reject_reasons'])
