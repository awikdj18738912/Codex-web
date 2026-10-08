"""Evaluation must preserve phrase boundaries and count failed recognition."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'experiments/scripts'
sys.path.insert(0,str(SCRIPTS))
import benchmark_multimed_entities as benchmark


def test_complete_procedure_and_cancer_names_are_not_scored_as_body_fragments():
    text='前列腺癌患者接受白内障手术。'
    labels={'前列腺':{'bod'},'白内障':{'dis'},'手术':{'pro'}}
    entities=benchmark.extract_entities(text,labels)
    assert [e['text'] for e in entities]==['前列腺癌','白内障手术']
    assert all(text[e['start']:e['end']]==e['text'] for e in entities)


def test_ascii_substrings_and_generic_words_are_not_entities():
    text='XMRSAy模型完成手术，MRSA感染需要治疗。'
    entities=benchmark.extract_entities(text,{'MRSA':{'ite'},'手术':{'pro'},'治疗':{'pro'}})
    assert [e['text'] for e in entities]==['MRSA感染']


def test_empty_asr_counts_as_empty_and_nonempty_corrupt_output_raises():
    record={'id':'silence','result':{'event':'final'},'final_payload':{'text':''}}
    assert benchmark.transcript(record,'clean_text')==''
    record['final_payload']['text']='真实转录'
    with pytest.raises(ValueError):benchmark.transcript(record,'clean_text')


def test_resume_requires_every_sample_once(tmp_path):
    p=tmp_path/'results.jsonl';p.write_text(json.dumps({'id':'a'})+'\n')
    with pytest.raises(ValueError):benchmark.require_complete(p,['a','b'])
    p.write_text(json.dumps({'id':'a'})+'\n'+json.dumps({'id':'a'})+'\n')
    with pytest.raises(ValueError):benchmark.require_complete(p,['a'])
