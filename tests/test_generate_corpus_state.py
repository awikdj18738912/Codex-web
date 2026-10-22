"""Durable count, quota and snapshot checks for the retained-size generator."""
import importlib.util
from pathlib import Path
import sqlite3
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('corpus', ROOT/'pipeline/scripts/generate_corpus.py')
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


def test_responses_and_retained_batch_survive_reopen_and_cache_is_case_specific(tmp_path):
    path = tmp_path/'state.sqlite3'
    store = corpus.CorpusStore(path)
    kwargs = {'system_prompt':'S','user_prompt':'U'}
    context = {'stage':'oral','id':'one'}
    key = corpus.digest(corpus.canonical({'context':context,'request':kwargs}))
    other = corpus.digest(corpus.canonical({'context':dict(context,id='two'),'request':kwargs}))
    store.save_response(key,context,'local',kwargs,'answer',1.0)
    record = {'id':'one','input':'源文本','output':{'refined_text':'目标文本'},'meta':{'scene':'daily_chat'}}
    store.commit_batch([{'serial':0,'scene':'daily_chat','status':'retained'}],[(0,record,'intent')])
    assert store.next_serial()==1
    store.db.close()
    store=corpus.CorpusStore(path)
    assert store.response(key)=='answer' and store.response(other) is None
    assert list(store.retained_rows())==[record]
    store.save_error(context,'local','temporary failure',.5)
    assert store.db.execute('SELECT COUNT(*) FROM request_errors').fetchone()[0]==1
    store.db.close()


def test_failed_duplicate_batch_rolls_back_attempts_atomically(tmp_path):
    store=corpus.CorpusStore(tmp_path/'state.sqlite3')
    item={'id':'one','input':'源','output':{'refined_text':'目标'},'meta':{'scene':'daily_chat'}}
    store.commit_batch([{'serial':0,'scene':'daily_chat','status':'retained'}],[(0,item,'intent')])
    with pytest.raises(sqlite3.IntegrityError):
        store.commit_batch([{'serial':1,'scene':'daily_chat','status':'retained'}],[(1,item,'intent')])
    assert store.next_serial()==1
    assert store.db.execute('SELECT COUNT(*) FROM retained').fetchone()[0]==1
    store.db.close()


def test_quota_allocation_exact_and_batches_do_not_overschedule_scenes():
    weights={'a':.12,'b':.33,'c':.55}
    quotas=corpus.scene_quotas(100000,weights)
    assert sum(quotas.values())==100000
    small=corpus.scene_quotas(17,weights)
    counts={s:0 for s in small}; serial=0
    while sum(counts.values())<17:
        jobs=corpus.plan_batch(small,counts,4,serial,7)
        assert jobs==corpus.plan_batch(small,counts,4,serial,7)
        for n,scene in jobs:
            assert n==serial
            serial+=1; counts[scene]+=1
            assert counts[scene]<=small[scene]
    assert counts==small


def test_snapshot_remains_immutable_and_detects_tampering(tmp_path,monkeypatch):
    source=tmp_path/'workspace';out=tmp_path/'output';out.mkdir()
    for rel in ['pipeline/scripts/generate_corpus.py','pipeline/configs/settings.py',
                'pipeline/configs/prompts.json','system/refinement_guard.py','zh-itn/zh-itn/build/libzh_itn_bridge.so']:
        p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
    monkeypatch.setattr(corpus,'ROOT',source)
    monkeypatch.setattr(corpus.sys,'argv',['generate_corpus.py','--output',str(out)])
    commands=[]
    monkeypatch.setattr(corpus.subprocess,'call',lambda cmd,env:commands.append(cmd) or 0)
    args=types.SimpleNamespace(output=out)
    assert corpus.freeze(args)==0 and commands[0][-1]=='--frozen'
    (source/'pipeline/configs/settings.py').write_text('new workspace version')
    assert corpus.freeze(args)==0
    frozen=out/'source_snapshot/pipeline/configs/settings.py'
    assert frozen.read_text()=='original'
    frozen.write_text('corrupted')
    with pytest.raises(RuntimeError,match='Frozen source changed'):
        corpus.freeze(args)
