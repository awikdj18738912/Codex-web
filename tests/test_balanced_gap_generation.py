from collections import Counter
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
from generate_balanced_refiner_gaps import BalancedGapRunner
from generate_refiner_100k import NearIndex
from selfgen_ledger import Ledger


class StubLedger:
    def __init__(self):
        self.counts=Counter(E03=3999,E06=2999,B01=1)
        self.seq=0
    def stats(self):
        result={c:{'accepted':self.counts[c],'total':self.counts[c]}
                for c in ('E03','E06','B01')}
        result['total']={'accepted':sum(self.counts.values())}
        return result
    def next_sequence(self):
        self.seq+=1;return self.seq


def test_balanced_caps_continue_beyond_old_limits_and_skip_completed_classes():
    runner=BalancedGapRunner.__new__(BalancedGapRunner)
    runner.args=SimpleNamespace(workers=16,batch_size=4)
    runner.categories=['E03','E06','B01']
    runner.quotas={'E03':4001,'E06':3001,'B01':1}
    runner.ledger=StubLedger()
    runner.initial_stats=runner.ledger.stats()
    runner.initial_count=sum(runner.ledger.counts.values())
    runner.last_export=runner.initial_count
    runner.audit_deferred=False;runner.stop_event=threading.Event();runner.failure_streak=0
    states=[];submitted=[]
    runner.status=lambda state='running',*a,**kw:states.append(state)
    runner.retry_deferred=lambda:True
    runner.export_review=lambda **kw:None
    def batch(category,seq,n):
        submitted.append((category,n))
        return {'rows':[{'id':f'{seq}-{i}','category':category,'rejection_reason':None} for i in range(n)]}
    def commit(row):
        runner.ledger.counts[row['category']]+=1
        return 'accepted'
    runner.batch=batch;runner.commit_record=commit
    runner.run()
    assert runner.ledger.counts==Counter(E03=4001,E06=3001,B01=1)
    assert {c for c,_ in submitted}=={'E03','E06'}
    assert states[-1]=='balanced_gaps_complete_pending_human_review'


def seed_runner(tmp_path,rows):
    runner=BalancedGapRunner.__new__(BalancedGapRunner)
    selected=tmp_path/'selected.jsonl';d02=tmp_path/'d02.jsonl'
    selected.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    d02.write_text('')
    runner.args=SimpleNamespace(selected=selected,d02_snapshot=d02)
    runner.out=tmp_path;runner.quotas={'E06':2,'D02':2};runner.config_hash='config'
    runner.config={'seed_inputs':{'test':'hash'}}
    runner.ledger=Ledger(tmp_path/'ledger.sqlite3',runner.quotas,'config')
    runner.near=NearIndex();runner.heldout=NearIndex();runner.heldout_exact=set()
    return runner


def test_import_keeps_prior_native_provenance_and_is_idempotent(tmp_path):
    rows=[{'id':'old:1','category':'E06','group_id':'old:group1','dataset':'Gemma4',
           'input':'请将材料寄给__ENTITY_001__。','reference':'请将材料寄给__ENTITY_001__。',
           'quality_basis':'saved_model_audit','selection_state':'selected_pending_human_review',
           'source_id':'original-id','audit':{'original':'audit-evidence'}}]
    runner=seed_runner(tmp_path,rows)
    runner.import_seeds();runner.import_seeds()
    assert runner.ledger.stats()['E06']['accepted']==1
    item=next(runner.ledger.iter_accepted())
    assert item['payload']['source_id']=='original-id'
    assert item['payload']['audit']=={'original':'audit-evidence'}
    assert item['payload']['human_review_status']=='pending'
    assert not item['payload']['training_ready']


def test_unlabeled_public_seed_is_rejected_before_ledger_import(tmp_path):
    rows=[{'id':'ChineseHP:1','category':'E06','group_id':'hp:1','dataset':'ChineseHP',
           'input':'测试公开输入内容','reference':'测试公开目标内容',
           'selection_state':'selected_pending_human_review'}]
    runner=seed_runner(tmp_path,rows)
    with pytest.raises(ValueError,match='public_seed_missing_explicit_native_label'):
        runner.import_seeds()
    assert runner.ledger.stats()['total']['accepted']==0
