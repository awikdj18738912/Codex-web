import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
from generate_refiner_50k import CAP, capped_seeds


def row(i, dataset='Gemma4', status='accepted', native=None):
    return {'id':str(i), 'category':'A01', 'group_id':str(i), 'source':f'输入{i}',
            'target':f'答案{i}', 'status':status,
            'payload_json':json.dumps({'dataset':dataset, 'native_labels':native,
                                       'audit':{'evidence':'original'}})}


def test_cap_prefers_public_and_preserves_audit_without_rejected_records():
    rows = [row(i) for i in range(CAP+10)]
    rows += [row('public', 'CS2W', native={'disfluency_label':'1'}), row('bad', status='rejected')]
    selected = list(capped_seeds(rows))
    assert len(selected) == CAP
    assert selected[0]['id'] == 'public'
    assert all(r['id'] != 'bad' for r in selected)
    assert all(r['prior_ledger_payload']['audit'] == {'evidence':'original'} for r in selected)
    assert all(not r['training_ready'] for r in selected)
    assert [r['id'] for r in selected] == [r['id'] for r in capped_seeds(reversed(rows))]


def test_public_without_matching_native_label_is_never_imported():
    with pytest.raises(ValueError, match='public_seed_missing_explicit_native_label'):
        list(capped_seeds([row('unlabeled', 'ChineseHP')]))
