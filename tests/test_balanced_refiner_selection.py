import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
from select_balanced_refiner_data import (
    cs2w_classify, dedup_text, fcgec_gate, iter_chinesehp, iter_cs2w,
    enforce_public_label_policy, record,
)


def csrow(source, target, labels):
    return {'id': 1, 'content': source, 'ref': target,
            'disfluency_label': labels, 'grammatical_error_label': '0'*len(source),
            'colloquial_word_label': '0'*len(source), 'asr_error_label': '0'*len(source)}


def test_rtype_word_repeat_is_not_automatically_a_character_stutter():
    source = '他们他们明天再来确认这个安排'
    category, eligible, reason = cs2w_classify(csrow(source, source[2:], '2'*4+'0'*(len(source)-4)))
    assert category == 'A03'
    assert not eligible
    assert 'semantic' in reason


def test_only_deleted_native_filler_span_can_qualify():
    source, target = '呃我明天送文件给你', '我明天送文件给你'
    assert cs2w_classify(csrow(source, target, '1'+'0'*(len(source)-1)))[:2] == ('A01', True)
    assert not cs2w_classify(csrow(source, target, '0'*(len(source)-1)+'1'))[1]


def test_missing_word_stays_in_semantic_review_queue():
    assert fcgec_gate({'category':'C03','valid':True,'input':'我们养成卫生习惯。',
                      'reference':'我们养成讲卫生的习惯。'}) == (False, 'missing_word_requires_semantic_review')


def test_reused_native_ids_do_not_overwrite_distinct_cs2w_rows(tmp_path):
    path = tmp_path/'train.jsonl'
    first = csrow('呃我明天送文件给你','我明天送文件给你','1'+'0'*8)
    second = dict(first, content='呃你明天送文件给我', ref='你明天送文件给我')
    path.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in (first,second))+'\n')
    rows = list(iter_cs2w(path,'revision'))
    assert len({row['id'] for row in rows}) == 2
    assert [row['native_id'] for row in rows] == [1,1]
    assert [row['source_line'] for row in rows] == [1,2]


def test_fixed_first_asr_hypothesis_never_uses_reference_to_pick_nbest(tmp_path):
    path = tmp_path/'hp.jsonl'
    path.write_text(json.dumps({'id':'a','corpus':'train','reference':'这次会议已经结束',
        'nbest':['这次会意已经结束','这次会议已经结束']},ensure_ascii=False)+'\n')
    row = next(iter_chinesehp(path, 'rev'))
    assert row['input'] == '这次会意已经结束'
    assert row['nbest_index'] == 0
    assert row['category'] == 'unclassified'
    assert not row['eligible']


def test_rtype_cannot_admit_inferred_stutter_or_auxiliary_labels():
    row = record('CS2W','s','A02','我我明天去办公室','我明天去办公室',eligible=True,
                 native_labels={'disfluency_label':'220000000'},auxiliary_labels=['E03'])
    enforce_public_label_policy(row)
    assert not row['eligible'] and row['category']=='unclassified'
    assert row['suggested_category_not_approved']=='A02'
    assert row['auxiliary_labels']==[]
    assert row['inferred_protection_signals']==['E03']


def test_explicit_native_labels_are_required_in_addition_to_quality_checks():
    for category, native in [('C02','CR'),('C03','CM'),('C04','IWO'),('E07','*')]:
        row=record('FCGEC','a',category,'样本文本','样本文本',eligible=False,
                   reason='needs_semantic_review',native_labels=native)
        enforce_public_label_policy(row)
        assert row['category']==category and row['category_label_evidence']['code']==native
        assert not row['eligible'] # Label evidence never overrides failed quality gates.


def test_entity_number_variants_share_dedup_key_without_mutating_text():
    raw = '把材料交给__ENTITY_018__，再通知__ENTITY_207__。'
    other = '把材料交给__ENTITY_901__，再通知__ENTITY_004__。'
    assert dedup_text(raw) == dedup_text(other)
    assert '__ENTITY_018__' in raw


def test_complete_small_run_caps_classes_and_preserves_partition_provenance(tmp_path):
    public = tmp_path/'public'
    (public/'CS2W').mkdir(parents=True)
    (public/'FCGEC').mkdir()
    manifest = {'retrieval_notes':{'repositories':{'CS2W':{'revision':'csrev'},'FCGEC':{'revision':'fcrev'}}}}
    (public/'manifest.json').write_text(json.dumps(manifest))
    s = '呃我明天送文件给你'
    (public/'CS2W/train.jsonl').write_text(json.dumps(csrow(s,s[1:],'1'+'0'*(len(s)-1)),ensure_ascii=False)+'\n')
    for split in ('val','test'):
        h = csrow('这是官方保留的评测句子。','这是官方保留的评测句子。','0'*13)
        (public/f'CS2W/{split}.jsonl').write_text(json.dumps(h,ensure_ascii=False)+'\n')
    gold = '请在明天下午把资料交给办公室同事。'
    native = {str(i):{'sentence':txt,'error_flag':0,'error_type':'*','operation':'[]'}
              for i,txt in enumerate((gold,'下周五上午请在会议室确认采购清单。'))}
    (public/'FCGEC/FCGEC_train_filtered.json').write_text(json.dumps(native,ensure_ascii=False))
    (public/'FCGEC/FCGEC_valid.json').write_text('{}')
    source = tmp_path/'source.sqlite3'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE samples(id TEXT,category TEXT,source TEXT,target TEXT,payload_json TEXT,status TEXT)')
        audit = {k:True for k in ('source_valid','category_valid','meaning_preserved','minimal_edit','cleanup_complete')}
        audit.update(issues=[],expected_target=gold)
        payload = {'group_id':'synthetic-group','audit':audit,'raw_teacher_sample':{'reference':gold},'plan':{}}
        db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?)',('s1','E07',gold,gold,json.dumps(payload,ensure_ascii=False),'accepted'))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path/'out'
    script = Path(__file__).resolve().parents[1]/'experiments/scripts/select_balanced_refiner_data.py'
    result = subprocess.run([sys.executable,str(script),'--snapshot',str(source),'--public-root',str(public),
        '--output',str(out),'--target-per-class','1'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out/'summary.json').read_text())
    assert summary['selected_total'] == 2
    assert summary['by_category']['E07']['selected'] == 1
    assert summary['by_category']['A01']['selected'] == 1
    assert summary['by_category']['B02']['shortfall'] == 1
    assert summary['human_reviewed_total'] == 0 and not summary['training_ready']
    identifiers = []
    for name in ('selected.pending_review.jsonl','reserve.jsonl','public_needs_review.jsonl',
                 'synthetic_needs_review.jsonl','excluded.jsonl'):
        identifiers += [json.loads(line)['id'] for line in (out/name).read_text().splitlines()]
    assert len(identifiers) == len(set(identifiers)) == 4
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
