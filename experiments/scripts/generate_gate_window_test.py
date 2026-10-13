"""Freeze a new synthetic probe bank after window-only routing is fixed."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts import synthetic_gate_data as d, collect_gate_pairs as c
from experiments.scripts.expand_gate_templates import delexical

BASIC = (
    ('rain_gauge', '雨量计的读数要填在交接表末尾。', '雨量计的读数，嗯，要填在交接表末尾。', '雨量计的读读数要填在交接表末尾。'),
    ('night_lamp', '夜间照明的开关装在楼梯转角处。', '夜间照明的那个开关装在楼梯转角处。', '夜间照明的开开关装在楼梯转角处。'),
    ('sealed_bag', '封存袋要等复核人员到场以后再打开。', '封存袋要等，呃，复核人员到场以后再打开。', '封存袋要等复核人员到场以后再再打开。'),
    ('vent_mesh', '通风网拆下以后先放到清洗槽旁边。', '通风网拆下以后，嗯，先放到清洗槽旁边。', '通风网拆下以后先先放到清洗槽旁边。'),
)
PRESERVE = (
    ('urgent_delivery', '这批包裹非常非常急，今晚必须送到。'),
    ('never_open', '不要打开铅封，不要打开铅封，值班员再次强调这条要求。'),
    ('rolls', '这批胶片要一卷一卷地检查，发现破损先登记。'),
    ('quote', '发言人说的是明白明白，需要逐字记录。'),
)
LONG = (
    ('late_named_alarm', '警报解除，警报解除。随后通知巡逻组。安排人员回到岗亭。前面的警报解除说了两遍，请原样保留。'),
    ('late_named_receipt', '凭证收到，凭证收到。先核对印章。接着填写收件日期。前面的凭证收到说了两遍，需要保留重复。'),
    ('clean_sequence', '先查看门禁指示灯。然后登记进入时间。离开时带上工具箱。最后把登记本交给管理员。'),
    ('mixed_cleanup', '已经备案，已经备案。接着确认确认货位。资料交给仓库主管。前面的已经备案说了两遍，需要原样保留。'),
)


def generate(experiment: Path, policy_root: Path):
    experiment, policy_root = experiment.resolve(), policy_root.resolve()
    output = experiment / 'test_data'
    if output.exists():
        raise FileExistsError('Cannot replace an existing probe bank')
    runtime = experiment / 'runtime/pairs_full'
    c.verify_frozen(runtime)
    selection = policy_root / 'training/selection.json'
    seen = set()
    for path in (ROOT / 'results').rglob('texts.jsonl'):
        for row in d.read_rows(path):
            for key in ('spoken_text', 'intended_text'):
                if isinstance(row.get(key), str):
                    seen.add(delexical(row[key]))
    rows = []
    def add(group, category, spoken, intended):
        if delexical(spoken) in seen or delexical(intended) in seen:
            raise ValueError('Previous sentence reused: ' + group)
        row = dict(seed_id=f'window_fresh_20261011_{len(rows):03d}',
                   template_group='window_fresh_v1:' + group, category=category,
                   spoken_text=spoken, intended_text=intended, synthetic_split='test',
                   domain='general', audio_script_verified=False,
                   content_role='fictional_controlled_probe',
                   scenario_role='script_construction_not_operation_label')
        row['seed_sha256'] = d.digest(row)
        rows.append(row)
    for group, clean, filler, stutter in BASIC:
        for category, spoken in [('clean', clean), ('filler', filler), ('stutter', stutter)]:
            add(group, category, spoken, clean)
    for group, text in PRESERVE:
        add(group, 'preserve', text, text)
    for group, text in LONG:
        add(group, 'multi_window', text, text.replace('确认确认', '确认'))
    output.mkdir()
    for row in rows:
        d.append_row(output / 'texts.jsonl', row)
    training = experiment / 'training'
    training.mkdir(exist_ok=True)
    shutil.copy2(selection, training / 'selection.json')
    d.write_json(output / 'manifest.json', dict(
        count=len(rows), seed=20261011, templates=len(BASIC)+len(PRESERVE)+len(LONG),
        texts_sha256=d.file_hash(output/'texts.jsonl'),
        policy_sha256_before_test_generation=d.file_hash(selection),
        runtime_protocol_sha256=c.file_hash(runtime/'protocol.json'),
        runner_sha256=c.file_hash(ROOT/'experiments/scripts/evaluate_gate_streaming.py'),
        generator_sha256=c.file_hash(Path(__file__)), created_unix=time.time(),
        routing_scope='window_only_local_native', frozen_before_audio=True,
        synthetic_splits={'test':len(rows)}, source_delexical_overlap=0,
        scope='new_synthetic_sentence_bank_unverified_audio_no_semantic_domain_disjointness_claim',
        test_threshold_tuning_prohibited=True, expected_gain_labels_not_assigned=True))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--policy-root', type=Path, required=True)
    a = p.parse_args()
    print(generate(a.experiment, a.policy_root))
