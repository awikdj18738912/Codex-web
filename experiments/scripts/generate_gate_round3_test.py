"""New unobserved sentence families, frozen before any new test audio."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import synthetic_gate_data as d
from experiments.scripts.expand_gate_templates import delexical

BASIC=(
    ('battery','备用电池已经放进应急柜的下层。','备用电池，那个，已经放进应急柜的下层。','备用电池已经已经放进应急柜的下层。'),
    ('seal_label','取走样品之前，要在封口贴上标识。','取走样品之前，嗯，要在封口贴上标识。','取走样品之前，要在封口贴上标标识。'),
    ('coolant','冷却液的剩余量写在白板右侧。','冷却液的那个剩余量写在白板右侧。','冷却液的剩剩余量写在白板右侧。'),
    ('cabinet_key','归还钥匙以后，把柜门锁紧。','归还钥匙以后，呃，把柜门锁紧。','归还钥匙以后，把柜门门锁紧。'),
    ('sample_rack','样品架的位置已经标在平面图上。','样品架的位置呢，已经标在平面图上。','样品架的位置已经标标在平面图上。'),
    ('packing_photo','外包装的照片需要存入质检台账。','外包装的照片需要，嗯，存入质检台账。','外包装的照片需要需要存入质检台账。'),
    ('clock_check','值班室的挂钟需要重新校准。','值班室的那个挂钟需要重新校准。','值班室的挂钟需要重新校校准。'),
    ('cart_path','推车经过坡道时要控制速度。','推车经过坡道时，呃，要控制速度。','推车经过坡道时要控控制速度。'),
)
PRESERVE=(
    ('fragile_seal','这个密封圈特别特别脆弱，安装时不能挤压。'),
    ('power_warning','不要直接通电，不要直接通电，说明书把这条警告写了两次。'),
    ('volume_scan','档案必须一册一册地扫描，每册结束后核对页码。'),
    ('quoted_pause','原声里说的是暂停暂停，请把这句话逐字记录。'),
)
LONG=(
    ('pressure_steps','第一步检查阀门。接着核对接头。然后然后读取压力值。读完后登记结果。最后关闭设备。',
     '第一步检查阀门。接着核对接头。然后读取压力值。读完后登记结果。最后关闭设备。'),
    ('late_intent','接收完成，接收完成。接下来登记编号。记录存入档案盒。前面的接收完成说了两遍，需要原样保留。',
     '接收完成，接收完成。接下来登记编号。记录存入档案盒。前面的接收完成说了两遍，需要原样保留。'),
    ('quantity_correction','先读第七号标签。实际数量是两千四百三十。不对，正确数量是两千四百三十一。请在末栏注明单位。',
     '先读第七号标签。正确数量是两千四百三十一。请在末栏注明单位。'),
    ('folder_warning','先把暂存目录交给值班组。目录目录暂时不能删除。等备份校验结束再清理。负责人又提醒，不能删除，不能删除。',
     '先把暂存目录交给值班组。目录暂时不能删除。等备份校验结束再清理。负责人又提醒，不能删除，不能删除。'),
)


def generate(root:Path):
    selection=root/'training/selection.json';frozen=json.loads(selection.read_text())
    if frozen['event']!='round3_policy_frozen_before_test_audio':raise ValueError('Policy must be frozen first')
    output=root/'test_data'
    if output.exists():raise FileExistsError('New test source must not overwrite a consumed test')
    old=[]
    sources=[ROOT/'results/qwen_tts_asr_gpu_pilot_20261010',ROOT/'results/gate_template_expansion_20261010',
             ROOT/'results/gate_round2_20261010/dev_data',ROOT/'results/gate_round2_20261010/risk_dev_data',ROOT/'results/gate_round2_20261010/test_data']
    sources+=list({p.parent for p in (ROOT/'results').rglob('test_consumed.json') if (p.parent/'texts.jsonl').exists()})
    for source in sources:
        old.extend(d.seeds(source))
    seen={delexical(r[k]) for r in old for k in ['spoken_text','intended_text']};rows=[]
    def add(group,category,spoken,intended):
        if delexical(spoken) in seen or delexical(intended) in seen:raise ValueError('Existing sentence template reused')
        identity=f'round3_fresh_20261011_{len(rows):03d}'
        row=dict(seed_id=identity,template_group='fresh_round3_v1:'+group,category=category,
            spoken_text=spoken,intended_text=intended,synthetic_split='test',domain='general',audio_script_verified=False,
            content_role='fictional_controlled_probe',scenario_role='script_construction_not_operation_label')
        row['seed_sha256']=d.digest(row);rows.append(row)
    for group,clean,filler,stutter in BASIC:
        for category,spoken in [('clean',clean),('filler',filler),('stutter',stutter)]:add(group,category,spoken,clean)
    for group,text in PRESERVE:add(group,'preserve',text,text)
    for group,spoken,intended in LONG:add(group,'multi_window',spoken,intended)
    output.mkdir();
    for row in rows:d.append_row(output/'texts.jsonl',row)
    d.write_json(output/'manifest.json',dict(count=len(rows),seed=20261011,texts_sha256=d.file_hash(output/'texts.jsonl'),
        templates=len({r['template_group'] for r in rows}),synthetic_splits={'test':len(rows)},
        policy_sha256_before_test_generation=d.file_hash(selection),freeze_unix=frozen['freeze_unix'],created_unix=time.time(),
        scope='new_sentence_family_synthetic_test_unverified_audio_small_sample',source_delexical_overlap=0,
        expected_gain_labels_not_assigned=True,test_threshold_tuning_prohibited=True))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(generate(a.root))
