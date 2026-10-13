"""One declared development-only risk probe batch; roles never assign labels."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.scripts import synthetic_gate_data as data

PROBES=[
    ('columns','请把这三列数据一列一列地对照，发现差异再记录。'),
    ('contracts','每份合同都要一份一份地核对签名，不能成批略过。'),
    ('playback','观察员一段一段地听回放，分别标注说话人的变化。'),
    ('retention','这条记录真的真的不能丢失，发言人刻意说了两遍。'),
    ('check_step','这个校验步骤特别特别关键，讲解者反复强调。'),
    ('attachment','邮件里注明附件特别特别重要，请保留原始文件。'),
    ('careful','大家要仔细仔细再仔细，提醒语里的重复不能省去。'),
    ('waiting','我们等了很久很久，才拿到审核结果。'),
    ('support','支架一点一点往下移动，每移动一次就暂停测量。'),
    ('entry','请每个小组一组一组地入场，以免门口拥挤。'),
    ('uncertainty','这个结论不一定成立，不一定成立，讲师重复提醒了两次。'),
    ('delete_warning','文件不能删除，不能删除，记录中保留了两次完整警告。'),
    ('not_pass','这不是通行证，不是通行证，工作人员重复说明。'),
    ('sealed','文档写了千万千万不能拆封，这两个提醒词都是原文。'),
]


def prepare(output):
    if output.exists() and any(output.iterdir()):raise FileExistsError('Never overwrite a risk-dev pack')
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(slug,text) in enumerate(PROBES):
        row=dict(seed_id=f'risk_dev_20261010_{i:03d}',category='development_risk_probe',
                 template_group='risk_development_v1:'+slug,spoken_text=text,intended_text=text,
                 synthetic_split='dev',domain='general',audio_script_verified=False,
                 content_role='fictional_controlled_transcription_probe',
                 scenario_role='risk_candidate_only_not_harm_label')
        row['seed_sha256']=data.digest(row);rows.append(row);data.append_row(output/'texts.jsonl',row)
    manifest=dict(schema_version=1,count=len(rows),seed=20261013,texts_sha256=data.file_hash(output/'texts.jsonl'),
                  requested_split='dev',synthetic_splits={'dev':len(rows)},generator_sha256=data.file_hash(Path(__file__)),
                  purpose='supplement_missing_development_risk_before_model_fit_not_independent_acceptance',
                  label_rule='actual_conservative_pair_outcome_only',human_gold=0)
    data.write_json(output/'manifest.json',manifest)
    (output/'preparation_sources').mkdir();(output/'preparation_sources'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/gate_round2_20261010/risk_dev_data')
    a=p.parse_args();print('risk_dev_candidates',len(prepare(a.output.resolve())))
