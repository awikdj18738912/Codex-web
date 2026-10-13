"""Development-only text regressions; never synthetic speech or held-out gold."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as collector
from experiments.scripts.fork_gate_replay_source import fork

PRESERVE=(
    '线路非常非常复杂，先画接线图。',
    '这份档案很珍贵很珍贵，请妥善保存。',
    '资源极其极其有限，优先处理告警。',
    '备件需要一箱一箱清点。',
    '验收记录要一页一页阅读。',
    '现场不能拍照，不能拍照，警示牌保留了这两句话。',
    '样本尚未完成审核，尚未完成审核。',
    '不得泄露，不得泄露。',
    '成功了，成功了。原样记录现场说的话。',
    '已经到达，已经到达。负责人特意说了四次。',
    '我们可以继续，我们可以继续。这句话需要逐字记录。',
    '不要转交2份，不要转交2份。',
)
CLEANUP=(
    ('把附件发到采购采购组。','把附件发到采购组。'),
    ('现场需要需要登记数量。','现场需要登记数量。'),
    ('请核对核对收件人。','请核对收件人。'),
    ('机房的门牌牌号已经更新。','机房的门牌号已经更新。'),
    ('备件清单存放在办办公桌上。','备件清单存放在办公桌上。'),
    ('稍后在仓仓库门口碰面。','稍后在仓库门口碰面。'),
    ('调试结束后请联系联系值班人员。','调试结束后请联系值班人员。'),
    ('换班时需要记录记录温度。','换班时需要记录温度。'),
    ('这份材料已已由负责人签收。','这份材料已由负责人签收。'),
    ('请转交交给下一位审核人。','请转交给下一位审核人。'),
    ('非常非常关键，请确认确认编号。','非常非常关键，请确认编号。'),
    ('无需重发，无需重发。请登记登记问题。','无需重发，无需重发。请登记问题。'),
)


def prepare(parent: Path,output: Path,modules: list[str]):
    fork(parent,output,modules)
    rows=[]
    for index,(kind,raw,expected) in enumerate([('preserve',s,s) for s in PRESERVE]+[('cleanup',a,b) for a,b in CLEANUP]):
        identity=f'text_development:repetition_preservation_{index:03d}'
        rows.append(dict(seed_id=identity,recording_group=identity,domain='general',data_kind='text_development_probe',
            raw_text=raw,spoken_text=raw,intended_text=expected,audio_script_verified=False,
            expectation_kind=kind,has_audio=False,supervision='developer_expectation_not_audio_gold'))
    collector_path=output/'pairs_full';input_path=collector_path/'inputs/text_probes.jsonl'
    text=''.join(__import__('json').dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (output/'asr.jsonl').write_text(text);input_path.write_text(text)
    (collector_path/'cases.jsonl').write_text('')
    import json
    protocol=json.loads((collector_path/'protocol.json').read_text())
    protocol['inputs']=[dict(original=str(output/'asr.jsonl'),frozen='inputs/text_probes.jsonl',sha256=collector.file_hash(input_path))]
    protocol['frozen_hashes']['cases.jsonl']=collector.file_hash(collector_path/'cases.jsonl')
    protocol['development_patch']['scope']='text_only_development_probe_no_audio_no_ASR_not_fresh_holdout'
    protocol['source_records_after_dedup']=len(rows)
    collector.write_json(collector_path/'protocol.json',protocol)
    collector.write_json(output/'probe_manifest.json',dict(cases=len(rows),preserve=len(PRESERVE),cleanup=len(CLEANUP),
        scope=protocol['development_patch']['scope'],expectations_are_manual_text_specs=True,
        expected_labels_not_model_inputs=True,original_audio_test_used_as_runtime_template_only=True))
    collector.verify_frozen(collector_path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--modules',nargs='*',default=[])
    args=parser.parse_args();prepare(args.parent,args.output,args.modules)


if __name__=='__main__':main()
