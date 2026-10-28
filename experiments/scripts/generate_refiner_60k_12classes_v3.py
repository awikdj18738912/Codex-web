"""Extend the preserved 12-class candidate pool to 60k; never export training data."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import requests

import generate_refiner_100k as base
import generate_balanced_refiner_gaps as gaps
from generate_refiner_multigpu import EndpointTransport

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ['A01','A02','A03','B01','B02','C01','D01','D02','E03','E05','E06','E07']
CAP = 5000
SOURCE = ROOT/'results/refiner_60k_12classes_20261027/resume_seed_snapshot_v3.jsonl'
FROZEN = Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02/split_audit.jsonl')


def prepare(inputs):
    if inputs.exists():
        manifest = json.loads((inputs/'manifest.json').read_text())
        for name, expected in manifest['input_hashes'].items():
            if gaps.file_hash(inputs/name) != expected:
                raise ValueError('seed_snapshot_changed:'+name)
        if manifest['entrypoint_sha256'] != gaps.file_hash(Path(__file__)):
            raise ValueError('entrypoint_changed')
        return manifest
    inputs.mkdir(parents=True)
    frozen = {r['id']:r for r in map(json.loads,FROZEN.open())}
    historical_ids={r['id'] for r in map(json.loads,Path('/home/aim0/data/datasets/ASR/refiner_balanced_13classes_20261026_no_A04_B03_C03_C04/candidates.pending_review.jsonl').open())}

    counts = Counter(); splits = Counter(); seen = set()
    with (inputs/'selected.pending_review.jsonl').open('w') as stream:
        for row in map(json.loads,SOURCE.open()):
            if row['category'] not in CATEGORIES:
                continue
            if row['id'] in seen:
                raise ValueError('duplicate_seed_id')
            seen.add(row['id'])
            old = frozen.get(row['id'])
            # Missing frozen IDs were isolated from the prior training experiment.
            split = old['split'] if old else ('isolated_review_do_not_train' if row['id'] in historical_ids else 'new_candidate')
            if old and (row['input'],row['reference']) != (old['input'],old['reference']):
                raise ValueError('frozen_pair_mismatch:'+row['id'])
            row.update(selection_state='selected_pending_human_review',
                       prior_frozen_split=split,
                       split_reservation='preserve_existing_split_or_isolation',
                       human_review_status='pending',training_ready=False)
            counts[row['category']]+=1; splits[split]+=1
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    if any(not 2941<=counts[c]<=CAP for c in CATEGORIES) or len(counts)!=12:
        raise ValueError('unexpected_seed_counts:'+str(counts))
    (inputs/'empty_d02.jsonl').write_text('')
    shutil.copyfile(FROZEN,inputs/'frozen_split_audit.jsonl')
    shutil.copyfile(__file__,inputs/'generate_refiner_60k_12classes_v3.py')
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'source':str(SOURCE),'source_sha256':gaps.file_hash(SOURCE),
        'frozen_split_source':str(FROZEN),'frozen_split_sha256':gaps.file_hash(FROZEN),
        'entrypoint_sha256':gaps.file_hash(Path(__file__)),
        'seed_total':sum(counts.values()),'seed_by_category':dict(counts),
        'prior_split_reservations':dict(splits),'target_total':60000,'target_per_class':CAP,
        'additional_by_category':{c:CAP-counts[c] for c in CATEGORIES},
        'original_files_modified':False,'training_started':False,'human_review_status':'pending',
        'input_hashes':{p.name:gaps.file_hash(p) for p in inputs.iterdir() if p.is_file()}}
    base.atomic_json(inputs/'manifest.json',manifest)
    return manifest


class Runner(gaps.BalancedGapRunner):
    def __init__(self,args):
        super().__init__(args)
        # Every old candidate is already indexed against new input AND target.
        # Keep all old splits explicitly, and also index frozen split records.
        for row in map(json.loads,(args.inputs/'frozen_split_audit.jsonl').open()):
            if row['split'] in ('dev','test'):
                for field in ('input','reference'):
                    self.heldout_exact.add(gaps.normalized(row[field]))
                    self.heldout.add('frozen:'+row['id']+':'+field,gaps.dedup_text(row[field]))
        with self.ledger._connect() as db:
            db.execute("UPDATE counters SET value=max(value,5000000) WHERE key='sequence'")
            db.commit()
        shutil.copyfile(__file__,self.out/'code_snapshot'/Path(__file__).name)
        base.atomic_json(self.out/'expansion_manifest.json',{
            'inputs_manifest':str(args.inputs/'manifest.json'),
            'inputs_manifest_sha256':gaps.file_hash(args.inputs/'manifest.json'),
            'entrypoint_sha256':gaps.file_hash(Path(__file__)),
            'target_total':60000,'target_per_class':CAP,'active_categories':CATEGORIES,
            'seed_total':35292,'planned_additional':24708,'gpu_ids':[0],'workers':16,
            'old_splits_and_isolation_must_be_preserved':True,'training_started':False})

    def status(self,state='running_60k_expansion',error=None,force=False):
        now=time.monotonic()
        if not force and now-self.last_status<15:
            return
        self.last_status=now
        stats=self.ledger.stats(); n=stats['total']['accepted']
        progress={'status':state,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'target':60000,'target_per_class':CAP,'active_categories':CATEGORIES,
            'gpu_ids':[0],'workers':16,'stats':stats,'seed_total':35292,
            'selected_or_machine_passed_pending_human_review':n,
            'new_candidates_net':n-35292,'remaining':60000-n,
            'new_since_process_start':n-self.initial_count,
            'accepted_per_hour_this_run':round((n-self.initial_count)/max(now-self.started,1)*3600,1),
            'remaining_by_active_category':{c:max(0,CAP-stats[c]['accepted']) for c in CATEGORIES},
            'error':error,'training_started':False,'human_review_status':'pending'}
        base.atomic_json(self.out/'progress.json',progress)
        lines=['# 12类补齐至6万条：运行状态','',
            '> 新增候选经过程序检查、教师独立请求审核与去重，仍待人工复核。旧候选保留原质量记录。','',
            f'状态：{state}；GPU0，16路并发。',
            f'当前候选：{n:,} / 60,000；净增：{n-35292:,}；剩余：{60000-n:,}。','',
            '| 小类 | 名称 | 已保留 | 目标 | 缺口 |','| --- | --- | ---: | ---: | ---: |']
        for c in CATEGORIES:
            count=stats[c]['accepted']
            lines.append(f'| {c} | {gaps.NAMES[c]} | {count} | {CAP} | {CAP-count} |')
        lines+=['','沿用40个场景；达到每类配额即停止新增。',
            '原始数据与冻结划分保留。未启动训练，计数不代表人工审核通过。']
        if error: lines+=['','需要处理：'+error]
        temp=self.out/'运行状态.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(self.out/'运行状态.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); args.inputs=args.inputs.resolve()
    prepare(args.inputs)
    if args.prepare_only:return
    # Process-local scope: historical scripts/ledgers remain untouched.
    gaps.CATEGORIES=CATEGORIES
    base.QUOTAS={c:CAP for c in CATEGORIES}
    base.RULES={c:base.RULES[c] for c in CATEGORIES}
    base.RULES['A03']='生成input、reference、reason。输入含一处至少2字的多字词语重复；重复强调也允许压缩。不把正常动词重叠当错误。答案只删除重复片段及附带停顿，不换词，不删除有独立意义的信息或程度词。'
    preservation='\n不得删掉正常的名词、宾语、主语、权利或范围信息，例如“的插曲”“权利”“有目共睹”“以上”“左右”。数字、程度、时间及列表各项必须保留。即时改口必须有明确改口证据，历史事实变化不能当作改口。只压缩重复本身，不能借强调之名删除有独立意义的程度信息。'
    base.GEN_SYSTEM+=preservation
    base.AUDIT_SYSTEM+=preservation
    base.AUDIT_SYSTEM+='\n本项目类别的完整定义如下，必须依这些定义判断，不能推测编号的含义：'+json.dumps(base.RULES,ensure_ascii=False)
    base.AUDIT_SYSTEM+='\nD01是异常误插逗号的修复；D02是句内异常误插句号的修复，均不要求同时存在口吃、冗余词或语音卡顿。把。宾语、交给。对象、放在。地点这类必需句法成分间插入句号属于D02定义。合法句界、完整分句和自然停顿仍须拒收；不要因只改标点本身就否认类别。'
    base.RULES['A02']+=' reference正文必须直接以给定stutter_char的实际汉字开头；不要输出plan.stutter_char等变量名，也不要额外添加口癖。'

    args.selected=args.inputs/'selected.pending_review.jsonl';args.d02_snapshot=args.inputs/'empty_d02.jsonl'
    args.public_root=Path('/home/aim0/data/datasets/ASR/public_refiner_sources_20261025')
    args.fcgec_test=Path('/home/aim0/data/datasets/ASR/refiner_balanced_17x5882_20261025/heldout_sources/FCGEC_test.json')
    args.api='http://127.0.0.1:8014/v1';args.model='gemma4-31b-teacher'
    args.gpu_ids=[0];args.workers=16;args.per_server_limit=16;args.batch_size=4
    args.target_per_class=CAP;args.seed=2026102706;args.request_timeout=900;args.stop_after=0
    args.categories=','.join(CATEGORIES)
    requests.get('http://127.0.0.1:8014/health',timeout=10).raise_for_status()
    models=requests.get(args.api+'/models',timeout=10);models.raise_for_status()
    if args.model not in {row['id'] for row in models.json()['data']}:
        raise ValueError('teacher_model_name_mismatch')
    runner=Runner(args)
    base.requests=EndpointTransport(args.api,[args.api],16,
        source_hash=gaps.file_hash(Path(__file__)),request_timeout=900)
    runner.run()

if __name__=='__main__':main()
