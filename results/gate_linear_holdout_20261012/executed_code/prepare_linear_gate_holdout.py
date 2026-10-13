"""New controlled text families plus untouched local AISHELL dev recordings."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.expand_gate_templates import delexical

BASIC=(
 ('route_card','巡检路线的起点写在记录卡背面。','巡检路线的起点，嗯，写在记录卡背面。','巡检路线的起起点写在记录卡背面。'),
 ('humidity','库房湿度超过限值以后要开启除湿机。','库房湿度超过限值以后，呃，要开启除湿机。','库房湿度超过限值以后要开开启除湿机。'),
 ('pendant','吊牌上的批次信息必须和箱单一致。','吊牌上的批次信息，嗯，必须和箱单一致。','吊牌上的批次信息必须必须和箱单一致。'),
 ('emergency_exit','紧急出口的箭头朝向走廊尽头。','紧急出口的那个箭头朝向走廊尽头。','紧急出口的箭箭头朝向走廊尽头。'),
 ('shelf_weight','货架底层的承重要求贴在立柱上。','货架底层的承重要求，嗯，贴在立柱上。','货架底层的承重要求贴贴在立柱上。'),
 ('return_slip','退货单盖章之后需要交到复核窗口。','退货单盖章之后，嗯，需要交到复核窗口。','退货单盖章之后需要交交到复核窗口。'),
)
PRESERVE=(
 ('critical_anchor','这个支点很重要，很重要，不能随意移动。'),
 ('separate_sheets','凭条应当一张一张地清点，清点完再装订。'),
 ('verbatim_wait','录音中的回答是等等等等，要求逐字转写。'),
)
LONG=(
 ('late_archive','归档结束，归档结束。随后检查索引。索引交给值班员。前面的归档结束说了两遍，需要原样保留。'),
 ('late_clear','现场清空，现场清空。接着核对名单。名单送到调度室。前面的现场清空说了两遍，请保留重复。'),
 ('step_filler','先断开电源。再检查接地线。然后，嗯，登记绝缘阻值。核验结束后签字。'),
)


def collect_used_ids(root,unused_preparation=None):
    used=set();provenance={};pattern=re.compile(r'BAC009S\d{4}W\d{4}')
    for path in (root/'results').rglob('*'):
        if not path.is_file() or path.suffix not in {'.json','.jsonl'}:continue
        if unused_preparation is not None and path.is_relative_to(unused_preparation):continue
        # Avoid manifests/weights; include prepared pools conservatively too.
        if path.name in {'integrity_check.json','recordings.json'}:continue
        with path.open(encoding='utf-8') as stream:
            hits=set()
            for line in stream:hits.update(pattern.findall(line))
        if hits:used.update(hits);provenance[str(path)]=c.file_hash(path)
    return used,provenance


def prepare(output,policy_root,human_count):
    if output.exists():raise FileExistsError('Never overwrite a holdout bank')
    selection_path=policy_root/'linear_training/selection.json';selection=json.loads(selection_path.read_text())
    primary=selection['primary'];model_path=policy_root/'linear_training'/primary/'model.pkl'
    if c.file_hash(model_path)!=selection['variants'][primary]['model_sha256']:raise ValueError('Frozen classifier changed')
    runtime=policy_root/'dev_collection/runtime';c.verify_frozen(runtime)
    unused_preparation=output.with_name(output.name+'_superseded_speaker_order')
    if unused_preparation.exists():
        if (unused_preparation/'test_data/tts.jsonl').exists() or (unused_preparation/'streaming').exists():
            raise ValueError('Superseded preparation has already observed test audio')
    else:unused_preparation=None
    seen=set();old_texts={}
    for path in (ROOT/'results').rglob('texts.jsonl'):
        if unused_preparation is not None and path.is_relative_to(unused_preparation):continue
        for row in d.read_rows(path):
            for key in ('spoken_text','intended_text'):
                if isinstance(row.get(key),str):seen.add(delexical(row[key]))
        old_texts[str(path)]=c.file_hash(path)
    synth=[]
    def add(group,category,spoken,intended):
        if delexical(spoken) in seen or delexical(intended) in seen:raise ValueError('Existing text family: '+group)
        row=dict(seed_id=f'linear_holdout_20261012_{len(synth):03d}',template_group='linear_holdout_v1:'+group,
            category=category,spoken_text=spoken,intended_text=intended,synthetic_split='test',domain='general',
            data_kind='synthetic_tts',human_gold=False,audio_script_verified=False,
            content_role='fictional_controlled_probe',scenario_role='script_construction_not_operation_label')
        row['seed_sha256']=d.digest(row);synth.append(row)
    for group,clean,filler,stutter in BASIC:
        for category,spoken in [('clean',clean),('filler',filler),('stutter',stutter)]:add(group,category,spoken,clean)
    for group,text in PRESERVE:add(group,'preserve',text,text)
    for group,text in LONG:add(group,'multi_window',text,text.replace('，嗯，',''))
    corpus=Path('/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell')
    transcript=corpus/'transcript/aishell_transcript_v0.8.txt'
    references={}
    for line in transcript.read_text().splitlines():
        key,value=line.split(maxsplit=1);references[key]=''.join(value.split())
    used,prior=collect_used_ids(ROOT,unused_preparation);groups=defaultdict(list)
    for path in sorted((corpus/'wav/dev').rglob('*.wav')):
        if path.stem not in used and path.stem in references:groups[path.parent.name].append(path)
    if not groups:raise ValueError('No untouched AISHELL dev recordings')
    rng=random.Random(20261012)
    for pool in groups.values():rng.shuffle(pool)
    speakers=sorted(groups);rng.shuffle(speakers)
    chosen=[]
    while len(chosen)<human_count and any(groups.values()):
        for speaker in speakers:
            if groups[speaker] and len(chosen)<human_count:chosen.append(groups[speaker].pop())
    if len(chosen)!=human_count:raise ValueError('Insufficient untouched reference recordings')
    output.mkdir(parents=True);(output/'logs').mkdir();(output/'executed_code').mkdir()
    shutil.copytree(runtime,output/'runtime');shutil.copytree(policy_root/'linear_training'/primary,output/'frozen_model')
    shutil.copy2(selection_path,output/'selection.json')
    source=output/'test_data';source.mkdir()
    for row in synth:c.append_row(source/'texts.jsonl',row)
    c.write_json(source/'manifest.json',dict(count=len(synth),seed=20261012,texts_sha256=c.file_hash(source/'texts.jsonl'),
        templates=len(BASIC)+len(PRESERVE)+len(LONG),synthetic_splits={'test':len(synth)},
        policy_sha256_before_test_generation=c.file_hash(output/'selection.json'),source_delexical_overlap=0,
        runtime_protocol_sha256=c.file_hash(output/'runtime/protocol.json'),frozen_before_audio=True,
        test_threshold_tuning_prohibited=True,human_gold=False,audio_script_verified=False))
    human=output/'human_sources.jsonl'
    for path in chosen:
        row=dict(seed_id='aishell_dev_holdout:'+path.stem,recording_id=path.stem,speaker=path.parent.name,
            template_group='aishell_dev_speaker:'+path.parent.name,category='human_read_speech',domain='general',
            data_kind='human_read_speech',reference=references[path.stem],intended_text=references[path.stem],
            spoken_text=references[path.stem],audio_path=str(path),audio_sha256=c.file_hash(path),
            human_gold=True,reference_kind='local_official_AISHELL_transcript',audio_script_verified=False,
            project_prior_id_overlap=False,split='heldout')
        c.append_row(human,row)
    (output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    c.write_json(output/'manifest.json',dict(synthetic_audio=len(synth),human_audio=len(chosen),
        human_speakers=len({p.parent.name for p in chosen}),human_sources_sha256=c.file_hash(human),
        official_transcript=str(transcript),official_transcript_sha256=c.file_hash(transcript),
        selection_sha256=c.file_hash(output/'selection.json'),model_sha256=c.file_hash(output/'frozen_model/model.pkl'),
        threshold=selection['variants'][primary]['threshold'],runtime_protocol_sha256=c.file_hash(output/'runtime/protocol.json'),
        prior_text_sources_sha256=old_texts,prior_AISHELL_mentions_sha256=prior,
        generator_sha256=c.file_hash(Path(__file__)),created_unix=time.time(),
        streaming_adapter_sha256=c.file_hash(ROOT/'experiments/scripts/stream_linear_gate_holdout.py'),
        threshold_tuning_prohibited=True,ASR_pretraining_overlap_unknown=True,
        scope='new_downstream_holdout_synthetic_controls_and_project_unused_official_read_speech'))
    print('Frozen',len(synth),'synthetic clips and',len(chosen),'human clips')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--policy-root',type=Path,required=True)
    p.add_argument('--human-count',type=int,default=24);a=p.parse_args();prepare(a.output.resolve(),a.policy_root.resolve(),a.human_count)
