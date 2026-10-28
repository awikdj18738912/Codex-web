"""Synthetic heldout balanced1000; reuse exactly the prior vLLM deployment runner."""
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

import benchmark_vllm_aasr_gpu0 as runner

ROOT=runner.ROOT
OUT=ROOT/'results/vllm_synthetic_balanced1000_20261028'


def read(path):return [json.loads(s) for s in Path(path).open()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    from system.zh_itn_bridge import ZhITNNormalizer
    OUT.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'results/vllm_aasr_refiners_gpu0_20261028'
    shutil.copy2(previous/'libzh_itn_bridge.so',OUT/'libzh_itn_bridge.so')
    normalizer=ZhITNNormalizer(OUT/'libzh_itn_bridge.so')
    pool_path=ROOT/'results/refiner13_no_C02_gate_20261027/all3056_refiner/predictions.jsonl'
    pool=read(pool_path)
    frozen=Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02/split_audit.jsonl')
    # Reject exact overlap with both input and reference of train and dev.
    forbidden=set()
    for r in read(frozen):
        if r['split'] in ('train','dev'):
            for key in ('input','reference'):
                forbidden.add(r[key]);forbidden.add(normalizer.normalize(r[key]).text)
    eligible={0:[],1:[]};seen=set();excluded=Counter()
    for r in pool:
        if r['source']!='Gemma4' or r['split']!='test':continue
        text,reference=r['text'],r['reference']
        baseline=normalizer.normalize(text).text
        normalized=normalizer.normalize(reference).text
        if text in forbidden or baseline in forbidden:
            excluded['train_or_dev_overlap']+=1;continue
        if text in seen or baseline in seen:
            excluded['duplicate_raw_or_normalized_input']+=1;continue
        if bool(r['label'])!=(baseline!=normalized):
            excluded['label_changed_after_numeric_rules']+=1;continue
        seen.update((text,baseline))
        eligible[r['label']].append({'id':r['id'],'source_record_id':r['id'],'source':'Gemma4',
            'split':'test','category':r['category'],'scene':r['category'],'label':r['label'],
            'derived_clean':r.get('derived_clean',r['id'].endswith(':clean_reference')),
            'text':text,'oral':text,'clean':normalized,'reference':normalized,'original_reference':reference,
            'rule_baseline':baseline})
    rng=random.Random(20261028)
    def stratified(group,n):
        bins=defaultdict(list)
        for r in group:bins[r['category']].append(r)
        for a in bins.values():rng.shuffle(a)
        selected=[]
        while len(selected)<n:
            progressed=False
            for category in sorted(bins):
                if bins[category]:selected.append(bins[category].pop());progressed=True
                if len(selected)==n:break
            if not progressed:raise RuntimeError('Insufficient eligible samples')
        return selected
    positive=stratified(eligible[1],500);negative=stratified(eligible[0],500)
    selected=positive+negative;rng.shuffle(selected)
    assert len({r['text'] for r in selected})==1000
    assert len({r['rule_baseline'] for r in selected})==1000
    (OUT/'text_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    old=json.loads((previous/'protocol.json').read_text())
    latency=positive[:50]+negative[:50];rng.shuffle(latency)
    old.update(dataset='Synthetic Gemma4 heldout1000 500need/500clean',source=str(pool_path),
        source_protocol=old['source_protocol'],latency_ids=[r['id'] for r in latency],
        latency='100 balanced heldout texts (50/50) single request; warm pipeline wall clock',
        throughput='all1000 unique inputs at8 and16 closed-loop concurrency, identical source order',
        comparison_limit='Same implementation and gate mutex as AASR vLLM run; only dataset changed. No new Transformers timing baseline.',
        quality='Weak candidate references; custom4-dimension rubric, NOT official AASR benchmark score. Score16-concurrency outputs.',
        report_script='experiments/scripts/report_balanced1000_vllm.py',
        manifest_sha256=sha(OUT/'text_manifest.jsonl'),library_sha256=sha(OUT/'libzh_itn_bridge.so'),
        script_sha256=sha(Path(__file__)),runner_sha256=sha(Path(runner.__file__)))
    (OUT/'protocol.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    audit={'selection_seed':20261028,'selection':'Round-robin stratified by category, labels fixed before inference; no selection by gate/refiner outcomes',
        'source':str(pool_path),'source_sha256':sha(pool_path),'frozen_split':str(frozen),'frozen_split_sha256':sha(frozen),
        'eligible_counts':{str(k):len(v) for k,v in eligible.items()},'excluded':dict(excluded),
        'selected_labels':dict(Counter(r['label'] for r in selected)),
        'selected_categories':dict(Counter(r['category'] for r in selected)),
        'derived_clean_count':sum(r['derived_clean'] for r in selected),
        'all_from_Gemma4_test':True,'raw_and_normalized_inputs_unique':True,'no_exact_input_overlap_with_train_dev_input_or_reference':True,
        'limitations':['Candidate references not all manually audited','Derived clean texts are synthetic reference targets, not independently collected natural clean utterances',
            'This heldout pool was previously inspected; no claim of new blind independent generalization','Reference texts are excluded from inference prompts']}
    (OUT/'selection_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    rubric=[]
    for r in selected:
        definitions=[
            ('content','输出是否保留clean中的关键事实、实体、数量、否定、条件和最终意图，且没有添加原文不支持的信息？'),
            ('format','输出是否保留正确数字数值、单位、范围、英文名称、代码与占位符？数字写法以clean中的本机规则库结果为准；没有这些内容时，不得引入错误。'),
            ('filter','输出是否清理了oral中需要清理的口癖、口吃、无意义重复等，同时保留有意义指代和正常强调？若oral本来干净，应保持内容；不把正常强调作为必须删除的错误。'),
            ('rephrase','若oral明确自我修正或多次改口，输出是否保留最终意图并删除明确放弃的旧候选？若无此现象，应保留原意，不得擅自改写、总结或补全。')]
        qa=[{'id':f'q{i:03d}','category':category,'question':question,
             'evidence':'oral为输入；clean为同一数字规则处理后的候选参考。以忠实、最小必要修改为准，允许等义表达。',
             'options':[{'text':'满足本项要求','score':1},{'text':'不满足本项要求','score':0}]} for i,(category,question) in enumerate(definitions,1)]
        rubric.append({'source_record_id':r['id'],'audio':'text-only-no-audio','scene':r['category'],
            'oral':r['text'],'clean':r['reference'],'status':'success','qa':qa})
    (OUT/'custom_rubric.json').write_text(json.dumps({'items':rubric,'qualification':'Custom four-dimension binary rubric, equal dimension weight, not official AASR atomic rubric'},ensure_ascii=False,indent=2)+'\n')
    print('Prepared1000syntheticheldout,500/500; customrubric4questions/sample.',flush=True)


def main():
    if not OUT.exists():prepare()
    os.environ['REFINER_SPEED_OUT']=str(OUT)
    os.environ['REFINER_SPEED_PORT']='8028'
    runner.OUT=OUT;runner.PORT=8028
    runner.orchestrate()


if __name__=='__main__':main()
