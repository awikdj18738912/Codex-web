"""Full local podcast comparison; no reference transcript and no claimed CER.

Reuses the real Web/ASR/Refiner runner. Both glossaries are frozen before
decoding, and every replay uses identical recorded ASR responses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

from benchmark_multimed_entities import Runner, digest, require_complete, transcript, CONTEXT_WORDS, MODEL
from evaluate_entity_library import ROOT, write_json
from system.entity_store import EntityStore

OUT=ROOT/'results/oncology_podcast_20261008'
MEDICAL=ROOT/'data/entity_benchmarks/multimed_chinese_test'
ONCOLOGY=ROOT/'data/entity_benchmarks/oncology_podcast_20261008'
# Fixed by the file's stated topic before any recognition, never error aliases.
TOPIC_TERMS='''靶向药 靶向治疗 免疫治疗 放射治疗 化学治疗 恶性肿瘤 良性肿瘤
肿瘤分期 癌症分期 肺癌 肝癌 胃癌 乳腺癌 结直肠癌 直肠癌 胰腺癌
食管癌 鼻咽癌 宫颈癌 卵巢癌 甲状腺癌 黑色素瘤 淋巴瘤 白血病
小细胞肺癌 非小细胞肺癌 肺腺癌 肺鳞癌 鳞状细胞癌 癌前病变
原位癌 浸润癌 远处转移 淋巴结转移 骨转移 肝转移 脑转移
淋巴结 肿瘤标志物 癌胚抗原 基因突变 基因检测 基因测序
驱动基因 突变基因 耐药性 耐药突变 靶向药物 分子靶向治疗
免疫检查点 免疫检查点抑制剂 免疫细胞 肿瘤微环境 细胞毒性
单克隆抗体 表皮生长因子受体 血管内皮生长因子 受体酪氨酸激酶
酪氨酸激酶抑制剂 细胞周期 细胞凋亡 肿瘤抑制基因 癌基因
奥希替尼 吉非替尼 厄洛替尼 克唑替尼 阿来替尼 贝伐珠单抗
曲妥珠单抗 帕博利珠单抗 纳武利尤单抗 阿替利珠单抗
紫杉醇 顺铂 卡铂 培美曲塞 多西他赛 卡培他滨
辅助治疗 新辅助治疗 姑息治疗 同步放化疗 立体定向放射治疗
质子治疗 重离子治疗 骨髓抑制 放射性肺炎 免疫相关不良反应
EGFR ALK ROS1 HER2 PD-1 PD-L1 CTLA-4 KRAS BRAF CAR-T'''.split()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=next(ROOT.glob('【视频播客】*.mp3'))
    if (OUT/'manifest.json').exists():
        if json.loads((OUT/'manifest.json').read_text())['dataset_sha256']!=digest(source):
            raise ValueError('Podcast changed; use a fresh output directory')
        return
    import soundfile as sf
    (OUT/'audio').mkdir(exist_ok=True)
    full=OUT/'audio/full.wav'
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-ac','1','-ar','16000','-c:a','pcm_f32le',str(full)],check=True)
    wave,rate=sf.read(full,dtype='float32');assert rate==16000
    samples=[]
    for index,start in enumerate(range(0,len(wave),60*rate)):
        chunk=wave[start:start+60*rate];rid=f'oncology_podcast_{index:04d}'
        path=OUT/'audio'/f'{rid}.wav';sf.write(path,chunk,rate,subtype='FLOAT')
        samples.append(dict(id=rid,row=index,reference='',audio_path=str(path),
            start_seconds=start/rate,end_seconds=(start+len(chunk))/rate,
            duration_seconds=len(chunk)/rate,pcm_sha256=digest_pcm(chunk)))
    write_json(OUT/'manifest.json',dict(dataset=str(source),dataset_sha256=digest(source),
        sample_count=len(samples),duration_seconds=len(wave)/rate,samples=samples,
        no_reference=True,segmentation='Contiguous 60-second windows; no overlap or filtering',
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'system').rglob('*.py')},
        formal_db_sha256=digest('/home/aim0/桌面/tan/ASR/AgenticASR_entity_e5bedb3/data/entities.db')))
    # Independent copy of the previous 42-term evaluation library, no changes.
    medical=OUT/'medical_library';medical.mkdir(exist_ok=True)
    for name in ['entities.db','entities.json','provenance.json','sample_entities.json','vocabulary_evidence.json']:
        shutil.copy2(MEDICAL/name,medical/name)
    ONCOLOGY.mkdir(parents=True,exist_ok=True)
    if (ONCOLOGY/'entities.db').exists():raise FileExistsError('Refuse to overwrite an existing oncology glossary')
    store=EntityStore(ONCOLOGY/'entities.db')
    definitions={r['canonical_text']:r for r in json.loads((MEDICAL/'entities.json').read_text())}
    for name in TOPIC_TERMS:
        definitions.setdefault(name,dict(canonical_text=name,entity_type='TERM',domain='medical',
            normalization_policy='normalize',priority=0,enabled=True,aliases=[],
            source='Predeclared cancer podcast topic vocabulary; not extracted from recognition'))
    oncology_context=tuple(dict.fromkeys((*CONTEXT_WORDS,'癌症','肿瘤','癌细胞','化疗','放疗','靶向','免疫','药物','基因','分期','转移')))
    for name,d in sorted(definitions.items()):
        entity=store.upsert_entity(name,entity_type='TERM',domain='medical',normalization_policy='normalize',source=d['source'])
        if re.fullmatch(r'[\u3400-\u9fff]{3,16}',name):
            keywords={w:1.0 for w in oncology_context if w not in name and name not in w}
            store.set_spelling_context(entity.entity_id,keywords=keywords,minimum_score=1.0,source='Fixed cancer scene before ASR; no near/omission permission')
            d['spelling_context']=dict(keywords=keywords,minimum_score=1.0,allow_internal_omission=False,allow_near_spelling=False)
    write_json(ONCOLOGY/'entities.json',[definitions[n] for n in sorted(definitions)])
    write_json(ONCOLOGY/'provenance.json',dict(vocabulary_size=len(definitions),
        construction='Prior 42 medical terms plus predeclared title-topic oncology terms; before baseline',
        dataset_sha256=digest(source),reference_at_inference=False,error_aliases=False,asr_context=False,
        frozen_hashes={name:digest(ONCOLOGY/name) for name in ['entities.db','entities.json']}))
    print('PREPARED',len(samples),'windows',len(definitions),'oncology terms',flush=True)


def digest_pcm(wave):
    import hashlib
    return hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()


def report():
    import csv
    import difflib
    manifest=json.loads((OUT/'manifest.json').read_text());ids=[r['id'] for r in manifest['samples']]
    rows={k:require_complete(OUT/f'{k}.jsonl',ids) for k in ['baseline','control_replay','medical_library','entity_library']}
    totals={};comparisons=[]
    for k,records in rows.items():
        normalizations=[n for r in records.values() for n in r['result'].get('entity_normalizations',[])]
        candidates=[c for r in records.values() for c in r['result'].get('entity_candidates',[])]
        protected=[e for r in records.values() for e in r['result'].get('protected_entities',[])
            if str(e.get('source','')).startswith('database:')]
        totals[k]=dict(samples=len(records),refiner_calls=sum(len(r['model_calls']) for r in records.values()),
            changed_samples=sum(transcript(r,'clean_text')!=transcript(rows['control_replay'][rid],'clean_text') for rid,r in records.items()),
            entity_normalization_events=len(normalizations),candidate_decisions=dict(Counter(c['decision'] for c in candidates)),
            case_normalization_events=sum(n['observed'].casefold()==n['canonical'].casefold() for n in normalizations),
            database_protection_events=len(protected),database_hit_names=sorted({e['replacement'] for e in protected}))
    for sample in manifest['samples']:
        rid=sample['id'];base=rows['baseline'][rid]
        for k in rows:
            r=rows[k][rid]
            if r['pcm_sha256']!=base['pcm_sha256'] or r['chunk_payloads']!=base['chunk_payloads'] or r['final_payload']!=base['final_payload']:
                raise ValueError(f'{rid}: unequal ASR or audio inputs')
        comparisons.append(dict(id=rid,start_seconds=sample['start_seconds'],end_seconds=sample['end_seconds'],
            raw=transcript(base,'raw_text'),texts={k:transcript(rows[k][rid],'clean_text') for k in rows},
            normalizations={k:rows[k][rid]['result'].get('entity_normalizations',[]) for k in rows},
            candidates={k:rows[k][rid]['result'].get('entity_candidates',[]) for k in rows}))
    write_json(OUT/'summary.json',totals);write_json(OUT/'comparison.json',comparisons)
    text_changes=[]
    for r in comparisons:
        before=r['texts']['control_replay']
        for k in ['medical_library','entity_library']:
            after=r['texts'][k]
            if before==after:continue
            edits=[dict(before=before[i:j],after=after[a:b],context=before[max(0,i-15):min(len(before),j+15)])
                for op,i,j,a,b in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if op!='equal']
            text_changes.append(dict(id=r['id'],condition=k,start_seconds=r['start_seconds'],end_seconds=r['end_seconds'],edits=edits))
    write_json(OUT/'text_changes.json',text_changes)
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['id','start_seconds','end_seconds','raw','baseline','control_replay','medical_42','oncology'])
        for r in comparisons:writer.writerow([r['id'],r['start_seconds'],r['end_seconds'],r['raw'],*r['texts'].values()])
    lines=['# 癌症视频播客：当前实体库整段对照','',
        f"原音 {manifest['duration_seconds']/60:.2f} 分钟，全部 {len(ids)} 个连续 60 秒窗口。无标准字幕，不计算 CER／实体准确率。",'',
        '两组词库均在首遍识别前冻结；所有回放使用相同真实 ASR 响应、真实精修模型。词库辅助识别关闭，当前纠正规则未修改。', '',
        '| 条件 | 窗口数 | 命中不同库名 | 最终文字变化（相对无库回放） | 实体规范事件 | 精修调用 |','|---|---:|---:|---:|---:|---:|']
    labels={'baseline':'无库首次转录','control_replay':'无库回放控制','medical_library':'此前 42 词医疗库','entity_library':f"肿瘤专题库 {json.loads((ONCOLOGY/'provenance.json').read_text())['vocabulary_size']} 词"}
    for k,s in totals.items():lines.append(f"| {labels[k]} | {s['samples']} | {len(s['database_hit_names'])} | {s['changed_samples']} | {s['entity_normalization_events']} | {s['refiner_calls']} |")
    s=totals['entity_library']
    if s['entity_normalization_events'] and s['entity_normalization_events']==s['case_normalization_events']:
        lines+=['','专题库的实体规范记录全部为大小写统一，不能视为实体识别纠错。']
    esc=lambda s:str(s).replace('|','\\|').replace('\n',' ')
    lines+=['','## 最终文字变化','', '| 时间窗 | 条件 | 无库回放 | 有库 |','|---|---|---|---|']
    for r in comparisons:
        for k in ['medical_library','entity_library']:
            if r['texts'][k]!=r['texts']['control_replay']:
                lines.append(f"| {r['start_seconds']:.0f}–{r['end_seconds']:.0f} 秒 | {labels[k]} | {esc(r['texts']['control_replay'])} | {esc(r['texts'][k])} |")
    lines+=['','## 实体替换记录（含中间过程）','', '| 时间窗 | 条件 | 替换记录 |','|---|---|---|']
    for r in comparisons:
        for k in ['medical_library','entity_library']:
            if r['normalizations'][k]:lines.append(f"| {r['start_seconds']:.0f}–{r['end_seconds']:.0f} 秒 | {labels[k]} | {esc(json.dumps(r['normalizations'][k],ensure_ascii=False))} |")
    lines+=['','## 模糊候选决策','', '| 时间窗 | 条件 | 原文 | 候选 | 决策 | 原因 |','|---|---|---|---|---|---|']
    for r in comparisons:
        for k in ['medical_library','entity_library']:
            seen=set()
            for c in r['candidates'][k]:
                signature=(c['observed'],c['canonical'],c['decision'],tuple(c.get('reasons',[])))
                if signature in seen:continue
                seen.add(signature)
                lines.append(f"| {r['start_seconds']:.0f}–{r['end_seconds']:.0f} 秒 | {labels[k]} | {esc(c['observed'])} | {esc(c['canonical'])} | {c['decision']} | {esc(', '.join(c.get('reasons',[])))} |")
    lines+=['','事件数可能包含对中间识别结果的重复处理，不能当作正确纠错次数；词库中出现更多标准名称也不能自动算准确率提升。', '',
        '全部文字及审计记录见 comparison.csv、comparison.json 和各条件 JSONL；紧凑文字差异见 text_changes.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for k in rows:
        text='\n\n'.join(f"[{r['start_seconds']:.0f}–{r['end_seconds']:.0f}s]\n{r['texts'][k]}" for r in comparisons)
        (OUT/f'{k}.txt').write_text(text+'\n')
    write_json(OUT/'scope_verification.json',dict(
        production_code_unchanged=all(digest(ROOT/p)==v for p,v in manifest['source_hashes'].items()),
        formal_library_unchanged=digest('/home/aim0/桌面/tan/ASR/AgenticASR_entity_e5bedb3/data/entities.db')==manifest['formal_db_sha256'],
        paired_audio_and_asr_identical=True))
    print(json.dumps(totals,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','report']);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.stage in {'prepare','run'}:prepare()
    if args.stage=='run':
        settings=SimpleNamespace(output=OUT,library=OUT/'medical_library',model=MODEL,device=args.device,
            asr_url='http://127.0.0.1:8766',mode='streaming',chunk_seconds=2.0)
        runner=Runner(settings)
        for condition in ['baseline','control_replay','medical_library']:runner.run(condition)
        settings.library=ONCOLOGY;runner.run('entity_library')
    if args.stage in {'run','report'}:report()


if __name__=='__main__':main()
