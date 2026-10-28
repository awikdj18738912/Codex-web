"""Clarify overlapping protection dimensions; preserve original audit evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path
import time

import audit_refiner_50k_sample as audit


CLARIFICATION = '''
最高优先级的类别澄清（之前审核可能混淆了分类轴，必须按下面判断）：
E03/E05/E06是保护型训练维度，可与A01口癖清理同时出现，不要求修改被保护的内容。
E03：input含否定、条件、情态或立场，reference保持该意义，可以完全不修改，也可以只删除明确无意义口癖。不能因为只删口癖就判类别错；只有语义未体现本类或错误清理才标记。
E05：包含正确英文名称或型号且reference原样保留，允许只清理口癖。这正是本类目标。英文名称未修改是正确行为，绝不能作为category_correct=false的理由，也不能仅因删口癖改判A01。
E06：实体占位符原样保留、顺序不变，允许只清理周围口癖或口吃。不能仅因修改属于口癖就改判A01。
这三类不要求原文中的否定、英文或实体有错误！拒绝给它们虚构“必须纠正英文或否定”的要求。
仍须严格检查场景自然性、目标正确、最小修改和原意保留。有意义的指代不可删除。软件、库名、模型名不能毫无说明地当作可搬运物品、衣料或宠物；不能靠臆造品牌、绰号或隐喻替不自然样本辩护。根据实际文字判断。
'''


def main():
    root=Path('/home/aim0/data/datasets/ASR/refiner_50k_sample_audit_20261026')
    while True:
        r=json.loads((root/'progress.json').read_text())
        if r['status']=='complete':break
        time.sleep(5)
    config=json.loads((root/'config.json').read_text())
    rows=[json.loads(l) for l in (root/'sample.jsonl').open()]
    rows=[r for r in rows if r['category'] in ['E03','E05','E06']]
    out=root/'protection_clarification';out.mkdir(exist_ok=True)
    audit.PROMPT += CLARIFICATION
    config.update(system_prompt=audit.PROMPT, sample_total=len(rows), per_category=30,
                  sample_by_source=dict(Counter(r['dataset'] for r in rows)),
                  revision_reason='Protection dimensions overlap cleanup; original auditor incorrectly required protected text to change',
                  revised_categories=['E03','E05','E06'],original_audit_preserved=True,
                  script_sha256=audit.digest(Path(__file__)))
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (out/'script.py').write_bytes(Path(__file__).read_bytes())
    records=[];path=out/'results.jsonl'
    if path.exists():records=[json.loads(l) for l in path.open()]
    seen={r['id'] for r in records}
    with path.open('a') as stream, ThreadPoolExecutor(max_workers=8) as pool:
        pending=[pool.submit(audit.review,row,config,out) for row in rows if row['id'] not in seen]
        for future in as_completed(pending):
            r=future.result();records.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            audit.save_summary(out,config,records)
            print(r['id'],r['status'],flush=True)
    audit.save_summary(out,config,records,final=True)
    revised={r['id']:r for r in records}
    combined=[revised.get(r['id'],r) for r in map(json.loads,(root/'results.jsonl').open())]
    final=root/'final_review';final.mkdir(exist_ok=True)
    original=json.loads((root/'config.json').read_text())
    (final/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined))
    audit.save_summary(final,original,combined,final=True)
    (final/'provenance.json').write_text(json.dumps({'original_results':str(root/'results.jsonl'),
        'revised_results':str(path),'revised_categories':['E03','E05','E06'],
        'reason':config['revision_reason'],'same_teacher_second_pass':True,
        'production_dataset_modified':False},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
