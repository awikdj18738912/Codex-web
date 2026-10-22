"""Merge independent reviews with explicit lead adjudications; export proposals only."""
import csv,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return [json.loads(l) for l in (ROOT/name).read_text().splitlines() if l.strip()]
def write(name,rows):
    (ROOT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

proposals={r['case_no']:r for r in read('teacher_proposals.jsonl')}
keys={r['case_no']:r for r in read('blind_key.jsonl')}
reviews=sum((read(name) for name in ['blind_review_first20.jsonl','blind_review_021_060.jsonl','blind_review_061_100.jsonl']),[])
assert len(reviews)==len({r['case_no'] for r in reviews})==97

# Grades distinguish stale correction narration (MINOR) from wrong final assertions (FAIL).
# Conflicting interpretations of artificial input are quarantined rather than assigned a gold label.
OVERRIDES={
 'P005':({'old':'PASS','proposal':'PASS'},'TIE','拼音仅辅助实体识别，可删除。'),
 'P011':({'old':'AMBIGUOUS','proposal':'AMBIGUOUS'},'UNRESOLVED','记错地址可指回忆纠正，也可能指填写错误；审查者分歧，隔离而不强造填写历史。'),
 'P017':({'old':'MINOR','proposal':'MINOR'},'TIE','旧目标仍叙述被撤回的回滚提议，但未用过去完成时断言已执行；列清理不全而非确定事实错误。'),
 'P020':({'old':'MINOR','proposal':'MINOR'},'TIE','Sony明确写成记混而非真实购买；存在废弃候选残留。'),
 'P023':({'old':'MINOR','proposal':'PASS'},'proposal','旧目标明确纠正为R²，属于改口未收敛，不应把否定过的旧值当作最终肯定。'),
 'P027':({'old':'MINOR','proposal':'MINOR'},'TIE','候选最后明确林明远；前面的谢昭然与拼字是未清理改口，不是最终人物错误。'),
 'P030':({'old':'MINOR','proposal':'MINOR'},'proposal','新目标删重启但仍加入403检查及条件顺序，未完全收敛到最后的先确认状态再helm更新；不纳入可用替换。'),
 'P034':({'old':'AMBIGUOUS','proposal':'AMBIGUOUS'},'UNRESOLVED','系统更新叙述后又说还是先说修鞋，无法确认是暂缓独立话题还是撤回话题；旧目标还用所以牵连。隔离。'),
 'P039':({'old':'MINOR','proposal':'MINOR'},'TIE','docker计划后明确说搞混了且最后选择tail；是废弃命令残留，最终操作未被反转。'),
 'P057':({'old':'MINOR','proposal':'PASS'},'proposal','旧金额被明确不对否定，最终金额正确；旧标清理不全，新标删废弃数值。'),
 'P059':({'old':'MINOR','proposal':'MINOR'},'proposal','旧目标明确纠正1.9%为广告点击率；新目标解决该残留但仍重复误报率5%以上。'),
 'P062':({'old':'FAIL','proposal':'MINOR'},'proposal','新目标修好并列关系，仍保留已撤回的腾讯新闻开头；候选需进一步清理。'),
 'P073':({'old':'AMBIGUOUS','proposal':'AMBIGUOUS'},'UNRESOLVED','原文明确把IFS纠正成听雨轩，末尾又另提天台咖啡屋；不能凭地理常识强断设施归属，隔离。'),
}
final=[];adjudications=[]
for r in reviews:
    r=dict(r);k=keys[r['case_no']]
    grades={k['a_is']:r['a_verdict'],k['b_is']:r['b_verdict']}
    winner={'A':k['a_is'],'B':k['b_is']}.get(r['winner'],r['winner'])
    if r['case_no'] in OVERRIDES:
        grades,winner,note=OVERRIDES[r['case_no']]
        adjudications.append({'case_no':r['case_no'],'scope':'pilot','original_review':r,'final_grades':grades,'final_winner':winner,'reason':note})
    else:note=None
    final.append({**r,'old_verdict':grades['old'],'proposal_verdict':grades['proposal'],'winner_identity':winner,
                  'lead_note':note,'teacher_decision':k['teacher_decision'],'is_control':k['is_control']})
write('pilot_review_final.jsonl',final)

ckeys={r['case_no']:r for r in read('confirmed_blind_key.jsonl')}
confirmed=read('confirmed_repair_proposals.jsonl')
creviews=read('confirmed_blind_review.jsonl')
# Main agent inspected all eight full pairs, including cases where the second reviewer read
# a retracted address as actual history or counted raw copied disfluencies as clean.
CFINAL={
 'P001':('MINOR','PASS','repair','最后意图收敛，撤回话题已移除。'),
 'P003':('FAIL','PASS','repair','补回独立网课经历，无额外事实。'),
 'P004':('FAIL','MINOR','repair','保留暂缓意思但回添大量停顿/填充词，暂不替换。'),
 'P011':('AMBIGUOUS','AMBIGUOUS','UNRESOLVED','对原先填写地址的指代审查分歧，且修订回添填充词，隔离。'),
 'P013':('FAIL','MINOR','repair','删除无依据角色但复制大段未清理口语，暂不替换。'),
 'P017':('MINOR','MINOR','old','修订几乎完整复制原口语，回滚撤回仍未清理，旧版清理较充分。'),
 'P019':('FAIL','PASS','repair','恢复上上个月，移除从订单号推断的日期。'),
 'P020':('MINOR','MINOR','TIE','删Sony候选但回添啊那个那个等填充与重复，不接受为最终标签。'),
}
cfinal=[]
for r in creviews:
    old,new,winner,note=CFINAL[r['case_no']]
    cfinal.append({**r,'old_verdict':old,'repair_verdict':new,'winner_identity':winner,'lead_note':note})
    adjudications.append({'case_no':r['case_no'],'scope':'confirmed_defect_repair','original_review':r,
                         'final_grades':{'old':old,'repair':new},'final_winner':winner,'reason':note})
write('confirmed_review_final.jsonl',cfinal)
write('lead_adjudications.jsonl',adjudications)

# Explicit allowlist after full source/old/new inspection; do not mass-accept teacher KEEP.
selected_pilot={'P023','P041','P054','P057','P068'}
selected_confirmed={'P001','P003','P019'}
accepted=[]
for r in final:
    if r['case_no'] in selected_pilot:
        row=proposals[r['case_no']]
        accepted.append({'case_no':r['case_no'],'id':row['id'],'scene':row['scene'],'input':row['input'],
                         'old_target':row['existing_target'],'proposed_target':row['proposal']['clean_text'],
                         'method':'teacher_target_review','review_note':r['reason'],'status':'verified_improvement_pending_application'})
for row in confirmed:
    if row['case_no'] in selected_confirmed:
        accepted.append({'case_no':row['case_no'],'id':row['id'],'scene':row['scene'],'input':row['input'],
                         'old_target':row['existing_target'],'proposed_target':row['clean_text'],
                         'method':'independent_defect_then_teacher_repair','review_note':row['confirmed_defect'],
                         'status':'verified_improvement_pending_application'})
write('verified_improvements_8.jsonl',sorted(accepted,key=lambda r:r['case_no']))
with (ROOT/'verified_improvements_8.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['case_no','id','scene','input','old_target','proposed_target','method','status']
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(accepted)

def tally(rows,field):return dict(Counter(r[field] for r in rows))
summary={'screened_train_records':84875,'heuristic_risk_records':13412,'pilot_records':100,
         'teacher_request_errors_or_schema_conflicts':sum(bool(r['error']) for r in proposals.values()),
         'teacher_valid_decisions':dict(Counter(r['proposal']['decision'] for r in proposals.values() if not r['error'])),
         'independently_compared':len(final),'old_verdicts':tally(final,'old_verdict'),
         'proposal_verdicts':tally(final,'proposal_verdict'),'paired_winners':tally(final,'winner_identity'),
         'controls':{'records':sum(r['is_control'] for r in proposals.values()),
                     'changed_targets':sum(bool(r['is_control'] and r['proposal'] and r['proposal'].get('clean_text')!=r['existing_target']) for r in proposals.values())},
         'confirmed_defect_followup':{'records':8,'old_verdicts':tally(cfinal,'old_verdict'),
                                     'repair_verdicts':tally(cfinal,'repair_verdict'),'paired_winners':tally(cfinal,'winner_identity')},
         'verified_improvements':len(accepted),'production_dataset_modified':False,
         'limitations':['Risk-enriched diagnostic pilot, not an estimate of dataset error prevalence.',
                        'Independent reviewers disagree on some artificial correction histories; source ambiguity is quarantined.',
                        'Verified improvements are isolated candidates, not a newly qualified full training corpus.',
                        'One first-stage JSON failure did not retain raw response; later requests preserve raw on parse errors.',
                        'No student retraining or final unseen-test claim in this experiment.']}
(ROOT/'final_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
