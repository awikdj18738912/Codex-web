"""Assemble blinded GPT audit and unmask only after all review files are final."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(name, rows):
    (ROOT / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

cases = {row['case_no']: row for row in read_jsonl(ROOT / 'blinded_cases_256.jsonl')}
keys = {row['case_no']: row for row in read_jsonl(ROOT / 'model_identity_key.jsonl')}
review = {}
for start in (1, 65, 129, 193):
    end = start + 63
    rows = read_jsonl(ROOT / f'adjudicated_{start:03d}_{end:03d}.jsonl')
    assert len(rows) == 64
    for row in rows:
        n = row['case_no']
        assert start <= n <= end and n not in review
        assert row['id'] == cases[n]['id'] == keys[n]['id']
        assert row['source_status'] in {'CLEAR', 'AMBIGUOUS', 'UNDERDETERMINED'}
        for label in ('reference', 'A', 'B'):
            text = cases[n]['reference'] if label == 'reference' else cases[n]['output_' + label]
            judged = row[label]
            assert judged['verdict' if label == 'reference' else 'overall'] in {'PASS', 'MINOR', 'FAIL', 'UNJUDGEABLE'}
            for issue in judged.get('issues', []):
                assert issue['quote'] in text, (n, label, issue)
            if judged['verdict' if label == 'reference' else 'overall'] == 'FAIL':
                assert judged['issues'], (n, label, 'FAIL without evidence')
        review[n] = row
assert set(review) == set(range(1, 257))
# Policy adjudication uses the actual system prompt and occurs before unmasking.
policy_path = ROOT / 'policy_adjudication_proposals.jsonl'
assert policy_path.exists(), 'Policy adjudication must finish before assembly'
policy_overrides = read_jsonl(policy_path)
for proposed in policy_overrides:
    replacement = proposed.get('replacement', proposed)
    n = replacement['case_no']
    assert replacement['id'] == cases[n]['id']
    assert all(k in replacement for k in ('source_status','reference','A','B','winner','winner_reason'))
    review[n] = replacement
for n, row in review.items():
    for label in ('reference','A','B'):
        original_text = cases[n]['reference'] if label == 'reference' else cases[n]['output_' + label]
        for evidence in row[label].get('issues', []):
            assert evidence['quote'] in original_text, (n,label,evidence)

merged, curated, queue, changes = [], [], [], []
for n in range(1, 257):
    c, r, key = cases[n], review[n], keys[n]
    record = {**c, 'audit': r, 'model_identity': {'A': key['A'], 'B': key['B']}}
    for label in ('A', 'B'):
        record[key[label]] = {'output': c['output_' + label], **r[label]}
    winner = r['winner']
    record['winner_unmasked'] = key[winner] if winner in ('A', 'B') else winner
    ref = r['reference']
    if r['source_status'] == 'CLEAR' and ref['verdict'] == 'PASS':
        reference = c['reference']
        origin = 'original_reference_gpt_accepted'
    elif r['source_status'] == 'CLEAR' and ref['verdict'] in ('MINOR', 'FAIL') and ref.get('proposed_reference'):
        reference = ref['proposed_reference']
        origin = 'gpt_revised_reference'
    else:
        reference = None
        origin = 'quarantined_for_ambiguity_or_unresolved_reference'
    record['curation_status'] = origin
    record['curated_reference'] = reference
    merged.append(record)
    if reference is not None:
        curated.append({'id': c['id'], 'scene': c['scene'], 'input': c['input'], 'reference': reference,
                        'reference_origin': origin, 'original_reference': c['reference'],
                        'review_reason': ref['reason'], 'human_validated': False,
                        'split_role': 'development_validation_previously_used_for_checkpoint_selection'})
        if reference != c['reference']:
            changes.append({'case_no': n, 'id': c['id'], 'input': c['input'], 'original_reference': c['reference'],
                            'revised_reference': reference, 'previous_reference_verdict': ref['verdict'], 'reason': ref['reason']})
    else:
        queue.append(record)

clear = [r for r in merged if r['audit']['source_status'] == 'CLEAR']
counts = lambda rows, fn: dict(Counter(fn(row) for row in rows))

def model_stats(rows, model):
    return {'denominator': len(rows), 'overall': counts(rows, lambda r: r[model]['overall']),
            'meaning': counts(rows, lambda r: r[model]['meaning']),
            'cleanup': counts(rows, lambda r: r[model]['cleanup']),
            'fidelity': counts(rows, lambda r: r[model]['fidelity']),
            'any_meaning_or_fidelity_FAIL': sum(r[model]['meaning'] == 'FAIL' or r[model]['fidelity'] == 'FAIL' for r in rows),
            'PASS_or_MINOR': sum(r[model]['overall'] in ('PASS', 'MINOR') for r in rows)}

controls = [r for r in clear if r['input'] == r['reference'] and r['audit']['reference']['verdict'] == 'PASS']
report = {
    'policy_adjusted_records': len(policy_overrides), 'human_validated': False, 'actual_system_prompt_sha256': hashlib.sha256((ROOT/'actual_refiner_system_prompt.txt').read_bytes()).hexdigest(), 'records': len(merged), 'source_status': counts(merged, lambda r: r['audit']['source_status']),
    'reference_verdict': counts(merged, lambda r: r['audit']['reference']['verdict']),
    'curated_development_records': len(curated), 'revised_reference_records': len(changes), 'quarantined_records': len(queue),
    'comparison_all': {m: model_stats(merged, m) for m in ('base', 'epoch2')},
    'comparison_clear_source_only': {m: model_stats(clear, m) for m in ('base', 'epoch2')},
    'preference_all': counts(merged, lambda r: r['winner_unmasked']),
    'preference_clear_source_only': counts(clear, lambda r: r['winner_unmasked']),
    'verified_original_identity_controls': {'records': len(controls), **{m: {
        'text_changed': sum(r[m]['output'] != r['input'] for r in controls),
        'meaning_or_fidelity_FAIL': sum(r[m]['meaning'] == 'FAIL' or r[m]['fidelity'] == 'FAIL' for r in controls)
    } for m in ('base', 'epoch2')}},
    'limitations': [
        'GPT judgments and proposed references are not independently human-validated gold labels.',
        'The 256 examples are scene-balanced synthetic validation samples, not a prevalence sample of production traffic.',
        'This validation split was already used for checkpoint selection; it is a development audit, not an unseen final test.',
        'All 256 inputs equal their linked generated oral_text. This audit does not establish robustness to real ASR errors.',
        'Punctuation, numeric formatting and reasonable paraphrases are not semantic errors; altered values and identifiers are errors.',
        'CLEAR-only statistics exclude ambiguous sources. FAIL may mean cleanup failure, factual/meaning error, or identifier loss; dimension counts must be read separately.',
        'Teacher-model differences and 4bit quantization effects have not been isolated in a controlled comparison.'
    ],
    'files': {'audit': str(ROOT/'audit_256.jsonl'), 'curated_dev': str(ROOT/'curated_development_256_subset.jsonl'),
              'queue': str(ROOT/'manual_review_queue.jsonl'), 'changes': str(ROOT/'reference_changes.jsonl')},
    'policy_adjustments_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    'review_sha256': {f'adjudicated_{s:03d}_{s+63:03d}.jsonl': hashlib.sha256((ROOT/f'adjudicated_{s:03d}_{s+63:03d}.jsonl').read_bytes()).hexdigest() for s in (1,65,129,193)}
}
write_jsonl('audit_256.jsonl', merged)
write_jsonl('curated_development_256_subset.jsonl', curated)
write_jsonl('manual_review_queue.jsonl', queue)
write_jsonl('reference_changes.jsonl', changes)
(ROOT/'audit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

columns = ['case_no','id','scene','source_status','reference_verdict','reference_reason','original_reference','curated_reference',
           'base_overall','base_meaning','base_cleanup','base_fidelity','base_reason','base_output',
           'epoch2_overall','epoch2_meaning','epoch2_cleanup','epoch2_fidelity','epoch2_reason','epoch2_output',
           'winner','winner_reason','input']
flat=[]
for r in merged:
    item={'case_no':r['case_no'],'id':r['id'],'scene':r['scene'],'source_status':r['audit']['source_status'],
          'reference_verdict':r['audit']['reference']['verdict'],'reference_reason':r['audit']['reference']['reason'],
          'original_reference':r['reference'],'curated_reference':r['curated_reference'],
          'winner':r['winner_unmasked'],'winner_reason':r['audit']['winner_reason'],'input':r['input']}
    for m in ('base','epoch2'):
        for k in ('overall','meaning','cleanup','fidelity','reason','output'):
            item[m+'_'+k]=r[m][k]
    flat.append(item)
with (ROOT/'audit_256.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(flat)

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
wb=Workbook();ws=wb.active;ws.title='逐条审核';ws.append(columns)
for r in flat:ws.append([r.get(k) for k in columns])
ws.freeze_panes='D2';ws.auto_filter.ref=ws.dimensions
for j,key in enumerate(columns,1):
    ws.column_dimensions[get_column_letter(j)].width=75 if key in ('input','original_reference','curated_reference','base_output','epoch2_output') else (52 if 'reason' in key else 20)
for cell in ws[1]:cell.fill=PatternFill('solid',fgColor='1F4E78');cell.font=Font(bold=True,color='FFFFFF')
for row in ws.iter_rows(min_row=2):
    ws.row_dimensions[row[0].row].height=70
    for cell in row:
        cell.alignment=Alignment(vertical='top',wrap_text=True)
        color={'PASS':'D9EAD3','MINOR':'FFF2CC','FAIL':'F4CCCC','UNJUDGEABLE':'FCE5CD','AMBIGUOUS':'FCE5CD'}.get(str(cell.value))
        if color:cell.fill=PatternFill('solid',fgColor=color)
summary=wb.create_sheet('汇总');summary.append(['项目','结果'])
for key in ['records','source_status','reference_verdict','curated_development_records','revised_reference_records','quarantined_records','preference_clear_source_only','verified_original_identity_controls']:
    summary.append([key,json.dumps(report[key],ensure_ascii=False)])
for m in ('base','epoch2'):summary.append([m+' CLEAR统计',json.dumps(report['comparison_clear_source_only'][m],ensure_ascii=False)])
for note in report['limitations']:summary.append(['限制',note])
summary.column_dimensions['A'].width=40;summary.column_dimensions['B'].width=130
for row in summary:
    for cell in row:cell.alignment=Alignment(vertical='top',wrap_text=True)
q=wb.create_sheet('待人工确认');q.append(['case_no','id','input','original_reference','source_status','reason'])
for r in queue:q.append([r['case_no'],r['id'],r['input'],r['reference'],r['audit']['source_status'],r['audit']['reference']['reason']])
q.freeze_panes='A2';q.auto_filter.ref=q.dimensions
for j,width in enumerate([12,28,95,95,22,80],1):q.column_dimensions[get_column_letter(j)].width=width
for row in q:
    for cell in row:cell.alignment=Alignment(vertical='top',wrap_text=True)
change_sheet=wb.create_sheet('参考修订');change_cols=['case_no','id','input','original_reference','revised_reference','previous_reference_verdict','reason'];change_sheet.append(change_cols)
for r in changes:change_sheet.append([r[k] for k in change_cols])
change_sheet.freeze_panes='A2';change_sheet.auto_filter.ref=change_sheet.dimensions
for j in range(1,len(change_cols)+1):change_sheet.column_dimensions[get_column_letter(j)].width=90 if j in (3,4,5,7) else 25
for row in change_sheet:
    for cell in row:cell.alignment=Alignment(vertical='top',wrap_text=True)
wb.save(ROOT/'audit_256.xlsx')

lines=['# 256条验证输出 GPT盲审','',
       '模型身份逐条随机分配为A/B，分段审查与主审复核结束后再解码。下面判断为GPT审核结果，尚未由独立人工标注验证。', '',
       f"共{len(merged)}条；源意图可确定{len(clear)}条；隔离{len(queue)}条。导出开发参考{len(curated)}条，其中修订{len(changes)}条。",'',
       '## 可确定源文上的比较','',
       '| 模型 | PASS | MINOR | FAIL | 无法判断 | 语义或标识保真FAIL |',
       '|---|---:|---:|---:|---:|---:|']
for m in ('base','epoch2'):
    s=report['comparison_clear_source_only'][m];v=s['overall']
    lines.append(f"| {m} | {v.get('PASS',0)} | {v.get('MINOR',0)} | {v.get('FAIL',0)} | {v.get('UNJUDGEABLE',0)} | {s['any_meaning_or_fidelity_FAIL']} |")
lines+=['', '偏好计数：'+json.dumps(report['preference_clear_source_only'],ensure_ascii=False), '',
        'FAIL包括清理不足与事实/实体损坏，不能把总FAIL率都解释为语义错误。原样复制通常保留事实但可能无法去除口癖和改口。', '',
        '## 原参考判定','',json.dumps(report['reference_verdict'],ensure_ascii=False), '',
        '## 使用范围','',
        '这些样本属于已用于选择checkpoint的合成验证集，适合开发比较，不是全新测试集。256条input均与对应oral_text相同，本轮不能推断真实ASR错误修复表现。修订版仅导出源意图明确且有完整参考的记录；歧义项进入人工队列，不猜补事实。', '',
        '## 文件','',
        '- audit_256.xlsx：审核、汇总、参考修订、人工队列。',
        '- audit_256.jsonl：全部原文、匿名输出、解码身份和逐项证据。',
        '- curated_development_256_subset.jsonl：可用于下一轮开发对照的GPT复核参考。',
        '- manual_review_queue.jsonl：待人工澄清，当前不计入CLEAR比较。',
        '- reference_changes.jsonl：完整参考修订记录。',
        '- audit_report.json：机器可读统计和限制。', '',
        '本次未重跑模型推理或训练，未修改原训练/验证数据文件。']
(ROOT/'audit_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['records','source_status','reference_verdict','curated_development_records','revised_reference_records','quarantined_records','comparison_clear_source_only','preference_clear_source_only','verified_original_identity_controls']},ensure_ascii=False,indent=2))
