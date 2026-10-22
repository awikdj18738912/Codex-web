import json
from collections import Counter,defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (OUT/'attempts.jsonl').read_text().splitlines() if x.strip()]
retained=[json.loads(x) for x in (OUT/'retained.jsonl').read_text().splitlines() if x.strip()]
summary=json.loads((OUT/'summary.json').read_text())
serials=[r['serial'] for r in rows]
status=Counter(r['status'] for r in rows);by_scene=defaultdict(Counter);by_phen=defaultdict(Counter);boundary_scene=defaultdict(Counter);boundary=Counter();audit_reason=Counter();issue_kinds=Counter();unsupported_matches=[]
for r in rows:
 st=r['status'];by_scene[r['scene']][st]+=1
 oral=r.get('oral') or {};labels=oral.get('speech_phenomena',[]);labels=labels if isinstance(labels,list) else [labels]
 for label in labels:by_phen[str(label)][st]+=1
 if 'boundary_punctuation' in labels:boundary[st]+=1;boundary_scene[r['scene']][st]+=1
 if st=='audit_rejected':
  reason=r.get('reason','');prefix=reason.split(':',2)[1] if reason.startswith('audit_failed:') and ':' in reason else reason.split(':',1)[0]
  audit_reason[prefix]+=1
  qa=r.get('quality_audit') or {}
  for issue in qa.get('issues',[]) if isinstance(qa,dict) else []:issue_kinds[str(issue.get('kind','unknown'))]+=1
  if 'unsupported_content_edits' in reason or 'unsupported_content_edits' in json.dumps(qa,ensure_ascii=False):unsupported_matches.append(r['id'])
retained_scene=Counter(x['meta']['scene'] for x in retained)
retained_change=Counter('different' if x['input']!=x['output']['refined_text'] else 'same' for x in retained)
report={'candidate_count':len(rows),'unique_serials':len(set(serials)),'serial_range':[min(serials),max(serials)],'missing_serials':sorted(set(range(200))-set(serials)),'duplicate_serials':sorted(n for n,c in Counter(serials).items() if c>1),'candidate_status_counts':dict(status),'clean_rejected':status['clean_rejected'],'keyword_rejected':status['keyword_rejected'],'asr_rejected':status['asr_rejected'],'assembly_rejected':status['assembly_rejected'],'export_rejected':status['export_rejected'],'audit_rejected_reason_categories':dict(audit_reason),'audit_issue_kind_counts':dict(issue_kinds),'unsupported_content_edits_occurrences':len(unsupported_matches),'candidate_by_scene_and_status':{s:dict(c) for s,c in by_scene.items()},'retained_by_scene':dict(retained_scene),'candidate_tags_by_status':{k:dict(v) for k,v in by_phen.items()},'boundary_punctuation':{'candidate_count':sum(boundary.values()),'status_counts':dict(boundary),'by_scene':{s:dict(c) for s,c in boundary_scene.items()},'quality_pass':boundary['quality_pass'],'retained':sum(1 for x in retained if 'boundary_punctuation' in (x.get('meta',{}).get('speech_phenomena') or []))},'passthrough_vs_teacher':{'retained_target_origin_counts':summary.get('target_origin_counts',{}),'retained_input_equal_output':retained_change['same'],'retained_input_differs_from_output':retained_change['different']},'api_calls':summary['api_calls_total'],'api_calls_by_stage':summary['api_calls_by_stage'],'full_audit':summary['full_structural_audit']}
(OUT/'posthoc_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
summary.update({'serial_integrity':{k:report[k] for k in ('candidate_count','unique_serials','serial_range','missing_serials','duplicate_serials')},'clean_rejected':report['clean_rejected'],'other_stage_rejections':{k:report[k] for k in ('keyword_rejected','asr_rejected','assembly_rejected','export_rejected')},'audit_rejected_reason_categories':dict(audit_reason),'unsupported_content_edits_occurrences':len(unsupported_matches),'candidate_by_scene_and_status':report['candidate_by_scene_and_status'],'boundary_punctuation_summary':report['boundary_punctuation'],'posthoc_report':'posthoc_report.json','comparison_to_previous_original_boundary_round':{'previous_retained':39,'previous_denominator':200,'previous_rate':0.195,'current_retained':summary['retained'],'current_denominator':200,'current_rate':summary['retained']/200,'retained_delta':summary['retained']-39,'rate_delta_percentage_points':100*(summary['retained']/200-0.195)}})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
