import os,sys,json,re,hashlib
from pathlib import Path
from collections import Counter,defaultdict
OUT=Path(__file__).resolve().parent; SNAP=OUT/'source_snapshot'; DATA=Path('/home/aim0/data/datasets/ASR/refiner_natural_200_20261017')
os.environ['REFINER_TRAIN_ZH_ITN_LIBRARY']=str(SNAP/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
sys.path[:0]=[str(SNAP/'pipeline'),str(SNAP)]
from src.utils.numeric_policy import normalize_source
from src.generators.clean_text_generator import _parse_clean_response
spec=__import__('importlib').util.spec_from_file_location('frozen_audit',SNAP/'pipeline/scripts/audit_dataset.py'); audit=__import__('importlib').util.module_from_spec(spec);spec.loader.exec_module(audit)
def readj(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
attempts=readj(OUT/'attempts.jsonl'); calls=readj(OUT/'api_calls.jsonl'); retained=readj(OUT/'retained.jsonl')
# Stored full records are separate from compact retention decisions.
byid=defaultdict(list)
for c in calls:byid[c.get('id')].append(c)
clean_rejects=[r for r in attempts if r['status']=='clean_rejected']
classes=Counter(); mismatch_examples=[]
term_chars=set('，。；、：')
def terminal_only(a,b):
 if a==b:return False
 if len(a)==len(b):return len(a)>0 and a[:-1]==b[:-1] and a[-1] in term_chars and b[-1] in term_chars
 if len(a)==len(b)+1:return a[:-1]==b and a[-1] in term_chars
 if len(b)==len(a)+1:return b[:-1]==a and b[-1] in term_chars
 return False
for r in clean_rejects:
 expected=((r.get('oral') or {}).get('meta') or {}).get('generation_plan',{}).get('expected_clean')
 cc=[x for x in byid.get(r['id'],[]) if x.get('stage')=='clean']
 if not cc: classes['no_clean_api_call_other_guard_or_passthrough']+=1;continue
 last=cc[-1]
 if 'error' in last:classes['api_call_error']+=1;continue
 parsed=_parse_clean_response(last.get('response',''))
 if parsed is None:classes['response_parse_failure']+=1;continue
 if not isinstance(expected,str):classes['missing_expected_clean']+=1;continue
 normalized,_=normalize_source(parsed['clean_text'])
 if normalized==expected:classes['normalized_exact_match_other_guard']+=1
 elif terminal_only(normalized,expected):
  classes['terminal_separator_only_mismatch']+=1
  mismatch_examples.append({'id':r['id'],'category':'terminal_separator_only_mismatch','normalized_clean':normalized,'expected_clean':expected})
 else:
  classes['other_normalized_text_mismatch']+=1
  if len(mismatch_examples)<30:mismatch_examples.append({'id':r['id'],'category':'other_normalized_text_mismatch','normalized_clean':normalized,'expected_clean':expected})
# Full structural audit, exact corpus count required.
audit_report,samples=audit.audit(OUT/'retained.jsonl',OUT/'audit_report.json',OUT/'review_sample.jsonl',min(200,len(retained)),2026101720,min_records=len(retained))
# Intentional sample size equals all available records, so reviewer artifact is complete.
(OUT/'audit_report.json').write_text(json.dumps(audit_report,ensure_ascii=False,indent=2)+'\n')
with (OUT/'review_sample.jsonl').open('w',encoding='utf8') as f:
 for x in samples:f.write(json.dumps(x,ensure_ascii=False)+'\n')
# Parse OralTextGenerator retry diagnostics. These are retry-error counts, distinct from one final candidate status.
log_text=(OUT/'runner.log').read_text(encoding='utf8',errors='replace')
oral_diag=re.compile(r'Oral generation attempt (\d+) failed for id=(\d+): (.*)')
oral_error_counts=Counter(); oral_last_by_id={}
for match in oral_diag.finditer(log_text):
 attempt_no,raw_id,message=match.groups(); key=int(raw_id); kind=message.split(':',1)[0].strip()
 oral_error_counts[kind]+=1; oral_last_by_id[key]=kind
oral_terminal=Counter(oral_last_by_id.get(r['serial'],'unknown') for r in attempts if r['status']=='oral_rejected')
api_stages=Counter(c.get('stage','unknown') for c in calls)
api_errors=Counter(c.get('stage','unknown') for c in calls if c.get('error'))
status=Counter(r['status'] for r in attempts)
actual=Counter('different' if r.get('input')!=r.get('output',{}).get('refined_text') else 'same' for r in retained)
origins=Counter((r.get('meta') or {}).get('target_origin','unknown') for r in retained)
phenomena=Counter(str(x) for r in retained for x in (r.get('meta') or {}).get('speech_phenomena',[]))
summary=json.loads((OUT/'summary.json').read_text())
summary.update({'candidate_status_counts':dict(status),'oral_retry_error_counts':dict(oral_error_counts),'oral_rejected_last_validation_error_counts':dict(oral_terminal),'api_calls_by_stage':dict(api_stages),'api_call_errors_by_stage':dict(api_errors),'clean_rejected_offline_comparison':{'candidate_count':len(clean_rejects),'categories':dict(classes),'definition_terminal_separator_only':'After normalize_source, texts differ only by adding/removing/replacing the final character, where changed character(s) are from ，。；、：; internal content is identical. ?/! are not ignored.','examples':mismatch_examples},'retained_target_origin_counts':dict(origins),'retained_speech_phenomena_counts':dict(phenomena),'retained_input_vs_output':{'same':actual['same'],'different':actual['different'],'interpretation':'Different means stored final input differs from refined_text; rule-only numeric conversions are excluded by comparing the post-rule stored training pair.'},'full_structural_audit':{'passed':audit_report['passed'],'errors':audit_report['errors'],'counts':audit_report['counts'],'sha256':audit_report['input_sha256']},'review_sample_records':len(samples)})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
progress={'stage':'complete','target':200,'completed':200,'accepted_before_dedup':8,'retained':len(retained),'pass_rate_fixed_denominator':len(retained)/200,'scene_quotas':summary['scene_quotas'],'candidate_status_counts':dict(status),'api_calls':sum(api_stages.values()),'api_calls_by_stage':dict(api_stages),'updated_utc':summary['completed_utc'],'pid':2787809}
(OUT/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':dict(status),'clean_reject_classes':dict(classes),'api_stages':dict(api_stages),'audit_passed':audit_report['passed'],'audit_errors':audit_report['errors'],'retained_same_diff':dict(actual),'target_origins':dict(origins),'summary':str(OUT/'summary.json')},ensure_ascii=False,indent=2))
