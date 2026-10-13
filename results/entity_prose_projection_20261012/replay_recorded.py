"""Offline comparison with fixed historical model responses; no new inference."""
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from system.entity_pipeline import prepare_entity_segment,finalize_entity_segment
from system.entity_matcher import EntityCandidateMatcher
from system.entity_store import EntityStore
from system.protection import EntityProtector
from system.refinement_protocol import apply_structured_patch

OUT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('system._projection_before',OUT/'before_entity_pipeline.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
BASE=ROOT/'results/large_entity_benchmark_v1_20261010/full_evaluation/service_logs'
store=EntityStore(ROOT/'results/large_entity_benchmark_v1_20261010/dataset/library/entities.db')
contexts={};counts=Counter();changes=[]
for condition in ('entity_rules','entity_audio_repair'):
 domains={r['trace_id']:r['entity_domain'] for r in map(json.loads,(BASE/f'{condition}.jsonl').open())}
 for row in map(json.loads,(BASE/f'{condition}.stream_trace.jsonl').open()):
  if row.get('event')!='window_result':continue
  domain=domains.get(row['trace_id'])
  if domain is None:continue
  if domain not in contexts:
   definitions=store.list_entities(domain=domain)
   contexts[domain]=(EntityProtector(definitions),EntityCandidateMatcher(definitions,selected_domain=domain,mode='auto'))
  protector,matcher=contexts[domain]
  for segment in row.get('segments',[]):
   if not segment.get('model_calls'):continue
   prepared=prepare_entity_segment(segment['source'],protector,matcher,allow_auto=True)
   prepared=replace(prepared,baseline_text=segment['baseline'],trusted_baseline_text=segment['baseline'])
   for call in segment['model_calls']:
    candidate,patch,issue=apply_structured_patch(call['input'],call['output'])
    if issue:counts['protocol_rejected']+=1;continue
    before=old.finalize_entity_segment(candidate,prepared,protector,entity_patch=patch,numeric_rule_mode=True)
    after=finalize_entity_segment(candidate,prepared,protector,entity_patch=patch,numeric_rule_mode=True)
    counts[condition+'_responses']+=1
    if before.text!=after.text or before.normalizations!=after.normalizations:
     changes.append(dict(condition=condition,trace_id=row['trace_id'],source=segment['source'],response=call['output'],before=before.text,after=after.text,changes=after.normalizations))
counts['changed_responses']=len(changes)
(OUT/'historical_replay.json').write_text(json.dumps({'counts':dict(counts),'changes':changes,'scope':'Recorded window responses individually finalized. Not a new full audio ASR run, nor a full-Web sequential replay. Current matcher/protector held fixed for both arms.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(counts),ensure_ascii=False),flush=True)
