"""Batch admission rejects malformed data and preserves fragment provenance."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def isolated(code):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'pipeline'), str(ROOT)]))
    run = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


PRELUDE = '''
import importlib.util, json, sys, tempfile, types
from pathlib import Path
sys.modules['openai']=types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
spec=importlib.util.spec_from_file_location('dataset_audit','pipeline/scripts/audit_dataset.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from src.utils.synthetic_oral import build_oral
intent='请使用FastAPI开发接口。'
plan=dict(construction_version='synthetic-oral-v1',raw_intent=intent,
          construction_phenomena=['oral:low'],correction_values=[],spelling_parts=[],expected_clean=intent)
source=build_oral(intent,[],[],['oral:low'])
item=dict(id='one',input=source,oral_text=source,output={'refined_text':intent,'keyword_list':[]},
          meta=dict(scene='vibe_coding',language='zh',is_fragment=False,generation_plan=plan,
                    quality_audit=dict(source_valid=True,meaning_preserved=True,cleanup_complete=True,
                                       issues=[],cleaned_target=intent)))
'''


def test_valid_batch_is_reproducible_and_wrong_stored_audit_is_rejected():
    isolated(PRELUDE + '''
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'data.jsonl'; report=Path(temp)/'report.json'
    p.write_text(json.dumps(item,ensure_ascii=False)+'\\n')
    a,samples=mod.audit(p,report,None,30,7)
    b,again=mod.audit(p,report,None,30,7)
    assert a['passed'] and a==b and samples==again and len(samples)==1
    item['meta']['quality_audit']['cleaned_target']='另外一句话。'
    p.write_text(json.dumps(item,ensure_ascii=False)+'\\n')
    a,_=mod.audit(p,report,None,30,7)
    assert not a['passed'] and a['errors']['by_type']['quality_audit_failed']==1
''')


def test_malformed_ids_and_empty_corpus_return_errors_without_sample_crash():
    isolated(PRELUDE + '''
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'data.jsonl'; report=Path(temp)/'report.json'
    p.write_text('')
    a,_=mod.audit(p,report,None,30,7)
    assert a['errors']['by_type']['empty_dataset']==1
    item['id']=['unhashable']
    item['meta']['is_fragment']='false'
    p.write_text(json.dumps(item,ensure_ascii=False)+'\\n{broken\\n')
    a,_=mod.audit(p,report,None,30,7)
    assert not a['passed']
    assert {'invalid_id_type','missing_or_invalid_fragment_flag','invalid_json'} <= a['errors']['by_type'].keys()
''')


def test_duplicate_ids_conflicts_and_non_normalized_storage_are_rejected():
    isolated(PRELUDE + '''
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'data.jsonl'; report=Path(temp)/'report.json'
    one=json.dumps(item,ensure_ascii=False)
    item['output']['refined_text']='请使用Flask开发接口。'
    p.write_text(one+'\\n'+json.dumps(item,ensure_ascii=False)+'\\n')
    a,_=mod.audit(p,report,None,30,7)
    assert a['errors']['by_type']['duplicate_id']==1
    assert a['errors']['by_type']['same_input_different_target']==1
    from src.utils.numeric_policy import normalize_source
    raw='明天下午三点开会。'; target=normalize_source(raw)[0]
    numeric=dict(item,id='number',input=raw,oral_text=raw,output={'refined_text':target},
                 meta=dict(item['meta'],generation_plan={'expected_clean':target}))
    p.write_text(json.dumps(numeric,ensure_ascii=False)+'\\n')
    a,_=mod.audit(p,report,None,30,7)
    assert a['errors']['by_type']['stored_pair_not_rule_normalized']==1
''')


def test_finalize_keeps_id_and_oral_for_later_fragment_validation():
    isolated(PRELUDE + '''
spec=importlib.util.spec_from_file_location('finalize','pipeline/scripts/05_finalize.py')
final=importlib.util.module_from_spec(spec);spec.loader.exec_module(final)
with tempfile.TemporaryDirectory() as temp:
    base=Path(temp); processed=base/'processed'; processed.mkdir(); out=base/'final'
    (processed/'vibe_coding.jsonl').write_text(json.dumps(item,ensure_ascii=False)+'\\n')
    final.PROCESSED_DIR=str(processed); final.FINAL_DIR=str(out)
    final.SCENE_DISTRIBUTION={'vibe_coding':1.0}
    sys.argv=['05_finalize.py','--skip-qc','--total','1']
    final.main()
    saved=json.loads((out/'train.jsonl').read_text())
    assert saved['id']==item['id'] and saved['oral_text']==source
    report,_=mod.audit(out/'train.jsonl',out/'report.json',None,30,7)
    assert report['passed']
''')


def test_batch_count_requirement_and_scene_local_ids():
    isolated(PRELUDE + '''
spec=importlib.util.spec_from_file_location('assemble','pipeline/scripts/04_assemble.py')
assemble=importlib.util.module_from_spec(spec);spec.loader.exec_module(assemble)
one={'meta':{}};two={'meta':{}}
assemble._copy_record_id(one,{'id':1},'daily_chat')
assemble._copy_record_id(two,{'id':1},'meeting')
assert one['id']!=two['id'] and one['meta']['source_record_id']==1
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'data.jsonl'; report=Path(temp)/'report.json'
    p.write_text(json.dumps(item,ensure_ascii=False)+'\\n')
    report,_=mod.audit(p,report,None,30,7,min_records=100000)
    assert not report['passed'] and report['errors']['by_type']['insufficient_records']==1
''')


def test_orchestrator_runs_acceptance_after_finalize_with_count_requirement():
    isolated(PRELUDE + '''
import pipeline.run_pipeline as runner
commands=[]
runner.run_cmd=lambda cmd,name: commands.append((cmd,name)) or True
args=types.SimpleNamespace(base_url=None,min_final_records=100000)
assert runner.step5_finalize(args)
assert len(commands)==2 and '--save-rejects' in commands[0][0]
assert commands[1][0][-2:]==['--min-records','100000']
assert commands[1][0][1].endswith('audit_dataset.py')
''')
