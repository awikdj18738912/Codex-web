"""Known labels remove redundant reconstruction, never the semantic gate."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def isolated(code):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "pipeline"), str(ROOT)]))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


PRELUDE = '''
import asyncio, copy, json, sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
import src.generators.clean_text_generator as clean_module
clean_module._QC_ENABLED = False
from src.utils.synthetic_oral import build_oral
from src.pipeline.training_quality import audit_pair
intent = '使用FastAPI开发接口。'
plan = dict(construction_version='synthetic-oral-v1', raw_intent=intent,
            construction_phenomena=['correction:rollback', 'oral:low'],
            correction_values=['FastAPI', 'Flask', 'FastAPI'], spelling_parts=[], expected_clean=intent)
source = build_oral(intent, plan['correction_values'], [], plan['construction_phenomena'])
item = dict(id='demo', oral_text=source, meta={'generation_plan':plan}, selected_seeds=['FastAPI'])
class Client:
    def __init__(self): self.calls=[]
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({'clean_text':intent})
'''


def test_known_target_needs_no_clean_llm_call_and_keeps_proof():
    isolated(PRELUDE + '''
client=Client()
gen=clean_module.CleanTextGenerator(client, target_mode='construction')
out=asyncio.run(gen.generate_one('vibe_coding',item))
assert not client.calls and out['clean_text']==intent
assert out['meta']['generation_plan']==plan
assert out['meta']['target_origin']=='verified-construction-v1'
assert out['keyword_list']==['FastAPI']
''')


def test_broken_constructed_proofs_do_not_fall_back_to_guessed_labels():
    isolated(PRELUDE + '''
client=Client(); gen=clean_module.CleanTextGenerator(client, target_mode='construction')
for changed in [dict(item,oral_text=source+'额外内容'),
                dict(item,meta={'generation_plan':dict(plan,construction_version='bad')}),
                dict(item,meta={'generation_plan':dict(plan,expected_clean='使用Flask开发接口。')})]:
    assert asyncio.run(gen.generate_one('vibe_coding',changed)) is None
assert not client.calls
''')


def test_teacher_mode_and_legacy_input_still_reconstruct():
    isolated(PRELUDE + '''
client=Client(); gen=clean_module.CleanTextGenerator(client,target_mode='teacher')
assert asyncio.run(gen.generate_one('vibe_coding',item))['clean_text']==intent
assert len(client.calls)==1
legacy=dict(item,meta={})
legacy['oral_text']='嗯，使用FastAPI开发接口。'
client=Client(); gen=clean_module.CleanTextGenerator(client,target_mode='construction')
out=asyncio.run(gen.generate_one('vibe_coding',legacy))
assert out['clean_text']==intent and len(client.calls)==1
assert out['meta']['generation_plan']['generation_mode']=='upstream-teacher-v1'
assert out['meta']['generation_plan']['expected_clean']==intent
assert 'intent_text' not in client.calls[0]['user_prompt']
from src.utils.numeric_policy import prepare_training_pair
assert prepare_training_pair(legacy['oral_text'],intent,out['meta']['generation_plan']) is not None
''')


def test_upstream_teacher_target_reaches_semantic_qc_past_surface_guards():
    isolated(PRELUDE + '''
from src.utils.numeric_policy import prepare_training_pair
source='明天下午3点用Python写接口。'
target='明天下午4点用TypeScript写接口。'
plan={'generation_mode':'upstream-teacher-v1','source_oral':source,'expected_clean':target}
assert prepare_training_pair(source,target,plan) is not None
assert prepare_training_pair(source,target,None) is None
''')


def test_construction_context_does_not_override_semantic_rejection():
    isolated(PRELUDE + '''
class Judge:
    def __init__(self): self.calls=[]
    async def generate(self,**kwargs):
        self.calls.append(kwargs)
        return json.dumps(dict(source_valid=False,meaning_preserved=True,cleanup_complete=True,
                               issues=[{'kind':'source','quote':'Flask'}],cleaned_target=intent))
judge=Judge()
ok,reason,_=asyncio.run(audit_pair(source,intent,judge,generation_plan=plan))
assert not ok and reason.startswith('audit_failed:source_valid')
assert 'verified_construction' in json.loads(judge.calls[0]['user_prompt'])
for badsource, badtarget, repair in [(source+'额外',intent,False),
                                     (source,'使用Flask开发接口。',False),
                                     (source,intent,True)]:
    before=len(judge.calls)
    ok,reason,_=asyncio.run(audit_pair(badsource,badtarget,judge,
                                      recognition_repair=repair,generation_plan=plan))
    assert not ok and reason=='invalid_construction_proof'
    assert len(judge.calls)==before
''')


def test_v1_ascii_period_intent_proof_rebuilds_without_double_period():
    isolated('''
from src.utils.synthetic_oral import build_oral
from src.utils.numeric_policy import lock_training_target
intent='Run docker build.'
plan=dict(construction_version='synthetic-oral-v1',raw_intent=intent,
          construction_phenomena=['oral:low'],correction_values=[],spelling_parts=[],expected_clean=intent)
source=build_oral(intent,[],[],plan['construction_phenomena'])
assert source=='嗯，Run docker build.'
assert lock_training_target(source,intent,plan)==intent
''')


def test_chinese_short_target_threshold_counts_content_not_punctuation_or_fillers():
    isolated('''
import importlib.util
spec=importlib.util.spec_from_file_location('assemble','pipeline/scripts/04_assemble.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert module._record_min_length('general', {}) == 4
assert module._record_min_length('english_general', {'language':'en'}) == 6
assert module._record_content_length('嗯，呃！', 'general', {}) == 0
assert module._record_content_length('嗯，呃！今天', 'general', {}) == 2
assert module._record_content_length('嗯，呃！今天出发', 'general', {}) == 4
assert module._record_content_length('哈哈哈哈', 'general', {}) == 0
assert module._record_content_length('那个女孩', 'general', {}) == 4
''')


def test_legacy_flat_seed_pool_can_be_extended_without_changing_entries():
    isolated(PRELUDE + '''
import tempfile
from pathlib import Path
from src.generators.seed_generator import SeedGenerator
with tempfile.TemporaryDirectory() as temp:
    generator=SeedGenerator(Client());generator.seeds_dir=Path(temp)
    path=Path(temp)/'explanation.json'
    path.write_text(json.dumps(['钟宁','林安','钟宁'],ensure_ascii=False))
    pool=generator.load_seeds('explanation')
    assert pool=={'legacy_seeds':['钟宁','林安']}
    assert '钟宁' in generator._build_exclusion_hint(pool)
    generator.merge_seeds('explanation',{'person_names':['张三']})
    assert generator.load_seeds('explanation')['legacy_seeds']==['钟宁','林安']
''')
