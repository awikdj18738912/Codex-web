"""A model's fragment flag is insufficient without actual prefix alignment."""
import os
from pathlib import Path
import subprocess
import sys


def test_full_sentence_and_nonprefix_are_not_fragments():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "pipeline") + os.pathsep + str(root)}
    code = '''
import sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from src.generators.asr_simulator import valid_partial_pair
full_input = '嗯，请导出日报，然后发给财务组。'
full_target = '请导出日报，然后发给财务组。'
assert valid_partial_pair(full_input, full_target, '嗯，请导出日报。', '请导出日报。')
assert not valid_partial_pair(full_input, full_target, full_input, full_target)
assert not valid_partial_pair(full_input, full_target, '请导出日报。', full_target)
assert not valid_partial_pair(full_input, full_target, '发给财务组。', '发给财务组。')
assert not valid_partial_pair('预算是1,250,000元，明天汇款。', '预算是1,250,000元，明天汇款。', '预算是1,250,', '预算是1,250,')
assert not valid_partial_pair('请执行f(alpha,beta)，然后返回。', '请执行f(alpha,beta)，然后返回。', '请执行f(alpha,', '请执行f(alpha,')
'''
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_llm_fragment_uses_aligned_observed_prefix_and_fixed_target():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "pipeline") + os.pathsep + str(root)}
    code = r'''
import asyncio, json, sys, types
sys.modules['openai'] = types.SimpleNamespace(AsyncOpenAI=object, OpenAI=object)
from src.generators.asr_simulator import ASRSimulator

class FirstChoice:
    def choice(self, values): return values[0]
    def random(self): return 0

class FakeClient:
    def __init__(self, target): self.calls = []; self.target = target
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({'input': '嗯，请导出上周销日报', 'target': self.target}, ensure_ascii=False)

oral = '嗯，请导出上周销售日报，然后发给财务组。'
clean = '请导出上周销售日报，然后发给财务组。'
fixed = '请导出上周销售日报，'
client = FakeClient(fixed)
sim = ASRSimulator(async_client=client, mode='llm_sim', simulation_ratio=1, rng=FirstChoice())
input_text, target = asyncio.run(sim._simulate_with_llm(oral, clean))
prompt = client.calls[0]['user_prompt']
assert '嗯，请导出上周销售日报' in prompt
assert '然后发给财务组' not in prompt
assert target == fixed
assert input_text == '嗯，请导出上周销日报'

no_boundary_client = FakeClient('没有自然边界的完整句子')
no_boundary = ASRSimulator(async_client=no_boundary_client, mode='llm_sim', simulation_ratio=1, rng=FirstChoice())
result = asyncio.run(no_boundary.simulate_one('daily_chat', {
    'id': 'no_boundary', 'oral_text': '请尽快处理这个请求', 'clean_text': '请尽快处理这个请求',
}))
assert result['input'] == '请尽快处理这个请求'
assert result['output']['refined_text'] == '请尽快处理这个请求'
assert result['meta']['is_fragment'] is False and no_boundary_client.calls == []

class BrokenJsonClient:
    async def generate(self, **kwargs): return 'not json'
broken = ASRSimulator(async_client=BrokenJsonClient(), mode='llm_sim', simulation_ratio=1, rng=FirstChoice())
result = asyncio.run(broken.simulate_one('daily_chat', {
    'id': 'broken_json', 'oral_text': oral, 'clean_text': clean,
}))
assert result == {}

unaligned_client = FakeClient(fixed)
unaligned = ASRSimulator(async_client=unaligned_client, mode='llm_sim', simulation_ratio=1, rng=FirstChoice())
try:
    asyncio.run(unaligned._simulate_with_llm('嗯，另外一段不相关口语。', clean))
except ValueError as exc:
    assert 'cannot be aligned' in str(exc)
else:
    raise AssertionError('unaligned clean prefix was accepted')
assert unaligned_client.calls == []

drift_client = FakeClient(clean)
drift = ASRSimulator(async_client=drift_client, mode='llm_sim', simulation_ratio=1, rng=FirstChoice())
try:
    asyncio.run(drift._simulate_with_llm(oral, clean))
except ValueError as exc:
    assert 'changed the fixed fragment target' in str(exc)
else:
    raise AssertionError('model target drift was silently accepted')
'''
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
