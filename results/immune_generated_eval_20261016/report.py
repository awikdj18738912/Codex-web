"""Paired synthetic audio evaluation with intended-script entity mentions."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path.cwd()))
from experiments.scripts.report_entity_repair_material import project_reference_span
from system.entity_mixed_spelling import mixed_key
p=Path(__file__).resolve().parent
r=json.loads((p/'quality/off.json').read_text());m=json.loads((p/'manifest.json').read_text());ann=json.loads((p/'reference_mentions.json').read_text())
ref=Path(m['reference_path']).read_text();before=r['primary_text'];after=r['text'];patches=r['entity_audio_repair']['changes']
assert hashlib.sha256(Path(m['library_path']).read_bytes()).hexdigest()==m['library_sha256']==r['library_sha256']
assert hashlib.sha256(Path(m['pcm_path']).read_bytes()).hexdigest()==m['pcm_sha256']==r['pcm_sha256']
freeze=json.loads((p/'frozen_source_hashes.json').read_text())
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in freeze.items())
rebuilt=before
for c in sorted(patches,key=lambda c:c['start'],reverse=True):
 assert before[c['start']:c['end']]==c['observed']
 rebuilt=rebuilt[:c['start']]+c['canonical']+rebuilt[c['end']:]
assert rebuilt==after
(p/'primary_before.txt').write_text(before+'\n');(p/'entity_after.txt').write_text(after+'\n')
# Legal read-out equivalence used ONLY by scoring; it is not repair logic.
def form(s):
 if s is None:return None
 for a,b in [('伽马德尔塔','γδ'),('伽马delta','γδ'),('阿尔法贝塔','αβ')]:s=s.replace(a,b)
 return ''.join(mixed_key(s)[0].split())
rows=[]
for mention in ann['mentions']:
 c=mention['canonical'];a=mention['start'];b=mention['end']
 assert ref[a:b]==mention['surface']
 raw=project_reference_span(ref,before,a,b);fixed=project_reference_span(ref,after,a,b)
 target=''.join(c.casefold().split())
 rows.append(dict(mention,before=raw,after=fixed,strict_before=raw==target,strict_after=fixed==target,equivalent_before=form(raw)==form(c),equivalent_after=form(fixed)==form(c),projection_before=raw is not None,projection_after=fixed is not None))
counts=Counter((c['observed'],c['canonical']) for c in patches)
summary=dict(duration_seconds=m['duration_seconds'],entities=31,reference_mentions=len(rows),retained_patches=len(patches),distinct_patched_entities=len(set(c['canonical'] for c in patches)),strict_before=sum(x['strict_before'] for x in rows),strict_after=sum(x['strict_after'] for x in rows),equivalent_before=sum(x['equivalent_before'] for x in rows),equivalent_after=sum(x['equivalent_after'] for x in rows),unprojected_before=sum(not x['projection_before'] for x in rows),unprojected_after=sum(not x['projection_after'] for x in rows),strict_gained=sum(not x['strict_before'] and x['strict_after'] for x in rows),strict_lost=sum(x['strict_before'] and not x['strict_after'] for x in rows),equivalent_gained=sum(not x['equivalent_before'] and x['equivalent_after'] for x in rows),equivalent_lost=sum(x['equivalent_before'] and not x['equivalent_after'] for x in rows),review_checks=r['entity_audio_repair']['checks'],errors=r['entity_audio_repair']['incremental']['errors'],source_hashes_match=True,output_reconstructed=True,scope='Frozen-rule synthetic speech, intended script mentions with conservative sequence-alignment projection, no human auditory adjudication. Strict normalization ignores case/space/punctuation; equivalence additionally accepts code-local Chinese digits and declared Greek read-outs. Unprojected mentions remain separate; no held-out natural speech accuracy or latency claim.')
patch_rows=[dict(c,context_before=before[max(0,c['start']-40):c['end']+40]) for c in patches]
(p/'comparison.json').write_text(json.dumps(dict(summary=summary,patches=patch_rows,mentions=rows),ensure_ascii=False,indent=2))
lines=['# 31实体合成音频：当前模块测试','','完整804.32秒音频，相同冻结31项词库、别名、上下文许可和代码。没有根据本轮转录补词、增别名或调参。CTC关闭，默认直接拼写与原音复核启用，主ASR无实体提示。前后文本来自同一次真实Qwen识别。','','## 最终保留替换','','| 原始转录 | 替换结果 | 次数 |','|---|---|---:|']
for (a,b),n in counts.items():lines.append(f'| {a} | {b} | {n} |')
if not counts:lines.append('| 无 | 无 | 0 |')
lines+=['',f"合计{len(patches)}处，涉及{summary['distinct_patched_entities']}种实体。原音复核{summary['review_checks']}次，处理异常{summary['errors']}。",'','## 逐次提及对照','','参考稿31种实体，共111次最长非重叠提及。全局对齐投影保守处理，无法可靠投影的提及单列。这里的严格拼写忽略大小写、空白和标点；读法等价另接受代码局部中文数字、伽马德尔塔/阿尔法贝塔读法。评分归一未注入修复模块。','','| 指标 | 原始转录 | 实体修复后 |','|---|---:|---:|',f"| 严格规范拼写匹配 | {summary['strict_before']}/111 | {summary['strict_after']}/111 |",f"| 接受上述合法读法后匹配 | {summary['equivalent_before']}/111 | {summary['equivalent_after']}/111 |",f"| 无法可靠投影 | {summary['unprojected_before']} | {summary['unprojected_after']} |",'',f"严格拼写新增匹配{summary['strict_gained']}处，丢失匹配{summary['strict_lost']}处；读法等价新增匹配{summary['equivalent_gained']}处，丢失匹配{summary['equivalent_lost']}处。",'','| 规范实体 | 提及数 | 严格：原始→修复 | 读法等价：原始→修复 |','|---|---:|---|---|']
for c in sorted(set(x['canonical'] for x in rows)):
 q=[x for x in rows if x['canonical']==c]
 lines.append(f"| {c} | {len(q)} | {sum(x['strict_before'] for x in q)}→{sum(x['strict_after'] for x in q)} | {sum(x['equivalent_before'] for x in q)}→{sum(x['equivalent_after'] for x in q)} |")
lines+=['','这是人工编写文本的合成语音测试，参考稿不是对实际合成发音的独立人工听审。统计描述脚本拼写恢复，不能把所有差异归为ASR错误，或直接当作自然语音的准确率。加速回放不能用于实时延迟结论。','','[原始转录](primary_before.txt) · [修复后全文](entity_after.txt) · [逐项明细与上下文](comparison.json)']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
