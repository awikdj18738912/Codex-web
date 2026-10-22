"""Report frozen-glossary evaluation without inferring an accuracy denominator."""
import json,hashlib
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import sys
sys.path.insert(0,str(Path.cwd()))
from system.entity_store import EntityStore
p=Path(__file__).resolve().parent
r=json.loads((p/'quality/off.json').read_text());m=json.loads((p/'manifest.json').read_text())
s=r['entity_audio_repair']; patches=s['changes'];before=r['primary_text'];after=r['text']
assert r['library_sha256']==m['library_sha256']
assert hashlib.sha256(Path(m['library_path']).read_bytes()).hexdigest()==m['library_sha256']
assert hashlib.sha256(Path(m['pcm_path']).read_bytes()).hexdigest()==m['pcm_sha256']
assert r['pcm_sha256']==m['pcm_sha256']
frozen=json.loads((p/'frozen_source_hashes.json').read_text())
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in frozen.items())
rebuilt=before
for c in sorted(patches,key=lambda c:c['start'],reverse=True):
 assert before[c['start']:c['end']]==c['observed']
 rebuilt=rebuilt[:c['start']]+c['canonical']+rebuilt[c['end']:]
assert rebuilt==after
(p/'primary_before.txt').write_text(before+'\n');(p/'entity_after.txt').write_text(after+'\n')
(p/'registered_entities.json').write_text(json.dumps([asdict(d) for d in EntityStore(m['library_path']).list_entities()],ensure_ascii=False,indent=2))
rows=[]
for c in patches:
 rows.append(dict(c,context_before=before[max(0,c['start']-55):min(len(before),c['end']+55)]))
targets=[]
for raw,canonical in [('LC一','ILC1'),('LC二','ILC2'),('LC三','ILC3'),('FAST','FAS')]:
 targets.append(dict(observed=raw,canonical=canonical,literal_count_in_primary=before.count(raw),retained_repair_count=sum(c['observed']==raw and c['canonical']==canonical for c in patches)))
summary=dict(duration_seconds=m['duration_seconds'],primary_characters=len(before),patches=len(patches),changed_canonicals=len(set(c['canonical'] for c in patches)),rules=dict(Counter(c.get('direct_rule',c.get('evidence_source','')) for c in patches)),review_checks=s['checks'],errors=s['incremental']['errors'],targets=targets,source_and_dictionary_unchanged=True,output_reconstructed=True,scope='Fresh full audio, frozen prior31 glossary and code, accelerated quality; no independent human-adjudicated gold or precision/recall claim')
(p/'comparison.json').write_text(json.dumps(dict(summary=summary,patches=rows),ensure_ascii=False,indent=2))
lines=['# 免疫系统2：冻结规则与词库测试','','用户指定新音频，完整处理。使用上次免疫系统1实验的相同31项实体、别名、上下文许可、阈值和代码；测试过程中未根据新转录补词或调参。CTC关闭，直接实体规则和既有原音复核启用。原始转写无实体提示，与修复文本来自同一条主识别流。','',f"时长 {m['duration_seconds']:.2f} 秒；主转写 {len(before)} 字符。最终保留 {len(patches)} 处补丁，涉及 {summary['changed_canonicals']} 个不同规范实体。原音复核 {s['checks']} 次，处理异常 {summary['errors']}。",'', '这些是实际改写数；没有人工听审gold，不等于正确恢复数或准确率。词库未覆盖的新实体不会被自动补进本轮。加速回放不用于声称实时延迟。','','[原始转录](primary_before.txt) · [实体修复后全文](entity_after.txt)','','## 全部最终保留改写','','| 原文 | 修复 | 路径/规则 |','|---|---|---|']
for c in patches:lines.append(f"| {c['observed']} | {c['canonical']} | {c.get('direct_rule',c.get('evidence_source',''))} |")
if not patches:lines.append('| 无 | 无 | 无 |')
lines+=['','## 前次四种错写在本段的情况','','| 原始写法 | 原文出现次数 | 实际修复次数 |','|---|---:|---:|']
for c in targets:lines.append(f"| {c['observed']} | {c['literal_count_in_primary']} | {c['retained_repair_count']} |")
lines+=['','某种错写没有出现，就没有为其泛化能力提供新的证据。上表原文次数为字面子串计数，实际修复按完整源跨度统计。','','[包含上下文的修改明细](comparison.json)']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
