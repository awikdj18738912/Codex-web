import sys,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path.cwd()))
from system.zh_itn_bridge import ZhITNNormalizer,DEFAULT_LIBRARY
from system.entity_store import EntityStore
from system.entity_direct_spelling import DirectEntitySpeller
from system.refinement_guard import split_for_refinement
root=Path('results/immune_numeric_first_20261016')
source=Path('results/immune_system_raw_20261015/raw_transcript.json')
raw=json.loads(source.read_text())['final_payload']['text']
normalizer=ZhITNNormalizer()
speller=DirectEntitySpeller(EntityStore('results/immune_system_mixed_repair_20261015/immune_entities.db').list_entities())
windows=split_for_refinement(raw,one_punctuation_window=True)
assert ''.join(windows)==raw
rows=[]; a=[];b=[]; offset=0; elapsed=0
for i,w in enumerate(windows):
 t=time.perf_counter();n=normalizer.normalize(w);elapsed+=time.perf_counter()-t
 before=speller.select(w,final=True);after=speller.select(n.text,final=True)
 a.extend(before);b.extend(after)
 rows.append(dict(index=i,raw_start=offset,raw=w,numeric_first=n.text,numeric_changes=[asdict(c) for c in n.changes],entity_before=before,entity_numeric_first=after))
 offset+=len(w)
 if i%50==0: print('windows',i,'/',len(windows),flush=True)
from collections import Counter
key=lambda p:(p['observed'],p['canonical'])
ca,cb=Counter(map(key,a)),Counter(map(key,b))
summary=dict(scope='Saved full-audio transcript; paired sentence/window direct entity comparison, no ASR rerun or acoustic replay; not a streaming end-to-end latency test',windows=len(windows),raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),library_sha256=hashlib.sha256(DEFAULT_LIBRARY.read_bytes()).hexdigest(),rule_pack=normalizer.rule_pack,numeric_changes=sum(len(r['numeric_changes']) for r in rows),numeric_seconds=elapsed,entity_before=len(a),entity_numeric_first=len(b),added=[dict(observed=k[0],canonical=k[1],count=v) for k,v in (cb-ca).items()],lost=[dict(observed=k[0],canonical=k[1],count=v) for k,v in (ca-cb).items()])
(root/'comparison.json').write_text(json.dumps(dict(summary=summary,windows=rows),ensure_ascii=False,indent=2))
(root/'numeric_first.txt').write_text(''.join(r['numeric_first'] for r in rows))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
