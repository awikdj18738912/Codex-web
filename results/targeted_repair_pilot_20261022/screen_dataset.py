#!/usr/bin/env python3
"""Deterministic risk-only screen for reviewed Chinese ASR refinement pairs."""
import json, re, hashlib
from decimal import Decimal, InvalidOperation
from collections import Counter, defaultdict, deque
from pathlib import Path

DATA = Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')
OUT = Path(__file__).resolve().parent
SCRIPT_VERSION = '1.1.1'
RISK_ORDER = ['english_entity_or_code_loss', 'arabic_number_value_loss', 'time_qualifier_loss', 'target_residual_correction']

# Stream the top-level JSON array: only one training row is resident at a time.
def iter_json_array(path):
    dec = json.JSONDecoder()
    with open(path, encoding='utf-8') as f:
        buf = ''
        pos = 0
        started = False
        eof = False
        while True:
            if not started:
                while pos < len(buf) and buf[pos].isspace(): pos += 1
                if pos == len(buf):
                    more=f.read(1024*1024)
                    if not more: return
                    buf += more; continue
                if buf[pos] != '[': raise ValueError('Expected JSON array')
                pos += 1; started=True
            while True:
                while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ','): pos += 1
                if pos < len(buf) and buf[pos] == ']': return
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    pos=end
                    yield obj
                    break
                except json.JSONDecodeError:
                    if eof: raise
                    more=f.read(1024*1024)
                    if not more: eof=True
                    else: buf=buf[pos:] + more; pos=0

# Train manifest rows are keyed to the JSON output index; a compact mapping is sufficient.
manifest = {}
with open(DATA/'split_manifest.jsonl', encoding='utf-8') as f:
    for line in f:
        row=json.loads(line)
        if row.get('split') == 'train':
            manifest[int(row['output_index'])] = row

NUM_RE = re.compile(r'(?<![A-Za-z0-9])(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?:%|％|[年月日天周季度季小时钟分钟秒元万亿])?')
# Terms are intentionally finite and explicit; units must still appear in target.
TIME_TERMS = ['下季度','上季度','下周','上周','明年','去年','下个月','上个月','本季度','本周']
TIME_UNITS = {'下季度':'季度','上季度':'季度','本季度':'季度','下周':'周','上周':'周','本周':'周','明年':'年','去年':'年','下个月':'月','上个月':'月'}
CORRECTION_RE = re.compile(r'不对|说错了|改口|最后改成')
# Capitalized or code-like ASCII spans: excludes isolated letters and ordinary lowercase words.
ASCII_RE = re.compile(r'(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9]*(?:[._+#/-][A-Za-z0-9]+)*)(?![A-Za-z0-9])')

def normnum(s):
    # Compare the Arabic numeric value only. Formatting and units are outside this flag.
    number=re.match(r'^[\d,，]+(?:\.\d+)?', s.replace(' ',''))
    if not number: return None
    try: return Decimal(number.group().replace(',','').replace('，',''))
    except InvalidOperation: return None

def risks_for(src, tgt):
    risks=[]
    tgt_lower=tgt.lower()
    src_entities=[]
    for m in ASCII_RE.finditer(src):
        s=m.group()
        letters=sum(c.isalpha() for c in s)
        # Preserve case-sensitive naming/code signal; lower-case words are only kept when mixed with digits/symbols.
        named=(any(c.isupper() for c in s) and letters >= 2) or (any(c.isdigit() for c in s) and letters >= 1) or any(c in '._+#/' for c in s)
        if named and s.lower() not in tgt_lower:
            src_entities.append(s)
    if src_entities:
        risks.append({'type':RISK_ORDER[0], 'evidence':{'input_exact':src_entities, 'target_exact':[]}})

    src_nums=[]
    target_nums={normnum(m.group()) for m in NUM_RE.finditer(tgt)}
    for m in NUM_RE.finditer(src):
        s=m.group()
        value=normnum(s)
        if value is not None and value not in target_nums:
            src_nums.append(s)
    if src_nums:
        risks.append({'type':RISK_ORDER[1], 'evidence':{'input_exact':src_nums, 'target_exact':[]}})

    time_lost=[]
    for term in TIME_TERMS:
        if term in src and term not in tgt and TIME_UNITS[term] in tgt:
            time_lost.append(term)
    if time_lost:
        risks.append({'type':RISK_ORDER[2], 'evidence':{'input_exact':time_lost, 'target_exact':[u for t in time_lost for u in [TIME_UNITS[t]] if u in tgt]}})

    residual=[m.group() for m in CORRECTION_RE.finditer(tgt)]
    if residual:
        risks.append({'type':RISK_ORDER[3], 'evidence':{'input_exact':[m.group() for m in CORRECTION_RE.finditer(src)], 'target_exact':residual}})
    return risks

# Stable hash tie-breaker avoids dependence on runtime hash randomization.
def stable_key(row): return hashlib.sha256(str(row['id']).encode()).hexdigest()

risk_rows=[]; controls=[]; counts=Counter(); scene_counts=Counter(); n=0
for idx, record in enumerate(iter_json_array(DATA/'train_sft.json')):
    meta=manifest.get(idx)
    if not meta: continue
    conv=record.get('conversations',[])
    if len(conv)<3: continue
    src=conv[1].get('value',''); tgt=conv[2].get('value','')
    rs=risks_for(src,tgt)
    in_hash=hashlib.sha256(src.encode('utf-8')).hexdigest()
    out_hash=hashlib.sha256(tgt.encode('utf-8')).hexdigest()
    if in_hash != meta.get('input_sha256') or out_hash != meta.get('target_sha256'):
        raise ValueError(f'Hash mismatch at output_index={idx} source_id={meta.get("source_id")}')
    row={'id':meta.get('source_id'), 'output_index':idx, 'scene':meta.get('scene'), 'input':src, 'existing_target':tgt, 'risks':rs, 'is_control':not bool(rs)}
    n+=1; scene_counts[row['scene']]+=1
    if rs:
        counts.update(r['type'] for r in rs); risk_rows.append(row)
    else: controls.append(row)

# Choose 80 with type priority and scene balance: round-robin scenes within each risk type,
# then take types in declared order, cycling until the quota is full.
buckets={t:defaultdict(deque) for t in RISK_ORDER}
for row in risk_rows:
    primary=next(r['type'] for r in row['risks'] if r['type'] in RISK_ORDER)
    buckets[primary][str(row['scene'])].append(row)
for t in RISK_ORDER:
    for scene in buckets[t]: buckets[t][scene]=deque(sorted(buckets[t][scene], key=stable_key))
selected=[]
while len(selected)<min(80,len(risk_rows)):
    progress=False
    for t in RISK_ORDER:
        scenes=sorted(buckets[t])
        for scene in scenes:
            if buckets[t][scene] and len(selected)<80:
                selected.append(buckets[t][scene].popleft()); progress=True
    if not progress: break
control_buckets=defaultdict(deque)
for row in sorted(controls,key=stable_key): control_buckets[str(row['scene'])].append(row)
selected_controls=[]
while len(selected_controls)<min(20,len(controls)):
    progressed=False
    for scene in sorted(control_buckets):
        if control_buckets[scene] and len(selected_controls)<20:
            selected_controls.append(control_buckets[scene].popleft()); progressed=True
    if not progressed: break
controls=selected_controls
# Emit risk queue in source order by primary risk, scene, stable ID.
risk_rows.sort(key=lambda x:(RISK_ORDER.index(x['risks'][0]['type']),str(x['scene']),stable_key(x)))
with open(OUT/'risk_queue.jsonl','w',encoding='utf-8') as f:
    for r in risk_rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
# Deterministic pilot ordering: risk cases by primary type/scene balanced selection order, then controls.
pilot=[]
for i,r in enumerate(selected+controls,1):
    pilot.append({'case_no':f'P{i:03d}','id':r['id'],'scene':r['scene'],'input':r['input'],'existing_target':r['existing_target'],'risks':r['risks'],'is_control':r['is_control']})
with open(OUT/'pilot_sample_100.jsonl','w',encoding='utf-8') as f:
    for r in pilot: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
report={'script_version':SCRIPT_VERSION,'source_sha256':{'train_sft.json':file_sha(DATA/'train_sft.json'),'split_manifest.jsonl':file_sha(DATA/'split_manifest.jsonl')},'hash_validation':'all rows matched manifest input_sha256 and target_sha256','data_source':str(DATA),'input_rows_scanned':n,'manifest_train_rows':len(manifest),'risk_rows':len(risk_rows),'control_rows':n-len(risk_rows),'risk_counts_by_type':dict(counts),'risk_counts_by_primary_type':dict(Counter(r['risks'][0]['type'] for r in risk_rows)),'scene_counts':dict(sorted(scene_counts.items())),'pilot_requested':100,'pilot_risk_count':len(selected),'pilot_control_count':len(controls),'pilot_actual_count':len(pilot),'selection':'deterministic risk-type priority with scene round-robin; stable SHA-256 tie-break; controls scene-round-robin','screening_scope':'heuristic risk flags only; no judgment that any data is bad and no bad-data-rate estimate','known_limitations':['Chinese numeral expression equivalence is not normalized.','Numbers from withdrawn candidates can be flagged if absent in target; surrounding context needs human review.'],'files':{'script':str(OUT/'screen_dataset.py'),'risk_queue':str(OUT/'risk_queue.jsonl'),'pilot_sample':str(OUT/'pilot_sample_100.jsonl'),'report':str(OUT/'screening_report.json')}}
with open(OUT/'screening_report.json','w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps({k:report[k] for k in ['input_rows_scanned','manifest_train_rows','risk_rows','pilot_risk_count','pilot_control_count','pilot_actual_count','risk_counts_by_type']},ensure_ascii=False))
