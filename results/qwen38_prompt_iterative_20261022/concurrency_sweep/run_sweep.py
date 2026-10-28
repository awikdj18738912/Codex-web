import json,time,subprocess,sys
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'concurrency_sweep'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
rows=[json.loads(l) for l in (ROOT/'second_pass/predictions.jsonl').read_text().splitlines() if l.strip()]
sample=[{k:r[k] for k in ('case_no','id','input','baseline_target','scene')} for r in sorted(rows,key=lambda r:r['case_no'])]
(OUT/'sample40.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample))
save('status.json',{'phase':'waiting_for_server'})
for _ in range(180):
 try:
  with urlopen('http://127.0.0.1:8012/v1/models',timeout=2) as f: json.load(f)
  break
 except Exception: time.sleep(5)
else: raise RuntimeError('server startup timeout')
results=[]
for concurrency in (4,8,12,16):
 save('status.json',{'phase':'benchmark','concurrency':concurrency,'completed_runs':results})
 path=OUT/f'c{concurrency}.jsonl';start=time.monotonic()
 with (OUT/f'c{concurrency}.log').open('w') as log:
  proc=subprocess.run([sys.executable,str(ROOT/'generate_round.py'),'--sample',str(OUT/'sample40.jsonl'),'--prompt',str(ROOT/'second_pass/prompt.json'),'--output',str(path),'--concurrency',str(concurrency),'--max-tokens','768'],stdout=log,stderr=subprocess.STDOUT)
 wall=time.monotonic()-start
 if proc.returncode:raise RuntimeError(f'benchmark c{concurrency} failed')
 output=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 old={r['id']:r['clean_text'] for r in rows}
 result={'concurrency':concurrency,'records':len(output),'wall_seconds':round(wall,2),'records_per_hour':round(len(output)*3600/wall),'errors':sum(bool(r['error']) for r in output),'completion_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in output),'exact_matches_prior_development_output':sum(old[r['id']]==r.get('clean_text') for r in output)}
 result['output_tokens_per_second']=round(result['completion_tokens']/wall,2)
 results.append(result);save('results.json',results);print(json.dumps(result),flush=True)
save('status.json',{'phase':'complete','results':results})
