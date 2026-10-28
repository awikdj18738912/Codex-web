import subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
started=time.monotonic()
while time.monotonic()-started<5400:
 if (P/'pilot500_results.complete.json').exists() and (P/'spotcheck_blind.jsonl').exists():
  subprocess.run([sys.executable,str(P/'summarize_pilot.py')],check=True)
  (P/'stage_ready_for_independent_review.txt').write_text('Teacher audit/repair/rechecks complete. Blind checks exported. Await independent model spot checks and primary adjudication. Formal training admission remains disabled.\n')
  print('Summary and blind review prepared; formal training admission disabled',flush=True);break
 time.sleep(30)
else:raise RuntimeError('pilot completion wait timeout')
