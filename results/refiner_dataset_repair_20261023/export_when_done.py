from pathlib import Path
import time,subprocess,sys
p=Path(__file__).resolve().parent
while not (p/'all_batches_complete.json').exists():time.sleep(30)
subprocess.run([sys.executable,str(p/'export_version.py')],check=True)
