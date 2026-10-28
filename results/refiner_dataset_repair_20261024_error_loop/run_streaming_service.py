"""Run the resumed client and export candidates only after full completion."""
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(P / 'run_error_loop.py'), '--concurrency-per-gpu', '16',
                '--record-workers', '64'], check=True)
subprocess.run([sys.executable, str(P / 'export_candidates.py')], check=True)
