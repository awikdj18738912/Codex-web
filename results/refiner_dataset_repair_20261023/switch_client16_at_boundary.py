import json,os,signal,time
from pathlib import Path
p=Path(__file__).resolve().parent
while not(p/'outputs/batch_0001.complete.json').exists():time.sleep(5)
(p/'selected_scheduling.json').write_text(json.dumps({'worker':str(p/'concurrency_trial/client_per_gpu_16/run_specialists.py'),'client_concurrency_per_gpu':16,'total_client_concurrency':32,'server_max_num_seqs_per_gpu':16,'cudagraph':'FULL_AND_PIECEWISE'}))
os.kill(1527965,signal.SIGCONT)
(p/'client16_switched.json').write_text(json.dumps({'at':time.strftime('%Y-%m-%d %H:%M:%S'),'after_batch':1}))
