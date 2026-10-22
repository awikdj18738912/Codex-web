"""Check actual fully formatted request lengths without truncation."""
import json
from collections.abc import Mapping
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
tokenizer=AutoTokenizer.from_pretrained('/home/aim0/data/ASR/models/Qwen3.8-27B-AWQ-INT4',local_files_only=True)
prompt=json.loads((ROOT/'prompt_snapshot.json').read_text());lengths=[]
for row in map(json.loads,(ROOT/'sample_60.jsonl').read_text().splitlines()):
    messages=[{'role':'system','content':prompt['system_prompt']},{'role':'user','content':prompt['user_template'].format(oral_text=row['input'])}]
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,preserve_thinking=False)
    ids=encoded['input_ids'] if isinstance(encoded,Mapping) else encoded
    assert isinstance(ids,list) and all(isinstance(i,int) for i in ids)
    lengths.append({'case_no':row['case_no'],'prompt_tokens':len(ids)})
report={'records':len(lengths),'max_prompt_tokens':max(x['prompt_tokens'] for x in lengths),
        'max_prompt_plus_output':max(x['prompt_tokens']+768 for x in lengths),'configured_context':3072,
        'all_fit':all(x['prompt_tokens']+768<=3072 for x in lengths)}
(ROOT/'token_length_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
