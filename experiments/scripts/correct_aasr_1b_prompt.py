"""Use the local AgenticASR default refiner prompt for a raw-oral 1B baseline.

The offline CLI's alternate prompt says numbers were already processed. It is
inappropriate for the 1B raw-input baseline. Retain that first pass as an ablation.
"""
import hashlib
import json
from pathlib import Path
import shutil

import benchmark_aasr_refiners_gpu0 as benchmark

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/aasr_bench_refiners_gpu0_20261028'


def main():
    from system.refinement_protocol import REFINER_SYSTEM_PROMPT
    control=OUT/'agentic1b_default_prompt_control'
    control.mkdir(exist_ok=False)
    original_protocol=json.loads((OUT/'protocol.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol['prompts']['agentic1b']=REFINER_SYSTEM_PROMPT
    protocol['agentic1b_pipeline']='Local system.refiner.SYSTEM_PROMPT / refinement_protocol.REFINER_SYSTEM_PROMPT, raw oral input, model owns numeric conversion, no native numeric rules; KEY stripped from body'
    protocol['agentic1b_prompt_source']=str(ROOT/'system/refinement_protocol.py')
    protocol['corrected_1b_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (control/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    for name in ('text_manifest.jsonl','libzh_itn_bridge.so'):
        shutil.copy2(OUT/name,control/name)
    benchmark.OUT=control
    benchmark.VARIANTS=['agentic1b']
    benchmark.MODELS={'agentic1b':benchmark.MODELS['agentic1b']}
    benchmark.run()
    replacement=[json.loads(s) for s in (control/'predictions.jsonl').open()]
    replacement_times=[json.loads(s) for s in (control/'batch_timings.jsonl').open()]
    assert len(replacement)==542 and len(replacement_times)==96
    archive=OUT/'first_pass_1b_offline_cli_prompt'
    archive.mkdir(exist_ok=False)
    for filename in ('predictions.jsonl','batch_timings.jsonl','protocol.json'):
        shutil.copy2(OUT/filename,archive/filename)
    for filename,new in [('predictions.jsonl',replacement),('batch_timings.jsonl',replacement_times)]:
        old=[json.loads(s) for s in (OUT/filename).open()]
        merged=[r for r in old if r['variant']!='agentic1b']+new
        assert len(merged)==len(old)
        (OUT/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
    protocol['original_first_pass_protocol']=str(archive/'protocol.json')
    protocol['agentic1b_correction_reason']='First-pass offline CLI prompt states numbers already processed while 1B gets raw input; replaced only 1B with the actual local AgenticASR default raw-input refiner prompt. Other four model outputs/timings unchanged.'
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    audit={'original_1b_prompt':original_protocol['prompts']['agentic1b'],
        'used_1b_prompt':REFINER_SYSTEM_PROMPT,'raw_input_1b_default_source':'system.refiner.SYSTEM_PROMPT',
        'unchanged_variants':['lora','lora_gate','full','full_gate'],
        'old_pass_archive':str(archive),'new_1b_runtime':str(control/'runtime.json'),
        'generation_protocol':'BF16 SDPA greedy cache, same32 latency samples and510 input batch groups, GPU0 warm serial inference'}
    (OUT/'agentic1b_prompt_correction.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print('1B raw-input default prompt correction complete; first pass retained.',flush=True)


if __name__=='__main__':main()
