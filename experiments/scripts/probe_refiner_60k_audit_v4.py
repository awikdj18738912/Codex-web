"""Regression probe: semantic stutter review and harmful noun deletion."""
from concurrent.futures import ThreadPoolExecutor
import json,time
from pathlib import Path
import requests
import generate_refiner_60k_12classes_v4 as driver
import generate_refiner_100k as base

OUT=Path(__file__).resolve().parents[2]/'results/refiner_60k_12classes_20261027/workflow_resume_20261028/audit_probe.json'

def main():
    driver.configure_prompts()
    cases=[
      ('A02','请请帮我把这次捐赠的矿泉水和方便面重新清点一遍，把具体数量登记在表上。','请帮我把这次捐赠的矿泉水和方便面重新清点一遍，把具体数量登记在表上。',True),
      ('A02','我我需要核对这份合同，确认签字日期。','我需要核对这份合同，确认签字日期。',True),
      ('A02','先先把送到仓库的这三箱零件清点好。','先把送到仓库的这三箱零件清点好。',True),
      ('A02','我我需要保留领取报酬的权利。','我需要保留领取报酬。',False),
      ('E03','这就导致了各种意外的插曲。','这就导致了各种意外。',False),
      ('D02','这份快递请交给。门口的工作人员。','这份快递请交给门口的工作人员。',True),
    ]
    def check(item):
        i,(category,source,target,wanted)=item
        payload={'id':f'probe-v4-{i}','category':category,'input':source,'reference':target}
        body={'model':'gemma4-31b-teacher','messages':[{'role':'system','content':base.AUDIT_SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
              'temperature':0,'max_tokens':600,'chat_template_kwargs':{'enable_thinking':False},
              'response_format':{'type':'json_schema','json_schema':{'name':'probe','strict':True,'schema':base.schema(category,1,True)}}}
        response=requests.post('http://127.0.0.1:8014/v1/chat/completions',json=body,timeout=(10,600));response.raise_for_status()
        raw=response.json();choice=raw['choices'][0];assert choice['finish_reason']=='stop'
        judge=json.loads(choice['message']['content']);passed=base.audit_pass(judge,target)
        return dict(payload,expected_pass=wanted,actual_pass=passed,probe_matched=passed==wanted,judge=judge)
    requests.get('http://127.0.0.1:8014/health',timeout=10).raise_for_status()
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,enumerate(cases)))
    success=all(r['probe_matched'] for r in results)
    OUT.write_text(json.dumps({'status':'passed' if success else 'failed','cases':results,'qualification':'Six regression probes only; not dataset quality certification.'},ensure_ascii=False,indent=2)+'\n')
    print('Probe:',len(results),'matched',sum(r['probe_matched'] for r in results),flush=True)
    if not success:raise RuntimeError('Audit regression probe failed; do not start bulk generation')

if __name__=='__main__':main()
