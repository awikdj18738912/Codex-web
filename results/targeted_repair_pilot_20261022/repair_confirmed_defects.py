"""Teacher writes fixes for independently identified defects, isolated from SFT."""
import asyncio,json,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parent
DEFECTS={
 'P001':'已有目标把已撤回的时间序列预测、自注意力、单目视频重建话题保留成有效叙述。最终意图是扩散Transformer生成3D形状，原文标识为“扩散Transformer for 3D shape generation”。只表达最终意图，不虚构早先思考经历。',
 'P003':'目标遗漏了未撤回的整段独立叙述：同学在网课，手机自动切网络，视频卡成PPT，一节课后脑子转不动。补回这件事，保留另外两件事。',
 'P004':'目标删除了“算了，先别管它了，回头再说”，使心电贴片算法工作从暂缓变成当前待办。保留暂缓语义和其余独立任务，不将暂缓误改为取消永久任务。',
 'P011':'把“之前填北京地址”的撤回回忆写成实际填写历史；最终确认地址是山西软件园B座。保留原文“好像”办理手续的不确定性，不编造实际填错地址的历史。',
 'P013':'目标新增“项目负责人”“苏景然团队”，原文只有机构、其项目经费以及苏景然搞的技术。删除无根据的头衔和团队，人物关系按原文表达，不补背景。',
 'P017':'目标把临时提议又撤回的回滚写成有效步骤。最终决定重新执行原git commit命令，然后kubectl查看。保留真实报错及末尾关于Hapi是否启动的疑问，不改成确定诊断。',
 'P019':'目标从订单号2024111733445566擅自推断出拍照日期2024年11月17日，原文只说上上个月。删掉编造日期，保留上上个月、原订单号和最终重新出照片的请求。',
 'P020':'Sony耳机、2023年9月10号为明确撤回的候选，目标却加入“以为买Sony”的思考历史。只保留最后确认的旅行收纳包、2024年4月22号旅行使用及原文有效细节，不保留撤回候选。',
}
SYSTEM='''依据原始口语和已确认的具体缺陷修复已有精修目标。你只生成完整修订文本，不做一般常识事实纠错，不添加推断。精修须表达说话者最终意图，明确撤回内容应删；未撤回的独立叙述应保留，推迟与取消应区分。名称/代码命令准确保留。只作必要修改，保持口语原意与不确定语气。输出JSON：{"clean_text":"完整修订文本"}。'''
def request(row,defect):
    user={'input':row['input'],'existing_target':row['existing_target'],'confirmed_defect':defect}
    body={'model':'refiner-teacher','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(user,ensure_ascii=False)}],
          'temperature':0,'max_tokens':1000,'response_format':{'type':'json_object'}}
    req=Request('http://127.0.0.1:8011/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=600) as f:return json.load(f)
async def main():
    rows={r['case_no']:r for r in map(json.loads,(ROOT/'pilot_sample_100.jsonl').read_text().splitlines())}
    sem=asyncio.Semaphore(4)
    (ROOT/'confirmed_defects.json').write_text(json.dumps(DEFECTS,ensure_ascii=False,indent=2))
    (ROOT/'confirmed_repair_prompt.txt').write_text(SYSTEM)
    async def one(case,defect):
        async with sem:
            row=rows[case];raw=None;error=None;clean=None;result=None
            try:
                result=await asyncio.to_thread(request,row,defect)
                raw=result['choices'][0]['message']['content'];clean=json.loads(raw)['clean_text']
                if not isinstance(clean,str) or not clean.strip():raise ValueError('empty clean_text')
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            item={**row,'confirmed_defect':defect,'clean_text':clean,'raw_content':raw,'error':error,
                  'usage':result.get('usage') if result else None}
            with (ROOT/'confirmed_repair_proposals.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
            print(case,'error=',error,flush=True)
    await asyncio.gather(*(one(c,d) for c,d in DEFECTS.items()))
if __name__=='__main__':asyncio.run(main())
