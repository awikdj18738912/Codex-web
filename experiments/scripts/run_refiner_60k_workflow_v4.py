"""GPU0-only generation -> audit -> SFT -> gate -> comparison, with owned-service cleanup."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/refiner_60k_12classes_20261027/workflow_resume_20261028'
POOL=Path('/home/aim0/data/datasets/ASR/refiner_balanced_12classes_60k_20261027_v4')
DATA=Path('/home/aim0/data/datasets/ASR/refiner60k_qwen25_full_20261027')
MODEL=Path('/home/aim0/data/ASR/training/qwen25_05b_refiner60k_full_20261027')
GATE=Path('/home/aim0/data/ASR/training/refiner60k_gate_rbt3_20261027')
CONFIG=ROOT/'experiments/llamafactory/refiner60k_qwen25_full_20261027.yaml'
TRAINPY=ROOT/'.runtime/llamafactory-env/bin/python'
GENUNIT='refiner-generation-60k-12classes-v4-20261028.service'
TEACHER='refiner-teacher-gpu0-60k-20261027.service'
OLDGEN='refiner-generation-60k-12classes-20261027.service'
CATEGORIES=['A01','A02','A03','B01','B02','C01','D01','D02','E03','E05','E06','E07']
CHILD=None
STATE={'status':'starting','completed_stages':[],'training_started':False,'human_review_status':'pending'}


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(path)
def update(status,**extra):
    STATE.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**extra);atomic(RUN/'progress.json',STATE)
    (RUN/'运行状态.md').write_text('# 6万条实验工作流\n\n状态：'+status+'\n\n已完成阶段：'+', '.join(STATE['completed_stages'])+'\n\n'+(STATE.get('error') or '')+'\n\n所有自动审核结果均非人工确认。GPU1未参与。\n')
def unit_state(unit):
    r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True,capture_output=True,timeout=15);return r.stdout.strip()
def stop_unit(unit):
    subprocess.run(['systemctl','--user','stop',unit],timeout=1050,check=True)
def cleanup():
    errors=[]
    for unit in [GENUNIT,OLDGEN,TEACHER]:
        try:
            if unit_state(unit) in ('active','activating','deactivating'):stop_unit(unit)
        except Exception as e:errors.append(unit+':'+str(e))
    atomic(RUN/'cleanup.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'owned_units':{u:unit_state(u) for u in [GENUNIT,OLDGEN,TEACHER]},'errors':errors,'gpu1_used':False})
    return errors

def step(name,cmd,env=None,timeout=12*3600):
    global CHILD
    update(name,current_command=[str(x) for x in cmd]);print('STAGE',name,flush=True)
    launch=os.environ.copy();launch.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTORCH_ALLOC_CONF='expandable_segments:True')
    if env:launch.update(env)
    log=RUN/(name+'.log')
    with log.open('a') as stream:
        CHILD=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,env=launch,stdout=stream,stderr=subprocess.STDOUT)
        try:code=CHILD.wait(timeout=timeout)
        except BaseException:
            CHILD.terminate()
            try:CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
            raise
        finally:CHILD=None
    if code:raise RuntimeError(f'{name} exit={code}; see {log}')
    STATE['completed_stages'].append(name);update(name+'_complete')

def await_generation():
    update('waiting_generation')
    start=time.monotonic()
    while time.monotonic()-start<72*3600:
        path=POOL/'progress.json'
        if path.exists():
            p=json.loads(path.read_text());update('waiting_generation',candidate_count=p['selected_or_machine_passed_pending_human_review'],generation_status=p['status'])
            if p['status']=='balanced_gaps_complete_pending_human_review':
                assert p['selected_or_machine_passed_pending_human_review']==60000
                assert set(p['stats'])==set(CATEGORIES)|{'total'}
                assert all(p['stats'][c]['accepted']==5000 for c in CATEGORIES)
                assert (POOL/'candidates.pending_review.jsonl').is_file()
                # Wait until the coordinator has fully exited, not just exported.
                if unit_state(GENUNIT) not in ('active','activating','deactivating'):
                    STATE['completed_stages'].append('generation_60000');return
            if p['status'] in ('needs_attention','waiting_teacher','stopped_after_draining'):
                raise RuntimeError('Generation stopped: '+p['status']+' '+str(p.get('error')))
        if unit_state(GENUNIT) not in ('active','activating','deactivating'):
            raise RuntimeError('Generation service ended before complete')
        time.sleep(30)
    raise TimeoutError('Generation exceeded 72h; cleanup and report incomplete')

def summary_report(success,error=None):
    lines=['# 6万条扩充实验结果','',
        '状态：'+('自动实验完成，等待人工确认' if success else '未完成，已停止本轮计算任务'),'']
    if error:lines+=['原因：'+error,'']
    lines+=['## 已执行阶段','']+['- '+x for x in STATE['completed_stages']]
    lines+=['','## 数据与审核','',
        '生成目标为12类各5,000条候选。质量复核后实际训练量可能减少；不补入被隔离的疑点凑数。',
        '默认跳过公开数据语义机器审核；合成数据抽样及风险样本由同一教师独立请求复核，均不能视为人工确认。',
        '人工仍需确认抽样、疑点配对和真实音频参考。旧验证、测试及隔离用途保留；不自动发布或部署新模型。','']
    mf=DATA/'manifest.json'
    if mf.exists():
        m=json.loads(mf.read_text());lines+=['实际冻结划分：`'+json.dumps(m.get('split_counts',{}),ensure_ascii=False)+'`。','']
    state=MODEL/'trainer_state.json'
    if state.exists():
        s=json.loads(state.read_text());lines+=['最佳checkpoint：`'+str(s.get('best_model_checkpoint'))+'`；最佳验证loss：'+str(s.get('best_metric'))+'。','']
    scores=RUN/'evaluation/scores.json'
    if scores.exists():
        s=json.loads(scores.read_text());lines+=['## 精修对照（候选参考匹配，不是人工准确率）','',
            '| 测试范围 | 模型 | 数量 | 完全匹配 | 字符差异率 | 干净候选改动 |','| --- | --- | ---: | ---: | ---: | ---: |']
        for suite in ['old1757','new_test','new_balanced_50_50']:
            for model,v in s['suites'].get(suite,{}).items():
                if not v.get('n'):continue
                lines.append(f"| {suite} | {model} | {v['n']} | {v.get('exact_match_rate',0):.2%} | {v.get('normalized_char_difference_rate',0):.4%} | {v.get('identity_changed_count',0)}/{v.get('identity_count',0)} |")
        lines+=['','门控组合细节与漏掉的可修正句见 evaluation/scores.json 和对应逐句JSONL；真实音频1见 evaluation/audio1_comparison.md。','']
    lines+=['## 结果位置','',f'- 冻结数据：{DATA}',f'- 新全量模型：{MODEL}',f'- 新RBT3门控：{GATE}',f'- 抽样文档：{RUN}/sample_report/sample1200_review.md',f'- 待人工复核：{RUN}/audit/待人工复核.md',f'- 实验评估：{RUN}/evaluation',f'- 服务关闭记录：{RUN}/cleanup.json','',
        '方法参考：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[数据泄漏防范](https://scikit-learn.org/stable/common_pitfalls.html)。']
    (RUN/'实验报告.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--validate-plan',action='store_true');a=p.parse_args()
    required=[ROOT/'experiments/scripts'/n for n in ['prepare_refiner_60k_experiment.py','audit_refiner_60k_experiment.py','preflight_refiner_60k_experiment.py','evaluate_refiner_60k_experiment.py','train_refiner13_pregate.py']]
    for path in required:
        assert path.exists(),path;compile(path.read_text(),str(path),'exec')
    assert TRAINPY.is_file() and CONFIG.is_file()
    if a.validate_plan:print('Workflow scripts/config found and compile; no services touched.');return
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RUN/'progress.json').exists():raise RuntimeError('Refusing implicit rerun; inspect prior workflow state')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in ()).throw(InterruptedError('Workflow interrupted')))
    success=False;error=None
    try:
        await_generation()
        # Normalize source/scene for the CPU preparation; preserve full original payload.
        original=POOL/'candidates.pending_review.jsonl';enriched=RUN/'candidates.enriched.jsonl'
        with enriched.open('w') as stream:
            for line in original.open():
                row=json.loads(line);row['source']=row.get('dataset') or row.get('source')
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        step('prepare_audit_sample',[sys.executable,required[0],'--candidates',enriched,'--output',RUN/'sample_data','--report-dir',RUN/'sample_report','--sample-only'],env={'CUDA_VISIBLE_DEVICES':''})
        step('teacher_second_audit',[sys.executable,required[1],'--sample',RUN/'sample_data/sample1200.jsonl','--risk',RUN/'sample_report/review_queue.jsonl','--output',RUN/'audit'],env={'CUDA_VISIBLE_DEVICES':''},timeout=24*3600)
        stop_unit(TEACHER);STATE['completed_stages'].append('teacher_stopped_for_training');update('teacher_stopped_for_training')
        step('freeze_filtered_data',[sys.executable,required[0],'--candidates',enriched,'--output',DATA,'--report-dir',RUN/'frozen_data_report','--audit-results',RUN/'audit/results.jsonl'],env={'CUDA_VISIBLE_DEVICES':''})
        # Verify exact original test is independent before expensive training.
        sys.path.insert(0,str(ROOT/'experiments/scripts'))
        # Avoid importing CUDA/evaluation libraries into supervisor: CPU subprocess preflight handles actual checks.
        step('preflight',[TRAINPY,required[2],'--config',CONFIG,'--report',RUN/'preflight.json'],env={'CUDA_VISIBLE_DEVICES':''})
        STATE['training_started']=True
        step('full_train',[TRAINPY,'-u','-c','from llamafactory.cli import main; main()','train',CONFIG])
        step('full_evaluation',[TRAINPY,required[3],'--data',DATA,'--model',MODEL,'--output',RUN/'evaluation','--review-sample',RUN/'sample_data/sample1200.jsonl','--inference-only'])
        step('rbt3_gate_train',[TRAINPY,required[4]],env={'PREGATE_DATA':str(DATA),'PREGATE_OUTPUT':str(GATE),'PREGATE_ENCODER':'/home/aim0/data/models/ASR/TextGate/hfl-rbt3','PREGATE_DEVICE':'cuda:0','PREGATE_REFINER_PREDICTIONS':str(RUN/'evaluation/new_test_new_full.jsonl')})
        step('gate_combined_evaluation',[TRAINPY,required[3],'--data',DATA,'--model',MODEL,'--output',RUN/'evaluation','--review-sample',RUN/'sample_data/sample1200.jsonl','--gate-model',GATE,'--reuse-inference'])
        success=True
    except BaseException as e:
        error=f'{type(e).__name__}: {e}';traceback.print_exc();update('needs_attention',error=error)
    finally:
        errors=cleanup()
        if errors:success=False;error=(error or '')+' Cleanup errors: '+str(errors)
        summary_report(success,error)
        update('automatic_experiments_complete_pending_human_review_services_stopped' if success else 'incomplete_services_stopped',error=error)
    if not success:raise SystemExit(1)

if __name__=='__main__':main()
