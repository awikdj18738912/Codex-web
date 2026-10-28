"""Use the repository's AASR rubric parser/scorer with a local 4-bit Gemma judge.

Same sample + same output is evaluated once and shared across variants. No cloud API.
This is a text-only, quantized-judge adaptation, not a reproduction of paper scores.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from benchmark_io import load_rubric, match_records
from client import ClientConfig, GemmaJudgeClient, JudgeError, build_user_prompt, _extract_content
from result_store import StoreConfig, append_rows, evaluate_sample, write_summary

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/aasr_bench_refiners_gpu0_20261028'
MODEL = '/home/aim0/data/ASR/models/gemma-4-31B-it-qat-w4a16-ct'
VARIANTS = ('lora','lora_gate','full','full_gate','agentic1b')
LOG_LOCK = threading.Lock()


def read(path):return [json.loads(s) for s in Path(path).open()]


def write(name, value):
    p=OUT/name
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(p)


class LocalJudge(GemmaJudgeClient):
    def __init__(self):
        from transformers import AutoTokenizer
        super().__init__(ClientConfig(api_url='http://127.0.0.1:8024/v1/chat/completions',
            model='aasr-rubric-gemma4', timeout_seconds=600, max_attempts=3))
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.tokenizer_lock=threading.Lock()

    @property
    def evaluation_identity(self):
        script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        return 'gemma4-31B-QAT-W4A16-local-AASR-no-thinking-'+script_hash

    def _messages(self,sample,output):
        return [{'role':'system','content':self._config.system_prompt},
                {'role':'user','content':build_user_prompt(sample,output)}]

    def _length(self,messages):
        with self.tokenizer_lock:
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,
                add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        return len(ids)

    def judge(self,sample,output):
        # Preserve all original questions. Split only an exceptional overlong
        # request; original question IDs/options/scores and scorer remain unchanged.
        length=self._length(self._messages(sample,output))
        if length+len(sample.questions)*18+64>3072 and len(sample.questions)>1:
            middle=len(sample.questions)//2
            a=replace(sample,questions=sample.questions[:middle])
            b=replace(sample,questions=sample.questions[middle:])
            return self.judge(a,output)+self.judge(b,output)
        return super().judge(sample,output)

    def _request(self,sample,output):
        messages=self._messages(sample,output)
        input_tokens=self._length(messages)
        budget=min(768,3072-input_tokens-16)
        if budget<32:
            raise JudgeError(f'Local judge context overflow: {sample.sample_id}, {input_tokens}')
        payload={'model':self._config.model,'messages':messages,'temperature':0,
            'max_tokens':budget,'chat_template_kwargs':{'enable_thinking':False}}
        request=urllib.request.Request(self._api_url,
            data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json'},method='POST')
        started=time.perf_counter()
        try:
            with urllib.request.urlopen(request,timeout=600) as response:
                body=json.load(response)
        except (urllib.error.URLError,TimeoutError) as error:
            raise JudgeError(f'Local Gemma judge failed: {error}') from error
        text=_extract_content(body)
        entry={'id':sample.sample_id,'output_sha256':hashlib.sha256(output.encode()).hexdigest(),
            'question_ids':[q.question_id for q in sample.questions],
            'input_tokens_estimate':input_tokens,'elapsed_seconds':time.perf_counter()-started,
            'finish_reason':body['choices'][0].get('finish_reason'),'usage':body.get('usage'),
            'raw_judge_answer':text}
        with LOG_LOCK:
            with (OUT/'judge_raw_responses.jsonl').open('a') as stream:
                stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
        return text


def main():
    assert json.loads((OUT/'progress.json').read_text())['phase']=='complete'
    rows=[r for r in read(OUT/'predictions.jsonl') if r['mode']=='throughput']
    rubric=load_rubric(OUT/'original_rubric.json')
    samples_by_variant={}
    for variant in VARIANTS:
        export=[{'source_record_id':r['source_record_id'],'output':r['output']} for r in rows if r['variant']==variant]
        assert len(export)==510 and len({r['source_record_id'] for r in export})==510
        (OUT/(variant+'.for_rubric.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in export))
        matched=match_records(rubric,export,'output')
        assert len(matched)==510
        assert sum(len(s.rubric.questions) for s in matched)==3416
        samples_by_variant[variant]=matched
    client=LocalJudge()
    rubric_hash=hashlib.sha256((OUT/'original_rubric.json').read_bytes()).hexdigest()
    identity=client.evaluation_identity
    def key(sample):
        return hashlib.sha256(json.dumps([sample.rubric.sample_id,sample.output,identity,
            rubric_hash],ensure_ascii=False).encode()).hexdigest()
    # Cache identity covers rubric, judge adapter and exact output; no approximate matching.
    representatives={}
    for samples in samples_by_variant.values():
        for sample in samples:representatives.setdefault(key(sample),sample)
    cache_path=OUT/'judge_sample_cache.jsonl'
    cache={r['cache_key']:r['rows'] for r in read(cache_path)} if cache_path.exists() else {}
    pending=[(k,s) for k,s in representatives.items() if k not in cache]
    write('judge_protocol.json',{'official_functions':['benchmark_io.load_rubric','benchmark_io.match_records',
        'result_store.evaluate_sample','result_store.write_summary'],
        'judge_model':MODEL,'judge_identity':client.evaluation_identity,'quantization':'QAT W4A16 compressed tensors',
        'local_api':'http://127.0.0.1:8024/v1','workers':4,'temperature':0,'enable_thinking':False,
        'rubric_reference':'Original clean/oral; NOT number-normalized or model-edited',
        'unique_sample_outputs':len(representatives),'total_variant_samples':2550,
        'questions_per_variant':3416,'questions_all_variants':17080,
        'duplicate_policy':'Same id + exact output + same rubric + same judge identity reuses exactly the same judgement',
        'context_policy':'3072 context; adapt max output tokens, recursively split question group only on exceptional overflow; original QA and scores unchanged',
        'qualification':'Official rubric and repository scoring functions with local quantized judge and direct oral text. NOT paper full-audio/full-precision reproduction.'})
    errors=[]
    print('Judging unique sample outputs:',len(representatives),'pending:',len(pending),flush=True)
    started=time.perf_counter()
    with cache_path.open('a') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate_sample,sample,client):k for k,sample in pending}
        for future in as_completed(futures):
            k=futures[future]
            try:
                result=future.result()
            except Exception as error:
                errors.append({'cache_key':k,'error':str(error)})
                print('Judge failed',k,str(error),flush=True)
            else:
                cache[k]=result
                stream.write(json.dumps({'cache_key':k,'rows':result},ensure_ascii=False)+'\n');stream.flush()
            write('judge_progress.json',{'phase':'judging','completed_unique':len(cache),
                'total_unique':len(representatives),'failed_unique':len(errors),
                'elapsed_seconds_this_run':time.perf_counter()-started})
            if (len(cache)+len(errors))%25==0:print('Judged',len(cache),'/',len(representatives),'failed',len(errors),flush=True)
    if errors:
        write('judge_errors.json',errors)
        write('judge_progress.json',{'phase':'failed','completed_unique':len(cache),'total_unique':len(representatives),'errors':len(errors)})
        raise RuntimeError('Some judge requests failed; no aggregate published. Restart resumes the exact-output cache.')
    scores={}
    for variant,samples in samples_by_variant.items():
        results_path=OUT/(variant+'.rubric_results.jsonl')
        results_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for sample in samples for row in cache[key(sample)]))
        summary_path=OUT/(variant+'.rubric_summary.json')
        write_summary(StoreConfig(asr_path=OUT/(variant+'.for_rubric.jsonl'),rubric_path=OUT/'original_rubric.json',
            results_path=results_path,summary_path=summary_path,text_field='output',judge_model=client.model,
            judge_identity=client.evaluation_identity,input_asr_sample_count=510,out_of_benchmark_asr_sample_count=0),samples)
        scores[variant]=json.loads(summary_path.read_text())
    write('rubric_scores_by_variant.json',{'status':'complete','qualification':'Official AASR rubric with LOCAL QUANTIZED Gemma4 judge; text-only oral input, not paper reproduction',
        'judge_protocol':str(OUT/'judge_protocol.json'),'scores':scores})
    write('judge_progress.json',{'phase':'complete','completed_unique':len(cache),'total_unique':len(representatives),
        'variant_samples':2550,'questions_all_variants':17080})
    print('All five rubric evaluations complete.',flush=True)


if __name__=='__main__':main()
