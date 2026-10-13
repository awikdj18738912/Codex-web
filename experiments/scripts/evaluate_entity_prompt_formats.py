"""Controlled ASR context-format repair; terms and all other settings fixed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,append_jsonl,write_json,normalized
from evaluate_medical_entities import lexicon_occurrences
from system.entity_asr_context import context_digest

OUT=ROOT/'results/entity_prompt_format_20261009'
PREVIOUS=ROOT/'results/entity_methods_comparison_20261009/medical'
PILOT_ROWS={13,116,165,166,173,174,182,200,207}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    prior=json.loads((PREVIOUS/'protocol.json').read_text());terms=prior['context_terms']
    protocol=dict(terms=terms,variants=dict(off='',legacy=prior['context_text'],
        json_terms=json.dumps(terms,ensure_ascii=False),comma_terms=', '.join(terms)),
        pilot_samples=[s for s in prior['samples'] if s['row'] in PILOT_ROWS],all_samples=prior['samples'],
        library=str(PREVIOUS/'library'),library_hashes=prior['library_hashes'],
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'system').rglob('*.py')},
        variable='Only context serialization changes; exact same 16 standard names, order, PCM and ASR parameters',
        asr_url='http://127.0.0.1:8766',language='Chinese',chunk_seconds=2.0,
        reference_at_inference=False,aliases_at_inference=False)
    write_json(path,protocol);return protocol


def decode(stage,variant=None):
    import httpx
    import soundfile as sf
    p=prepare();folder=OUT/stage;folder.mkdir(exist_ok=True)
    variants=list(p['variants']) if stage=='pilot' else [variant]
    if any(v not in p['variants'] for v in variants):raise ValueError('Unknown context format')
    samples=p['pilot_samples'] if stage=='pilot' else p['all_samples']
    done={v:{r['id'] for r in read_jsonl(folder/f'{v}.jsonl')} for v in variants}
    with httpx.Client(timeout=180) as client:
        health=client.get(p['asr_url']+'/health').raise_for_status().json()
        if health.get('context_supported') is not True:raise ValueError('Context unavailable')
        write_json(folder/'asr_health.json',health)
        for index,sample in enumerate(samples):
            wave,rate=sf.read(sample['audio_path'],dtype='float32');assert rate==16000
            pcm=wave.astype('<f4').tobytes()
            if hashlib.sha256(pcm).hexdigest()!=sample['pcm_sha256']:raise ValueError('PCM changed')
            order=variants[index%len(variants):]+variants[:index%len(variants)]
            for v in order:
                if sample['id'] in done[v]:continue
                context=p['variants'][v];params=dict(language=p['language'],chunk_seconds=p['chunk_seconds'])
                if context:params['context']=context
                started=time.perf_counter()
                ready=client.post(p['asr_url']+'/stream/start',params=params).raise_for_status().json()
                if ready.get('context_applied')!=bool(context) or ready.get('context_digest')!=context_digest(context):raise ValueError('Context acknowledgement differs')
                sid=ready['session_id'];chunks=[]
                try:
                    for offset in range(0,len(pcm),64000):
                        chunks.append(client.post(p['asr_url']+'/stream/chunk',params=dict(session_id=sid),
                            content=pcm[offset:offset+64000]).raise_for_status().json())
                    final=client.post(p['asr_url']+'/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
                except BaseException:
                    client.post(p['asr_url']+'/stream/cancel',params=dict(session_id=sid));raise
                append_jsonl(folder/f'{v}.jsonl',dict(id=sample['id'],row=sample['row'],variant=v,reference=sample['reference'],
                    pcm_sha256=sample['pcm_sha256'],context_text=context,context_terms=p['terms'] if context else [],
                    ready=ready,chunk_payloads=chunks,final_payload=final,elapsed_seconds=time.perf_counter()-started))
                print(stage,index+1,'/',len(samples),v,sample['id'],final.get('text',''),flush=True)


def report(stage):
    from system.zh_itn_bridge import ZhITNNormalizer
    p=prepare();folder=OUT/stage;itn=ZhITNNormalizer();cache={}
    gt=json.loads((Path(p['library'])/'sample_entities.json').read_text())
    names=[r['canonical_text'] for r in json.loads((Path(p['library'])/'entities.json').read_text())]
    def norm(t):
        if t not in cache:cache[t]=normalized(itn.normalize(t).text)
        return cache[t]
    summary={};details=[]
    for v in p['variants']:
        path=folder/f'{v}.jsonl'
        if not path.exists():continue
        records=read_jsonl(path);stats=Counter()
        for r in records:
            text=r['final_payload'].get('text','');ref=norm(r['reference']);expected=Counter(e['text'] for e in gt[r['id']])
            have=Counter(e['text'] for e in lexicon_occurrences(text,names))
            rec={w:min(c,have[w]) for w,c in expected.items()}
            echo='参考术语仅用于辅助拼写' in text or '术语列表：' in text
            stats.update(samples=1,character_errors=Levenshtein.distance(ref,norm(text)),reference_chars=len(ref),
                expected_mentions=sum(expected.values()),recovered_mentions=sum(rec.values()),
                unreferenced_mentions=sum(max(0,c-expected[w]) for w,c in have.items()),instruction_echo_samples=int(echo))
            details.append(dict(id=r['id'],variant=v,reference=r['reference'],text=text,recovered_by_name=rec,
                unreferenced_names={w:c-expected[w] for w,c in have.items() if c>expected[w]},instruction_echo=echo))
        summary[v]=dict(stats,CER=stats['character_errors']/stats['reference_chars'])
    write_json(folder/'summary.json',summary);write_json(folder/'comparison.json',details)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','pilot','pilot-report','full','full-report'])
    parser.add_argument('--variant',choices=['json_terms','comma_terms']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='pilot':decode('pilot');report('pilot')
    elif args.stage=='pilot-report':report('pilot')
    elif args.stage=='full':
        if not args.variant:parser.error('--variant is required for full evaluation')
        decode('full',args.variant);report('full')
    else:report('full')


if __name__=='__main__':main()
