"""Offline scoring for the pure-ASR add-on experiment; never runs inference.

Default requires all 2,506 paired recordings. Use --partial to score only IDs
with one complete pure_asr final row. Scoring references are read from the parent
experiment's isolated scoring sidecar; they are not used to select inference data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import opencc
from rapidfuzz.distance import Levenshtein

OUT=Path(__file__).resolve().parent
EXPERIMENT=OUT.parent
ROOT=OUT.parents[2]
SOURCE=EXPERIMENT/'source_snapshot'
sys.path.insert(0,str(SOURCE))
from system.entity_audio_repair import normalized
from system.zh_itn_bridge import ZhITNNormalizer

CONVERTER=opencc.OpenCC('t2s')
ITN=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
METHODS=('pure_asr_raw','context_raw','context_clean','audio_raw','audio_clean')
OLD_ARMS=('context','audio')
CORPUS_MAIN='large_entity_benchmark_v1_20261010'
CORPUS_SUPPLEMENT='entity_reliable_data_20261012_pilot'


def norm(text):
    return normalized(CONVERTER.convert(ITN.normalize(text).text))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def percentile_stats(values):
    if not values:
        return {'count':0,'mean':None,'median':None,'p90':None}
    return {
        'count':len(values),
        'mean':float(np.mean(values)),
        'median':float(np.median(values)),
        'p90':float(np.percentile(values,90)),
    }


def score_text(text, normalized_ref, entities):
    normalized_text=norm(text)
    all_gold=Counter(norm(e['canonical']) for e in entities)
    by_type=defaultdict(Counter)
    for e in entities:
        by_type[e['entity_type']][norm(e['canonical'])]+=1
    hits=sum(min(count,normalized_text.count(word)) for word,count in all_gold.items())
    return {
        'text':text,
        'hits':hits,
        'expected_mentions':sum(all_gold.values()),
        'errors':Levenshtein.distance(normalized_ref,normalized_text),
        'reference_chars':len(normalized_ref),
        'hits_by_type':{
            typ:sum(min(count,normalized_text.count(word)) for word,count in gold.items())
            for typ,gold in by_type.items()
        },
        'expected_by_type':{typ:sum(gold.values()) for typ,gold in by_type.items()},
    }


def aggregate(scores, sample_ids, method):
    selected=[scores[sid]['values'][method] for sid in sample_ids]
    hits=sum(x['hits'] for x in selected)
    expected=sum(x['expected_mentions'] for x in selected)
    errors=sum(x['errors'] for x in selected)
    chars=sum(x['reference_chars'] for x in selected)
    return {
        'samples':len(selected),'hits':hits,'expected_mentions':expected,
        'entity_recall':hits/expected if expected else None,
        'errors':errors,'reference_chars':chars,'CER':errors/chars if chars else None,
    }


def group_keys(sample):
    return ('all','corpus:'+sample['corpus'],'cohort:'+sample['cohort'],
            'library:'+sample['library_key'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true',help='score only pure-ASR IDs with a complete final result row')
    args=parser.parse_args()

    parent_inputs=read_jsonl(EXPERIMENT/'inference_inputs.jsonl')
    samples={x['id']:x for x in parent_inputs}
    assert len(samples)==len(parent_inputs)==2506
    references=read_json(EXPERIMENT/'references.json')['samples']
    assert set(references)==set(samples)
    legacy_summary=read_json(EXPERIMENT/'summary.json')
    assert legacy_summary.get('complete') and legacy_summary['completed_pairs']==len(samples)

    # Reconstruct the old two arms from saved final Web results. Their raw and
    # clean transcripts share each mode's original run timing; they are separate
    # transcript scoring views, not separate latency observations.
    legacy_rows=read_jsonl(EXPERIMENT/'results.jsonl')
    old={}
    for row in legacy_rows:
        if row['arm'] not in OLD_ARMS:
            continue
        key=(row['id'],row['arm'])
        assert key not in old, f'duplicate old result row: {key}'
        assert row['id'] in samples
        assert row['pcm_sha256']==samples[row['id']]['pcm_sha256']
        assert row['result']['event']=='final'
        old[key]=row
    assert set(old)=={(sid,arm) for sid in samples for arm in OLD_ARMS}

    pure_bytes=(OUT/'results.jsonl').read_bytes()
    if not args.partial and pure_bytes and not pure_bytes.endswith(b'\n'):
        raise AssertionError('full pure-ASR results must end with a complete newline-terminated row')
    complete_pure_bytes=pure_bytes
    if args.partial and pure_bytes and not pure_bytes.endswith(b'\n'):
        complete_pure_bytes=pure_bytes[:pure_bytes.rfind(b'\n')+1] if b'\n' in pure_bytes else b''
    pure_rows=[json.loads(line) for line in complete_pure_bytes.decode('utf-8').splitlines() if line]
    pure={}
    request_totals=Counter()
    for row in pure_rows:
        assert row['arm']=='pure_asr', row.get('arm')
        sid=row['id']; key=(sid,'pure_asr')
        assert sid in samples
        assert key not in pure, f'duplicate pure-ASR result row: {sid}'
        assert row['pcm_sha256']==samples[sid]['pcm_sha256']
        result=row['result']
        assert result['event']=='final' and 'raw_text' in result and 'clean_text' in result
        assert result['raw_text']==result['clean_text']==result['asr_payload']['text'], f'pure-ASR output was changed after ASR: {sid}'
        assert result['session_refiner_call_count']==0, f'pure-ASR row has refiner calls: {sid}'
        assert result['session_refiner_retry_call_count']==0, f'pure-ASR row has refiner retries: {sid}'
        ready=row['ready']
        assert ready['entity_asr_context_requested'] is False
        assert ready['entity_asr_context_applied'] is False and ready['entity_asr_context_term_count']==0
        assert ready['entity_audio_repair_requested'] is False
        assert ready['refiner_enabled'] is False and ready['postprocessing_enabled'] is False
        assert result['entity_asr_context_requested'] is False and result['entity_asr_context_applied'] is False
        assert result['entity_audio_repair_requested'] is False
        assert result['numeric_normalization_enabled'] is False and result['text_cleanup_enabled'] is False
        requests=result['asr_requests']
        endpoints=[request['endpoint'] for request in requests]
        frames=row['transport']['frames']
        assert row['transport']['acks']==frames and row['transport']['max_outstanding']<=4
        assert endpoints.count('/stream/start')==1 and endpoints.count('/stream/finish')==1
        assert endpoints.count('/stream/chunk')==frames
        assert set(endpoints)<={'/stream/start','/stream/chunk','/stream/finish'}
        assert all(request['context_supplied'] is False for request in requests)
        request_totals.update(endpoints)
        request_totals['context_supplied_true']+=sum(request['context_supplied'] is True for request in requests)
        request_totals['validated_sessions']+=1
        pure[key]=row

    ids=sorted(sid for sid in samples if (sid,'pure_asr') in pure)
    if not args.partial:
        assert len(ids)==len(samples), f'incomplete pure-ASR results: {len(ids)}/{len(samples)}; use --partial to score a subset'
    assert ids
    assert set(references)==set(samples)

    per_id={}
    for sid in ids:
        sample=samples[sid]; label=references[sid]
        reference=label['reference']; normalized_ref=norm(reference)
        assert normalized_ref
        entities=label['gold_entities']
        gold=Counter(norm(e['canonical']) for e in entities)
        assert all(word and normalized_ref.count(word)>=count for word,count in gold.items()), ('gold/reference mismatch',sid)
        values={}
        pure_result=pure[sid,'pure_asr']['result']
        values['pure_asr_raw']=score_text(pure_result['raw_text'],normalized_ref,entities)
        for arm in OLD_ARMS:
            result=old[sid,arm]['result']
            values[arm+'_raw']=score_text(result['raw_text'],normalized_ref,entities)
            values[arm+'_clean']=score_text(result['clean_text'],normalized_ref,entities)
        per_id[sid]={
            'sample':sample,'reference':reference,'gold_entities':entities,'values':values,
            'timing':{
                'pure_asr':pure[sid,'pure_asr']['timing'],
                'context':old[sid,'context']['timing'],
                'audio':old[sid,'audio']['timing'],
            },
        }

    # For a full run, assert this scorer reproduces the legacy score's persisted
    # raw/clean counts before reporting the pure-ASR comparison.
    baseline_checks={}
    if len(ids)==len(samples):
        for arm in OLD_ARMS:
            raw=aggregate(per_id,ids,arm+'_raw')
            clean=aggregate(per_id,ids,arm+'_clean')
            saved=legacy_summary['groups']['all'][arm]
            baseline_checks[arm]={
                'raw_hits_match':raw['hits']==saved['raw_hits'],
                'raw_errors_match':raw['errors']==saved['raw_errors'],
                'clean_hits_match':clean['hits']==saved['hits'],
                'clean_errors_match':clean['errors']==saved['errors'],
            }
            assert all(baseline_checks[arm].values()), (arm,baseline_checks[arm])

    memberships=defaultdict(list)
    for sid in ids:
        for key in group_keys(samples[sid]):
            memberships[key].append(sid)

    groups={}
    for key,group_ids in memberships.items():
        group={method:aggregate(per_id,group_ids,method) for method in METHODS}
        timings={}
        for arm in ('pure_asr','context','audio'):
            timing_values=[per_id[sid]['timing'][arm] for sid in group_ids]
            timing_summary={}
            for field in ('total_seconds','session_setup_seconds','audio_stream_seconds','tail_seconds'):
                timing_summary[field]=percentile_stats([x[field] for x in timing_values])
            timings[arm]=timing_summary
        comparisons={}
        for baseline in ('context_raw','context_clean','audio_raw','audio_clean'):
            hit_delta=[];error_delta=[]
            for sid in group_ids:
                pure_v=per_id[sid]['values']['pure_asr_raw']
                other=per_id[sid]['values'][baseline]
                hit_delta.append(pure_v['hits']-other['hits'])
                error_delta.append(pure_v['errors']-other['errors'])
            comparisons['pure_asr_minus_'+baseline]={
                'hit_delta':sum(hit_delta),'error_delta':sum(error_delta),
                'pure_more_hits_samples':sum(x>0 for x in hit_delta),
                'baseline_more_hits_samples':sum(x<0 for x in hit_delta),
                'equal_hit_samples':sum(x==0 for x in hit_delta),
                'pure_lower_CER_samples':sum(x<0 for x in error_delta),
                'baseline_lower_CER_samples':sum(x>0 for x in error_delta),
                'equal_CER_samples':sum(x==0 for x in error_delta),
            }
        groups[key]={'metrics':group,'run_timing':timings,'paired_comparisons':comparisons}

    all_types=sorted({e['entity_type'] for sid in ids for e in per_id[sid]['gold_entities']})
    entity_types={}
    for typ in all_types:
        type_ids=[sid for sid in ids if any(e['entity_type']==typ for e in per_id[sid]['gold_entities'])]
        method_metrics={}
        for method in METHODS:
            hits=sum(per_id[sid]['values'][method]['hits_by_type'].get(typ,0) for sid in type_ids)
            expected=sum(per_id[sid]['values'][method]['expected_by_type'].get(typ,0) for sid in type_ids)
            full_transcript=aggregate(per_id,type_ids,method)
            method_metrics[method]={
                'samples_with_type':len(type_ids),'hits':hits,'expected_mentions':expected,
                'entity_recall':hits/expected if expected else None,
                'whole_transcript_errors_on_type_subset':full_transcript['errors'],
                'whole_transcript_reference_chars_on_type_subset':full_transcript['reference_chars'],
                'whole_transcript_CER_on_type_subset':full_transcript['CER'],
            }
        entity_types[typ]=method_metrics

    # Overall pure-versus-existing clean/raw comparisons and raw-run timing.
    id_text='\n'.join(ids)
    pure_result_bytes='\n'.join(json.dumps(pure[sid,'pure_asr'],ensure_ascii=False,sort_keys=True,separators=(',',':')) for sid in ids)
    result={
        'complete':len(ids)==len(samples),'scope':'full' if len(ids)==len(samples) else 'partial',
        'sample_count':len(ids),'planned_samples':len(samples),'paired_final_rows':len(ids)+2*len(ids),
        'id_set_sha256':hashlib.sha256(id_text.encode('utf-8')).hexdigest(),
        'pure_result_rows_sha256':hashlib.sha256(pure_result_bytes.encode('utf-8')).hexdigest(),
        'baseline_full_score_checks':baseline_checks,
        'pure_asr_request_validation':{
            'validated_sessions':request_totals['validated_sessions'],
            'start_requests':request_totals['/stream/start'],
            'chunk_requests':request_totals['/stream/chunk'],
            'finish_requests':request_totals['/stream/finish'],
            'context_supplied_true':request_totals['context_supplied_true'],
            'incomplete_trailing_bytes_ignored':len(pure_bytes)-len(complete_pure_bytes),
        },
        'metrics_definition':{
            'normalization':'same conservative zh-ITN, OpenCC t2s, NFKC/casefold and Unicode P/Z/C removal as the two-arm experiment',
            'entity_hits':'Exact normalized canonical substring occurrence, capped by frozen gold multiplicity; no identity linking or alias inference.',
            'by_type_CER':'Whole-transcript CER over recordings that contain the type; it is not error attribution to only that entity type.',
            'pure_transcript':'Pure-ASR raw_text and clean_text are required to be identical; scored once as pure_asr_raw.',
        },
        'design_limits':[
            'Pure ASR is a separately collected supplementary run, not a third arm interleaved with the completed context/audio pair collection.',
            'Its private WebSocket is a transparent forwarder to the same ASR service and measures only the raw ASR route; it has no browser/user-proxy time. Existing context/audio timings cover their full Web final path and are not raw-ASR-only timings.',
            'Accuracy differences compare complete mode pipelines; they cannot all be attributed to entity prompting/review alone because pure ASR omits entity assistance, refinement, and numeric processing.',
            'The historical run used 1-second PCM blocks, at most four outstanding blocks, and 2-second ASR chunks; see pure_asr/protocol.json for the recorded run settings when available.',
        ],
        'groups':groups,'entity_types':entity_types,
        'negative_controls':{
            cohort:groups['cohort:'+cohort]['metrics']
            for cohort in ('negative','no_registered_target') if 'cohort:'+cohort in groups
        },
    }
    summary_path=OUT/('partial_summary.json' if args.partial else 'summary.json')
    details_path=OUT/('partial_paired_details.jsonl' if args.partial else 'paired_details.jsonl')
    summary_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with details_path.open('w',encoding='utf-8') as f:
        for sid in ids:
            data=per_id[sid]
            f.write(json.dumps({
                'id':sid,'corpus':data['sample']['corpus'],'cohort':data['sample']['cohort'],
                'library_key':data['sample']['library_key'],'topics':data['sample']['topics'],
                'audio_seconds':data['sample']['duration_seconds'],'reference':data['reference'],
                'gold_entities':data['gold_entities'],'values':data['values'],'run_timing':data['timing'],
            },ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','scope','sample_count','planned_samples','id_set_sha256','pure_result_rows_sha256')},ensure_ascii=False,indent=2))
    print(json.dumps(groups['all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
