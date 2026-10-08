"""Compare the existing spelling-context feature on an independent drug probe."""
from collections import Counter
from dataclasses import replace
import json
from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,normalized
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.protection import EntityProtector

OUT=ROOT/'results/entity_ranker_20261006/drug_names'


def run():
    definitions=EntityStore(OUT/'context_drug_names.db').list_entities(domain='drug_names')
    configurations={'plain':tuple(replace(d,spelling_context=None) for d in definitions),'context':tuple(definitions)}
    matchers={k:EntityCandidateMatcher(v,selected_domain='drug_names',mode='auto') for k,v in configurations.items()}
    protector=EntityProtector(definitions);counts={k:Counter() for k in ['raw','plain','context']};changes=[]
    for row in read_jsonl(OUT/'asr.jsonl'):
        source=row['raw_text'];reference=normalized(row['reference']);results={'raw':source};matches={}
        blocked=[(s.start,s.end) for s in protector.protect(source).spans]
        for key,matcher in matchers.items():
            report=matcher.match(source,allow_auto=True,blocked_spans=blocked);output=source
            for match in sorted(report.auto_matches,key=lambda m:m.start,reverse=True):output=output[:match.start]+match.canonical+output[match.end:]
            results[key]=output;matches[key]=[m.public_dict() for m in report.auto_matches]
        before=Levenshtein.distance(reference,normalized(source))
        for key,output in results.items():
            count=counts[key];count['samples']+=1;count['character_errors']+=Levenshtein.distance(reference,normalized(output))
            count['reference_chars']+=len(reference);count['expected_mentions']+=len(row['entities'])
            count['recovered_mentions']+=sum(normalized(e['text']) in normalized(output) for e in row['entities'])
            count['improved_samples']+=Levenshtein.distance(reference,normalized(output))<before
            count['regressed_samples']+=Levenshtein.distance(reference,normalized(output))>before
        if results['context']!=source:
            changes.append(dict(id=row['id'],variant=row['variant'],source=source,output=results['context'],reference=row['reference'],matches=matches['context']))
    result=dict(counts={k:dict(v) for k,v in counts.items()},changes=changes,
                source='MMedFD Agent public TTS evaluation: not original human speech',
                dictionary='CMeEE training drug annotations only; no test misspelling aliases',
                existing_feature_only=True)
    write_json(OUT/'spelling_evaluation.json',result);print(json.dumps(result['counts'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
