"""Score the preregistered phonetic-edge rule on two recorded cohorts.

First 200 are development; second 200 have not supplied reference error scores
to rule selection. Real original ASR requests/prompt/PCM are unchanged.
"""
import asyncio
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from validate_entity_confirmation import classes,exercise,replay_request,OUT,BASE,read,write
from system.entity_store import EntityStore


async def main():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    destination=OUT/'short_expansion_v3'
    protocol=json.loads((destination/'protocol.json').read_text())
    candidate=destination/'candidate_entity_audio_repair.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==protocol['candidate_sha256']
    def decision_module(path):
        tree=ast.parse(path.read_text())
        tree.body=[node for node in tree.body if not (isinstance(node,ast.Assign) and any(
            isinstance(target,ast.Name) and target.id=='SHORT_EXPANSION_RULE' for target in node.targets))]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):node.keywords=[k for k in node.keywords if k.arg!='short_expansion_rule']
        return ast.dump(tree,include_attributes=False)
    assert decision_module(ROOT/'system/entity_audio_repair.py')==decision_module(candidate)
    cc=opencc.OpenCC('t2s');itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(s):return normalized(cc.convert(itn.normalize(s).text))
    _,new=classes();definitions=EntityStore(BASE/'dataset/library/entities.db').list_entities(domain='general')
    for cohort,folders,input_path in [('development_200',('holdout','holdout_news'),OUT/'holdout_audio.jsonl'),
                                     ('validation_200',('holdout_v2',),OUT/'short_expansion/holdout_audio.jsonl')]:
        samples={r['id']:r for folder in folders for r in read(OUT/folder/'samples.jsonl')}
        records=read(input_path);assert len(records)==len({r['id'] for r in records})==len(samples)==200
        outputs=[]
        for row in records:
            proposed=row['baseline']
            # All prior no-edit cases stay identical: the code only removes
            # confirmations after retrieval/verification, never adds one.
            if row['baseline']['entity_audio_repair']['changes']:
                pcm=Path(samples[row['id']]['pcm_path']).read_bytes()
                request,remaining,_=replay_request(row['baseline_trace'])
                proposed=await exercise(new,request,definitions,pcm);assert not remaining
            outputs.append(dict(id=row['id'],baseline=row['baseline'],proposed=proposed))
        assert all(r['proposed']['entity_audio_repair']['checks']==r['baseline']['entity_audio_repair']['checks'] for r in outputs)
        output_path=destination/(cohort+'.jsonl')
        output_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs))
        counts={arm:Counter() for arm in ('baseline','proposed')};changes=[]
        for row in outputs:
            sample=samples[row['id']];ref=norm(sample['reference']);expected=Counter(norm(e['canonical']) for e in sample['gold_entities'])
            for arm in counts:
                value=row[arm];compact=norm(value['text']);status=value['entity_audio_repair']
                counts[arm].update(samples=1,hits=sum(min(n,compact.count(w)) for w,n in expected.items()),
                    expected_mentions=sum(expected.values()),errors=Levenshtein.distance(ref,compact),reference_chars=len(ref),
                    replays=status['checks'],absent_changes=sum(norm(c['canonical']) not in ref for c in status['changes']),
                    negative_errors=Levenshtein.distance(ref,compact) if sample['cohort']=='negative' else 0)
            if row['baseline']['text']!=row['proposed']['text']:
                changes.append(dict(id=row['id'],reference=sample['reference'],before=row['baseline']['text'],after=row['proposed']['text']))
        for value in counts.values():value['CER']=value['errors']/value['reference_chars']
        write(destination/(cohort+'_summary.json'),counts);write(destination/(cohort+'_changes.json'),changes)
        print(cohort,json.dumps(counts,ensure_ascii=False),flush=True)
    write(destination/'module_verification.json',dict(candidate_sha256=protocol['candidate_sha256'],
        dictionary_sha256=hashlib.sha256((BASE/'dataset/library/entities.db').read_bytes()).hexdigest(),
        same_original_pcm_and_actual_ASR=True,no_extra_ASR_requests=True,
        validation_rule_fixed_before_reference_scoring=True))


if __name__=='__main__':asyncio.run(main())
