"""Score the new veto on the first 200 now-development recordings."""
import asyncio
from collections import Counter
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
    cc=opencc.OpenCC('t2s');itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(s):return normalized(cc.convert(itn.normalize(s).text))
    _,new=classes();definitions=EntityStore(BASE/'dataset/library/entities.db').list_entities(domain='general')
    samples={r['id']:r for folder in ('holdout','holdout_news') for r in read(OUT/folder/'samples.jsonl')}
    destination=OUT/'short_expansion/development_200.jsonl'
    done={r['id'] for r in read(destination)} if destination.exists() else set()
    for row in read(OUT/'holdout_audio.jsonl'):
        if row['id'] in done:continue
        sample=samples[row['id']];pcm=Path(sample['pcm_path']).read_bytes()
        request,remaining,_=replay_request(row['baseline_trace'])
        proposed=await exercise(new,request,definitions,pcm);assert not remaining
        with destination.open('a') as f:f.write(json.dumps(dict(id=row['id'],baseline=row['baseline'],proposed=proposed),ensure_ascii=False)+'\n')
        print('DEVELOPMENT_200',row['id'],row['baseline']['text']==proposed['text'],flush=True)
    counts={arm:Counter() for arm in ('baseline','proposed')};changes=[]
    for row in read(destination):
        sample=samples[row['id']];ref=norm(sample['reference']);expected=Counter(norm(e['canonical']) for e in sample['gold_entities'])
        for arm in counts:
            result=row[arm];value=norm(result['text']);status=result['entity_audio_repair']
            counts[arm].update(samples=1,hits=sum(min(n,value.count(w)) for w,n in expected.items()),expected_mentions=sum(expected.values()),
                errors=Levenshtein.distance(ref,value),reference_chars=len(ref),replays=status['checks'],
                absent_changes=sum(norm(c['canonical']) not in ref for c in status['changes']))
        if row['baseline']['text']!=row['proposed']['text']:
            changes.append(dict(id=row['id'],reference=sample['reference'],before=row['baseline']['text'],after=row['proposed']['text']))
    for value in counts.values():value['CER']=value['errors']/value['reference_chars']
    assert all(v['samples']==200 for v in counts.values())
    write(OUT/'short_expansion/development_200_summary.json',counts)
    write(OUT/'short_expansion/development_200_changes.json',changes)


if __name__=='__main__':asyncio.run(main())
