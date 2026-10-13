"""Audit paired entity-stage outputs and timings, including gold limitations."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/entity_local_replay_20261012'
sys.path.insert(0,str(ROOT))


def main():
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    cc=opencc.OpenCC('t2s');itn=ZhITNNormalizer(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    def norm(t):return normalized(cc.convert(itn.normalize(t).text))
    evaluations={}
    for name,folder in [('historical_dense_v1',OUT),('sparse_v2',OUT/'v2/sparse_development'),('unseen_v2',OUT/'v2/unseen_evaluation')]:
        labels={}
        if name=='unseen_v2':
            for child in ('unseen_long','unseen_long_supplement'):labels.update(json.loads((OUT/'v2'/child/'references.json').read_text()))
        else:labels=json.loads((folder/'references.json').read_text())
        rows=[json.loads(s) for s in (folder/'results.jsonl').read_text().splitlines() if s];pairs={}
        for r in rows:assert r['arm'] not in pairs.setdefault(r['id'],{});pairs[r['id']][r['arm']]=r
        assert all(set(p)=={'baseline','candidate'} for p in pairs.values())
        records=[];counts={a:Counter() for a in ('baseline','candidate')}
        for sid,arms in pairs.items():
            label=labels[sid];reference=label if isinstance(label,str) or label is None else label['reference']
            target=label.get('registered_entity',label.get('canonical')) if isinstance(label,dict) else None
            record=dict(id=sid,identical_text=arms['baseline']['final']['text']==arms['candidate']['final']['text'],
                        retained_change_records=arms['baseline']['final']['entity_audio_repair']['changes']==arms['candidate']['final']['entity_audio_repair']['changes'])
            for arm,r in arms.items():
                value=r['final'];status=value['entity_audio_repair'];c=counts[arm]
                c.update(samples=1,wall_seconds=r['wall_seconds'],replays=status['checks'],changes=status['change_count'])
                if reference is not None:
                    text=norm(value['text']);ref=norm(reference)
                    c.update(scored_samples=1,errors=Levenshtein.distance(ref,text),reference_chars=len(ref),
                             absent_changes=sum(norm(e['canonical']) not in ref for e in status['changes']))
                    if target:
                        expected=ref.count(norm(target));c.update(hits=min(expected,text.count(norm(target))),expected_mentions=expected)
                record[arm]=dict(seconds=r['wall_seconds'],mode=status.get('replay_mode','full' if status['checks'] else 'skipped'),
                                 replay_audio_seconds=status.get('replay_audio_seconds'),changes=status['change_count'])
            record['time_reduction_percent']=100*(1-record['candidate']['seconds']/record['baseline']['seconds'])
            records.append(record)
        for c in counts.values():
            if c['reference_chars']:c['CER']=c['errors']/c['reference_chars']
        evaluations[name]=dict(metrics=counts,records=records,
            all_text_identical=all(r['identical_text'] for r in records),all_changes_identical=all(r['retained_change_records'] for r in records),
            scope='Entity repair stage; timings are single alternating pairs, not whole transcription speed. Historical_dense usesV1 full-mode compatibility, not localized speed evidence.')
    (OUT/'paired_summary.json').write_text(json.dumps(evaluations,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{'metrics':v['metrics'],'same':v['all_text_identical']} for k,v in evaluations.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
