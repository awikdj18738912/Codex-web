"""Collect speed measurements and exact transcript comparisons per variant."""

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import statistics


def differences(source, target):
    result=[]
    for kind,a,b,c,d in SequenceMatcher(None,source,target,autojunk=False).get_opcodes():
        if kind!='equal':
            result.append(dict(kind=kind,source=source[a:b],target=target[c:d],
                source_context=source[max(0,a-20):min(len(source),b+20)],
                target_context=target[max(0,c-20):min(len(target),d+20)]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    groups=dict(baseline=['baseline_a1','baseline_a2'],transport=['transport_b1'],
                batch=['scoring_b1','scoring_b2'],crop_cache=['crop_c1','crop_c2'],
                uncompressed=['uncompressed_d1','uncompressed_d2'],deployed=['deployed_e1'])
    rows=[]
    for audio in [4,5]:
        baseline=json.loads((root/f'baseline_a2_测试音频{audio}.json').read_text())
        for group,labels in groups.items():
            runs=[]
            for label in labels:
                path=root/f'{label}_测试音频{audio}.json'
                if not path.exists():continue
                x=json.loads(path.read_text());m=x['_speed_benchmark']
                runs.append(dict(label=label,elapsed_s=m['elapsed_s'],feed_s=m['feed_s'],finish_s=m['finish_s'],
                    first_text_s=m['first_text_s'],first_refined_s=m['first_refined_s'],
                    refiner_wall_s=x['refiner_session_stats']['completed_wall_latency_ms']/1000,
                    refiner_calls=x['refiner_session_stats']['call_count'],
                    raw_equal=x['raw_text']==baseline['raw_text'],clean_equal=x['clean_text']==baseline['clean_text'],
                    raw_differences=differences(baseline['raw_text'],x['raw_text']),
                    clean_differences=differences(baseline['clean_text'],x['clean_text']),
                    rules_unchanged=m['configuration']['hashes']['zh-itn/zh-itn/build/libzh_itn_bridge.so']==baseline['_speed_benchmark']['configuration']['hashes']['zh-itn/zh-itn/build/libzh_itn_bridge.so']))
            if runs:
                rows.append(dict(audio=audio,variant=group,runs=runs,
                    mean_elapsed_s=statistics.mean(r['elapsed_s'] for r in runs),
                    mean_finish_s=statistics.mean(r['finish_s'] for r in runs),
                    mean_refiner_wall_s=statistics.mean(r['refiner_wall_s'] for r in runs),
                    all_raw_equal=all(r['raw_equal'] for r in runs),
                    all_clean_equal=all(r['clean_equal'] for r in runs),
                    all_rules_unchanged=all(r['rules_unchanged'] for r in runs)))
    (root/'comparison.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2)+'\n')
    table=['# 转录速度与精修内容对照','',
           '同一模型、v90 数字规则、2 秒解码间隔。基线、评分优化和关闭压缩各两轮；2 秒传输因内容变化被拒，仅一轮。','',
           '| 音频 | 方案 | 总耗时均值(s) | 收尾均值(s) | 精修调用耗时均值(s) | 原文一致 | 精修逐字一致 |',
           '| --- | --- | ---: | ---: | ---: | --- | --- |']
    names=dict(baseline='基线',transport='2秒传输（放弃）',batch='双句批处理（回退）',
               crop_cache='评分裁剪缓存（回退）',uncompressed='关闭压缩',deployed='部署后验证')
    for row in rows:
        table.append(f"| {row['audio']} | {names[row['variant']]} | {row['mean_elapsed_s']:.3f} | {row['mean_finish_s']:.3f} | {row['mean_refiner_wall_s']:.3f} | {'是' if row['all_raw_equal'] else '否'} | {'是' if row['all_clean_equal'] else '否'} |")
        print({k:v for k,v in row.items() if k!='runs'})
    (root/'comparison.md').write_text('\n'.join(table)+'\n')


if __name__=='__main__':main()
