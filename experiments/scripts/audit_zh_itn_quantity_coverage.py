"""Compare the native ITN against generated quantity and preservation examples."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.zh_itn_bridge import ZhITNNormalizer


def spoken(n):
    digits = '零一二三四五六七八九'
    if n < 10:
        return digits[n]
    if n < 100:
        return ('' if n // 10 == 1 else digits[n // 10]) + '十' + (digits[n % 10] if n % 10 else '')
    for value, unit in ((100000000, '亿'), (10000, '万'), (1000, '千'), (100, '百')):
        if n >= value:
            high, low = divmod(n, value)
            return spoken(high) + unit + (('零' if low < value // 10 else '') + spoken(low) if low else '')


def cases():
    rows = []
    def add(kind, source, expected):
        rows.append(dict(category=kind, input=source, expected=expected))
    for n in (2, 3, 5, 9, 10, 11, 13, 17, 20, 44, 50, 70, 99, 100, 101, 200, 256, 1000, 2000, 3000):
        for unit, noun in [('个', '法器'), ('条', '河流'), ('座', '城市'), ('种', '方案'), ('位', '同事'),
                           ('名', '嘉宾'), ('只', '鸟'), ('本', '书'), ('台', '设备'), ('首', '歌'),
                           ('张', '照片'), ('家', '企业'), ('次', '检查'), ('级', '泵站'), ('根', '管道'),
                           ('份', '材料'), ('册', '教材'), ('双', '鞋'), ('支', '笔'), ('杯', '茶')]:
            for prefix in ('现场有', '总计', '包含'):
                add('classifier', prefix + spoken(n) + unit + noun + '。', prefix + str(n) + unit + noun + '。')
        for modifier in ('多', '余', '来'):
            if n >= 10:
                for unit in ('条', '座', '次', '种', '年', '公里', '吨'):
                    add('approximation', spoken(n) + modifier + unit + '。', str(n) + modifier + unit + '。')
                for magnitude in ('万', '亿'):
                    for unit in ('条', '座', '吨', '立方米'):
                        add('infix_magnitude', spoken(n) + modifier + magnitude + unit + '。', str(n) + modifier + magnitude + unit + '。')
        for magnitude in ('万', '亿'):
            for unit in ('吨', '立方米', '公里'):
                add('large_measure', '总量是' + spoken(n) + magnitude + unit + '。', '总量是' + str(n) + magnitude + unit + '。')
        add('multiplier', '水量是平时的' + spoken(n) + '倍之多。', '水量是平时的' + str(n) + '倍之多。')
        if n < 10:
            for unit, noun in [('件', '设备'), ('辆', '车'), ('颗', '卫星'), ('箱', '货物'), ('批', '产品')]:
                add('small_existing_classifier', '共有' + spoken(n) + unit + noun + '。', '共有' + str(n) + unit + noun + '。')
    for source in ('一个方案。', '有一位同事。', '一条条河流。', '一首首歌。', '一根根线条。',
                   '三番五次、一五一十、七嘴八舌。', '一条龙服务。', '一刀切，一把手。', '万一、一度、一直以来。',
                   '一多就乱，三多一少。', '三四个人，二三十个人。', '两个两种选择。',
                   '二十年后的今天。', '八年过去了。',
                   '只有两个两种选择。', '三千多万一千条。', '二十年后的今天。',
                   '原文写作“三十个苹果”。', '《十六亿吨》是书名。', '百分之五。', '三心二意不是三个心两个意。',
                   '有两条约。', '有三种子。', '发动机有二点儿零T两种排量。', '一九九八年。',
                   '1到2米，450亿立方米。', '这里有四点五万条江河。'):
        expected = '5%。' if source == '百分之五。' else source
        if source == '二十年后的今天。': expected = '20年后的今天。'
        if source == '八年过去了。': expected = '8年过去了。'
        if source == '三心二意不是三个心两个意。': expected = '三心二意不是3个心2个意。'
        if source == '一九九八年。': expected = '1998年。'
        if source == '这里有四点五万条江河。': expected = '这里有4.5万条江河。'
        add('preserve', source, expected)
    for source, expected in [('我让他五个法器。再不行，我让他十个。', '我让他5个法器。再不行，我让他10个。'),
                             ('六十多万条沟谷，七十多年。', '60多万条沟谷，70多年。'),
                             ('从曾经的十六亿吨减少到了两亿吨。', '从曾经的16亿吨减少到了2亿吨。'),
                             ('相当于十七条黄河。', '相当于17条黄河。'),
                             ('十三级泵站一级一级地提水。', '13级泵站一级一级地提水。'),
                             ('二零一七年元旦前，票房就已突破十亿大关。', '2017年元旦前，票房就已突破10亿大关。'),
                             ('九十年代初，二千年即将到来。', '90年代初，2000年即将到来。'),
                             ('从一九七一年十二岁入伍，一直到二十五岁。', '从1971年12岁入伍，一直到25岁。'),
                             ('如今多达四十六处，跨越八个省份。', '如今多达46处，跨越8个省份。'),
                             ('水位更低两米，凌空而过九公里。', '水位更低2米，凌空而过9公里。'),
                             ('相当于二点六个三峡，建起三座三峡。', '相当于2.6个三峡，建起3座三峡。'),
                             ('多达六百多万人，陈灿八十五环、郝淑芬一百零三环。', '多达600多万人，陈灿85环、郝淑芬103环。'),
                             ('如果你有一百万，一千万呢？', '如果你有100万，1000万呢？'),
                             ('三到四个人，一到两米。', '3到4个人，1到2米。'),
                             ('共有两份材料和两百条河流。', '共有2份材料和200条河流。'),
                             ('十六亿吨下降到两亿吨。', '16亿吨下降到2亿吨。'),
                             ('十六亿吨到两亿吨。', '16亿吨到2亿吨。'),
                             ('十六亿吨至两亿吨。', '16亿吨至2亿吨。'),
                             ('十六亿吨~两亿吨。', '16亿吨~2亿吨。'),
                             ('一万平方公里至两万立方米。', '一万平方公里至两万立方米。'),
                             ('十六亿吨减少到了两亿立方米。', '十六亿吨减少到了两亿立方米。'),
                             ('一百年好合，十万火急，千万别失望。', '一百年好合，十万火急，千万别失望。'),
                             ('七零八零年代的战友情。', '七零八零年代的战友情。'),
                             ('只有两个两种选择。', '只有两个两种选择。')]:
        add('recording', source, expected)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    normalizer = ZhITNNormalizer(args.library.resolve())
    totals = {}
    rows = []
    for case in cases():
        result = normalizer.normalize(case['input'])
        repeated = normalizer.normalize(result.text).text
        row = dict(case, actual=result.text, passed=result.text == case['expected'], idempotent=repeated == result.text)
        rows.append(row)
        counts = totals.setdefault(case['category'], {'total': 0, 'passed': 0})
        counts['total'] += 1
        counts['passed'] += int(row['passed'])
    report = dict(rule_pack=normalizer.rule_pack, total=len(rows), passed=sum(r['passed'] for r in rows),
                  idempotent=sum(r['idempotent'] for r in rows), categories=totals, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, ensure_ascii=False))
    for row in [r for r in rows if not r['passed']][:12]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
