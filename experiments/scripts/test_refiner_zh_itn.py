#!/usr/bin/env python3
"""Compare Refiner-only output with number-protected Refiner + zh-itn output.

The generated cases cover numeric normalization, mixed cleanup/numeric text,
and non-numeric transcript cleanup. Numeric-looking spans are masked before
the Refiner runs, restored verbatim, and then normalized by the zh-itn CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import random
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "itn_refiner_integration_test.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from postprocess_asr import ModelConfig, TransformersPostprocessor  # noqa: E402
from postprocess_asr import SYSTEM_PROMPT as BASE_REFINER_SYSTEM_PROMPT  # noqa: E402


_NUMERAL_CHARS = "0-9０-９零〇一二两三四五六七八九十百千万亿幺洞拐勾壹贰叁肆伍陆柒捌玖拾佰仟萬億几"
_NUMBER_TOKEN = rf"[{_NUMERAL_CHARS}]+(?:[.．点:：][{_NUMERAL_CHARS}]+)*"
_NUMBER_UNITS = (
    "个百分点", "百分点", "平方公里", "立方米", "平方米", "平方厘米",
    "人民币", "美元", "欧元", "港币", "块钱", "千米", "公里", "厘米",
    "毫米", "公斤", "千克", "毫升", "小时", "分钟", "秒钟", "毫秒",
    "星期", "季度", "世纪", "年代", "月份", "个月", "分钟", "天",
    "年", "月", "日", "号", "点", "时", "分", "秒", "元", "块",
    "角", "个", "只", "件", "本", "张", "台", "条", "位", "名",
    "辆", "套", "箱", "袋", "杯", "份", "枚", "颗", "粒", "栋",
    "间", "户", "艘", "架", "门", "次", "把", "双", "家", "人",
    "层", "场", "趟", "轮", "群", "阵", "束", "串", "吨", "升",
    "米", "克", "度", "岁", "倍", "成", "折", "%", "％",
)
_UNIT_PATTERN = "(?:" + "|".join(
    re.escape(unit) for unit in sorted(set(_NUMBER_UNITS), key=len, reverse=True)
) + ")"
_KEY_SUFFIX_RE = re.compile(r"\s*<KEY>\s*\[[^\]]*\]\s*$")
_NUMBERISH_RE = re.compile(
    rf"(?<![{_NUMERAL_CHARS}])"
    rf"(?P<prefix>(?:百分之|零下|负|第|初|星期|周)\s*)?"
    rf"(?P<number>{_NUMBER_TOKEN})"
    rf"(?P<range>\s*(?:到|至|[-~～])\s*{_NUMBER_TOKEN})?"
    rf"(?P<approx>半|多|余|来|几)?"
    rf"(?P<unit>{_UNIT_PATTERN})?"
    rf"(?P<approx_after>半)?"
)
_SPOKEN_GB_RE = re.compile(
    rf"(?<![{_NUMERAL_CHARS}])(?P<number>[{_NUMERAL_CHARS}]+)\s*G(?:\s*B)?(?![A-Za-z])"
)
_LATIN_IDENTIFIER_RE = re.compile(
    rf"(?P<prefix>(?:[A-Za-z][A-Za-z0-9]*|F杠)\s*)"
    rf"(?P<number>[{_NUMERAL_CHARS}]+)"
)
_DAYPARTS = ("凌晨", "清晨", "早上", "上午", "中午", "下午", "晚上")
_DATE_END_RE = re.compile(rf"[{_NUMERAL_CHARS}]+月[{_NUMERAL_CHARS}]+(?:日|号)$")
_CLOCK_RE = re.compile(rf"[{_NUMERAL_CHARS}]+点")
_PLACEHOLDER_RE = re.compile(r"__ENTITY_\d{3}__")
REFINER_TEST_SYSTEM_PROMPT = (
    BASE_REFINER_SYSTEM_PROMPT.replace(
        "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。",
        "此测试不抽取实体，不要在末尾追加 <KEY>、实体列表或其他元数据。",
    )
    + "本测试中的数字表达已临时替换为不可编辑的 __ENTITY_NNN__ 标记；"
    "数字原文稍后恢复并交由 zh_itn 规范化。请原样保留每个标记一次且保持顺序，"
    "只做最小的口语清理：删除明显重复、语气词和口误，必要时调整标点；"
    "保留其他原文的词序、语义锚点和句子结构，不要重写、概括或拆成编号列表。"
)


@dataclass(frozen=True, slots=True)
class TestCase:
    case_id: str
    category: str
    text: str
    expected_fragments: tuple[str, ...] = ()


def generated_cases() -> tuple[TestCase, ...]:
    """Return 100 reproducible smoke and long-sentence stress cases."""

    core_cases = (
        TestCase(
            "date_percent",
            "numeric",
            "二零二四年五月三日增长百分之十。",
            ("2024年5月3日", "10%"),
        ),
        TestCase(
            "time",
            "numeric",
            "下午三点半开始。",
            ("下午3点半",),
        ),
        TestCase(
            "money_range",
            "numeric",
            "合计需要四千到五千元。",
            ("4000到5000元",),
        ),
        TestCase(
            "phone",
            "numeric",
            "联系电话幺三九九幺二六三七九七。",
            ("13991263797",),
        ),
        TestCase(
            "software_version",
            "numeric",
            "软件版本为一点零点三。",
            ("1.0.3",),
        ),
        TestCase(
            "duration",
            "numeric",
            "设备运行十六小时三十三分钟后停止。",
            ("16小时33分钟",),
        ),
        TestCase(
            "large_unit_decimal",
            "numeric",
            "合计获配两千零二十四点九七万股。",
            ("2024.97万股",),
        ),
        TestCase(
            "unanchored_count_preserved",
            "numeric",
            "我买了三本书。",
            ("三本书",),
        ),
        TestCase(
            "idiom_preserved",
            "numeric",
            "我们一五一十地把经过说清楚。",
            ("一五一十",),
        ),
        TestCase(
            "ambiguous_quantity_preserved",
            "numeric",
            "大概二三十个人会来。",
            ("二三十个人",),
        ),
        TestCase(
            "filler_repetition_with_number",
            "mixed",
            "呃，我我还需要十到十五分钟，后面再确认一下。",
            ("10到15分钟",),
        ),
        TestCase(
            "filler_with_percent",
            "mixed",
            "这份报告其实写得不错，呃，里面提到百分之三十七，结论还要核实。",
            ("37%",),
        ),
        TestCase(
            "repetition_with_count",
            "mixed",
            "这个方案我们我们至少十五秒检查一遍，先按原计划推进。",
            ("至少15秒",),
        ),
        TestCase(
            "filler",
            "non_numeric_cleanup",
            "嗯，我觉得这个方案可以，呃，先按这个方向推进吧。",
        ),
        TestCase(
            "repetition",
            "non_numeric_cleanup",
            "这个问题我们我们已经讨论过了，先按原计划推进。",
        ),
        TestCase(
            "self_correction",
            "non_numeric_cleanup",
            "我本来想把文件发给张伟，不对，应该发给李明。",
        ),
        TestCase(
            "punctuation",
            "non_numeric_cleanup",
            "其实我觉得这个方案挺好只是细节还需要再确认",
        ),
        TestCase(
            "mixed_language",
            "non_numeric_cleanup",
            "我用的是Qwen-ASR模型，它识别的文本还需要再整理一下。",
        ),
        TestCase(
            "filler_and_false_start",
            "non_numeric_cleanup",
            "那个，我刚才想说，呃，我们先把重点说清楚。",
        ),
    )

    # These phrase-level fixtures have positive expectations in the zh-itn
    # C++ rule tests. Combining three independent clauses per sample stresses
    # long-context marker preservation and multiple ITN decisions in one run.
    facts: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("在二零二零年一月十四号举行", ("2020年1月14号",)),
        ("一九五五年九月二十七号成立", ("1955年9月27号",)),
        ("公元二九九九年十二月三十一日", ("2999年12月31日",)),
        ("当地时间三月十三日下午", ("3月13日下午",)),
        ("七月六日下午三点", ("7月6日下午3点",)),
        ("二月二十九日", ("2月29日",)),
        ("八月中旬", ("8月中旬",)),
        ("占比百分之九十八点五", ("98.5%",)),
        ("增长百分之一百", ("100%",)),
        ("约百分之零点五", ("0.5%",)),
        ("占比百分之二十到百分之二十五", ("20%到25%",)),
        ("增长百分之三十到四十", ("30%到40%",)),
        ("占比百分之一点五至二点五", ("1.5%至2.5%",)),
        ("波动区间百分之五十-六十", ("50%-60%",)),
        ("三月一日上午九点和三月二日下午三点", ("3月1日上午9点", "3月2日下午3点")),
        ("十六小时三十三分钟和二十小时十四分钟", ("16小时33分钟", "20小时14分钟")),
        ("十到十五分钟", ("10到15分钟",)),
        ("三十六到四十八小时", ("36到48小时",)),
        ("每小时二十五到三十公里", ("每小时25到30公里",)),
        ("四千到五千元", ("4000到5000元",)),
        ("二十三到二十八万元", ("23到28万元",)),
        ("一千到两千美元", ("1000到2000美元",)),
        ("一百万美元至一千万美元", ("100万美元至1000万美元",)),
        ("十九点六八万元-二十三点六八万元", ("19.68万元-23.68万元",)),
        ("二十四点九九到二十九点九九万元", ("24.99到29.99万元",)),
        (
            "两千三百八十七点四亿元增长至两千七百零一点一亿元",
            ("2387.4亿元", "2701.1亿元"),
        ),
        ("海外采购预算不少于两千万美元", ("2000万美元",)),
        ("普通商品价格为一百至两百元", ("100至200元",)),
        ("反射面约在一百平方米", ("100平方米",)),
        ("大部分地区录得超过八十毫米雨量", ("80毫米",)),
        ("路线大概十公里", ("10公里",)),
        ("全程一百公里前后", ("100公里",)),
        ("道路每一百公里设置一个服务区", ("每100公里",)),
        ("至少十五秒", ("至少15秒",)),
        ("二十分钟以上它就凉了", ("20分钟以上",)),
        ("软件版本为一点零点三", ("1.0.3",)),
        ("驾考宝典版本为七点八点五", ("7.8.5",)),
        ("版本号二十点零三", ("20.03",)),
        ("联系电话幺三九九幺二六三七九七", ("13991263797",)),
        ("联系电话：零二九八六五五四二六二", ("02986554262",)),
        ("联系电话四零零零二九三零六零", ("4000293060",)),
        ("订票服务热线九五三三九办理", ("95339",)),
        ("我的电话是一二三四五六七", ("1234567",)),
        ("铁路一二三零六网站已开放", ("12306",)),
        ("消费者可通过一二三幺五投诉举报", ("12315",)),
        ("中国移动一零零八六发送短信", ("10086",)),
        ("差点打了一二零", ("120",)),
        ("必须戴N九五口罩", ("N95",)),
        ("奔驰四S店已经开业", ("4S店",)),
        ("二零零一年索尼PS二两千万台的订单", ("2001年", "PS2")),
        ("美国达美航空公司一架波音七三七八零零客机降落", ("波音737-800",)),
        ("一架F杠二十二隐形战斗机起飞", ("F-22",)),
        ("米二直升机完成任务", ("米2直升机",)),
        ("服务中心安排二十四小时值班", ("24小时",)),
        ("维护时间大约二十分钟", ("大约20分钟",)),
        ("培训至少三十分钟", ("至少30分钟",)),
        ("体验路线大约一点五公里", ("大约1.5公里",)),
        ("运输距离不超过一百公里", ("不超过100公里",)),
        ("普通商品价格为一百元", ("100元",)),
        ("纪念品价格约为一百元", ("约为100元",)),
        ("充电用时不到十分钟", ("不到10分钟",)),
        ("接近八十五分钟", ("接近85分钟",)),
        ("煮三十分钟左右", ("煮30分钟左右",)),
        ("位于以南约四百五十公里", ("约450公里",)),
        ("二零一四年和二零一五年", ("2014年", "2015年")),
        ("已有16小时另用二十小时十四分钟", ("已有16小时", "20小时14分钟")),
        ("在呼叫幺二零后", ("呼叫120",)),
        ("微信投诉一二三四五打电话投诉", ("12345",)),
        ("电话是四零零零二九三零六零，备用热线九五三三九", ("4000293060", "95339")),
        ("预算为一千到两千美元，额外预留四千到五千元", ("1000到2000美元", "4000到5000元")),
        ("占比百分之九十八点五，增长百分之三十到四十", ("98.5%", "30%到40%")),
        ("设备从三月一日上午九点运行到三月二日下午三点", ("3月1日上午9点", "3月2日下午3点")),
        ("型号为N九五，软件版本为一点零点三", ("N95", "1.0.3")),
        ("我买了三本书", ("三本书",)),
        ("我们一五一十地说明经过", ("一五一十",)),
        ("大概二三十个人会来", ("二三十个人",)),
        ("失败率达到百分之九十多", ("百分之九十多",)),
        ("缓存大小五 G B", ("五 G B",)),
    )
    templates = (
        "跨部门复盘会上，主持人回顾时提到{a}，运营组接着补充{b}，财务组最后核对{c}，请把这段连续口述整理后写入正式纪要。",
        "为了整理项目周报，我先听到{a}，旁边同事又补了一句{b}，收尾时还提到{c}，麻烦把口语部分理顺后发给负责人审核。",
        "会议录音里先后说到{a}、{b}和{c}，整理人员需要保留每处日期、比例、金额或联系方式，再把整段内容合并成清楚的说明。",
        "项目负责人解释背景时先说{a}，随后又提到{b}，最后确认了{c}；这些信息都来自同一段连续口述，不能在精修时遗漏。",
        "我把本周的记录重新捋了一遍，里面提到{a}，还说了{b}，另一处补充的是{c}，会后需要整理成可以直接归档的文字。",
    )
    fact_combinations = list(combinations(range(len(facts)), 3))
    selected_combinations = random.Random(20260928).sample(fact_combinations, 70)
    long_numeric_cases: list[TestCase] = []
    for index in range(70):
        first, second, third = selected_combinations[index]
        selected = (facts[first], facts[second], facts[third])
        text = templates[index % len(templates)].format(
            a=selected[0][0], b=selected[1][0], c=selected[2][0]
        )
        if index < 12:
            text = "呃，我我先把记录说完整，" + text
            category = "mixed"
        else:
            category = "numeric"
        expected = tuple(fragment for _, fragments in selected for fragment in fragments)
        long_numeric_cases.append(
            TestCase(f"long_stress_{index + 1:03d}", category, text, expected)
        )

    long_non_numeric_texts = (
        "嗯，关于项目说明，我我已经和运营同事核对过了，细节确认以后再发给你，免得大家拿到不同版本。",
        "这个结论其实我觉得整体上没有问题，呃，只是表达还可以更清楚一些，最后一段最好再顺一遍。",
        "我本来准备把会议纪要发给周老师，不对，应该先交给负责审核的林老师看完以后再统一转发。",
        "那个，我刚才想说的是先把背景交代清楚，然后再讨论具体方案，大家听完以后再一起判断。",
        "这段内容我们已经已经整理过一次，麻烦再检查语气和前后衔接，确认以后放进最终版本。",
        "我用的是语音识别出来的初稿，里面有些停顿和重复表达，整理时保留原意就可以，不必改得太书面。",
        "其实这个安排挺合适的只是前面那句话读起来不太顺我建议把顺序稍微调整一下再发给团队",
        "嗯嗯，我先把这部分内容看完，呃，然后再把需要补充的地方集中整理出来，免得遗漏重点。",
        "刚刚，那个，我想确认一下，这个文件是不是已经给客户看过了，如果还没有我先帮忙检查。",
        "你你先别着急，我把刚才的修改意见重新说一遍，前面的部分不用动，后面的说明再顺一下。",
        "我们这边已经和设计同事沟通好了，接下来由我负责整理文字，完成以后再请你确认是否合适。",
    )
    long_non_numeric_cases = tuple(
        TestCase(f"long_non_numeric_{index + 1:02d}", "non_numeric_cleanup", text)
        for index, text in enumerate(long_non_numeric_texts)
    )
    cases = core_cases + tuple(long_numeric_cases) + long_non_numeric_cases
    if len(cases) != 100:
        raise AssertionError(f"expected exactly 100 test cases, got {len(cases)}")
    return cases


def _mask_numeric_spans(text: str) -> tuple[str, list[dict[str, str]]]:
    if _PLACEHOLDER_RE.search(text):
        raise ValueError("input text already contains a reserved __ENTITY_NNN__ marker")
    raw_matches: list[tuple[int, int]] = []
    for match in _NUMBERISH_RE.finditer(text):
        if not match.group(0):
            continue
        if (
            match.group("prefix") is None
            and match.group("range") is None
            and match.group("approx") is None
            and match.group("approx_after") is None
            and match.group("unit") is None
            and match.group("number") in "零〇一二两三四五六七八九几"
        ):
            # A bare 一 in lexical phrases such as “一下” is not a numeric
            # span. Contextual counts, ordinals, weekdays, and multi-digit
            # numerals are still protected.
            continue
        start = match.start()
        for daypart in _DAYPARTS:
            if start >= len(daypart) and text[start - len(daypart) : start] == daypart:
                start -= len(daypart)
                break
        raw_matches.append((start, match.end()))
    raw_matches.extend(
        (match.start(), match.end()) for match in _SPOKEN_GB_RE.finditer(text)
    )
    raw_matches.extend(
        (match.start(), match.end()) for match in _LATIN_IDENTIFIER_RE.finditer(text)
    )
    raw_matches.sort()

    matches: list[tuple[int, int]] = []
    for start, end in raw_matches:
        if matches:
            previous_start, previous_end = matches[-1]
            gap = text[previous_end:start]
            previous_text = text[previous_start:previous_end]
            joins_percentage_range = gap == "之"
            joins_date_and_clock = (
                gap in _DAYPARTS
                and _DATE_END_RE.search(previous_text) is not None
                and _CLOCK_RE.match(text[start:end]) is not None
            )
            if start <= previous_end or joins_percentage_range or joins_date_and_clock:
                matches[-1] = (previous_start, max(previous_end, end))
                continue
        matches.append((start, end))

    # A trailing daypart is an ITN context cue even when the time has no clock
    # value (for example, “当地时间三月十三日下午”). Keep it with the date.
    extended_matches: list[tuple[int, int]] = []
    for start, end in matches:
        matched_text = text[start:end]
        if _DATE_END_RE.search(matched_text):
            daypart = next(
                (value for value in _DAYPARTS if text.startswith(value, end)),
                None,
            )
            if daypart is not None:
                end += len(daypart)
        if extended_matches and extended_matches[-1][1] >= start:
            previous_start, previous_end = extended_matches[-1]
            extended_matches[-1] = (previous_start, max(previous_end, end))
        else:
            extended_matches.append((start, end))
    matches = extended_matches
    if not matches:
        return text, []

    spans: list[dict[str, str]] = []
    pieces: list[str] = []
    cursor = 0
    for index, (start, end) in enumerate(matches):
        if index >= 1000:
            raise ValueError("one test sentence contains more than 999 numeric spans")
        marker = f"__ENTITY_{index:03d}__"
        pieces.append(text[cursor:start])
        pieces.append(marker)
        spans.append({"marker": marker, "text": text[start:end]})
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), spans


def _restore_markers(candidate: str, source: str, spans: list[dict[str, str]]) -> tuple[str, str | None]:
    expected = [span["marker"] for span in spans]
    actual = _PLACEHOLDER_RE.findall(candidate)
    if actual != expected or any(candidate.count(marker) != 1 for marker in expected):
        return source, "protected_numeric_marker_missing_duplicated_or_reordered"
    restored = candidate
    for span in spans:
        restored = restored.replace(span["marker"], span["text"], 1)
    return restored, None


def _strip_key_metadata(candidate: str) -> tuple[str, str | None]:
    """Remove the Refiner's optional entity-list suffix from transcript text."""

    if "<KEY>" not in candidate:
        return candidate, None
    cleaned, replacements = _KEY_SUFFIX_RE.subn("", candidate, count=1)
    if replacements != 1 or "<KEY>" in cleaned:
        return candidate, "malformed_refiner_key_metadata"
    return cleaned.rstrip(), None


def _find_itn_cli(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"zh_itn executable not found: {path}")
        return path
    candidates = (
        PROJECT_ROOT / "zh-itn" / "zh-itn" / "build" / "zh_itn",
        PROJECT_ROOT / "zh-itn" / "zh-itn" / "build" / "Release" / "zh_itn.exe",
        PROJECT_ROOT / "zh-itn" / "zh-itn" / "build" / "zh_itn.exe",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "zh_itn executable was not found. Build it with "
        "cmake -S zh-itn/zh-itn -B /tmp/zh-itn-build -DCMAKE_BUILD_TYPE=Release "
        "and cmake --build /tmp/zh-itn-build --config Release; then pass "
        "--itn-cli /tmp/zh-itn-build/zh_itn."
    )


def _run_itn(executable: Path, texts: list[str]) -> list[dict[str, Any]]:
    request = {
        "schema_version": 3,
        "locale": "zh-CN",
        "profile": "conservative",
        "segments": [
            {"id": index, "text": text}
            for index, text in enumerate(texts)
        ],
    }
    completed = subprocess.run(
        [str(executable)],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"zh_itn returned invalid JSON (exit={completed.returncode}): "
            f"{completed.stdout[:500]} {completed.stderr[:500]}"
        ) from error
    if completed.returncode != 0:
        raise RuntimeError(f"zh_itn failed: {response}")
    if response.get("schema_version") != 3 or not isinstance(response.get("segments"), list):
        raise RuntimeError(f"unexpected zh_itn response: {response}")
    segments = response["segments"]
    if len(segments) != len(texts):
        raise RuntimeError("zh_itn returned a different number of segments")
    for index, segment in enumerate(segments):
        if segment.get("id") != index or segment.get("status") == "error":
            raise RuntimeError(f"zh_itn segment {index} failed: {segment}")
    return segments


def _generate_batches(
    processor: TransformersPostprocessor,
    texts: list[str],
    batch_size: int,
) -> tuple[list[str], list[str | None]]:
    outputs: list[str] = []
    errors: list[str | None] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        try:
            generated, _latency_ms = processor.generate(
                batch,
                system_prompt=REFINER_TEST_SYSTEM_PROMPT,
            )
        except Exception as error:  # Keep a source fallback for the diagnostic report.
            message = f"{type(error).__name__}: {error}"
            outputs.extend(batch)
            errors.extend([message] * len(batch))
        else:
            outputs.extend(generated)
            errors.extend([None] * len(generated))
    return outputs, errors


def _case_result(
    case: TestCase,
    masked_text: str,
    spans: list[dict[str, str]],
    refiner_masked_output: str,
    restored_text: str,
    placeholder_error: str | None,
    itn_segment: dict[str, Any],
    refiner_error: str | None,
    refiner_only_text: str | None,
) -> dict[str, Any]:
    itn_text = itn_segment["output"]
    missing = [fragment for fragment in case.expected_fragments if fragment not in itn_text]
    return {
        **asdict(case),
        "expected_fragments": list(case.expected_fragments),
        "masked_text": masked_text,
        "numeric_spans": spans,
        "refiner_masked_output": refiner_masked_output,
        "refiner_placeholder_error": placeholder_error,
        "refiner_error": refiner_error,
        "refiner_only_text": refiner_only_text,
        "refiner_restored_text": restored_text,
        "itn_text": itn_text,
        "itn_status": itn_segment.get("status"),
        "itn_mappings": itn_segment.get("mappings", []),
        "expected_fragments_pass": not missing,
        "missing_expected_fragments": missing,
        "same_as_refiner_only": (
            refiner_only_text == itn_text if refiner_only_text is not None else None
        ),
        "refiner_changed_context": (
            refiner_masked_output != masked_text if refiner_error is None else False
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run generated Chinese transcript cases through a Refiner with numeric "
            "spans protected, then normalize them using zh_itn."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--itn-cli", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    return args


def run() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Refiner model not found: {model_path}")
    itn_cli = _find_itn_cli(args.itn_cli)
    cases = generated_cases()
    masked: list[str] = []
    spans_by_case: list[list[dict[str, str]]] = []
    for case in cases:
        masked_text, spans = _mask_numeric_spans(case.text)
        masked.append(masked_text)
        spans_by_case.append(spans)

    config = ModelConfig(
        model_path=str(model_path),
        device_map="auto" if args.device_map == "auto" else args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    processor = TransformersPostprocessor(config)

    non_numeric_indices = [
        index for index, case in enumerate(cases)
        if case.category == "non_numeric_cleanup"
    ]
    non_numeric_texts = [cases[index].text for index in non_numeric_indices]
    non_numeric_control, control_errors = _generate_batches(
        processor, non_numeric_texts, args.batch_size
    )
    non_numeric_locked, locked_errors = _generate_batches(
        processor, [masked[index] for index in non_numeric_indices], args.batch_size
    )

    for offset, candidate in enumerate(non_numeric_control):
        cleaned, metadata_error = _strip_key_metadata(candidate)
        non_numeric_control[offset] = cleaned
        if metadata_error is not None:
            control_errors[offset] = metadata_error
    for offset, candidate in enumerate(non_numeric_locked):
        cleaned, metadata_error = _strip_key_metadata(candidate)
        non_numeric_locked[offset] = cleaned
        if metadata_error is not None:
            locked_errors[offset] = metadata_error

    non_numeric_control_by_index = dict(zip(non_numeric_indices, non_numeric_control))
    non_numeric_errors_by_index = dict(zip(non_numeric_indices, control_errors))
    locked_by_index: dict[int, str] = {}
    refiner_errors_by_index: dict[int, str | None] = {}
    for index, output, error in zip(non_numeric_indices, non_numeric_locked, locked_errors):
        locked_by_index[index] = output
        refiner_errors_by_index[index] = error

    remaining_indices = [
        index for index, case in enumerate(cases)
        if case.category != "non_numeric_cleanup"
    ]
    remaining_outputs, remaining_errors = _generate_batches(
        processor, [masked[index] for index in remaining_indices], args.batch_size
    )
    for offset, candidate in enumerate(remaining_outputs):
        cleaned, metadata_error = _strip_key_metadata(candidate)
        remaining_outputs[offset] = cleaned
        if metadata_error is not None:
            remaining_errors[offset] = metadata_error
    for index, output, error in zip(remaining_indices, remaining_outputs, remaining_errors):
        locked_by_index[index] = output
        refiner_errors_by_index[index] = error

    restored_by_index: dict[int, str] = {}
    placeholder_errors: dict[int, str | None] = {}
    for index, case in enumerate(cases):
        restored, placeholder_error = _restore_markers(
            locked_by_index[index], masked[index], spans_by_case[index]
        )
        if placeholder_error is not None:
            restored = case.text
        restored_by_index[index] = restored
        placeholder_errors[index] = placeholder_error

    itn_segments = _run_itn(itn_cli, [restored_by_index[index] for index in range(len(cases))])
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        rows.append(
            _case_result(
                case,
                masked[index],
                spans_by_case[index],
                locked_by_index[index],
                restored_by_index[index],
                placeholder_errors[index],
                itn_segments[index],
                refiner_errors_by_index[index],
                non_numeric_control_by_index.get(index),
            )
        )

    non_numeric_rows = [
        row for row in rows if row["category"] == "non_numeric_cleanup"
    ]
    mixed_rows = [row for row in rows if row["category"] == "mixed"]
    non_numeric_pass = all(
        row["refiner_error"] is None
        and non_numeric_errors_by_index.get(index) is None
        and row["refiner_placeholder_error"] is None
        and row["same_as_refiner_only"] is True
        and row["itn_status"] == "unchanged"
        for index, row in enumerate(rows)
        if row["category"] == "non_numeric_cleanup"
    )
    expected_pass = all(row["expected_fragments_pass"] for row in rows)
    non_numeric_cleanup_observed = any(
        row["refiner_changed_context"] for row in non_numeric_rows
    )
    mixed_cleanup_observed = any(
        row["refiner_changed_context"] for row in mixed_rows
    )
    mixed_refinement_pass = (
        mixed_cleanup_observed
        and all(
            row["refiner_error"] is None
            and row["refiner_placeholder_error"] is None
            for row in mixed_rows
        )
    )
    refiner_integrity_pass = all(
        row["refiner_error"] is None and row["refiner_placeholder_error"] is None
        for row in rows
    )
    refiner_fallback_cases = [
        row["case_id"] for row in rows
        if row["refiner_error"] is not None or row["refiner_placeholder_error"] is not None
    ]
    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    passed = (
        non_numeric_pass
        and expected_pass
        and non_numeric_cleanup_observed
        and mixed_refinement_pass
        and refiner_integrity_pass
    )
    report = {
        "status": "passed" if passed else "failed",
        "case_count": len(rows),
        "category_counts": category_counts,
        "numeric_span_count": sum(len(row["numeric_spans"]) for row in rows),
        "itn_applied_case_count": sum(row["itn_status"] == "applied" for row in rows),
        "refiner_model": str(model_path),
        "zh_itn_cli": str(itn_cli),
        "refiner_integrity_pass": refiner_integrity_pass,
        "refiner_fallback_cases": refiner_fallback_cases,
        "non_numeric_isolation_pass": non_numeric_pass,
        "non_numeric_cleanup_observed": non_numeric_cleanup_observed,
        "mixed_refinement_pass": mixed_refinement_pass,
        "expected_itn_outputs_pass": expected_pass,
        "mixed_refiner_fallback_cases": [
            row["case_id"] for row in mixed_rows
            if row["refiner_error"] is not None
            or row["refiner_placeholder_error"] is not None
        ],
        "cases": rows,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Status: {report['status']}")
    print(f"Cases: {report['case_count']} {report['category_counts']}")
    print(
        f"Protected numeric spans: {report['numeric_span_count']}; "
        f"zh-itn changed {report['itn_applied_case_count']} cases"
    )
    print(
        "Refiner/marker integrity: "
        f"{'PASS' if refiner_integrity_pass else 'FAIL'} "
        f"(fallback cases={refiner_fallback_cases})"
    )
    print(
        "Non-numeric Refiner isolation: "
        f"{'PASS' if non_numeric_pass else 'FAIL'} "
        f"({len(non_numeric_rows)} cases)"
    )
    print(f"zh-itn expected outputs: {'PASS' if expected_pass else 'FAIL'}")
    print(
        "Refiner cleanup observed: "
        f"non-numeric={non_numeric_cleanup_observed}, "
        f"mixed={mixed_cleanup_observed}"
    )
    print(f"Mixed-case marker/refiner fallback: {report['mixed_refiner_fallback_cases']}")
    for row in rows:
        print(
            f"[{row['category']}] {row['case_id']}: "
            f"{row['text']} -> {row['itn_text']} "
            f"(itn={row['itn_status']}, marker_error={row['refiner_placeholder_error']})"
        )
    print(f"Report: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
