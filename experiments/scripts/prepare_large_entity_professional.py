#!/usr/bin/env python3
"""Prepare manually curated professional-term audio evidence from CV/SpeechIO.

The seed vocabulary is editorially curated. A term is admitted only if its exact
canonical spelling occurs in an eligible reference. POS/regex candidates never
enter the reviewed entity table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/large_entity_benchmark_v1_20261010/staging/professional"
CV_TSV = Path("/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN/test.tsv")
CV_ROOT = CV_TSV.parent
SPEECHIO = ROOT / "results/speechio_professional_entity_eval_20261010/source"

# These are manually selected standard Chinese technical terms, not extracted
# POS/regex output. Admissions are still gated on exact audio-reference evidence.
SEEDS = {
    "科技计算机通信": "人工智能 机器学习 深度学习 神经网络 计算机视觉 自然语言处理 量子计算机 云计算 边缘计算 区块链 大数据 数据挖掘 数据库 操作系统 编程语言 软件工程 开源软件 搜索引擎 互联网 物联网 移动互联网 无线通信 光纤通信 卫星通信 第五代移动通信 第四代移动通信 电子商务 网络安全 信息安全 人脸识别 语音识别 计算机程序 机器人 自动驾驶 半导体 集成电路 微处理器 芯片 光刻机 液晶显示器 触摸屏 虚拟现实 增强现实 操作系统内核 超级计算机 计算机网络 通信协议 数字信号处理 计算机图形学 人工神经网络 机器翻译 密码学 计算机科学 计算机软件 程序设计 信息技术",
    "工业交通": "高速铁路 铁路运输 城市轨道交通 地铁 有轨电车 磁悬浮列车 电动汽车 混合动力汽车 燃料电池汽车 自动驾驶汽车 内燃机 蒸汽机 涡轮发动机 喷气发动机 航空发动机 船舶工程 造船工业 集装箱 港口 起重机 挖掘机 推土机 数控机床 工业机器人 自动化生产线 智能制造 工业设计 机械工程 土木工程 建筑工程 桥梁工程 隧道工程 水利工程 水力发电 风力发电 太阳能发电 核电站 输电线路 电力系统 电动机 发电机 变压器 轴承 齿轮 液压系统 制冷系统 电气工程 无人机 航空航天工程 卫星导航 全球定位系统 铁路系统 高速公路 铁路车站 机车车辆 交通运输 航空运输 公共交通 汽车制造 内燃机车 机械制造",
    "金融经济": "国内生产总值 国民生产总值 通货膨胀 通货紧缩 货币政策 财政政策 资本市场 股票市场 债券市场 期货市场 外汇市场 证券交易所 中央银行 商业银行 投资银行 货币基金 指数基金 共同基金 风险投资 私募股权 资产证券化 量化交易 高频交易 股票指数 消费者价格指数 生产者价格指数 失业率 劳动生产率 汇率 利率 基准利率 存款准备金率 货币供应量 国际贸易 贸易顺差 贸易逆差 经济增长 经济衰退 宏观经济 微观经济 供给侧改革 国民收入 税收政策 社会保险 养老保险 住房公积金 电子支付 移动支付 数字货币 数字人民币 比特币 区块链金融 公司债券 国债 金融市场 经济学 经济学家 外汇储备",
    "法律政策": "中华人民共和国宪法 民法典 刑法 刑事诉讼法 民事诉讼法 行政诉讼法 行政处罚法 劳动合同法 消费者权益保护法 著作权法 专利法 商标法 反垄断法 个人信息保护法 数据安全法 网络安全法 环境保护法 物权法 合同法 公司法 选举法 立法法 国家赔偿法 司法解释 法律援助 行政许可 行政复议 刑事责任 民事责任 行政责任 无罪推定 正当防卫 紧急避险 证据规则 司法鉴定 人民法院 人民检察院 最高人民法院 最高人民检察院 全国人民代表大会 全国人民代表大会常务委员会 国务院 常务委员会 宪法修正案 法律责任 知识产权 诉讼时效 法定代表人 劳动争议 仲裁委员会 国际法 公共政策 民法 行政法 法律制度 法律规范 行政法规 法律顾问 律师事务所 刑事诉讼 民事案件 司法机关 立法机关 政策法规 经济政策 刑事法 民事法 法律法规 法律规定",
    "自然科学化学": "元素周期表 化学元素 原子核 电子 质子 中子 同位素 放射性元素 化学键 共价键 离子键 氢键 化学反应 氧化还原反应 酸碱反应 化学平衡 催化剂 化学方程式 有机化学 无机化学 分析化学 高分子化学 生物化学 物理化学 量子力学 相对论 广义相对论 狭义相对论 引力波 黑洞 暗物质 暗能量 太阳系 银河系 恒星行星 地球大气层 板块构造 火山喷发 地震波 光合作用 细胞分裂 脱氧核糖核酸 核糖核酸 基因组 染色体 遗传学 进化论 生态系统 生物多样性 电磁波 电磁感应 热力学第二定律 能量守恒定律 牛顿运动定律 化学工业 化学工程 化学物质 有机化合物 硫酸 盐酸 二氧化碳",
    "农业生态": "杂交水稻 水稻种植 小麦育种 玉米育种 转基因作物 有机农业 生态农业 精准农业 农业机械化 农业灌溉 滴灌技术 喷灌技术 土壤肥力 土壤改良 土壤侵蚀 水土保持 农作物病虫害 植物病理学 农业气象学 农业生态学 农业生物技术 植物保护 农药残留 除草剂 杀虫剂 杀菌剂 生物农药 有机肥料 化学肥料 氮肥 磷肥 钾肥 农业合作社 农产品加工 粮食安全 粮食储备 轮作制度 间作套种 设施农业 温室栽培 水产养殖 畜牧业 生态保护区 自然保护区 湿地保护 森林生态学 碳循环 温室气体 可持续农业 农田生态系统 农业技术 农田 耕地 水稻 小麦 玉米 大豆 棉花 花生 茶叶 果树 果园 经济作物 化肥 农家肥 有机肥 农药 林业",
    "历史教育": "世界历史 中国历史 古代史 近代史 现代史 中国近代史 欧洲中世纪 文艺复兴 工业革命 法国大革命 美国独立战争 第一次世界大战 第二次世界大战 冷战时期 丝绸之路 郑和下西洋 甲骨文 秦始皇兵马俑 敦煌莫高窟 京杭大运河 科举制度 活字印刷术 四大发明 中华文明 古希腊文明 古罗马文明 玛雅文明 古埃及文明 考古学 历史地理学 教育学 教育心理学 高等教育 义务教育 职业教育 素质教育 学前教育 远程教育 教育改革 教育公平 课程标准 教学大纲 学分制 学位制度 普通高等学校 全国高等学校 教育部 研究生教育 师范教育 教育技术学 史学 历史事件 历史时期 历史文化 罗马帝国 拜占庭帝国 清朝 明朝 唐朝 宋朝 秦朝 春秋战国 高等学校",
    "体育文化": "奥林匹克运动会 夏季奥林匹克运动会 冬季奥林匹克运动会 亚洲运动会 世界杯足球赛 国际足联 世界田径锦标赛 职业足球联赛 中国足球协会 篮球联赛 国家篮球协会 世界乒乓球锦标赛 羽毛球世界锦标赛 网球公开赛 温布尔登网球锦标赛 马拉松比赛 铁人三项 竞技体操 艺术体操 速度滑冰 花样滑冰 越野滑雪 运动生理学 体育心理学 体育教育 全民健身 竞技体育 体育锻炼 武术 太极拳 中国传统音乐 京剧 昆曲 越剧 黄梅戏 皮影戏 中国书法 中国画 文物保护 非物质文化遗产 文化遗产 博物馆 美术馆 交响乐团 古典音乐 民间音乐 民间文学 现代舞 芭蕾舞 电影艺术 戏剧艺术 体育赛事 体育场 体育馆 奥林匹克运动 亚运会 足球运动 足球联赛 足球队 篮球运动 篮球运动员 乒乓球运动 游泳运动员 田径运动 中国文化 文化产业 文化活动 中国戏曲 中国传统戏曲 民间传说"
}

# These manually reviewed exclusions avoid accepting a subword from an unrelated
# lexical compound or a metaphorical occurrence as the positive entity evidence.
DROP_TERMS = {"自然科学化学": {"电子", "中子", "催化剂"},
              "工业交通": {"内燃机", "发电机", "轴承"},
              "历史教育": {"教育部"},
              "体育文化": {"越剧", "足球队", "篮球运动员", "游泳运动员", "中国文化", "文化产业", "文化活动"}}
REJECT_REFERENCE_SUBSTRINGS = {"刑法": {"死刑法律"}, "世界历史": {"全世界历史最"}}

def load_exclusions():
    cv_ids, cv_refs, speech_ids, speech_refs = set(), set(), set(), set()
    for p in (ROOT / "results/generic_entity_blind_20261010/metadata.json",
              ROOT / "results/generic_entity_optimization_20261010/holdout/metadata.json"):
        if p.exists():
            for x in json.loads(p.read_text(encoding="utf-8"))["samples"]:
                if x.get("source", "").startswith("commonvoice"):
                    cv_ids.add(x.get("location", {}).get("sentence_id", ""))
                    cv_refs.add(x.get("reference", ""))
    for p in (ROOT / "results/speechio_professional_entity_eval_20261010").glob("*/protocol.json"):
        for x in json.loads(p.read_text(encoding="utf-8"))["samples"]:
            speech_ids.add(x.get("source_id", ""))
            speech_ids.add(x.get("id", ""))
            speech_refs.add(x.get("reference", ""))
    cv_ids.discard(""); cv_refs.discard(""); speech_ids.discard(""); speech_refs.discard("")
    return cv_ids, cv_refs, speech_ids, speech_refs

def cv_rows(cv_ids, cv_refs):
    with CV_TSV.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        wanted = {name: header.index(name) for name in ("sentence_id", "sentence", "path")}
        for line, row in enumerate(reader, start=2):
            sid, ref, clip = (row[wanted[k]] for k in ("sentence_id", "sentence", "path"))
            if sid in cv_ids or ref in cv_refs or not ref or not (CV_ROOT / "clips" / clip).is_file():
                continue
            yield {"source": "common_voice_zh-CN_test", "sample_id": sid,
                   "reference": ref, "duration_seconds": None,
                   "location": {"tsv": str(CV_TSV), "line": line, "sentence_id": sid,
                                "clip_path": "clips/" + clip,
                                "audio_path": str(CV_ROOT / "clips" / clip)}}

def speechio_rows(speech_ids, speech_refs):
    for f in sorted(SPEECHIO.glob("*.parquet")):
        table = pq.read_table(f, columns=["segment_id", "text", "duration"])
        for i, (sid, ref, dur) in enumerate(zip(table["segment_id"].to_pylist(),
                                                table["text"].to_pylist(),
                                                table["duration"].to_pylist())):
            if sid in speech_ids or ref in speech_refs or not ref:
                continue
            yield {"source": "SpeechIO", "sample_id": sid, "reference": ref,
                   "duration_seconds": float(dur),
                   "location": {"parquet": str(f), "row_index": i, "segment_id": sid}}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv_ids, cv_refs, speech_ids, speech_refs = load_exclusions()
    refs = list(cv_rows(cv_ids, cv_refs)) + list(speechio_rows(speech_ids, speech_refs))
    by_term = {}
    for domain, text in SEEDS.items():
        for term in text.split():
            if term in DROP_TERMS.get(domain, set()):
                continue
            by_term.setdefault(term, domain)

    # Prefer Common Voice for predictable local audio access; use SpeechIO when
    # it is the only eligible reference evidence. Iterate term order, then corpus.
    rows_by_source = {
        "common_voice_zh-CN_test": [r for r in refs if r["source"] == "common_voice_zh-CN_test"],
        "SpeechIO": [r for r in refs if r["source"] == "SpeechIO"],
    }
    entities, samples, missing = [], [], []
    used_terms = set()
    for domain, text in SEEDS.items():
        for term in text.split():
            if term in DROP_TERMS.get(domain, set()):
                continue
            if term in used_terms: continue
            used_terms.add(term)
            hit = None
            for src in ("common_voice_zh-CN_test", "SpeechIO"):
                hit = next((r for r in rows_by_source[src]
                            if term in r["reference"] and not any(bad in r["reference"] for bad in REJECT_REFERENCE_SUBSTRINGS.get(term, set()))), None)
                if hit: break
            if not hit:
                missing.append({"canonical": term, "topic": domain, "reason": "not_found_in_eligible_reference"})
                continue
            start = hit["reference"].find(term)
            entity_id = "term-" + hashlib.sha256(term.encode("utf-8")).hexdigest()[:12]
            entities.append({"entity_id": entity_id, "canonical": term, "type": "TERM",
                             "topic": domain, "source": hit["source"],
                             "reference": hit["reference"], "evidence": hit["reference"],
                             "sample_id": hit["sample_id"],
                             "review_method": "reference_curated",
                             "curation_note": "手工标准术语种子；逐条核对实际参考完整出现及跨度"})
            samples.append({"sample_id": hit["sample_id"], "source": hit["source"],
                            "reference": hit["reference"], "entity_id": entity_id,
                            "canonical": term, "entity_span": [start, start + len(term)],
                            "entity_surface": term, "topic": domain,
                            "review_method": "reference_curated",
                            "duration_seconds": hit["duration_seconds"],
                            "location": hit["location"]})

    # Add durations for the selected MP3 clips; all CV files are local and checked.
    for s in samples:
        if s["source"] == "common_voice_zh-CN_test":
            q = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", s["location"]["audio_path"]],
                               text=True, capture_output=True, check=True)
            s["duration_seconds"] = float(q.stdout.strip())

    # Validate exact SpeechIO row and embedded audio payload without exporting it.
    speechio_checks = []
    grouped = defaultdict(list)
    for s in samples:
        if s["source"] == "SpeechIO":
            grouped[s["location"]["parquet"]].append(s)
    for parquet_path, selected in grouped.items():
        pf = pq.ParquetFile(parquet_path)
        offsets, n, rowgroup_cache = [], 0, {}
        for rg in range(pf.metadata.num_row_groups):
            rows = pf.metadata.row_group(rg).num_rows
            offsets.append((n, n + rows, rg)); n += rows
        for s in selected:
            row = s["location"]["row_index"]
            lo, hi, rg = next(x for x in offsets if x[0] <= row < x[1])
            if rg not in rowgroup_cache:
                rowgroup_cache[rg] = pf.read_row_group(rg, columns=["segment_id", "text", "audio"])
            batch = rowgroup_cache[rg]
            local = row - lo
            audio = batch["audio"][local].as_py()
            payload_ok = bool(audio and (audio.get("bytes") or audio.get("path")))
            speechio_checks.append(bool(local < batch.num_rows and
                batch["segment_id"][local].as_py() == s["sample_id"] and
                batch["text"][local].as_py() == s["reference"] and payload_ok))

    # Never label algorithmic candidates as reviewed. This run does not create
    # POS/regex output; the file makes that boundary explicit for downstream work.
    (OUT / "draft_candidates.json").write_text(json.dumps({"status": "not_reviewed",
        "note": "No automatic POS/regex candidates admitted by this script.", "candidates": []},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    counts = Counter(e["topic"] for e in entities)
    by_topic_source = defaultdict(Counter)
    for s in samples: by_topic_source[s["topic"]][s["source"]] += 1
    stats = {"seed_count": len(used_terms), "reviewed_unique_entities": len(entities),
             "sample_count": len(samples), "missing_seed_count": len(missing),
             "excluded_commonvoice_sample_ids": len(cv_ids), "excluded_commonvoice_references": len(cv_refs),
             "excluded_speechio_identifiers": len(speech_ids), "excluded_speechio_references": len(speech_refs),
             "by_topic": dict(counts), "by_source": dict(Counter(s["source"] for s in samples)),
             "by_topic_source": {k: dict(v) for k, v in by_topic_source.items()},
             "duration_seconds_total": round(sum(s["duration_seconds"] for s in samples), 3),
             "missing_by_topic": dict(Counter(x["topic"] for x in missing)),
             "missing_seeds": missing,
             "validation": {"all_spans_match_reference": all(s["reference"][s["entity_span"][0]:s["entity_span"][1]] == s["canonical"] for s in samples),
                            "all_cv_audio_exists": all(Path(s["location"]["audio_path"]).is_file() for s in samples if s["source"] == "common_voice_zh-CN_test"),
                            "all_cv_durations_positive": all(s["duration_seconds"] > 0 for s in samples if s["source"] == "common_voice_zh-CN_test"),
                            "all_speechio_rows_and_audio_valid": bool(speechio_checks) and all(speechio_checks),
                            "speechio_audio_rows_checked": len(speechio_checks)}}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("""# 专业术语跨域样本 staging\n\n仅包含人工挑选的标准中文术语种子，且每个词条都在 Common Voice zh-CN test 或 SpeechIO 本地源参考中逐条找到完整字面证据。`review_method=reference_curated` 表示核对了参考文本和实体跨度，不表示人工听审音频。`draft_candidates.json` 明确不计入审核集合。\n\n` samples.jsonl` 给出音频文件路径或 Parquet 文件与零基行号定位、参考、跨度和时长。源文件未经修改。Common Voice 音频路径逐条检查并用 ffprobe 获取时长；SpeechIO 样本按源 parquet 精确行定位并使用其时长字段。排除了既有 Common Voice 两批样本及旧 SpeechIO domain protocol 的 sample ID/source ID 和完整参考句。\n\n查看 `stats.json` 的实际领域覆盖和缺失种子；只有其中 `reviewed_unique_entities` 可作为本 staging 已审核参考候选计数。\n""".replace("` samples", "`samples"), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("reviewed_unique_entities", "sample_count", "by_topic", "by_source", "missing_by_topic")}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
