"""Freeze independent new scenarios before audio generation or label inspection.

Scenario roles describe intended script construction, never predicted gate
labels. Original source holdouts and all prior experiment artifacts are intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import synthetic_gate_data as data

# Each basic tuple is one scenario family with clean/filler/stutter counterparts.
# Counterparts and all fictional slot values always share a split.
BASIC = [
    ("release_check", "上线之前请{name}再检查一次配置。", "上线之前，呃，请{name}再检查一次配置。", "上线之前请请{name}再检查一次配置。"),
    ("copy_backup", "把备份文件复制到{place}的测试服务器。", "把那个，备份文件复制到{place}的测试服务器。", "把备份文件复复制到{place}的测试服务器。"),
    ("work_order", "这张工作单需要{name}签字确认。", "这张工作单，嗯，需要{name}签字确认。", "这张工作单需要需要{name}签字确认。"),
    ("warehouse_delivery", "下午三点将样品送到{place}仓库。", "下午三点，呃，将样品送到{place}仓库。", "下午三点将样品送送到{place}仓库。"),
    ("budget_review", "我们先审查预算，再通知{name}采购。", "我们，嗯，先审查预算，再通知{name}采购。", "我们先审查查预算，再通知{name}采购。"),
    ("ticket_reply", "请{name}回复这条故障工单。", "请，呃，{name}回复这条故障工单。", "请{name}回回复这条故障工单。"),
    ("experiment_schedule", "这轮实验安排在{place}的实验室进行。", "这轮实验呢，安排在{place}的实验室进行。", "这轮实验安排在{place}的实验验室进行。"),
    ("change_register", "所有变更都要写进{name}负责的登记表。", "所有变更，嗯，都要写进{name}负责的登记表。", "所有变更都都要写进{name}负责的登记表。"),
    ("restart_notice", "重启设备前先通知{name}。", "重启设备前，那个，先通知{name}。", "重重启设备前先通知{name}。"),
    ("address_label", "请在箱子上注明{place}的收货地址。", "请在箱子上，呃，注明{place}的收货地址。", "请在箱子上注明{place}的收收货地址。"),
    ("handover", "{name}负责讲解今天的交接流程。", "{name}，嗯，负责讲解今天的交接流程。", "{name}负责讲讲解今天的交接流程。"),
    ("contact_list", "把{place}项目的联系人列表发给{name}。", "把{place}项目的那个，联系人列表发给{name}。", "把{place}项目的联系人列表发发给{name}。"),
    ("invoice_scan", "{name}会在下班之前上传发票扫描件。", "{name}会，呃，在下班之前上传发票扫描件。", "{name}会在下班之前上上传发票扫描件。"),
    ("test_account", "我们给{place}的同事开一个测试账号。", "我们，嗯，给{place}的同事开一个测试账号。", "我们给{place}的同事开一个测试试账号。"),
    ("checklist", "请{name}逐项勾选这份验收清单。", "请{name}，呃，逐项勾选这份验收清单。", "请{name}逐项勾勾选这份验收清单。"),
    ("microphone_test", "演示开始之前{name}要测试一下麦克风。", "演示开始之前，嗯，{name}要测试一下麦克风。", "演示开始之前{name}要测试一下麦麦克风。"),
    ("archive", "把旧版本的资料移到{place}项目的归档目录。", "把旧版本的那个，资料移到{place}项目的归档目录。", "把旧版本的资料移移到{place}项目的归档目录。"),
    ("cost_report", "这笔差旅费用由{name}汇总后提交。", "这笔差旅费用，呃，由{name}汇总后提交。", "这笔差旅费用由{name}汇汇总后提交。"),
    ("prototype_ship", "第一批原型机送往{place}进行试用。", "第一批原型机呢，送往{place}进行试用。", "第一批原型机送往{place}进行试试用。"),
    ("wire_check", "接线完成后请{name}检查端子是否松动。", "接线完成后，嗯，请{name}检查端子是否松动。", "接线完成后请{name}检查端端子是否松动。"),
    ("group_notice", "培训的报名链接由{name}发到工作群。", "培训的那个，报名链接由{name}发到工作群。", "培训的报名链接由{name}发到工作工作群。"),
    ("appointment", "下周一到{place}的服务中心预约安装。", "下周一，呃，到{place}的服务中心预约安装。", "下周一到{place}的服务中心预预约安装。"),
    ("account_close", "月底之前{name}需要完成账目核对。", "月底之前，嗯，{name}需要完成账目核对。", "月底之前{name}需要完成账账目核对。"),
    ("route_map", "请把去{place}的路线图打印两份。", "请把去{place}的那个，路线图打印两份。", "请把去{place}的路线图打印打印两份。"),
]
MEANINGFUL = [
    ("emphasis_deadline", "{name}强调这次期限非常非常紧，不能再延期。"),
    ("emphasis_detail", "每一个细节都很重要很重要，{name}要求全部记录。"),
    ("step_steps", "在{place}调试时要一步一步排查，不能跳过中间步骤。"),
    ("page_compare", "{name}把原件和复印件一张一张进行比对。"),
    ("batch_checks", "{place}仓库按一批一批的顺序检查货物。"),
    ("slow_care", "{name}慢慢拧紧螺丝，以免损坏接口。"),
    ("just_changed", "{name}刚刚修改了参数，请使用新的配置。"),
    ("every_person", "{place}团队人人都要参加这次演练。"),
    ("clear_explain", "{name}必须把故障经过清清楚楚地讲出来。"),
    ("long_queue", "{place}大厅排着长长的队伍。"),
    ("think_again", "{name}说先想一想，再决定是否购买。"),
    ("each_compare", "请{name}把两个方案逐条逐条对照。"),
    ("layer_process", "{place}的审核流程需要层层把关。"),
    ("minute_later", "{name}说要一遍一遍听录音，每一遍都检查不同的细节。"),
    ("never_repeat", "千万千万不要泄露密码，{name}特意重复提醒。"),
    ("emphasis_source", "这个原始文件很重要很重要，{name}连续强调了两遍。"),
]
PROTECTED = [
    ("not_deleted", "{name}没有删除记录，只是暂时隐藏了页面。"),
    ("not_approved", "审批没有通过，{name}还不能安排发货。"),
    ("two_numbers", "{name}登记的是{number}台设备，不是{other_number}台。"),
    ("two_prices", "{place}项目的单价是{amount}元，总价不是这个数。"),
    ("unless_sign", "除非{name}已经签字，否则不要更新正式配置。"),
    ("not_every", "并不是所有{place}的设备都需要更换电池。"),
    ("no_double_negative", "{name}并非没有提出意见，只是会议记录漏写了。"),
    ("quantity_range", "{place}这批样品不少于十个，也不多于二十个。"),
    ("wrong_contact", "联系人是{name}而不是{other_name}，请核对姓名。"),
    ("version_number", "{name}要求保留版本{number}，不要覆盖版本{other_number}。"),
    ("original_decimal", "{place}项目记录的测量值是{decimal}，小数点不能省略。"),
    ("drug_record", "{name}确认文档写的是阿莫西林，没有写阿奇霉素。"),
]
SYNTAX = [
    ("condition_ready", "只有{name}确认备份完整，我们才能恢复数据库。"),
    ("if_check", "如果{place}的检测结果不合格，就重新安排采样。"),
    ("because_delay", "因为{name}还在核对数量，所以暂时没有发货。"),
    ("although_alarm", "虽然{place}设备停止报警，仍然要检查日志。"),
    ("before_reset", "在{name}完成记录之前，不要重置这个计数器。"),
    ("after_sign", "等{name}签收以后，我们再结算运输费用。"),
    ("either_path", "无论去{place}还是留在本地，都需要先登记。"),
    ("purpose_note", "为了让{name}核对原始信息，附件保留了完整录音。"),
]


def scenarios():
    values = []
    for slug, clean, filler, stutter in BASIC:
        values.append(("basic_expand", slug, [("clean", clean, clean), ("filler", filler, clean), ("stutter", stutter, clean)]))
    for family, templates in (("meaningful_expand", MEANINGFUL), ("protected_expand", PROTECTED), ("syntax_expand", SYNTAX)):
        for slug, text in templates:
            values.append((family, slug, [(family, text, text)]))
    return values


def delexical(text):
    for name in sorted(data.NAMES, key=len, reverse=True):
        text = text.replace(name, "{name}")
    for place in sorted(data.PLACES, key=len, reverse=True):
        text = text.replace(place, "{place}")
    return re.sub(r"\d+(?:\.\d+)?", "{number}", data.lexical(text))


def generate(old_rows, seed=20261011, slots_per_template=1):
    if not 1 <= slots_per_template <= 8:
        raise ValueError("Use 1..8 fictional slot draws per independent scenario")
    rng = random.Random(seed)
    specs = scenarios()
    split_by_group = {}
    for family in sorted({s[0] for s in specs}):
        groups = sorted(s[1] for s in specs if s[0] == family)
        rng.shuffle(groups)
        train_n, dev_n = round(len(groups) * 0.6), max(1, round(len(groups) * 0.2))
        for index, slug in enumerate(groups):
            split_by_group[(family, slug)] = "train" if index < train_n else "dev" if index < train_n + dev_n else "test"
    old_pairs = {(delexical(r["spoken_text"]), delexical(r["intended_text"])) for r in old_rows}
    seen, rows = set(), []
    for family, slug, variants in specs:
        for draw in range(slots_per_template):
            for _attempt in range(1000):
                name, other = rng.sample(data.NAMES, 2)
                number, other_number = rng.sample(range(10, 990), 2)
                slots = dict(name=name, other_name=other, place=rng.choice(data.PLACES), number=number,
                             other_number=other_number, amount=number * 10, decimal=f"{number // 10}.{number % 10}")
                rendered = [(category, script.format(**slots), intended.format(**slots)) for category, script, intended in variants]
                if not any((spoken, target) in seen for _, spoken, target in rendered):
                    break
            else:
                raise ValueError("Scenario exhausted unique slot draws")
            for category, spoken, target in rendered:
                if (delexical(spoken), delexical(target)) in old_pairs:
                    raise ValueError("New scenario overlaps an old delexicalized template")
                key = spoken, target
                seen.add(key)
                row = dict(seed_id=f"expand_gate_{seed}_{len(rows):05d}", category=category,
                           template_group=f"{family}_v1:{slug}", spoken_text=spoken, intended_text=target,
                           synthetic_split=split_by_group[family, slug], domain="medical" if slug == "drug_record" else "general",
                           audio_script_verified=False, content_role="fictional_controlled_transcription_probe",
                           scenario_role="script_construction_only_not_gate_benefit_or_harm_label", slot_draw=draw)
                row["seed_sha256"] = data.digest(row)
                rows.append(row)
    return rows


def prepare(source, output, seed=20261011, slots_per_template=1):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite a frozen expansion")
    old_rows = data.seeds(source)
    rows = generate(old_rows, seed, slots_per_template)
    output.mkdir(parents=True, exist_ok=True)
    data.write_json(output / "scenario_specs.json", scenarios())
    for row in rows:
        data.append_row(output / "texts.jsonl", row)
    frozen = output / "preparation_sources"
    frozen.mkdir()
    for path in (Path(__file__), Path(data.__file__)):
        (frozen / path.name).write_bytes(path.read_bytes())
    manifest = dict(schema_version=1, count=len(rows), seed=seed, slots_per_template=slots_per_template,
                    texts_sha256=data.file_hash(output / "texts.jsonl"), generator_sha256=data.file_hash(Path(__file__)),
                    prior_texts_sha256=data.file_hash(source / "texts.jsonl"), prior_source=str(source.resolve()),
                    categories=dict(Counter(r["category"] for r in rows)), synthetic_splits=dict(Counter(r["synthetic_split"] for r in rows)),
                    unique_template_groups=len({r["template_group"] for r in rows}),
                    template_groups_by_split={split: len({r["template_group"] for r in rows if r["synthetic_split"] == split})
                                              for split in ("train", "dev", "test")},
                    split_unit="scenario_family_with_all_slots_and_disfluency_counterparts",
                    prior_delexicalized_template_overlap=0, reference_role="unverified_synthesis_scripts_not_human_gold",
                    intended_policy="remove_marked_filler_stutter_preserve_emphasis_negation_numbers_and_syntax",
                    predeclared_label_rule="existing_conservative_pair_effect_labels_not_scenario_categories",
                    scope="new_template_train_expansion_reserved_dev_test_not_synthesized")
    data.write_json(output / "manifest.json", manifest)
    print(__import__("json").dumps(manifest, ensure_ascii=False, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/qwen_tts_asr_gpu_pilot_20261010")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gate_template_expansion_20261010")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--slots-per-template", type=int, default=1)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.seed, args.slots_per_template)


if __name__ == "__main__":
    main()
