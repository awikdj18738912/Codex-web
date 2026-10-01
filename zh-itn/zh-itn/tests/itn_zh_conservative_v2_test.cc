/*
 * itn_zh_conservative_v2_test.cc - Identity-first Chinese ITN checks.
 */

#include "itn_zh_conservative_v2.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace zh_itn::itn;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

V2SegmentRequest request(const std::string& text) {
    return {text, "", "", "zh-CN", false, false};
}

}  // namespace

int main() {
    bool passed = true;
    const std::vector<std::string> identity_cases = {
        "",
        "普通中文文本",
        "一些问题一定要一起讨论",
        "一直以来一般民众都有一些意见",
        "四川三亚六安九寨沟",
        "十拿九稳十万火急一五一十",
        "一把手一刀切一锅端",
        "不管三七二十一七七八八三三两两",
        "三心二意不是三个心两个意",
        "一点意见和一块业务",
        "五月花与五月天",
        "七夕活动和九五至尊",
        "三个人第二次等了五分钟左右",
        "提高一个百分点",
        "这个包重五G",
        "emoji🙂和补充字符𠀀保持不变",
    };
    for (const auto& text : identity_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "identity request failed: " + text);
        passed &= expect(result.normalized.output == text, "identity output changed: " + text);
        std::string failure;
        passed &= expect(
            validate_safe_normalization(result.normalized, {}, &failure),
            "identity trace failed: " + text + " (" + failure + ")"
        );
    }

    const std::vector<std::pair<std::string, std::string>> second_batch_cases = {
        {"在二零二零年一月十四号举行", "在2020年1月14号举行"},
        {"一九五五年九月二十七号成立", "1955年9月27号成立"},
        {"公元二九九九年十二月三十一日", "公元2999年12月31日"},
        {"当地时间三月十三日下午", "当地时间3月13日下午"},
        {"七月六日下午三点", "7月6日下午3点"},
        {"二月二十九日", "2月29日"},
        {"占比百分之九十八点五", "占比98.5%"},
        {"增长百分之一百", "增长100%"},
        {"约百分之零点五", "约0.5%"},
        {"中国移动五G套餐客户数增长", "中国移动5G套餐客户数增长"},
        {"四G网络升级", "4G网络升级"},
        {"二G通信技术", "2G通信技术"},
        {"推进六G研究", "推进6G研究"},
        {"八月中旬", "8月中旬"},
    };
    for (const auto& [text, expected] : second_batch_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "second batch request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "second batch output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "second batch approval trace missing: " + text
        );
        passed &= expect(
            result.normalized.mappings.size() >= 1,
            "second batch mapping trace missing: " + text
        );
    }
    const auto month_period_5g = normalize_zh_conservative_v2(request(
        "中国电信也将在八月中旬开展五G体验包活动。"
    ));
    passed &= expect(
        month_period_5g.valid
            && month_period_5g.normalized.output
                == "中国电信也将在8月中旬开展5G体验包活动。"
            && month_period_5g.normalized.applied.size() == 2,
        "month-period and 5G combination failed"
    );
    const std::vector<std::pair<std::string, std::string>> spaced_identifier_cases = {
        {
            "技术展区展示五 G基站和六 G研究设备，工作人员佩戴 N 九五口罩。",
            "技术展区展示5G基站和6G研究设备，工作人员佩戴N95口罩。",
        },
        {"旧游戏区保留索尼 P S 二主机。", "旧游戏区保留索尼PS2主机。"},
        {"汽车服务区安排了一家四 S 店。", "汽车服务区安排了一家4S店。"},
    };
    for (const auto& [text, expected] : spaced_identifier_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "spaced identifier normalization failed: " + text
        );
    }

    const std::vector<std::string> second_batch_rejected_cases = {
        "二零二三年二月二十九日",
        "二零二四年二月三十日",
        "十三月一日",
        "两月三日",
        "五月一号店",
        "《二零二四年五月三日》",
        "文件名是二零二四年五月三日",
        "失败率达到百分之九十多",
        "占比大约百分之十几",
        "提高一个百分点",
        "百分之百分之一",
        "这个包重五G",
        "缓存大小五GB",
        "容量二百五十六G",
        "增长百分之四十八点六五G",
        "一点二五G手机",
        "第六幺四G突变病毒株",
        "设备重量约五G",
        "频率五G赫兹",
        "《五G时代》",
        "这个包重五 G",
        "缓存大小五 G B",
        "产品型号 N 九五",
        "编号 P S 二",
    };
    for (const auto& text : second_batch_rejected_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "second batch rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "second batch rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "second batch rejected candidate was approved: " + text
        );
    }

    const auto approved_with_rejected_candidate = normalize_zh_conservative_v2(
        request("在二零二四年发布二月三十日")
    );
    passed &= expect(
        approved_with_rejected_candidate.normalized.output == "在2024年发布二月三十日",
        "a rejected candidate blocked an independently approved rule"
    );
    const std::vector<std::pair<std::string, std::string>> generic_composition_cases = {
        {
            "二零二四年五月三日增长百分之十",
            "2024年5月3日增长10%",
        },
        {"三月一日和三月二日", "3月1日和3月2日"},
        {"上午九点和下午三点各开一场", "上午9点和下午3点各开一场"},
        {
            "三月一日上午九点和三月二日下午三点",
            "3月1日上午9点和3月2日下午3点",
        },
        {"和三月一号下午四点三十分", "和3月1号下午4点30分"},
        {"一点五秒和二点五秒", "1.5秒和2.5秒"},
        {
            "十六小时三十三分钟和二十小时十四分钟",
            "16小时33分钟和20小时14分钟",
        },
        {"已有16小时另用二十小时十四分钟", "已有16小时另用20小时14分钟"},
        {
            "文件名是二零二四年，产品于二零二五年发布",
            "文件名是二零二四年，产品于2025年发布",
        },
        {"二零一四年和二零一五年", "2014年和2015年"},
        {
            "纪念品价格约为一百元，普通商品价格为一百至两百元，"
            "海外采购预算不少于两千万美元。",
            "纪念品价格约为100元，普通商品价格为100至200元，"
            "海外采购预算不少于2000万美元。",
        },
        {
            "服务中心安排二十四小时值班，维护时间大约二十分钟，"
            "培训至少三十分钟。",
            "服务中心安排24小时值班，维护时间大约20分钟，"
            "培训至少30分钟。",
        },
        {
            "体验路线大约一点五公里，运输距离不超过一百公里，"
            "道路每一百公里设置一个服务区。",
            "体验路线大约1.5公里，运输距离不超过100公里，"
            "道路每100公里设置一个服务区。",
        },
    };
    for (const auto& [text, expected] : generic_composition_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "generic composition request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "generic composition output mismatch: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            !result.normalized.applied.empty(),
            "generic composition approval trace missing: " + text
        );
    }
    const std::vector<std::pair<std::string, std::string>> percentage_range_cases = {
        {"占比百分之二十到百分之二十五", "占比20%到25%"},
        {"增幅百分之二十至百分之二十五", "增幅20%至25%"},
        {"区间百分之二十-百分之二十五", "区间20%-25%"},
        {"占比从百分之五上升到百分之八十", "占比从5%上升到80%"},
        {"增长百分之三十到四十", "增长30%到40%"},
        {"占比百分之一点五至二点五", "占比1.5%至2.5%"},
        {"波动区间百分之五十-六十", "波动区间50%-60%"},
    };
    for (const auto& [text, expected] : percentage_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected
                && result.normalized.applied.size() == 1,
            "percentage range semantic object failed: " + text
        );
    }
    for (const std::string text : {
            "占比百分之二十到百分之二十五到百分之三十",
            "“百分之二十到百分之二十五”",
            "提高百分之二十到百分之二十五个百分点",
            "提高百分之二十到二十五个百分点",
            "占比百分之二十到三十到四十",
            "占比百分之二十到三十多",
            "提高百分之十个百分点",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "unsafe percentage range changed: " + text
        );
    }
    const std::vector<std::pair<std::string, std::string>> explicit_unit_range_cases = {
        {"十到十五分钟", "10到15分钟"},
        {"十五到三十分钟", "15到30分钟"},
        {"三十六到四十八小时", "36到48小时"},
        {"五十到六十公里", "50到60公里"},
        {"每小时二十五到三十公里", "每小时25到30公里"},
        {"四千到五千元", "4000到5000元"},
        {"二十三到二十八万元", "23到28万元"},
        {"一千到两千美元", "1000到2000美元"},
        {"一百万美元至一千万美元", "100万美元至1000万美元"},
        {"十九点六八万元-二十三点六八万元", "19.68万元-23.68万元"},
        {"二十四点九九到二十九点九九万元", "24.99到29.99万元"},
        {
            "两千三百八十七点四亿元增长至两千七百零一点一亿元",
            "2387.4亿元增长至2701.1亿元",
        },
    };
    for (const auto& [text, expected] : explicit_unit_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected
                && result.normalized.applied.size() == 1,
            "explicit unit range semantic object failed: " + text + " -> "
                + result.normalized.output
        );
    }
    for (const std::string text : {
            "三到五分钟",
            "五到十分钟",
            "大约十到十五分钟",
            "十到十五分钟左右",
            "十到十五到二十分钟",
            "《十到十五分钟》",
            "百分之二十一点七至二点七三四亿元",
            "五百万到一亿美元",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "unsafe explicit unit range changed: " + text + " -> "
                + result.normalized.output
        );
    }
    const std::vector<std::pair<std::string, std::string>> explicit_unit_threshold_cases = {
        {"不超过一百公里", "不超过100公里"},
        {"可以跑八百公里以上", "可以跑800公里以上"},
        {"大部分地区录得超过八十毫米雨量", "大部分地区录得超过80毫米雨量"},
        {"二十公里以下", "20公里以下"},
        {"充电用时不到十分钟", "充电用时不到10分钟"},
        {"不到二十分钟", "不到20分钟"},
        {"三十分钟以内能够送到", "30分钟以内能够送到"},
        {"二十分钟以上它就凉了", "20分钟以上它就凉了"},
        {"至少十五秒", "至少15秒"},
    };
    for (const auto& [text, expected] : explicit_unit_threshold_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected
                && result.normalized.applied.size() == 1,
            "explicit unit threshold failed: " + text + " -> "
                + result.normalized.output
        );
    }
    for (const std::string text : {
            "不超过三公里",
            "三分钟以内",
            "大约不超过一百公里",
            "不到十分钟左右",
            "综合续航超一千六百公里",
            "不超过十到十五分钟",
            "《超过八十毫米》",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "unsafe explicit unit threshold changed: " + text + " -> "
                + result.normalized.output
        );
    }
    const std::vector<std::pair<std::string, std::string>> explicit_unit_approximation_cases = {
        {"大约二十分钟", "大约20分钟"},
        {"接近八十五分钟", "接近85分钟"},
        {"煮三十分钟左右", "煮30分钟左右"},
        {"位于以南约四百五十公里", "位于以南约450公里"},
        {"路线大概十公里", "路线大概10公里"},
        {"反射面约在一百平方米", "反射面约在100平方米"},
        {"全程一百公里前后", "全程100公里前后"},
    };
    for (const auto& [text, expected] : explicit_unit_approximation_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected
                && result.normalized.applied.size() == 1,
            "explicit unit approximation failed: " + text + " -> "
                + result.normalized.output
        );
    }
    for (const std::string text : {
            "大约三分钟",
            "三分钟左右",
            "几十分钟",
            "大约不超过一百公里",
            "预约一百公里的行程",
            "约一百毫米汞柱",
            "《大约二十分钟》",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "unsafe explicit unit approximation changed: " + text + " -> "
                + result.normalized.output
        );
    }
    const std::vector<std::pair<std::string, std::string>> explicit_unit_periodic_cases = {
        {"全省均价每五百克为三点四六元", "全省均价每500克为3.46元"},
        {"每一百毫升含糖十克", "每100毫升含糖十克"},
        {"每隔二十分钟检查一次", "每隔20分钟检查一次"},
        {"每二十四小时更新", "每24小时更新"},
        {"每一百米设置一个标志", "每100米设置一个标志"},
    };
    for (const auto& [text, expected] : explicit_unit_periodic_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "explicit periodic unit failed: " + text + " -> "
                + result.normalized.output
        );
    }
    for (const std::string text : {
            "每一分钟检查一次",
            "每隔三分钟检查一次",
            "每几百克收费",
            "大约每五百克",
            "《每一百毫升》",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "unsafe periodic unit changed: " + text + " -> "
                + result.normalized.output
        );
    }
    auto unit_range_left = request("十到");
    unit_range_left.right_context = "十五分钟";
    unit_range_left.has_right_neighbor = true;
    const auto unit_range_left_result = normalize_zh_conservative_v2(unit_range_left);
    passed &= expect(
        unit_range_left_result.valid
            && unit_range_left_result.normalized.output == unit_range_left.text,
        "cross-segment explicit unit range changed its left endpoint"
    );
    auto unit_range_right = request("十五分钟");
    unit_range_right.left_context = "十到";
    unit_range_right.has_left_neighbor = true;
    const auto unit_range_right_result = normalize_zh_conservative_v2(unit_range_right);
    passed &= expect(
        unit_range_right_result.valid
            && unit_range_right_result.normalized.output == unit_range_right.text,
        "cross-segment explicit unit range changed its right endpoint"
    );

    const std::vector<std::pair<std::string, std::string>> third_batch_cases = {
        {"必须戴N九五口罩", "必须戴N95口罩"},
        {"奔驰四S店已经开业", "奔驰4S店已经开业"},
        {"二零零一年索尼PS二两千万台的订单", "2001年索尼PS2两千万台的订单"},
        {"软件版本为一点零点三", "软件版本为1.0.3"},
        {"驾考宝典版本为七点八点五", "驾考宝典版本为7.8.5"},
        {"版本号二十点零三", "版本号20.03"},
        {
            "版本一点零点三地址一二七点零点零点一",
            "版本1.0.3地址一二七点零点零点一"
        },
    };
    for (const auto& [text, expected] : third_batch_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "third batch request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "third batch output mismatch: " + text + " -> " + result.normalized.output
        );
        const size_t expected_applied = text.find("索尼PS") == std::string::npos ? 1 : 2;
        passed &= expect(
            result.normalized.applied.size() == expected_applied,
            "third batch approval trace missing: " + text
        );
    }
    const auto released_n95_percentage_pair = normalize_zh_conservative_v2(
        request("N九五口罩产能利用率达到百分之一百二十八")
    );
    passed &= expect(
        released_n95_percentage_pair.valid
            && released_n95_percentage_pair.normalized.output
                == "N95口罩产能利用率达到128%"
            && released_n95_percentage_pair.normalized.applied.size() == 2
            && released_n95_percentage_pair.decision_reason
                == "applied_multiple_semantic_objects",
        "released N95 and percentage pair failed"
    );

    const std::vector<std::string> third_batch_rejected_cases = {
        "项目代号N九五",
        "N九五六口罩",
        "《四S店》",
        "四S级考试",
        "PS二三主机",
        "我考了PS二分",
        "价格一点零点三",
        "版本一点",
        "版本一点零点三点四点五点六",
    };
    for (const auto& text : third_batch_rejected_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "third batch rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "third batch rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "third batch rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> contextual_identifier_cases = {
        {"铁路一二三零六网站已开放", "铁路12306网站已开放"},
        {"消费者可通过一二三幺五投诉举报", "消费者可通过12315投诉举报"},
        {"中国移动一零零八六发送短信", "中国移动10086发送短信"},
        {"差点打了一二零", "差点打了120"},
        {"美国达美航空公司一架波音七三七八零零客机降落",
         "美国达美航空公司一架波音737-800客机降落"},
        {"一架F杠二十二隐形战斗机起飞", "一架F-22隐形战斗机起飞"},
        {"米二直升机完成任务", "米2直升机完成任务"},
    };
    for (const auto& [text, expected] : contextual_identifier_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected
                && result.normalized.applied.size() == 1,
            "contextual identifier output mismatch: " + text + " -> "
                + result.normalized.output
        );
    }
    for (const auto& text : {
        "一二三零六名旅客", "铁路一二三零六号列车", "铁路项目代号一二三零六",
        "中国联通一零零八六发送短信",
        "打了一二零分", "波音七三七客机", "F杠二十二分", "米二号楼",
        "《米二直升机》",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text
                && result.normalized.applied.empty(),
            "unsafe contextual identifier changed: " + std::string(text) + " -> "
                + result.normalized.output
        );
    }

    const std::vector<std::pair<std::string, std::string>> fourth_batch_cases = {
        {"联系电话幺三九九幺二六三七九七", "联系电话13991263797"},
        {"联系电话：零二九八六五五四二六二", "联系电话：02986554262"},
        {"联系电话四零零零二九三零六零", "联系电话4000293060"},
        {
            "联系电话是四零零零二九。三零六零值班员还说。",
            "联系电话是4000293060值班员还说。",
        },
        {
            "联系电话是四零零零二九。\n三零六零值班员还说。",
            "联系电话是4000293060值班员还说。",
        },
        {"在呼叫幺二零后", "在呼叫120后"},
        {"微信投诉一二三四五打电话投诉", "微信投诉12345打电话投诉"},
        {"订票服务热线九五三三九办理", "订票服务热线95339办理"},
        {"我的电话是一二三四五六七", "我的电话是1234567"},
        {"联系电话幺三洞拐勾", "联系电话13079"},
    };
    for (const auto& [text, expected] : fourth_batch_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "fourth batch request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "fourth batch output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "fourth batch approval trace missing: " + text
        );
    }
    const auto telephone_pair = normalize_zh_conservative_v2(request(
        "联系电话幺三八零零一三八零零零或联系电话幺三九九幺二六三七九七"
    ));
    passed &= expect(
        telephone_pair.valid
            && telephone_pair.normalized.output
                == "联系电话13800138000或联系电话13991263797"
            && telephone_pair.normalized.applied.size() == 2
            && telephone_pair.decision_reason == "applied_multiple_semantic_objects",
        "released telephone pair failed"
    );
    const auto incomplete_split_400 = normalize_zh_conservative_v2(request(
        "联系电话是四零零零二九。三零六名员工参加培训。"
    ));
    passed &= expect(
        incomplete_split_400.valid
            && incomplete_split_400.normalized.output
                == "联系电话是400029。三零六名员工参加培训。",
        "incomplete split 400 number crossed the sentence boundary"
    );
    const std::vector<std::pair<std::string, std::string>> telephone_list_cases = {
        {
            "号码一零六九零六七零八或一零六九零五七零八或一零六九零四八八六",
            "号码106906708或106905708或106904886",
        },
        {
            "请拨打咨询服务电话零九九幺二二零二三七零或零九九幺二三五幺零零二",
            "请拨打咨询服务电话09912202370或09912351002",
        },
        {"他给幺幺九、幺二零、幺幺零。", "他给119、120、110。"},
        {"她给幺幺九幺二零幺幺零", "她给119、120、110"},
    };
    for (const auto& [text, expected] : telephone_list_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == expected
                && result.normalized.applied.size() == 1,
            "shared-anchor telephone list failed: " + text + " -> "
                + result.normalized.output
        );
    }

    const std::vector<std::string> fourth_batch_rejected_cases = {
        "手机高通骁龙八七零旗舰芯片",
        "像诺基亚幺幺零零这样的手提电话",
        "联系电话幺七七九五六九二六二七零二九八八五幺三八九八",
        "全球应急热线幺八六幺零幺二三零八",
        "《联系电话幺三九九幺二六三七九七》",
        "号码一零六九零六七零八或五万元",
        "联系电话幺五零二九八八八八零二零二九八四二六三七九二",
        "产品型号幺幺九、幺二零、幺幺零",
    };
    for (const auto& text : fourth_batch_rejected_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "fourth batch rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "fourth batch rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "fourth batch rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> fifth_batch_cases = {
        {"会议安排在上午九点", "会议安排在上午9点"},
        {"下午三点半开始", "下午3点半开始"},
        {"晚上八点十分播出", "晚上8点10分播出"},
        {"案发于凌晨一点二十分三十秒", "案发于凌晨1点20分30秒"},
        {"中午十二点整截止", "中午12点整截止"},
        {"夜里两点一刻醒来", "夜里2点一刻醒来"},
        {"清晨六点三刻出发", "清晨6点三刻出发"},
        {"上午九点零五分开会", "上午9点05分开会"},
        {"上午九点到十一点", "上午9点到11点"},
    };
    for (const auto& [text, expected] : fifth_batch_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "fifth batch request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "fifth batch output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "fifth batch approval trace missing: " + text
        );
    }

    const std::vector<std::string> fifth_batch_rejected_cases = {
        "晚上六七点钟",
        "下午四五点钟",
        "凌晨三点多",
        "下午五点左右",
        "下午十九点起飞",
        "上午十三点开会",
        "上午九点十秒",
        "《上午九点》",
    };
    for (const auto& text : fifth_batch_rejected_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "fifth batch rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "fifth batch rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "fifth batch rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> sixth_batch_cases = {
        {
            "二零二零年七月七号上午十一点三十分",
            "2020年7月7号上午11点30分"
        },
        {"截至三月三十号下午三点十二分", "截至3月30号下午3点12分"},
        {"至三月一号下午四点三十分", "至3月1号下午4点30分"},
        {
            "前文已经结束。至三月一号下午四点三十分",
            "前文已经结束。至3月1号下午4点30分"
        },
        {"二零二四年五月三日中午十二点整", "2024年5月3日中午12点整"},
        {"五月八号下午五点左右", "5月8号下午5点左右"},
        {
            "二零二零年四月十号凌晨三点四十分许",
            "2020年4月10号凌晨3点40分许"
        },
        {"七月二十八日下午十九点", "7月28日下午十九点"},
    };
    for (const auto& [text, expected] : sixth_batch_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "sixth batch request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "sixth batch output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "sixth batch approval trace missing: " + text
        );
    }

    const std::vector<std::string> sixth_batch_rejected_cases = {
        "《二零二四年五月三日上午九点》",
        "六月十三号上午十点至十四号上午十点",
        "到三月一号下午四点三十分",
        "至三月一号下午四点三十分左右",
        "二月一号上午九点至三月一号下午四点三十分",
    };
    for (const auto& text : sixth_batch_rejected_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "sixth batch rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "sixth batch rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "sixth batch rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> seventh_batch_cases = {
        {"参数最好是二点九秒", "参数最好是2.9秒"},
        {"耗时零点五秒", "耗时0.5秒"},
        {"实测延迟一点五毫秒", "实测延迟1.5毫秒"},
        {"下载用了二点五分钟", "下载用了2.5分钟"},
        {"周平均工作时间为四十四点三小时", "周平均工作时间为44.3小时"},
    };
    for (const auto& [text, expected] : seventh_batch_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "seventh batch request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "seventh batch output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "seventh batch approval trace missing: " + text
        );
    }

    const std::vector<std::string> seventh_batch_rejected_cases = {
        "不到一点五小时",
        "一点五小时到二小时",
        "二分十点五秒",
        "一点五小时三分钟",
        "一点二三四五秒",
        "《一点五小时》",
    };
    for (const auto& text : seventh_batch_rejected_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "seventh batch rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "seventh batch rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "seventh batch rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>>
        explicit_decimal_unit_approximation_cases = {
            {"但是它长度整个只有一点五公里左右", "但是它长度整个只有1.5公里左右"},
            {"约一点五公里", "约1.5公里"},
             {"大概一百零六点四六平方公里", "大概106.46平方公里"},
             {"大约一点五毫米", "大约1.5毫米"},
            {"大约一点五小时", "大约1.5小时"},
            {"二点九秒左右", "2.9秒左右"},
            {"接近十二点五分钟", "接近12.5分钟"},
        };
    for (const auto& [text, expected] : explicit_decimal_unit_approximation_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "decimal-unit approximation request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "decimal-unit approximation mismatch: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "decimal-unit approximation trace missing: " + text
        );
    }

    const std::vector<std::string> rejected_decimal_unit_approximation_cases = {
        "一点五公里多",
        "大约一点五公里以上",
        "预约一点五公里",
        "《一点五公里左右》",
        "一点五米每秒钟",
    };
    for (const auto& text : rejected_decimal_unit_approximation_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "decimal-unit approximation rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "decimal-unit approximation rejection changed: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "decimal-unit approximation rejection was approved: " + text
        );
    }
    auto split_decimal_approximation_prefix = request("一点五公里左右");
    split_decimal_approximation_prefix.left_context = "大约";
    split_decimal_approximation_prefix.has_left_neighbor = true;
    const auto split_decimal_approximation_prefix_result = normalize_zh_conservative_v2(
        split_decimal_approximation_prefix
    );
    passed &= expect(
        split_decimal_approximation_prefix_result.valid
            && split_decimal_approximation_prefix_result.normalized.output
                == split_decimal_approximation_prefix.text,
        "cross-segment decimal approximation prefix was approved"
    );
    auto split_decimal_approximation_suffix = request("一点五公里");
    split_decimal_approximation_suffix.right_context = "左右";
    split_decimal_approximation_suffix.has_right_neighbor = true;
    const auto split_decimal_approximation_suffix_result = normalize_zh_conservative_v2(
        split_decimal_approximation_suffix
    );
    passed &= expect(
        split_decimal_approximation_suffix_result.valid
            && split_decimal_approximation_suffix_result.normalized.output
                == split_decimal_approximation_suffix.text,
        "cross-segment decimal approximation suffix was approved"
    );

    const std::vector<std::pair<std::string, std::string>> year_cases = {
        {"在一九六三年发布", "在1963年发布"},
        {"二〇二四年度报告", "2024年度报告"},
        {"公元二九九九年", "公元2999年"},
        {"二零二四年度《报告》", "2024年度《报告》"},
        {"二零二六年", "2026年"},
        {"二零二四年五月发布", "2024年5月发布"},
        {"到二零一七年五月", "到2017年5月"},
    };
    for (const auto& [text, expected] : year_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "year request failed: " + text);
        passed &= expect(result.normalized.output == expected, "year output mismatch: " + text);
        passed &= expect(result.normalized.applied.size() == 1, "year approval trace missing");
        std::string failure;
        passed &= expect(
            validate_safe_normalization(
                result.normalized,
                {{
                    "zh.date.explicit_four_digit_year", "3", "zh-CN",
                    "date_time", "explicit_four_digit_year", nullptr,
                }},
                &failure
            ) == false,
            "validator accepted a registry without a trusted approver"
        );
    }

    const std::vector<std::pair<std::string, std::string>> year_range_cases = {
        {"从二零一四年至二零一五年", "从2014年至2015年"},
        {"二零一七年到二零一八年", "2017年到2018年"},
        {"一九九五年到一九九九年列装", "1995年到1999年列装"},
    };
    for (const auto& [text, expected] : year_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "year range request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "year range output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "year range approval trace missing: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> year_month_range_cases = {
        {"方案实施时间为二零二零年七月到十二月", "方案实施时间为2020年7月到12月"},
        {"二零二四年一月至三月", "2024年1月至3月"},
    };
    for (const auto& [text, expected] : year_month_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "year-month range request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "year-month range output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "year-month range approval trace missing: " + text
        );
    }
    const std::vector<std::string> rejected_year_month_range_cases = {
        "二零二零年一到七月",
        "一九六三年三至九月",
        "二零二零年十二月到三月",
        "二零二零年七月到七月",
        "二零二零年七月至二零二零年十二月",
        "二零二零年七月到十三月",
        "《二零二零年七月到十二月》",
    };
    for (const auto& text : rejected_year_month_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "year-month range rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "year-month range rejection changed: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "year-month range rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> full_date_range_cases = {
        {
            "二零二四年五月三日至二零二四年五月五日",
            "2024年5月3日至2024年5月5日",
        },
        {
            "二零一九年八月一号到二零二零年一月三十一号",
            "2019年8月1号到2020年1月31号",
        },
        {
            "签证在二零二零年一月二十四日至二零二零年七月三十一日到期",
            "签证在2020年1月24日至2020年7月31日到期",
        },
    };
    for (const auto& [text, expected] : full_date_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "full date range request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "full date range output mismatch: " + text + " -> " + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "full date range approval trace missing: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> shared_year_date_range_cases = {
        {
            "二零二四年五月三日至五月五日",
            "2024年5月3日至5月5日",
        },
        {
            "二零二零年七月三十一号至八月三号",
            "2020年7月31号至8月3号",
        },
        {
            "二零二四年五月三日至五日",
            "2024年5月3日至5日",
        },
        {
            "二零二四年的五月三日至五日",
            "2024年的5月3日至5日",
        },
        {
            "签证在二零二四年五月三日至五日到期",
            "签证在2024年5月3日至5日到期",
        },
    };
    for (const auto& [text, expected] : shared_year_date_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "shared-year date range request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "shared-year date range output mismatch: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "shared-year date range approval trace missing: " + text
        );
    }

    const std::vector<std::string> rejected_shared_year_date_range_cases = {
        "二零二四年十二月三十一日至一月一日",
        "二零二四年五月三日至五月三日",
        "二零二三年二月二十八日至二月二十九日",
        "二零二四年五月三日至五月五日上午",
        "《二零二四年五月三日至五月五日》",
        "二零二四年五月三日至五月五日至六月一日",
    };
    for (const auto& text : rejected_shared_year_date_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "shared-year date range rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "shared-year date range rejection changed: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "shared-year date range rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> shared_month_date_range_cases = {
        {"四月十五号到十九号", "4月15号到19号"},
        {"七月七号至八号", "7月7号至8号"},
        {"三月十六日至二十日陆续开学", "3月16日至20日陆续开学"},
    };
    for (const auto& [text, expected] : shared_month_date_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "shared-month date range request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "shared-month date range output mismatch: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "shared-month date range approval trace missing: " + text
        );
    }

    const std::vector<std::string> rejected_shared_month_date_range_cases = {
        "七月三十号至二号",
        "七月七号至七号",
        "四月三十日至三十一日",
        "四月一号至三号下午",
        "《四月一号至三号》",
        "四月一号至三号至五号",
        "一十月一号至三号",
    };
    for (const auto& text : rejected_shared_month_date_range_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "shared-month date range rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "shared-month date range rejection changed: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "shared-month date range rejected candidate was approved: " + text
        );
    }

    const std::vector<std::pair<std::string, std::string>> compound_duration_cases = {
        {"双十一当天仅用十六小时三十三分钟", "双十一当天仅用16小时33分钟"},
        {"全程足足要跑二十七小时十四分钟", "全程足足要跑27小时14分钟"},
    };
    for (const auto& [text, expected] : compound_duration_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "compound duration request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "compound duration output mismatch: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.size() == 1,
            "compound duration approval trace missing: " + text
        );
    }

    const std::vector<std::string> rejected_compound_duration_cases = {
        "一个小时二十分钟",
        "十二小时五分钟",
        "十六小时六十分钟",
        "大约十六小时三十三分钟",
        "十六小时三十三分钟左右",
        "十六到二十小时三十三分钟",
        "每十六小时三十三分钟",
        "十六小时三十三分钟二十秒",
        "片名《十六小时三十三分钟》",
    };
    for (const auto& text : rejected_compound_duration_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "compound duration rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "compound duration rejection changed: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "compound duration rejected candidate was approved: " + text
        );
    }

    const auto contextual_timestamp = normalize_zh_conservative_v2(request(
        "波音七三七又出事莫斯科时间五月六号九点二十五分"
    ));
    passed &= expect(contextual_timestamp.valid, "contextual timestamp request failed");
    passed &= expect(
        contextual_timestamp.normalized.output
            == "波音七三七又出事莫斯科时间5月6号9点25分",
        "contextual timestamp output mismatch: " + contextual_timestamp.normalized.output
    );
    passed &= expect(
        contextual_timestamp.normalized.applied.size() == 1,
        "contextual timestamp approval trace missing"
    );

    const auto contextual_full_date_timestamp = normalize_zh_conservative_v2(request(
        "北京时间二零二四年五月六号九点二十五分"
    ));
    passed &= expect(
        contextual_full_date_timestamp.normalized.output
            == "北京时间2024年5月6号9点25分",
        "contextual full-date timestamp output mismatch"
    );

    const std::vector<std::string> rejected_contextual_timestamp_cases = {
        "五月六号九点二十五分",
        "九点二十九分",
        "北京时间五月六号五点二十六分到五点二十八分",
        "北京时间五月六号五点十一分十二分的时候",
        "北京时间五月六号二十四点三十分",
        "北京时间五月六号二十三点六十分",
        "莫斯科时间《五月六号九点二十五分》",
    };
    for (const auto& text : rejected_contextual_timestamp_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "contextual timestamp rejection failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "contextual timestamp rejection changed: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "contextual timestamp rejected candidate was approved: " + text
        );
    }

    const auto contextual_approximate_timestamp = normalize_zh_conservative_v2(request(
        "截至北京时间八月十四日六点三十分左右"
    ));
    passed &= expect(
        contextual_approximate_timestamp.valid,
        "contextual approximate timestamp request failed"
    );
    passed &= expect(
        contextual_approximate_timestamp.normalized.output
            == "截至北京时间8月14日6点30分左右",
        "contextual approximate timestamp output mismatch: "
            + contextual_approximate_timestamp.normalized.output
    );
    passed &= expect(
        contextual_approximate_timestamp.normalized.applied.size() == 1,
        "contextual approximate timestamp approval trace missing"
    );
    for (const auto& [text, expected] : std::vector<std::pair<std::string, std::string>>{
             {"截至北京时间八月十四日六点三十分前后",
              "截至北京时间8月14日6点30分前后"},
             {"截至北京时间八月十四日六点三十分许",
              "截至北京时间8月14日6点30分许"},
         }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.normalized.output == expected,
                         "contextual approximate timestamp variant mismatch: " + text);
    }

    const std::vector<std::string> rejected_contextual_approximate_timestamp_cases = {
        "八月十四日六点三十分左右",
        "截至北京时间八月十四日二十四点三十分左右",
        "截至北京时间《八月十四日六点三十分左右》",
        "截至北京时间八月十四日六点三十分左右到七点",
    };
    for (const auto& text : rejected_contextual_approximate_timestamp_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid,
            "contextual approximate timestamp rejection failed: " + text
        );
        passed &= expect(
            result.normalized.output == text,
            "contextual approximate timestamp rejection changed: " + text + " -> "
                + result.normalized.output
        );
        passed &= expect(
            result.normalized.applied.empty(),
            "contextual approximate timestamp rejected candidate was approved: " + text
        );
    }

    for (const auto& [text, expected] : std::vector<std::pair<std::string, std::string>>{
             {"二零一九年七夕", "2019年七夕"},
             {"在二零二四年的十月份发布", "在2024年的10月份发布"},
             {"二零二四年五月份", "2024年5月份"},
             {"一九七零年代开始", "1970年代开始"},
             {"二零二四年五月三日上午十点至二零二四年五月五日下午三点",
              "2024年5月3日上午10点至2024年5月5日下午3点"},
             {"二零二四年五月三日上午九点三十分到二零二四年五月四日下午五点三十分",
              "2024年5月3日上午9点30分到2024年5月4日下午5点30分"},
         }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "expanded date rule failed: " + text);
        passed &= expect(result.normalized.output == expected,
                         "expanded date rule mismatch: " + text + " -> "
                             + result.normalized.output);
    }

    const std::vector<std::string> rejected_year_cases = {
        "《一九八四年》",
        "一九八四",
        "二零零二零一六年",
        "三零零零年",
        "零零零一年",
        "二0二四年",
        "《二零二四年度报告》",
        "〈二零二四年〉",
        "“二零二四年”",
        "‘二零二四年’",
        "「二零二四年」",
        "『二零二四年』",
        "【二零二四年】",
        "（二零二四年）",
        "(二零二四年)",
        "[二零二四年]",
        "\"二零二四年\"",
        "'二零二四年'",
        "文件名是二零二四年",
        "密码是二零二四年",
        "原文写的是二零二四年",
        "搜索词是二零二四年",
        "作品名叫二零二四年",
        "从二零一四年一五年开始",
        "二零一四至二零一五年",
        "但在二零一二年二零一三年开始",
        "一四年二零一五年开始",
        "这个是要等到一三一四年的时候",
        "我在一四一五年的时候",
        "我们是一五一四年底才成立的公司",
        "报告称一八四零年以来中国第一次在技术领域领先",
        "在二零二四年代发布",
        "从一九九零年那个年代就开始",
        "从零七年开始二零零七年开始",
        "从两千年到二零一八年结束",
        "一八四零年六月",
        "二零二四年一十月",
        "二零二四年十三月",
        "《二零二四年五月》",
        "文件名是二零二四年五月",
        "二零零二零一六年到二零一七年",
        "一八四零年到一八四九年",
        "《二零一四年至二零一五年》",
        "二零二四年二月二十九日至二零二三年二月二十九日",
        "《二零二四年五月三日至二零二四年五月五日》",
    };
    for (const auto& text : rejected_year_cases) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "rejected year caused failure: " + text);
        passed &= expect(result.normalized.output == text, "rejected year changed: " + text);
        passed &= expect(result.normalized.applied.empty(), "rejected year was approved: " + text);
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {
                "我说：“二零二六年七月，半导体巨头长兴科技登陆科创板。梁文峰通过旗下换方量化和九张资产，动用一百九十四只私募产品，参与长兴科技往下打新，合计获配两千零二十四点九七万股，配售金额约一点七五亿元，上市首日浮盈超八点二亿元。”",
                "我说：“2026年7月，半导体巨头长兴科技登陆科创板。梁文峰通过旗下换方量化和九张资产，动用194只私募产品，参与长兴科技往下打新，合计获配2024.97万股，配售金额约1.75亿元，上市首日浮盈超8.2亿元。”",
            },
            {
                "他表示：“二零二四年五月三日，预算一点五亿元。”",
                "他表示：“2024年5月3日，预算1.5亿元。”",
            },
            {
                "记者问：「项目在二零二六年七月启动吗？」",
                "记者问：「项目在2026年7月启动吗？」",
            },
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "direct-speech normalization failed: " + input + " -> "
                + result.normalized.output
        );
    }
    for (const auto& text : std::vector<std::string>{
            "“二零二四年”",
            "原文写作：“二零二四年五月三日，预算一点五亿元。”",
            "他说：“原文写作‘二零二四年’。”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "literal quotation changed: " + text + " -> " + result.normalized.output
        );
    }

    const auto once = normalize_zh_conservative_v2(request("在二零二六年发布"));
    const auto twice = normalize_zh_conservative_v2(request(once.normalized.output));
    passed &= expect(twice.normalized.output == once.normalized.output, "year rule is not idempotent");

    auto cross_segment_month = request("在二零二四年");
    cross_segment_month.right_context = "五月发布";
    cross_segment_month.has_right_neighbor = true;
    const auto cross_segment_month_result = normalize_zh_conservative_v2(cross_segment_month);
    passed &= expect(cross_segment_month_result.valid, "cross-segment month request failed");
    passed &= expect(
        cross_segment_month_result.normalized.output == "在2024年",
        "complete cross-segment month did not release the year"
    );

    auto cross_segment_month_right = request("五月发布");
    cross_segment_month_right.left_context = "在二零二四年";
    cross_segment_month_right.has_left_neighbor = true;
    const auto cross_segment_month_right_result = normalize_zh_conservative_v2(
        cross_segment_month_right
    );
    passed &= expect(
        cross_segment_month_right_result.valid,
        "cross-segment month-right request failed"
    );
    passed &= expect(
        cross_segment_month_right_result.normalized.output == "5月发布",
        "complete cross-segment month did not normalize the month"
    );

    for (const std::string& text : {"十三月发布", "五月三日发布"}) {
        auto incomplete_month = request(text);
        incomplete_month.left_context = "在二零二四年";
        incomplete_month.has_left_neighbor = true;
        const auto result = normalize_zh_conservative_v2(incomplete_month);
        passed &= expect(result.valid, "cross-segment month guard request failed");
        passed &= expect(
            result.normalized.output == text,
            "invalid or continuing cross-segment month was normalized"
        );
    }

    for (const std::string& right_context : {"5月发布", "５月发布", "正月发布", "代发布"}) {
        auto continuation = request("在二零二四年");
        continuation.right_context = right_context;
        continuation.has_right_neighbor = true;
        const auto result = normalize_zh_conservative_v2(continuation);
        passed &= expect(result.valid, "cross-segment continuation request failed");
        passed &= expect(
            result.normalized.output == continuation.text,
            "cross-segment date/number continuation changed the year"
        );
    }

    auto cross_segment_full_date = request("在二零二四年");
    cross_segment_full_date.right_context = "的五月发布";
    cross_segment_full_date.has_right_neighbor = true;
    const auto cross_segment_full_date_result = normalize_zh_conservative_v2(
        cross_segment_full_date
    );
    passed &= expect(
        cross_segment_full_date_result.valid,
        "cross-segment full-date request failed"
    );
    passed &= expect(
        cross_segment_full_date_result.normalized.output == cross_segment_full_date.text,
        "cross-segment full-date continuation changed the year"
    );

    auto range_left = request("从二零二四年");
    range_left.right_context = "至二零二五年发布";
    range_left.has_right_neighbor = true;
    const auto range_left_result = normalize_zh_conservative_v2(range_left);
    passed &= expect(range_left_result.valid, "cross-segment range left request failed");
    passed &= expect(
        range_left_result.normalized.output == range_left.text,
        "cross-segment range changed its left year"
    );

    auto range_right = request("二零二五年发布");
    range_right.left_context = "从二零二四年至";
    range_right.has_left_neighbor = true;
    const auto range_right_result = normalize_zh_conservative_v2(range_right);
    passed &= expect(range_right_result.valid, "cross-segment range right request failed");
    passed &= expect(
        range_right_result.normalized.output == "2025年发布",
        "complete cross-segment range did not normalize its right year"
    );

    auto complete_range_left = request("从二零一四年至");
    complete_range_left.right_context = "二零一五年开始";
    complete_range_left.has_right_neighbor = true;
    const auto complete_range_left_result = normalize_zh_conservative_v2(complete_range_left);
    passed &= expect(
        complete_range_left_result.normalized.output == "从2014年至",
        "complete cross-segment year range did not normalize its left year"
    );

    auto complete_range_right = request("二零一五年开始");
    complete_range_right.left_context = "从二零一四年至";
    complete_range_right.has_left_neighbor = true;
    const auto complete_range_right_result = normalize_zh_conservative_v2(complete_range_right);
    passed &= expect(
        complete_range_right_result.normalized.output == "2015年开始",
        "complete cross-segment year range did not normalize its right year"
    );

    auto prior_year = request("二零二五年发布");
    prior_year.left_context = "截至二零二四年";
    prior_year.has_left_neighbor = true;
    const auto prior_year_result = normalize_zh_conservative_v2(prior_year);
    passed &= expect(prior_year_result.valid, "cross-segment prior-year request failed");
    passed &= expect(
        prior_year_result.normalized.output == prior_year.text,
        "cross-segment prior year changed the following year"
    );

    auto range_after_digit = request("二零二五年发布");
    range_after_digit.left_context = "从二零二四到";
    range_after_digit.has_left_neighbor = true;
    const auto range_after_digit_result = normalize_zh_conservative_v2(range_after_digit);
    passed &= expect(range_after_digit_result.valid, "numeric range context request failed");
    passed &= expect(
        range_after_digit_result.normalized.output == range_after_digit.text,
        "numeric range context changed its right year"
    );

    auto independent_year = request("在二零二四年");
    independent_year.right_context = "发布新版本";
    independent_year.has_right_neighbor = true;
    const auto independent_year_result = normalize_zh_conservative_v2(independent_year);
    passed &= expect(independent_year_result.valid, "independent year request failed");
    passed &= expect(
        independent_year_result.normalized.output == "在2024年",
        "unrelated cross-segment context blocked an independent year"
    );

    const auto duplicate_rule = normalize_zh_conservative_v2(
        request("二零二四年和二零二五年")
    );
    passed &= expect(duplicate_rule.valid, "duplicate-rule diagnostic request failed");
    passed &= expect(
        duplicate_rule.normalized.output == "2024年和2025年"
            && duplicate_rule.normalized.applied.size() == 2,
        "duplicate-rule diagnostic changed output"
    );
    passed &= expect(
        duplicate_rule.decision_reason == "applied_multiple_semantic_objects"
            && duplicate_rule.failure_reason.empty(),
        "duplicate-rule diagnostic reason is missing"
    );
    passed &= expect(
        duplicate_rule.audit_duplicate_rule_candidate_counts.size() == 1
            && duplicate_rule.audit_duplicate_rule_candidate_counts[0].first
                == "zh.date.explicit_four_digit_year"
            && duplicate_rule.audit_duplicate_rule_candidate_counts[0].second == 2
            && duplicate_rule.audit_total_candidate_count == 2,
        "duplicate-rule candidate count is missing"
    );
    const auto duplicate_rule_shadow = normalize_zh_conservative_v2_shadow_atomic(
        request("二零二四年和二零二五年")
    );
    passed &= expect(
        duplicate_rule_shadow.valid
            && duplicate_rule_shadow.normalized.output == "2024年和2025年"
            && duplicate_rule_shadow.normalized.applied.size() == 2
            && duplicate_rule_shadow.decision_reason
                == "shadow_atomic_applied_multiple_transformations",
        "duplicate-rule shadow did not atomically apply independent candidates"
    );

    const auto released_same_rule_pair = normalize_zh_conservative_v2(
        request("占比从百分之五上升到百分之八十")
    );
    passed &= expect(
        released_same_rule_pair.valid
            && released_same_rule_pair.normalized.output == "占比从5%上升到80%"
            && released_same_rule_pair.normalized.applied.size() == 1
            && released_same_rule_pair.decision_reason == "applied",
        "percentage range semantic object was not applied"
    );
    const auto same_rule_triple = normalize_zh_conservative_v2(
        request("百分之十和百分之二十和百分之三十")
    );
    passed &= expect(
        same_rule_triple.valid
            && same_rule_triple.normalized.output == "10%和20%和30%"
            && same_rule_triple.normalized.applied.size() == 3
            && same_rule_triple.decision_reason == "applied_multiple_semantic_objects",
        "independent same-rule triple was not applied"
    );
    const auto released_same_rule_triple = normalize_zh_conservative_v2(
        request("五G网络升级六G并兼容四G")
    );
    passed &= expect(
        released_same_rule_triple.valid
            && released_same_rule_triple.normalized.output
                == "5G网络升级6G并兼容4G"
            && released_same_rule_triple.normalized.applied.size() == 3
            && released_same_rule_triple.decision_reason
                == "applied_multiple_semantic_objects",
        "released same-rule triple was not applied"
    );
    const auto released_year_percentage_triple = normalize_zh_conservative_v2(
        request(
            "中指院的数据,二零二六年上半年,二十二个重点城市累计拿地金额中,"
            "央国企占比达到百分之五十四,一线城市及厦门等地更是超过百分之六十,"
            "广州、深圳甚至超过了百分之九十。"
        )
    );
    passed &= expect(
        released_year_percentage_triple.valid
            && released_year_percentage_triple.normalized.output
                == "中指院的数据,2026年上半年,二十二个重点城市累计拿地金额中,"
                   "央国企占比达到54%,一线城市及厦门等地更是超过60%,"
                   "广州、深圳甚至超过了90%。"
            && released_year_percentage_triple.normalized.applied.size() == 4
            && released_year_percentage_triple.decision_reason
                == "applied_multiple_semantic_objects",
        "released year and percentage triple combination was not applied"
    );
    const auto partially_approved_year_percentage_triple = normalize_zh_conservative_v2(
        request("二零二六年占比分别为百分之十、百分之二十和百分之三十")
    );
    passed &= expect(
        partially_approved_year_percentage_triple.valid
            && partially_approved_year_percentage_triple.normalized.output
                == "2026年占比分别为10%、20%和30%"
            && partially_approved_year_percentage_triple.normalized.applied.size() == 4
            && partially_approved_year_percentage_triple.decision_reason
                == "applied_multiple_semantic_objects",
        "independent year and percentage triple was not applied"
    );
    const auto unreleased_month_day_percentage_triple = normalize_zh_conservative_v2(
        request("七月六日占比分别为百分之十、百分之二十和百分之三十")
    );
    passed &= expect(
        unreleased_month_day_percentage_triple.valid
            && unreleased_month_day_percentage_triple.normalized.output
                == "7月6日占比分别为10%、20%和30%"
            && unreleased_month_day_percentage_triple.normalized.applied.size() == 4
            && unreleased_month_day_percentage_triple.decision_reason
                == "applied_multiple_semantic_objects",
        "independent month-day and percentage triple was not applied"
    );

    const auto multiple_approved = normalize_zh_conservative_v2(
        request("七月六日销售假冒N九五口罩")
    );
    passed &= expect(multiple_approved.valid, "multiple-approval diagnostic request failed");
    passed &= expect(
        multiple_approved.normalized.output == "7月6日销售假冒N95口罩"
            && multiple_approved.normalized.applied.size() == 2,
        "released distinct-rule pair was not applied"
    );
    passed &= expect(
        multiple_approved.decision_reason
            == "applied_multiple_semantic_objects",
        "released distinct-rule pair reason is missing"
    );
    const auto multiple_approved_shadow = normalize_zh_conservative_v2_shadow_atomic(
        request("七月六日销售假冒N九五口罩")
    );
    passed &= expect(
        multiple_approved_shadow.valid
            && multiple_approved_shadow.normalized.output == "7月6日销售假冒N95口罩"
            && multiple_approved_shadow.normalized.applied.size() == 2,
        "multiple-approval shadow atomic application failed"
    );

    const auto released_full_date_range_and_full_date = normalize_zh_conservative_v2(
        request(
            "签证有效期为二零二零年一月二十四日至二零二零年七月三十一日，"
            "复查日期为二零二零年八月三十一日"
        )
    );
    passed &= expect(
        released_full_date_range_and_full_date.valid
            && released_full_date_range_and_full_date.normalized.output
                == "签证有效期为2020年1月24日至2020年7月31日，"
                   "复查日期为2020年8月31日"
            && released_full_date_range_and_full_date.normalized.applied.size() == 2
            && released_full_date_range_and_full_date.decision_reason
                == "applied_multiple_semantic_objects",
        "released full-date-range and full-date pair was not applied"
    );
    const auto unreleased_full_date_range_date_and_n95 = normalize_zh_conservative_v2(
        request(
            "签证有效期为二零二零年一月二十四日至二零二零年七月三十一日，"
            "复查日期为二零二零年八月三十一日，需佩戴N九五口罩"
        )
    );
    passed &= expect(
        unreleased_full_date_range_date_and_n95.valid
            && unreleased_full_date_range_date_and_n95.normalized.output
                == "签证有效期为2020年1月24日至2020年7月31日，"
                   "复查日期为2020年8月31日，需佩戴N95口罩"
            && unreleased_full_date_range_date_and_n95.normalized.applied.size() == 3
            && unreleased_full_date_range_date_and_n95.decision_reason
                == "applied_multiple_semantic_objects",
        "independent full-date-range, full-date, and N95 objects were not applied"
    );

    const auto unreleased_pair = normalize_zh_conservative_v2(
        request("五G网络使用N九五口罩")
    );
    passed &= expect(
        unreleased_pair.valid
            && unreleased_pair.normalized.output == "5G网络使用N95口罩"
            && unreleased_pair.normalized.applied.size() == 2
            && unreleased_pair.decision_reason
                == "applied_multiple_semantic_objects",
        "independent distinct-rule pair was not applied"
    );

    const auto year_percentage_pair = normalize_zh_conservative_v2(
        request("公司在二零一九年基本增长百分之十")
    );
    passed &= expect(
        year_percentage_pair.valid
            && year_percentage_pair.normalized.output
                == "公司在2019年基本增长10%"
            && year_percentage_pair.normalized.applied.size() == 2
            && year_percentage_pair.decision_reason
                == "applied_multiple_semantic_objects",
        "released year and percentage pair failed"
    );

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"九点三十分开会", "9点30分开会"},
            {"会议时间为九点零五分", "会议时间为9点05分"},
            {"九点三十分左右开会", "9点30分左右开会"},
            {"发车时间约为八点十五分", "发车时间约为8点15分"},
            {"在二零一七年的十二月十六日发布", "在2017年的12月16日发布"},
            {"在二零二四年的六月一号发布", "在2024年的6月1号发布"},
            {"十一秒钟后继续", "11秒钟后继续"},
            {"大约十一秒钟", "大约11秒钟"},
            {"一点五秒钟", "1.5秒钟"},
            {"一点五秒钟左右", "1.5秒钟左右"},
            {"至少二点五小时", "至少2.5小时"},
            {"一点五小时以内", "1.5小时以内"},
            {"重量二十五公斤", "重量25公斤"},
            {"重量二十五千克", "重量25千克"},
            {"面积三十公顷", "面积30公顷"},
            {"体积十二立方米", "体积12立方米"},
            {"采样率一百九十二千赫兹", "采样率192千赫兹"},
            {"刷新率一百四十四赫兹", "刷新率144赫兹"},
            {"频率二点四吉赫兹", "频率2.4吉赫兹"},
            {"不超过一点五公里", "不超过1.5公里"},
            {"二点五千克以上", "2.5千克以上"},
            {"大约三十摄氏度", "大约30摄氏度"},
            {"不超过三十五摄氏度", "不超过35摄氏度"},
            {"温度零下十到五摄氏度", "温度零下10到5摄氏度"},
            {"三十摄氏度至四十摄氏度", "30摄氏度至40摄氏度"},
            {"设备正常温度应保持在六十到六十二度之间", "设备正常温度应保持在60到62度之间"},
        }) {
        const auto relaxed = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            relaxed.valid && relaxed.normalized.output == expected,
            "released strong-anchor relaxation produced an unexpected result"
        );
    }
    for (const auto& text : std::vector<std::string>{
            "比分九点三十分",
            "九点三十分左右",
            "九点三十分前后开会",
            "《二零二四年五月三日至五日》",
            "二零二四年五月三日至五日上午",
            "二零二四年五月五日至三日",
            "《九点三十分开会》",
            "二零一七年的十三月一日",
            "一秒钟后继续",
            "一千克苹果",
            "大约一点五小时以上",
            "《二十五千克》",
            "大约五摄氏度",
            "大约三十摄氏度以上",
            "三到五摄氏度",
            "零下十到五华氏度到六华氏度",
            "完成六十到六十二度大转弯",
            "洛氏硬度应保持在六十到六十二度之间",
        }) {
        const auto preserved = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            preserved.valid && preserved.normalized.output == text,
            "strong-anchor relaxation changed a negative case"
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {
                "标题: 二零二四年五月三日中的文字也不应改写",
                "标题: 2024年5月3日中的文字也不应改写",
            },
            {
                "标题是二零二四年五月三日工作报告",
                "标题是2024年5月3日工作报告",
            },
        }) {
        const auto title_date = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            title_date.valid
                && title_date.normalized.output == expected
                && title_date.normalized.applied.size() == 1,
            "title date was not normalized"
        );
    }
    const auto literal_file_name_date = normalize_zh_conservative_v2(
        request("文件名: 二零二四年五月三日")
    );
    passed &= expect(
        literal_file_name_date.valid
            && literal_file_name_date.normalized.output == "文件名: 二零二四年五月三日"
            && literal_file_name_date.normalized.applied.empty(),
        "literal file name after a colon was unexpectedly normalized"
    );

    const auto three_approved = normalize_zh_conservative_v2(
        request("七月六日并且五G网络并且百分之九十")
    );
    passed &= expect(
        three_approved.valid
            && three_approved.normalized.output
                == "7月6日并且5G网络并且90%"
            && three_approved.normalized.applied.size() == 3
            && three_approved.decision_reason
                == "applied_multiple_semantic_objects",
        "three independent transformations were not applied"
    );

    const std::vector<std::pair<std::string, std::string>> money_policy_matrix = {
        {"一点三亿美元左右", "1.3亿美元左右"},
        {"约为一万两千零四点八元", "约为12004.8元"},
        {"大约八十亿美元", "大约80亿美元"},
        {"一百七十万美元左右", "170万美元左右"},
        {"十亿美元以上", "10亿美元以上"},
        {"至少两千万美元", "至少2000万美元"},
        {"超过五千元", "超过5000元"},
        {"西安市累计达到五点五亿元", "西安市累计达到5.5亿元"},
        {"最高的预算不超过一百万元", "最高的预算不超过100万元"},
        {"大约五亿美元", "大约五亿美元"},
        {"大约十亿美元以上", "大约十亿美元以上"},
        {"几十万美元左右", "几十万美元左右"},
        {"一点五至二点五亿元", "1.5至2.5亿元"},
    };
    for (const auto& [input, expected] : money_policy_matrix) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "money policy matrix produced an unexpected result"
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"基本养老金一千五百元", "基本养老金1500元"},
            {"累计爬升九千七百三十八米", "累计爬升9738米"},
            {"二零二四年，基本养老金一千五百元", "2024年，基本养老金1500元"},
            {"二零二四年，累计爬升九千七百三十八米", "2024年，累计爬升9738米"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v49 batch A produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "基本养老金一元",
            "步行九千米",
            "海拔五千三百米五千八百米",
            "故障率为千分之零点二三",
            "抽样误差为千分之三",
            "海拔五千三百米五千八百米",
            "故障率为千分之几点",
            "抽样误差约为千分之几",
            "原文写作“基本养老金一千五百元”",
            "原文写作“累计爬升九千七百三十八米”",
            "原文写作“抽样误差为千分之三”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v49 batch A changed a protected or ambiguous case: " + text
        );
    }
    for (const auto& segment : std::vector<V2SegmentRequest>{
            {"基本养老金一千", "", "五百元", "zh-CN", false, true},
            {"累计爬升九千七百", "", "三十八米", "zh-CN", false, true},
            {"故障率为千分之", "", "零点二三", "zh-CN", false, true},
            {"抽样误差为千分", "", "之三", "zh-CN", false, true},
        }) {
        const auto result = normalize_zh_conservative_v2(segment);
        passed &= expect(
            result.valid && result.normalized.output == segment.text,
            "v49 batch A reconstructed an incomplete cross-segment object"
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"车辆以一百六十公里每小时的速度前进", "车辆以160公里每小时的速度前进"},
            {"G二四三八次列车到站", "G2438次列车到站"},
            {"T幺七九次列车已经出发", "T179次列车已经出发"},
            {"D七幺六六车次晚点", "D7166车次晚点"},
            {
                "G二四三八次列车到站。T幺七九次列车已经出发",
                "G2438次列车到站。T179次列车已经出发"
            },
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v50 batch B produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "G二次列车", "G二四三八号列车", "AG二四三八次列车",
            "G二四三八次列车A", "g二四三八次列车", "G拐九次列车",
            "《G二四三八次列车》",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v50 batch B changed an incomplete or protected train number: " + text
        );
    }
    const auto split_train = normalize_zh_conservative_v2(
        {"乘坐G二四三八", "", "次列车到站", "zh-CN", false, true}
    );
    passed &= expect(
        split_train.valid && split_train.normalized.output == "乘坐G二四三八",
        "v50 batch B reconstructed an incomplete cross-segment train number"
    );

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"搭乘HU七五四航班回国", "搭乘HU754航班回国"},
            {"暂停三U八三九二航班运行", "暂停3U8392航班运行"},
            {"航班号为CA一二三四", "航班号为CA1234"},
            {"查询CA九二零这个航班", "查询CA920这个航班"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v51 flight-number rule produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "编号为CA一二三四", "搭乘ca一二三四航班", "搭乘CCA一二三四航班",
            "搭乘CA一航班", "搭乘CA拐九航班", "搭乘一CA一二三四航班",
            "《CA一二三四航班》",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v51 flight-number rule changed an incomplete or protected case: " + text
        );
    }
    const auto split_flight = normalize_zh_conservative_v2(
        {"搭乘CA一二三四", "", "航班回国", "zh-CN", false, true}
    );
    passed &= expect(
        split_flight.valid && split_flight.normalized.output == "搭乘CA一二三四",
        "v51 flight-number rule reconstructed an incomplete cross-segment object"
    );

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"利率下调二十个基点", "利率下调20个基点"},
            {"利率下调二十基点", "利率下调20基点"},
            {"调降幅度达到十至二十个基点", "调降幅度达到10至20个基点"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v52 basis-point extension produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "利率下调五个基点", "利率下调五至二十个基点", "这是二十个基本点",
            "这个基点时代", "原文写作“利率下调二十个基点”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v52 basis-point extension changed a protected or uncertain case: " + text
        );
    }
    const auto split_basis_point_range = normalize_zh_conservative_v2(
        {"调降幅度达到十至二十", "", "个基点", "zh-CN", false, true}
    );
    passed &= expect(
        split_basis_point_range.valid
            && split_basis_point_range.normalized.output == "调降幅度达到十至二十",
        "v52 basis-point extension reconstructed an incomplete cross-segment range"
    );

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"今年五月发布报告", "今年5月发布报告"},
            {"明年十一月份复核", "明年11月份复核"},
            {"每年十一月到次年三月是采挖期", "每年11月到次年3月是采挖期"},
            {"每年十月份至次年二月份轮作", "每年10月份至次年2月份轮作"},
            {"铁棍山药穿越二千余年时光", "铁棍山药穿越2000余年时光"},
            {"这座建筑已有一百余年历史", "这座建筑已有100余年历史"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v54 calendar and large-year-span rules produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "一年种山药，十年无地利", "种一季地要歇十年", "这位有着二十年种植经验",
            "十年三个月", "二十余年", "今年十三月", "每年十一月到次年十三月",
            "《二千余年》",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v54 calendar and large-year-span rules changed a protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"动用一百九十四只私募产品参与打新", "动用194只私募产品参与打新"},
            {"合计获配两千零二十四点九七万股", "合计获配2024.97万股"},
            {"获配九十三点三三九万股", "获配93.339万股"},
            {"此后八月，宇树科技上市", "此后8月，宇树科技上市"},
            {"锁定期三十六个月", "锁定期36个月"},
            {"今年4月开启融资，在六月完成交割", "今年4月开启融资，在6月完成交割"},
            {"在二十万元及以上中国新能源汽车市场中", "在20万元及以上中国新能源汽车市场中"},
            {"二十万元及其他费用", "20万元及其他费用"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v55 finance and contextual-month rules produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "在六月完成交割", "今年4月开启融资。下次在六月完成交割",
            "服务周期是十二个月", "锁定期三个月", "九章资产参与投资",
            "锁定期或缩短为三年",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v55 finance and contextual-month rules changed a protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"第二轮融资投前估值甚至五千亿元", "第二轮融资投前估值甚至5000亿元"},
            {"提升三点一个百分点", "提升3.1个百分点"},
            {"第一季度交付九万五千一百四十二辆", "第一季度交付95142辆"},
            {"目前芯片交付已超五万颗", "目前芯片交付已超5万颗"},
            {"纯电车型理想I六累计下线超十八万辆", "纯电车型理想I6累计下线超18万辆"},
            {"二季度全新理想L九等高ASP车型", "二季度全新理想L9等高ASP车型"},
            {"全新L九、全新L八和新一代L六的换代上市", "全新L9、全新L8和新一代L6的换代上市"},
            {"自研马赫M一百芯片启动交付", "自研马赫M100芯片启动交付"},
            {"酷睿处理器I九幺零九零零K", "酷睿处理器I910900K"},
            {"搭载M二五四系列发动机", "搭载M254系列发动机"},
            {"七月底发布OTA九点一版本", "7月底发布OTA9.1版本"},
            {"七月单月交付三万零四百六十八辆", "7月单月交付30468辆"},
            {"五月到七月完成换代", "5月到7月完成换代"},
            {"截至八月累计下线", "截至8月累计下线"},
            {"融资规模三千至五千亿元", "融资规模3000至5000亿元"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v56 contextual finance and product rules produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "A一点意见", "L六个月",
            "方案B二组", "发动机有二点儿零T两种排量", "同比增长二点五",
            "一万九千三百四千七十二辆",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v56 contextual finance and product rules changed a protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"北京六月首宗宅地成交", "北京6月首宗宅地成交"},
            {"七月首批项目开工", "7月首批项目开工"},
            {"DX零零杠零二零二杠零一八八地块", "DX00-0202-0188地块"},
            {"楼面价每平方米二万三千零一十元", "楼面价每平方米23010元"},
            {"面积约二万一千四百一十六点零三平方米", "面积约21416.03平方米"},
            {"面积不超过三万八千五百四十八点八五平方米", "面积不超过38548.85平方米"},
            {"容积率一点八", "容积率1.8"},
            {"建筑限高三十六米", "建筑限高36米"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v57 real-estate rules produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "六月首饰展览", "DX零零杠零二零二项目", "DX零杠一地块",
            "一点八容积率", "三十六米布料", "土地一级开发项目",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v57 real-estate rules changed a protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {
                "今年3月启动，环比提升百分之三点四。7月底，OTA九点一版本，"
                "模型参数将成倍提升。",
                "今年3月启动，环比提升3.4%。7月底，OTA9.1版本，"
                "模型参数将成倍提升。",
            },
            {
                "同比增长二点五；环比提升百分之三点四。七月单月交付30468辆。",
                "同比增长二点五；环比提升3.4%。7月单月交付30468辆。",
            },
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v57 mixed numeric contexts produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{"今年3月启动", "今年5月交付"}) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text
                && result.audit_total_candidate_count == 0,
            "v57 accepted an already-normalized relative month candidate: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"仓库登记了三亿零四十万零八十件零件", "仓库登记了300400080件零件"},
            {"一号库有一亿零三万零五十件", "一号库有100030050件"},
            {"二号库有十三亿零八万零七百件", "二号库有1300080700件"},
            {"待复核区还剩五万零八十件", "待复核区还剩50080件"},
            {"一号生产线完成了一百零二批", "一号生产线完成了102批"},
            {"系统准确记录的是二百零五人", "系统准确记录的是205人"},
            {"还有六千五百万人的减税幅度", "还有6500万人的减税幅度"},
            {"中国有十三亿人口", "中国有13亿人口"},
            {"药物剂量为零点三五毫克", "药物剂量为0.35毫克"},
            {"压力范围是零点八到一点二兆帕", "压力范围是0.8到1.2兆帕"},
            {"转速不得低于每分钟一千五百转", "转速不得低于每分钟1500转"},
            {"视频帧率是每秒五十九点九四帧", "视频帧率是每秒59.94帧"},
            {"台风中心气压为九百八十百帕", "台风中心气压为980百帕"},
            {"血压为一百四十五毫米汞柱比九十五毫米汞柱", "血压为145毫米汞柱比95毫米汞柱"},
            {"贷款年利率从百分之三点四五降至百分之三点二五", "贷款年利率从3.45%降至3.25%"},
            {"材料按照一比三混合", "材料按照1比3混合"},
            {"新设备的效率是旧设备的一点五倍", "新设备的效率是旧设备的1.5倍"},
            {"平均偏差约为零点零零五", "平均偏差约为0.005"},
            {"传感器允许误差为正负零点零三", "传感器允许误差为正负0.03"},
            {"最低值是负二点五，最高值是正三点二", "最低值是负2.5，最高值是正3.2"},
            {"系统在十四点零五分三十秒启动", "系统在14点05分30秒启动"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v58 exact numeric structures produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "现场大概有两百多人", "一件事情", "一千九百二十人乘一千零八十人",
            "峰值性能最多提高两倍", "每分钟三转", "十四点零五分三十秒",
            "AI两千人工智能全球最具影响力学者榜单公布",
            "如果我给你一部手机只要一千人民币", "我觉得是放大一百倍",
            "原文写作“药物剂量为零点三五毫克”", "三心二意不是三个心两个意",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v58 exact numeric structures changed a protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {
                "交通管制从四月三十日十二时持续到五月一日二十二时",
                "交通管制从4月30日12时持续到5月1日22时",
            },
            {
                "活动一直持续到十一月十二日零点",
                "活动一直持续到11月12日0点",
            },
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "contextual date and whole-hour rule produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "交通管制从四月三十日十二时三十分开始",
            "活动持续到十一月十二日零点半",
            "活动持续到十一月三十一日零点",
            "原文写作“交通管制从四月三十日十二时开始”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "contextual date and whole-hour rule changed a protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"办公地点是B座四零二室", "办公地点是B座402室"},
            {"会议厅位于三号楼五层零六号房", "会议厅位于三号楼五层06号房"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "closed room identifier produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "四零二室设备", "B座四零二号设备",
            "原文写作“办公地点是B座四零二室”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "closed room identifier changed an unanchored or protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"小张语文考了九十八点五分", "小张语文考了98.5分"},
            {"平均得分为六十四点三四分", "平均得分为64.34分"},
            {"数学一百零二分，英语八十五分", "数学102分，英语85分"},
            {"主力选手得到三十五分", "主力选手得到35分"},
            {"替补选手砍下十二分", "替补选手砍下12分"},
            {"三十五分钟", "35分钟"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "contextual score rule produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "三分之一", "价格八十五分", "感情只有三分",
            "这篇文章值得八分", "语文考了九分", "原文写作“语文考了九十八点五分”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "contextual score rule changed an ambiguous or protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"总成绩排在全年级第三十五名", "总成绩排在全年级第35名"},
            {"主力选手排名第十二名", "主力选手排名第12名"},
            {"最终取得第一百零二名", "最终取得第102名"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "contextual rank rule produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "排名全国第九名", "第三十五名学生", "第三十五名参赛者",
            "第三十五层", "第三十五章", "第三十五页",
            "原文写作“总成绩排在全年级第三十五名”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "contextual rank rule changed an unanchored or protected case: " + text
        );
    }
    const auto ordinary_person_count = normalize_zh_conservative_v2(request("第三十五人"));
    passed &= expect(
        ordinary_person_count.valid
            && ordinary_person_count.normalized.output == "第35人"
            && ordinary_person_count.normalized.applied.size() == 1
            && ordinary_person_count.normalized.applied.front().candidate.rule_id
                != "zh.ordinal.contextual_rank",
        "contextual rank rule intercepted an ordinary person count"
    );

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"请订正试卷第十二页第八题", "请订正试卷第12页第八题"},
            {"请查看文件第三十五页第二条", "请查看文件第35页第二条"},
            {"报告第一百零二页列出了明细", "报告第102页列出了明细"},
            {
                "请订正试卷第三页第二题和第十二页第八题",
                "请订正试卷第三页第二题和第12页第八题",
            },
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "contextual document page rule produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "试卷第三页第二题", "第三十五页", "第三十五页纸", "这本书有三十五页",
            "第三十五章", "第十二题", "第十二条", "原文写作“文件第三十五页”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "contextual document page rule changed an unanchored or protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"验证码是一二三四五六", "验证码是123456"},
            {"邮编是一二三四五六", "邮编是123456"},
            {"工号是一二三四五六", "工号是123456"},
            {"学号是一二三四五六", "学号是123456"},
            {"合同号是一二三四五六", "合同号是123456"},
            {"序列号是一二三四五六", "序列号是123456"},
            {"设备号是一二三四五六", "设备号是123456"},
            {"代码是一二三四五六", "代码是123456"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "explicit identifier anchor coverage produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "验证码是一二", "邮编是一二", "工号是一二", "学号是一二",
            "合同号是一二", "序列号是一二", "设备号是一二", "代码是一二",
            "原文写作“验证码是一二三四五六”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "explicit identifier anchor coverage changed a short or protected case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"集团本季度营收为五百一十二亿六千万元", "集团本季度营收为512亿6000万元"},
            {"房子位于三十二层顶楼的零四室", "房子位于三十二层顶楼的04室"},
            {"也就是三二零四号房", "也就是3204号房"},
            {"会议厅位于零六号房", "会议厅位于06号房"},
            {"实际称重为二百三十五点四斤", "实际称重为235.4斤"},
            {"误差不到零点五斤", "误差不到0.5斤"},
            {"收货地址是南京市玄武区中山路三百零四号", "收货地址是南京市玄武区中山路304号"},
            {"电池容量是一百零二度", "电池容量是102度"},
            {"满电续航七百多公里", "满电续航700多公里"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v65 missed an approved long-form recovery structure: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "货物看起来有两三百斤", "大概有两三百斤", "四零二室设备",
            "原文写作“集团本季度营收为五百一十二亿六千万元”",
            "原文写作“也就是三二零四号房”",
            "中山路三百零四号文件", "完成一百八十度大转弯",
            "洛氏硬度只有五十五度", "七百多公里", "七百多个问题",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v65 changed an ambiguous or protected recovery case: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"G三六宁洛高速", "G36宁洛高速"},
            {"K洞马群枢纽至K二三拥庄枢纽", "K0马群枢纽至K23拥庄枢纽"},
            {
                "限行时间为四月三十日十二时至五月一日二十二时",
                "限行时间为4月30日12时至5月1日22时",
            },
            {"截至二月二十三号八时", "截至2月23号8时"},
            {"码率为三百二十千比特每秒", "码率为320千比特每秒"},
            {"声道格式为五点一环绕声", "声道格式为5.1环绕声"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v67 anchored road, timestamp, or audio structure produced an unexpected result: "
                + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "G三六项目组", "K二三章节", "五点一想法",
            "原文写作“G三六宁洛高速”", "原文写作“声道格式为五点一环绕声”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v67 changed an unanchored or protected road or audio structure: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"测试版本是四点零点八点二三点一", "测试版本是4.0.8.23.1"},
            {"升级到二零二六点八点二十八构建版本", "升级到2026.8.28构建版本"},
            {"安装包大小是二点三吉比特", "安装包大小是2.3吉比特"},
            {"缓存目录还有五百一十二兆字节", "缓存目录还有512兆字节"},
            {"服务器总存储容量为两太字节", "服务器总存储容量为2太字节"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v68 version or storage structure produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "数值是四点零点八点二三点一", "两只字节", "原文写作“二点三吉比特”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v68 changed an unanchored or protected version or storage structure: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"这只股票开盘价是十二块三毛五", "这只股票开盘价是12.35元"},
            {"最高涨到十三块零八分", "最高涨到13.08元"},
            {"收盘报十二块九毛六", "收盘报12.96元"},
            {"成交量为三亿零五百零八万股", "成交量为3亿508万股"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v69 stock price or share count produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "商品价格是十二块三毛五", "口袋里有十三块零八分",
            "这只股票写作“十二块三毛五”", "三亿零五百零八万股东",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v69 changed an unanchored, approximate, or protected stock structure: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"采用二纳米制程", "采用2纳米制程"},
            {"十二核CPU和十二核GPU", "12核CPU和12核GPU"},
            {"双十六核神经引擎", "双16核神经引擎"},
            {"最多三十六核CPU，最多八十核GPU", "最多36核CPU，最多80核GPU"},
            {"带宽达到每秒四点四TB", "带宽达到每秒4.4TB"},
            {"统一内存带宽为每秒一百七十GB", "统一内存带宽为每秒170GB"},
            {"容量最高五百一十二GB", "容量最高512GB"},
            {"E. P. Y. C. 九零零六Venice服务器芯片", "E. P. Y. C. 9006Venice服务器芯片"},
            {"M. I. 四五五XAI加速器", "M. I. 455XAI加速器"},
            {"M六是一颗入门芯片", "M6是一颗入门芯片"},
            {"M六统一内存带宽最高", "M6统一内存带宽最高"},
            {"比M五最高提升1.2倍，比M一提升2.4倍", "比M5最高提升1.2倍，比M1提升2.4倍"},
            {"比M三Ultra高50%", "比M3Ultra高50%"},
            {"容量相差十六倍", "容量相差16倍"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v70 technical specification produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "两颗超级核心", "四颗性能核心", "六颗能效核心",
            "提高六倍", "我觉得是放大一百倍", "容量五GB", "原文写作“十二核CPU”",
            "A. B. 三四五普通编号", "九零零六Venice服务器芯片",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v70 changed a natural single digit or protected technical expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"昨日新增本土病例三十六例", "昨日新增本土病例36例"},
            {"第五十届全国运动会", "第50届全国运动会"},
            {"已有六十四个代表团报名", "已有64个代表团报名"},
            {"预计接待观众超过三十五万人次", "预计接待观众超过35万人次"},
            {
                "预计发放总额二千九百七十二万八千零九十一元整",
                "预计发放总额2972万8091元整",
            },
            {
                "营业收入达到五千一百八十八万五千一百二十八元三角",
                "营业收入达到5188万5128元3角",
            },
            {
                "净利润七千八百二十九万一千一百七十三元一角",
                "净利润7829万1173元1角",
            },
            {"上证指数报二千五百一十二点", "上证指数报2512点"},
            {"两市成交额四万一千九百七十六亿", "两市成交额41976亿"},
            {
                "北向资金净流入四百零一万一千三百四十五元整",
                "北向资金净流入401万1345元整",
            },
            {
                "订单金额为八千二百一十一万三千六百六十八元七角",
                "订单金额为8211万3668元7角",
            },
            {
                "实付二千三百五十七万二千二百二十七元整",
                "实付2357万2227元整",
            },
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v71 news or financial structure produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "有三十六例外情况", "召开十届会议", "一元三角形函数",
            "预计费用二千元", "指数为三点五", "满十六减四百三十",
            "原文写作“第五十届全国运动会”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v71 changed an ambiguous or protected news or financial expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"取件码为九三二七", "取件码为9327"},
            {"包裹单号五九六八二八四五九八八三", "包裹单号596828459883"},
            {"快递单号九一三四一九五三八一八六", "快递单号913419538186"},
            {"工作电流三十七安", "工作电流37安"},
            {"平均值四万七千九百一十九点八七四", "平均值47919.874"},
            {"电阻约为三千一百五十欧姆", "电阻约为3150欧姆"},
            {"请到三十五站台候车", "请到35站台候车"},
            {"本次列车限速一百一十五千米每小时", "本次列车限速115千米每小时"},
            {"总价九千六百九十五万零五百零一元3角", "总价9695万501元3角"},
            {"预计运行十四小时", "预计运行14小时"},
            {"下半场补时四十分钟", "下半场补时40分钟"},
            {"本场比赛共有七万七千七百三十三名观众到场", "本场比赛共有77733名观众到场"},
            {"满分十四分", "满分14分"},
            {"本次考试得四十三分", "本次考试得43分"},
            {"录取分数线为七百二十八分", "录取分数线为728分"},
            {"本实验第四十六组数据中", "本实验第46组数据中"},
            {"这套房源位于第四十九层", "这套房源位于第49层"},
            {"第一五二七次列车即将发车", "第1527次列车即将发车"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v72 logistics, measurement, or score structure produced an unexpected result: "
                + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "有三十七安保人员", "平均值班时间很长", "总价可能九千元",
            "参加第四十六组朋友聚会",
            "原文写作“取件码为九三二七”",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v72 changed an unanchored or protected logistics or measurement expression: "
                + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"本次列车全程一千三百七十一米", "本次列车全程1371米"},
            {"全程约六百零五米", "全程约605米"},
            {"列车将于十一点五分从本站出发", "列车将于11点5分从本站出发"},
            {"这个月花了八千五百九十六万六千五百六十八元8角",
             "这个月花了8596万6568元8角"},
            {"昨天买了个七千八百五十一万七千三百五十一元整的包",
             "昨天买了个7851万7351元整的包"},
            {"帮我设一个四点三十五分的闹钟", "帮我设一个4点35分的闹钟"},
            {"把客厅空调调到十七度", "把客厅空调调到17度"},
            {"播放列表里第二十一首歌", "播放列表里第21首歌"},
            {"把手机音量调到二十五", "把手机音量调到25"},
            {"再定一个十九点四十五分的提醒", "再定一个19点45分的提醒"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v73 strong event or setting context produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "还差六百多就到目的地了", "这家店的东西大概九万多吧", "家里有四口人",
            "设置下午二十一点十分的日程", "设一个四点三十五分的目标",
            "参加第二十一首诗朗诵", "完成十七度大转弯", "音量大概二十五",
            "买了个一百元件样品",
        }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v73 changed an unanchored, vague, invalid, or lexical expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {
                "租金每月七千二百五十一万五千八百八十三元一缴，按季度支付",
                "租金每月7251万5883元一缴，按季度支付",
            },
            {"快递包裹已到达六十二号驿站", "快递包裹已到达62号驿站"},
            {"请到十二号窗口办理", "请到12号窗口办理"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v74 payment boundary or facility number produced an unexpected result: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "第二百一十五条", "按第四百七十七条约定", "本协议第九百九十条约定",
            "文件共有六十二号记录", "六十二号球员", "米二号楼",
            "一元一次方程", "原文写作“六十二号驿站”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v74 changed a legal, unanchored, lexical, or protected expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"每天上午八时至晚上十时", "每天上午8时至晚上10时"},
            {"中午十二点半休息", "中午12点半休息"},
            {"车机神经网络算力为五百一十二太拉弗洛普斯",
             "车机神经网络算力为512 TFLOPS"},
            {"AMD已抢先发布采用2纳米制成的EPYC 九零零六Venice服务器芯片与Instinct MI四五五XAI加速器。",
             "AMD已抢先发布采用2纳米制成的EPYC 9006 Venice服务器芯片与Instinct MI455X AI加速器。"},
            {"统一内存带宽为一点二太字节每秒", "统一内存带宽为1.2太字节每秒"},
            {"商品原价是两千三百九十九块九毛", "商品原价是2399.9元"},
            {"活动价是三百九十九块九毛", "活动价是399.9元"},
            {"订单总额为两百三十五块四毛", "订单总额为235.4元"},
            {"请把M6芯片装到B二号测试机上", "请把M6芯片装到B2号测试机上"},
            {"加载版本V四点零点八", "加载版本V4.0.8"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v75 strong time, specification, price, or identifier produced an unexpected result: "
                + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "总共运行了一小时四十分零三十秒", "我们从两点一刻忙到差五分三点",
            "不要拖到十二点以后", "算力大概五百一十二",
            "价格可能两千三百九十九块九毛", "B二号方案", "数值V四点零点八",
            "原文写作“上午八时至晚上十时”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v75 changed an anomalous, natural, vague, or protected expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"计数器从九千九百九十九跳到一万，又从九十九万九千九百九十九跳到一百万",
             "计数器从9999跳到10000，又从999999跳到1000000"},
            {"本轮疫情已持续一千七百七十天多", "本轮疫情已持续1770天多"},
            {"该楼盘位于第六十号楼", "该楼盘位于第60号楼"},
            {"主队暂列第三十三", "主队暂列第33"},
            {"全场射门将近六百零二次", "全场射门将近602次"},
            {"录取线将近六百一十二分", "录取线将近612分"},
            {"统计周期从二零二六年十二月三十一日二十三点开始，到二零二七年一月一日凌晨两点结束",
             "统计周期从2026年12月31日23点开始，到2027年1月1日凌晨2点结束"},
            {"改到十七点零五分", "改到17点05分"},
            {"视频采用H. 点二六四编码，无线网络标准是WiFi六E",
             "视频采用H.264编码，无线网络标准是WiFi 6E"},
            {"溶液pH值是七点四", "溶液pH值是7.4"},
            {"低于五个ppm，空气中含量二十个ppb", "低于5 ppm，空气中含量20 ppb"},
            {"盐水浓度为零点九克每百毫升", "盐水浓度为0.9克/100毫升"},
            {"当前汇率是一美元兑换七点二三元人民币",
             "当前汇率是1美元兑换7.23元人民币"},
            {"钢材报价为每吨三千八百五十元", "钢材报价为每吨3850元"},
            {"目标位置是东经一百二十一度二十八分二十五点三秒，北纬三十一度十三分四十九点四秒",
             "目标位置是东经121度28分25.3秒，北纬31度13分49.4秒"},
            {"公司本季度营收不足四千一百七十八万", "公司本季度营收不足4178万"},
            {"这款商品原价最多六千二百三十万四千九百八十元八角",
             "这款商品原价最多6230万4980元8角"},
            {"单人补贴金额四至九十二元", "单人补贴金额4至92元"},
            {"距地铁站四到八十公里", "距地铁站4到80公里"},
            {"下季度成交额九至四十亿元", "下季度成交额9至40亿元"},
            {"距您大约二千一百九十六米", "距您大约2196米"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v77 anchored cardinal, technical, or range structure produced an unexpected result: "
                + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "918事变纪念日", "五四运动纪念日", "一二九运动", "三一五晚会",
            "包裹大约上午二十三点送达", "本轮消费券发放总额达到七万六",
            "依据本合同七十八条", "绿化率七到九百分号",
            "车位配比大约四百五十", "计数器方案采用九千九百九十九",
            "该方案涉及第六十号记录", "射门大约六百零二种方案",
            "原文写作“WiFi六E”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v77 changed a fixed name, invalid time, truncation, or weak context: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"列车大约六点二十五分到站", "列车大约6点25分到站"},
            {"全程接近一千四百零三千米", "全程接近1403千米"},
            {"座位是十二车八A", "座位是12车8A"},
            {
                "收件地址是浙江省杭州市余杭区五常街道文一西路九六九号三号楼二单元",
                "收件地址是浙江省杭州市余杭区五常街道文一西路969号3号楼2单元",
            },
            {"周末常走将近二百六十七步", "周末常走将近267步"},
            {"帮我设个十二点一刻的闹钟", "帮我设个12点一刻的闹钟"},
            {"明天二十二点二十二分提醒我开会", "明天22点22分提醒我开会"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v78 strongly anchored travel, address, count, or clock structure failed: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "六点二十五分的长度", "一刻也不能耽误", "十二车货物",
            "五常街道文一西路", "三号方案二单元测试", "将近二百六十七种做法",
            "原文写作“文一西路九六九号”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v78 changed an unanchored clock, identifier, address, or count expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"假期期间十四时至二十二时实施交通管控",
             "假期期间14时至22时实施交通管控"},
            {"每日八时至二十二时", "每日8时至22时"},
            {"二零二六年四月三十日十二时至五月一日二十二时",
             "2026年4月30日12时至5月1日22时"},
            {"二零二六年五月一日九时至二十二时",
             "2026年5月1日9时至22时"},
            {"游客可乘坐地铁一号线、三号线、五号线",
             "游客可乘坐地铁1号线、3号线、5号线"},
            {"乘坐G三公交接驳专线", "乘坐G3公交接驳专线"},
            {"向东二百四十米三角掉头处", "向东240米三角掉头处"},
        }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v79 composite traffic time, route, or distance structure failed: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "十四时至二十二时的长度", "地铁一号方案", "一号线索",
            "G三项目组", "向东二百四十种方案", "七珍线乙山抚江台",
            "原文写作“每日八时至二十二时”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v79 changed an unanchored time, route, distance, or protected expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"当前时间是十五点三十分", "当前时间是15点30分"},
            {"视频时间码是零一小时二十三分四十五秒", "视频时间码是01:23:45"},
            {"药物剂量是二百五十微克", "药物剂量是250微克"},
            {"视频采用H点二六四编码", "视频采用H.264编码"},
            {"项目使用C加加十七、Python三点十二和CUDA十三点零",
             "项目使用C++17、Python 3.12和CUDA 13.0"},
            {"空腹血糖必须小于七点零", "空腹血糖必须小于7.0"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v81 timecode, technical notation, or clinical threshold failed: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "时间很宝贵，十五点三十分只是一个数字", "耗时零一小时二十三分四十五秒",
            "C加加十八", "Python三点十二个人",
            "数值必须小于七点零", "原文写作“视频时间码是零一小时二十三分四十五秒”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v81 changed an unanchored, unsupported, or protected expression: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"有效期为三十七年，期满前大概二千零四十五天可申请续签。",
             "有效期为37年，期满前大概2045天可申请续签。"},
            {"有效期限为三十年。", "有效期限为30年。"},
            {"到期前约一百八十天可以申请续期。", "到期前约180天可以申请续期。"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v82 anchored validity or renewal duration failed: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "这位有着二十年种植经验的老药农", "一年种山药，十年无地利",
            "有效期为三年", "大概二千零四十五天后完成",
            "原文写作“有效期为三十七年”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v82 changed a natural, unsupported, or protected duration: " + text
        );
    }

    for (const auto& [input, expected] : std::vector<std::pair<std::string, std::string>>{
            {"光的速度是每秒三十万公里，而声音每秒只能跑大约三百四十米。",
             "光的速度是每秒30万公里，而声音每秒只能跑大约340米。"},
            {"物体每秒移动约二十米。", "物体每秒移动约20米。"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(
            result.valid && result.normalized.output == expected,
            "v83 approximate rate distance failed: " + input
        );
    }
    for (const auto& text : std::vector<std::string>{
            "这个说法大约三百四十米", "他每秒只能跑大约三米",
            "原文写作“声音每秒只能跑大约三百四十米”",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(
            result.valid && result.normalized.output == text,
            "v83 changed an unanchored, single-digit, or protected distance: " + text
        );
    }

    for (const auto& [text, expected] : std::vector<std::pair<std::string, std::string>>{
            {"请在一点十分设置闹钟。", "请在1点10分设置闹钟。"},
            {"这一点十分明确，闹钟设在一点十分。",
             "这一点十分明确，闹钟设在1点10分。"},
            {"任务耗时十二分钟。", "任务耗时12分钟。"},
            {"现场有三十多人。", "现场有30多人。"},
            {"现场有一百多人。", "现场有100多人。"},
            {"路程大约十到二十公里。", "路程大约10到20公里。"},
            {"从二零二五年六月到二零二六年七月。",
             "从2025年6月到2026年7月。"},
            {"二零二四年五月三日一点十分开会，出席十二人。",
             "2024年5月3日1点10分开会，出席12人。"},
            {"计算结果是负三点二。", "计算结果是-3.2。"},
            {"测量结果为负十二。", "测量结果为-12。"},
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "v85 request failed: " + text);
        passed &= expect(
            result.normalized.output == expected,
            "v85 explicit alarm, duration, person count, or cross-year month range failed: "
                + text + " -> " + result.normalized.output
        );
        passed &= expect(
            !result.normalized.applied.empty(),
            "v85 approval trace missing: " + text
        );
    }
    for (const auto& text : std::vector<std::string>{
            "这一点十分重要。",
            "附近大概有三十多人。",
            "从二零二五年六月到二零二六年十三月。",
            "大约十到十五分钟",
            "这个结果可能是负三点二。",
            "计算结果是“负三点二”。",
            "负荆请罪。",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid, "v85 preservation request failed: " + text);
        passed &= expect(
            result.normalized.output == text,
            "v85 changed an ambiguous, weakly anchored, invalid, or approximate range: "
                + text + " -> " + result.normalized.output
        );
    }

    const std::vector<std::pair<std::string, std::string>> structural_quantity_cases = {
        {"洪水以每秒超过六万立方米的流量涌来。",
         "洪水以每秒超过6万立方米的流量涌来。"},
        {"最大库容有四百五十亿立方米。", "最大库容有450亿立方米。"},
        {"覆盖全流域的三万多个监测站点。", "覆盖全流域的3万多个监测站点。"},
        {"足以装下三千多个湖泊。", "足以装下3000多个湖泊。"},
        {"水位四十五点二二米。", "水位45.22米。"},
        {"这里有四点五万条江河。", "这里有4.5万条江河。"},
        {"蓄水能力相当于二点六个水库。", "蓄水能力相当于2.6个水库。"},
        {"让一点七亿人口用上了水。", "让1.7亿人口用上了水。"},
        {"人们将大堤加高了一到两米。", "人们将大堤加高了1到2米。"},
        {"需要三到四个人。", "需要3到4个人。"},
    };
    for (const auto& [input, expected] : structural_quantity_cases) {
        const auto result = normalize_zh_conservative_v2(request(input));
        passed &= expect(result.valid && result.normalized.output == expected,
            "structural quantity failed: " + input + " -> " + result.normalized.output
                + " (" + result.decision_reason + ")");
    }
    for (const auto& text : std::vector<std::string>{
            "这里有一个方案。", "他三番五次提醒我们。",
            "一首首歌传遍大街小巷。", "万一有变化就通知我。",
    }) {
        const auto result = normalize_zh_conservative_v2(request(text));
        passed &= expect(result.valid && result.normalized.output == text,
            "structural quantity changed non-quantity: " + text + " -> "
                + result.normalized.output);
    }

    auto wrong_locale = request("二零二六年");
    wrong_locale.locale = "en-US";
    const auto locale_result = normalize_zh_conservative_v2(wrong_locale);
    passed &= expect(!locale_result.valid, "unsupported locale was accepted");
    passed &= expect(
        locale_result.normalized.output == wrong_locale.text,
        "unsupported locale did not preserve input"
    );

    return passed ? 0 : 1;
}
