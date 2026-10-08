// Common numeric syntax, included inside the conservative rule namespace.
// Quantity recognition depends on the number and unit, not a preceding verb.
constexpr const char* kCommonNumericRule = "zh.number.complete_numeric_object";
constexpr const char* kCommonNumericRangeRule = "zh.number.complete_numeric_range";
constexpr const char* kCommonNumericSequenceRule = "zh.number.complete_numeric_sequence";
constexpr const char* kCommonNumericRuleVersion = "1";

enum class CommonUnitKind { count, duration, measure, money, magnitude, score, index };
struct CommonNumericUnit {
    std::string text;
    CommonUnitKind kind;
    std::string dimension;
    size_t width;
};

const std::vector<CommonNumericUnit>& common_numeric_units() {
    static const auto units = [] {
        std::vector<CommonNumericUnit> result;
        auto add = [&](CommonUnitKind kind, const std::string& dimension,
                       std::initializer_list<const char*> names) {
            for (const char* name : names) {
                const std::string text(name);
                result.push_back({text, kind, dimension.empty() ? text : dimension,
                    unicode_scalar_boundaries(text)->size() - 1});
            }
        };
        add(CommonUnitKind::count, "", {"个", "个人", "条", "座", "种", "位", "名", "只", "本", "台", "首", "张", "家", "次", "级", "根", "份", "册", "双", "支", "杯", "件", "辆", "颗", "箱", "批", "处", "人", "瓶", "间", "艘", "套", "头", "匹", "门", "笔", "碗", "袋", "包", "桶", "把", "段", "项", "场", "页", "排", "层", "组", "章", "节", "集", "部", "所", "棵", "株", "枚", "封", "幅", "束", "届", "期", "回", "题"});
        add(CommonUnitKind::duration, "time", {"年", "个月", "月", "周", "星期", "个星期", "天", "日", "夜", "小时", "个小时", "分钟", "分", "秒钟", "秒", "毫秒"});
        add(CommonUnitKind::measure, "age", {"岁"});
        add(CommonUnitKind::measure, "score", {"环"});
        add(CommonUnitKind::measure, "length", {"公里", "千米", "米", "厘米", "毫米", "微米", "纳米"});
        add(CommonUnitKind::measure, "area", {"平方公里", "平方米", "平方厘米", "公顷"});
        add(CommonUnitKind::measure, "volume", {"立方米", "立方厘米", "升", "毫升"});
        add(CommonUnitKind::measure, "mass", {"吨", "公斤", "千克", "克", "毫克", "微克", "斤"});
        add(CommonUnitKind::measure, "temperature", {"摄氏度", "华氏度"});
        add(CommonUnitKind::measure, "voltage", {"伏特", "伏", "千伏"});
        add(CommonUnitKind::measure, "current", {"安培", "毫安"});
        add(CommonUnitKind::measure, "power", {"瓦特", "瓦", "千瓦", "兆瓦"});
        add(CommonUnitKind::measure, "frequency", {"赫兹", "千赫兹", "兆赫兹", "吉赫兹"});
        add(CommonUnitKind::measure, "", {"欧姆", "毫安时", "兆帕", "百帕", "分贝", "毫米汞柱", "毫摩尔每升", "克每百毫升", "米每二次方秒", "公里每小时", "千米每小时", "米每秒", "帧每秒", "兆比特每秒", "千比特每秒", "太字节每秒", "太拉弗洛普斯", "牛顿米", "GB", "TB", "吉字节", "兆字节", "太字节", "吉比特", "比特", "字节", "个百分点", "百分点", "个基点", "基点", "个ppm", "个ppb", "人次", "个代表团", "名观众", "只私募产品", "核CPU", "核GPU", "核神经引擎", "万人", "亿人", "万股", "转", "帧", "步", "倍"});
        add(CommonUnitKind::money, "CNY", {"元", "元钱", "块钱", "人民币", "元人民币", "角", "毛", "分", "分人民币"});
        add(CommonUnitKind::money, "USD", {"美元"});
        add(CommonUnitKind::money, "HKD", {"港元"});
        add(CommonUnitKind::money, "JPY", {"日元"});
        add(CommonUnitKind::money, "EUR", {"欧元"});
        add(CommonUnitKind::count, "", {"块"});
        add(CommonUnitKind::magnitude, "magnitude", {"万", "亿", "个万", "个亿"});
        add(CommonUnitKind::score, "score", {"分"});
        add(CommonUnitKind::index, "index", {"号"});
        std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.width > b.width;
        });
        return result;
    }();
    return units;
}

bool standard_literal_delimiter(const std::string& input,
    const std::vector<size_t>& boundaries, size_t position, const std::string& opener) {
    if (opener == "《" || opener == "〈") return true;
    size_t left_end=position;
    while (left_end && contains_any(" ：:\t", {unicode_scalar_substring(input,boundaries,left_end-1,left_end)})) --left_end;
    const std::string left = left_clause_context(input, boundaries, left_end, 24);
    return contains_any(left, {"原文", "字面", "书名", "标题", "命名", "名称", "名字", "写作", "写成", "念作", "拼写", "引号里", "引号内"});
}

bool standard_numeric_lexical_guard(const std::string& input,
    const std::vector<size_t>& boundaries, size_t position) {
    // Shared lexical expressions already protected by the Python normalizer.
    static const std::vector<std::string> idioms = {
        "一五一十", "一心一意", "三心二意", "不三不四", "乱七八糟", "七上八下", "五湖四海", "四面八方", "九牛一毛", "十全十美", "十拿九稳", "十万火急", "百里挑一", "千方百计", "万无一失", "一举两得", "三番五次", "一清二楚", "说一不二", "十有八九", "七七八八", "三三两两", "五花八门", "五颜六色", "千言万语", "千军万马", "千变万化", "千山万水", "万水千山", "九死一生", "八九不离十", "四通八达", "四平八稳", "两全其美", "两肋插刀", "一针一线", "一朝一夕", "一刀两断", "一穷二白", "一来二去", "一干二净", "一百年大计", "一百年好合", "一千年一遇", "百年大计", "百年好合", "千年一遇", "双十一", "双十二"
    };
    const size_t byte = boundaries[position];
    // These are lexical words, not a count of steps or revolutions. This guard
    // applies to every quantity detector, including the older measurement rules.
    for (const std::string term : {"进一步", "一转眼"}) {
        size_t at = 0;
        while ((at = input.find(term, at)) != std::string::npos) {
            if (at <= byte && byte < at + term.size()) return true;
            at += term.size();
        }
    }
    const auto left=left_clause_context(input,boundaries,position,24);
    if (input.compare(byte,std::string("零件").size(),"零件")==0
        && !ends_with_any(left,{"数量是","数量为","件数是","件数为"})) return true;
    if (position && contains_any("幺洞拐",{unicode_scalar_substring(input,boundaries,position-1,position)})) return true;
    size_t numeric_end=position;
    while (numeric_end<boundaries.size()-1 && number_continuation(unicode_scalar_substring(input,boundaries,numeric_end,numeric_end+1))) ++numeric_end;
    const auto following=numeric_end<boundaries.size()-1 ? unicode_scalar_substring(input,boundaries,numeric_end,numeric_end+1) : "";
    // Deictic/indefinite dates (that day, someday) do not express a duration.
    // Actual durations and ranges keep using the complete numeric grammar.
    if (numeric_end == position + 1
        && unicode_scalar_substring(input,boundaries,position,numeric_end)=="一"
        && (following=="天" || following=="年" || following=="日")
        && ends_with_any(left,{"这","那","头","有","某","的"})) return true;
    if (following=="年" && numeric_end-position==4 && !ends_with_any(left,{"公元","西元"})
        && spoken_year_digit(unicode_scalar_substring(input,boundaries,position+1,position+2)).has_value()
        && spoken_year_digit(unicode_scalar_substring(input,boundaries,position+3,position+4)).has_value()
        && unicode_scalar_substring(input,boundaries,position,position+1)=="一"
        && unicode_scalar_substring(input,boundaries,position+2,position+3)=="一") return true;
    if (following=="元") {
        const auto after=right_clause_context(input,boundaries,numeric_end+1,16);
        const bool money_context=explicit_integer_money_context(input,boundaries,position,numeric_end+1,"元");
        const auto next=after.empty() ? "" : after.substr(0,unicode_scalar_boundaries(after)->at(1));
        const bool boundary=after.empty() || clause_boundary_scalar(next) || contains_any("、,）)]】\"”'",{next});
        if (!money_context && !boundary
            && !starts_with_any(after,{"钱","人民币","整","每","左右","上下","以内","以上","以下","及","或","和","与","起","呢","呀","吧","的"})) return true;
    }
    if (literal_scope_before_year(input,boundaries,position)
        && (following=="年" || contains_any(left,{"原文","原句","原样","文件名","文件夹名","目录名","书名","作品名","写的是","写作","搜索词","关键词","名称","名字","昵称","网名","名为","叫作","叫做"}))) return true;
    if (ends_with_any(left,{"品牌是","品牌为","品牌叫","公司名称是","公司名称为","名字是","名称是","标题是"})) return true;
    const bool explicit_count=ends_with_any(left,{"有","共","共有","总计","包含","包括","记录","发现"});
    if (!explicit_count) {
        for (const auto& unit:common_numeric_units()) {
            if (unit.kind!=CommonUnitKind::count || numeric_end+unit.width>boundaries.size()-1
                || input.compare(boundaries[numeric_end],unit.text.size(),unit.text)!=0) continue;
            const auto after=right_clause_context(input,boundaries,numeric_end+unit.width,8);
            if (starts_with_any(after,{"资产","科技","集团","公司","品牌"})) return true;
        }
    }
    if (position + 6 <= boundaries.size()-1
        && spoken_year_digit(unicode_scalar_substring(input,boundaries,position,position+1)).has_value()
        && *spoken_year_digit(unicode_scalar_substring(input,boundaries,position,position+1))>=3
        && !ends_with_any(left,{"公元","西元"})
        && unicode_scalar_substring(input,boundaries,position+1,position+2)=="零"
        && spoken_year_digit(unicode_scalar_substring(input,boundaries,position+2,position+3)).has_value()
        && unicode_scalar_substring(input,boundaries,position+3,position+6)=="零年代") return true;
    for (const auto& term : idioms) {
        size_t at = 0;
        while ((at = input.find(term, at)) != std::string::npos) {
            if (at <= byte && byte < at + term.size()) return true;
            at += term.size();
        }
    }
    if (input.compare(byte, std::string("四处").size(), "四处") == 0) {
        const auto left = left_clause_context(input, boundaries, position, 12);
        return !ends_with_any(left, {"有", "共", "总计", "发现", "查出", "存在"});
    }
    return false;
}

struct CommonNumber {
    std::string rendered;
    std::string expanded;
    int64_t integer;
    bool decimal;
};

bool common_number_scalar(const std::string& value) {
    return integer_measure_number_scalar(value) || value == "点" || value == "."
        || (value.size() == 1 && std::isdigit(static_cast<unsigned char>(value[0])));
}

std::optional<CommonNumber> common_number(const std::string& input,
    const std::vector<size_t>& boundaries, size_t start, size_t end,
    bool digit_sequence = false) {
    if (start >= end) return std::nullopt;
    size_t dot = end;
    for (size_t i = start; i < end; ++i) {
        const auto ch = unicode_scalar_substring(input, boundaries, i, i + 1);
        if (ch == "点" || ch == ".") {
            if (dot != end) return std::nullopt;
            dot = i;
        }
    }
    if (dot == start || (dot != end && dot + 1 == end)) return std::nullopt;
    bool ascii = true;
    for (size_t i = start; i < dot; ++i) {
        const auto ch = unicode_scalar_substring(input, boundaries, i, i + 1);
        if (ch.size() != 1 || !std::isdigit(static_cast<unsigned char>(ch[0]))) ascii = false;
    }
    int64_t integer = 0;
    std::string rendered;
    if (ascii) {
        const auto token = unicode_scalar_substring(input, boundaries, start, dot);
        try { integer = std::stoll(token); } catch (...) { return std::nullopt; }
        rendered = token;
    } else {
        bool all_digits = true;
        for (size_t i = start; i < dot; ++i) {
            if (!spoken_year_digit(unicode_scalar_substring(input, boundaries, i, i + 1)).has_value()) all_digits = false;
        }
        if (all_digits && dot - start > 1) {
            if (!digit_sequence || dot - start > 4) return std::nullopt;
            for (size_t i = start; i < dot; ++i) integer = integer * 10
                + *spoken_year_digit(unicode_scalar_substring(input, boundaries, i, i + 1));
            rendered = std::to_string(integer);
        } else {
            const auto value = parse_spoken_large_count_integer(input, boundaries, start, dot);
            const auto format = format_spoken_large_count_integer(input, boundaries, start, dot);
            if (!value.has_value() || !format.has_value()) return std::nullopt;
            integer = *value;
            rendered = *format;
        }
    }
    std::string expanded = std::to_string(integer);
    if (dot != end) {
        std::string fraction;
        for (size_t i = dot + 1; i < end; ++i) {
            const auto ch = unicode_scalar_substring(input, boundaries, i, i + 1);
            if (ch.size() == 1 && std::isdigit(static_cast<unsigned char>(ch[0]))) fraction += ch;
            else {
                const auto digit = spoken_year_digit(ch);
                if (!digit.has_value()) return std::nullopt;
                fraction += static_cast<char>('0' + *digit);
            }
        }
        rendered = expanded + "." + fraction;
        expanded = rendered;
    }
    return CommonNumber{rendered, expanded, integer, dot != end};
}

bool common_lexical_suffix(const CommonNumericUnit& unit, const std::string& right) {
    if (unit.text == "个") return starts_with_any(right, {"月", "小时", "星期", "百分点", "基点", "基本点", "ppm", "ppb"});
    if (unit.text == "人") return starts_with_any(right, {"工", "民币", "帮"});
    if (unit.text == "条") return starts_with_any(right, {"件", "例", "款", "理", "约", "码", "纹"});
    if (unit.text == "种") return starts_with_any(right, {"子", "植", "姓"});
    if (unit.text == "位") return starts_with_any(right, {"置", "移", "于"});
    if (unit.text == "名") return starts_with_any(right, {"字", "称", "义"});
    if (unit.text == "次") return starts_with_any(right, {"序", "数", "方", "元", "世界大战"});
    if (unit.text == "家") return starts_with_any(right, {"庭", "长"});
    if (unit.text == "台") return starts_with_any(right, {"风", "阶"});
    if (unit.text == "首") return starts_with_any(right, {"先", "都"});
    if (unit.text == "只") return starts_with_any(right, {"字节"});
    if (unit.text == "股") return starts_with_any(right, {"东","份","票","权","价","市"});
    if (unit.text == "号") return starts_with_any(right, {"召", "称", "啕"});
    if (unit.text == "元") return starts_with_any(right, {"旦", "年", "件", "素", "音", "组", "论", "化", "函数", "方程", "关系", "结构", "系统", "模型", "分析", "运算", "数据", "宇宙", "单位", "编码", "裤业"});
    if (unit.text == "角") return starts_with_any(right, {"形", "函数"});
    if (unit.text == "米") return starts_with_any(right, {"饭", "粉", "酒", "线"});
    return false;
}

struct CommonObject {
    size_t start, number_start, number_end, end;
    std::string rendered;
    const CommonNumericUnit* unit;
};
std::optional<CommonObject> common_range(const std::string&, const std::vector<size_t>&, size_t);

std::optional<CommonObject> common_object(const std::string& input,
    const std::vector<size_t>& b, size_t start, bool endpoint = false) {
    const size_t length = b.size() - 1;
    auto scalar = [&](size_t i) { return i < length ? unicode_scalar_substring(input, b, i, i + 1) : ""; };
    if (start >= length || inside_protected_delimiters(input, b, start)) return std::nullopt;
    const bool ordinal = scalar(start) == "第";
    std::string prefix = ordinal ? "第" : "";
    size_t number_start = start + (ordinal ? 1 : 0);
    if (!ordinal && (scalar(start) == "负" || scalar(start) == "-")) { prefix = "-"; ++number_start; }
    if (!ordinal && start + 2 < length && (unicode_scalar_substring(input,b,start,start+2)=="零下"
        || unicode_scalar_substring(input,b,start,start+2)=="零上")) {
        prefix=unicode_scalar_substring(input,b,start,start+2); number_start=start+2;
    }
    if (number_start >= length || !common_number_scalar(scalar(number_start))) return std::nullopt;
    if (start && (number_continuation(scalar(start - 1)) || contains_any("点.．数几第", {scalar(start - 1)})
        || contains_any("幺洞拐",{scalar(start-1)})
        || (scalar(start - 1).size() == 1 && std::isalnum(static_cast<unsigned char>(scalar(start - 1)[0]))))) return std::nullopt;
    if (!endpoint && start>=2 && range_connector(scalar(start-1)) && number_continuation(scalar(start-2))) return std::nullopt;
    const auto left=left_clause_context(input,b,start,24);
    const bool duration_anchor=contains_any(left,{"耗时","用时","用了","持续","历时"});
    if (start && scalar(start-1)=="年") {
        size_t previous=start-1;
        while (previous && spoken_year_digit(scalar(previous-1)).has_value()) --previous;
        if (start-1-previous>1 && start-1-previous<4) return std::nullopt;
    }
    if (start && scalar(start-1)=="分" && contains_any(left,{"点"}) && !duration_anchor) return std::nullopt;
    size_t maximum = number_start;
    while (maximum < length && common_number_scalar(scalar(maximum))) ++maximum;
    std::optional<CommonObject> best;
    size_t best_unit_width = 0;
    for (size_t number_end = number_start + 1; number_end <= maximum; ++number_end) {
        size_t unit_start = number_end;
        std::string modifier;
        if (contains_any("多余来", {scalar(unit_start)}) && !scalar(unit_start).empty()) modifier = scalar(unit_start++);
        std::string infix_magnitude;
        if (!modifier.empty() && (scalar(unit_start) == "万" || scalar(unit_start) == "亿")) infix_magnitude = scalar(unit_start++);
        for (const auto& unit : common_numeric_units()) {
            if (unit_start + unit.width > length || input.compare(b[unit_start], unit.text.size(), unit.text) != 0) continue;
            const size_t end = unit_start + unit.width;
            auto number = common_number(input, b, number_start, number_end,
                unit.text == "年" || (unit.kind==CommonUnitKind::duration && scalar(number_start)=="零"));
            if (!number.has_value()) continue;
            const auto right = right_clause_context(input, b, end, 12);
            const auto next = scalar(end);
            if (common_lexical_suffix(unit, right) || next == unit.text) continue;
            if (unit.kind==CommonUnitKind::index && start && scalar(start-1)=="米") continue;
            if (unit.kind==CommonUnitKind::count && starts_with_any(unit.text,{"只"}) && starts_with_any(right,{"字节"})) continue;
            const auto number_text=unicode_scalar_substring(input,b,number_start,number_end);
            if (starts_with_any(unit.text,{"万","亿"}) && contains_any(number_text,{"万","亿"})) continue;
            if (unit.text=="年") {
                bool digit_run=true;
                for (size_t i=number_start;i<number_end;++i) if (!spoken_year_digit(scalar(i)).has_value()) digit_run=false;
                const bool explicit_era=ends_with_any(left,{"公元","西元"});
                if (digit_run && number_end-number_start>1 && number_end-number_start<4 && !explicit_era) continue;
                if (digit_run && number_end-number_start==4 && scalar(number_start)=="一" && scalar(number_start+2)=="一" && !explicit_era) continue;
            }
            if (unit.kind==CommonUnitKind::count && (next=="乘" || (start && scalar(start-1)=="乘"))) continue;
            if (unit.text == "年" && starts_with_any(right, {"代"})) {
                const auto token = unicode_scalar_substring(input, b, number_start, number_end);
                if (number->integer > 9999 || number->integer%10!=0 || (number_end - number_start == 4
                    && *spoken_year_digit(scalar(number_start))>=3
                    && scalar(number_start + 1) == "零" && scalar(number_start + 3) == "零")) continue;
            }
            if (unit.text=="月" && !right.empty() && !clause_boundary_scalar(next) && !number_continuation(next)
                && !starts_with_any(right,{"份","底","初","上旬","中旬","下旬","前","后","以来","期间"})
                && !ends_with_any(left,{"在","于","今年","明年","去年","年"})) continue;
            bool adjacent_duration=false;
            if (number_continuation(next) && (unit.kind==CommonUnitKind::duration || unit.kind==CommonUnitKind::money || unit.kind==CommonUnitKind::measure)) {
                const auto following=common_object(input,b,end,endpoint);
                adjacent_duration=following.has_value() && following->unit
                    && following->unit->kind==unit.kind && following->unit->dimension==unit.dimension;
                if (!adjacent_duration && unit.text=="年") {
                    const auto months=common_range(input,b,end);
                    adjacent_duration=months.has_value() && months->unit && months->unit->text=="月";
                }
            }
            if (number_continuation(next) && !adjacent_duration && !(unit.kind == CommonUnitKind::count
                && classifier_followed_by_numbered_noun(input, b, end))) continue;
            if (!endpoint && range_connector(next)) continue;
            if (!ordinal && !endpoint && unit.kind == CommonUnitKind::count && number->integer == 1
                && !number->decimal && modifier.empty() && infix_magnitude.empty()) continue;
            if (unit.text == "月" && (number->integer < 1 || number->integer > 12 || number->decimal)) continue;
            if (unit.text == "分") {
                const bool money=contains_any(left,{"元","钱","价格","花了","费用","金额"});
                const bool score=contains_any(left,{"得分","总分","成绩","满分","考了","考得","评分"});
                if (unit.kind==CommonUnitKind::money && !money) continue;
                if (unit.kind==CommonUnitKind::score && !score) continue;
                if (unit.kind==CommonUnitKind::duration && !duration_anchor && !adjacent_duration) continue;
                if (unit.kind==CommonUnitKind::duration && number->decimal && !duration_anchor) continue;
            }
            if ((unit.text == "角" || unit.text == "毛") && !contains_any(left_clause_context(input, b, start, 16), {"钱", "元", "价格", "花了", "费用", "金额"})) continue;
            if (!ordinal && unit.kind == CommonUnitKind::count && start >= 2
                && structural_classifier_unit(scalar(start - 1)) && common_number_scalar(scalar(start - 2))) continue;
            std::string output = prefix + number->rendered + modifier + infix_magnitude + measure_output_unit(unit.text);
            CommonObject object{start, number_start, number_end, end, output, &unit};
            if (!best.has_value() || end > best->end || (end == best->end && unit.width > best_unit_width)) {
                best = object;
                best_unit_width = unit.width;
            }
        }
        // Colloquial major/minor currency syntax: N块M, N元M角, N块零M分.
        if (scalar(number_end) == "块") {
            const auto major = common_number(input, b, number_start, number_end);
            size_t cursor = number_end + 1;
            std::string fraction;
            while (cursor < length && fraction.size() < 2) {
                auto digit = spoken_year_digit(scalar(cursor));
                if (!digit.has_value()) break;
                fraction += static_cast<char>('0' + *digit); ++cursor;
            }
            const bool cents = scalar(cursor) == "分";
            const bool tenth = scalar(cursor) == "角" || scalar(cursor) == "毛";
            if ((cents || tenth) && !fraction.empty()) ++cursor;
            if (tenth && fraction.size()==1 && spoken_year_digit(scalar(cursor)).has_value()) {
                fraction+=static_cast<char>('0'+*spoken_year_digit(scalar(cursor++)));
                if (scalar(cursor)=="分") ++cursor;
            }
            const bool boundary = cursor == length || clause_boundary_scalar(scalar(cursor)) || contains_any("、,）)\"”", {scalar(cursor)});
            if (major.has_value() && !major->decimal && !fraction.empty()
                && boundary && ((fraction.size() == 1 && (tenth || boundary)) || ((cents || tenth) && fraction.size() <= 2))) {
                if (cents && fraction.size() == 1) fraction = "0" + fraction;
                const auto output = major->expanded + "." + fraction + "元";
                if (!best.has_value() || cursor > best->end) best = CommonObject{start, number_start, number_end, cursor, output, nullptr};
            }
        }
    }
    return best;
}

bool common_scalar_anchor(const std::string& left) {
    return ends_with_any(left, {"计算结果是", "计算结果为", "测量结果是", "测量结果为", "结果是", "结果为", "数值是", "数值为", "读数是", "读数为", "数量是", "数量为", "总数是", "总数为", "音量设为", "音量调到", "总分是", "总分为"});
}

std::optional<CommonObject> common_scalar(const std::string& input,
    const std::vector<size_t>& b, size_t start) {
    const size_t length = b.size() - 1;
    if (inside_protected_delimiters(input, b, start)) return std::nullopt;
    if (start && (number_continuation(unicode_scalar_substring(input,b,start-1,start))
        || unicode_scalar_substring(input,b,start-1,start)=="点")) return std::nullopt;
    const auto left = left_clause_context(input,b,start,24);
    if (ends_with_any(left,{"加加"})) return std::nullopt;
    const bool percent = start + 3 <= length && unicode_scalar_substring(input,b,start,start+3)=="百分之";
    const bool anchored=common_scalar_anchor(left);
    const size_t number_start = start + (percent ? 3 : 0);
    size_t end = number_start;
    while (end < length && common_number_scalar(unicode_scalar_substring(input,b,end,end+1))) ++end;
    const auto number = common_number(input,b,number_start,end);
    if (!number.has_value()) return std::nullopt;
    if (!percent && !anchored) {
        const auto token=unicode_scalar_substring(input,b,number_start,end);
        if (!contains_any(token,{"十","百","千","万","亿","点"})
            || ends_with_any(left,{"上午","下午","晚上","凌晨","时间是","时间为"})) return std::nullopt;
    }
    if (end < length) {
        const auto next = unicode_scalar_substring(input,b,end,end+1);
        if (number_continuation(next) || next=="点" || range_connector(next)) return std::nullopt;
        const bool numeric_operation=contains_any("加减乘除比",{next}) && end+1<length
            && common_number_scalar(unicode_scalar_substring(input,b,end+1,end+2));
        if (!percent && !numeric_operation && !clause_boundary_scalar(next) && !contains_any("、,）)\"”啊呀吧呢",{next})) return std::nullopt;
        if (percent && contains_any("多几",{next})) return std::nullopt;
        if (percent && ((next.size()==1 && std::isalnum(static_cast<unsigned char>(next[0])))
            || starts_with_any(right_clause_context(input,b,end,12),{"个百分点","百分点"}))) return std::nullopt;
    }
    if (start>=2 && range_connector(unicode_scalar_substring(input,b,start-1,start))
        && number_continuation(unicode_scalar_substring(input,b,start-2,start-1))) return std::nullopt;
    return CommonObject{start,number_start,end,end,percent ? number->expanded+"%" : number->rendered,nullptr};
}

std::optional<CommonObject> common_range(const std::string& input,
    const std::vector<size_t>& b, size_t start) {
    const size_t length = b.size()-1;
    if (start>=2 && range_connector(unicode_scalar_substring(input,b,start-1,start))
        && number_continuation(unicode_scalar_substring(input,b,start-2,start-1))) return std::nullopt;
    auto first = common_object(input,b,start,true);
    size_t bridge_start;
    std::string first_output;
    if (first.has_value()) { bridge_start=first->end; first_output=first->rendered; }
    else {
        if (inside_protected_delimiters(input,b,start) || (start && number_continuation(unicode_scalar_substring(input,b,start-1,start)))) return std::nullopt;
        bridge_start=start;
        while (bridge_start<length && common_number_scalar(unicode_scalar_substring(input,b,bridge_start,bridge_start+1))) ++bridge_start;
        const auto bare = common_number(input,b,start,bridge_start);
        if (!bare.has_value()) return std::nullopt;
        first_output=bare->rendered;
    }
    for (size_t second_start=bridge_start+1; second_start<=length && second_start<=bridge_start+12; ++second_start) {
        const auto bridge=unicode_scalar_substring(input,b,bridge_start,second_start);
        if (!(bridge.size() && structural_quantity_bridge(bridge))) continue;
        auto second=common_object(input,b,second_start,true);
        if (!second.has_value() || second->unit==nullptr) continue;
        if (first.has_value() && (first->unit==nullptr || first->unit->dimension!=second->unit->dimension)) continue;
        return CommonObject{start,start,bridge_start,second->end,first_output+bridge+second->rendered,second->unit};
    }
    return std::nullopt;
}

std::optional<CommonObject> common_sequence(const std::string& input, const std::vector<size_t>& b, size_t start) {
    auto first=common_object(input,b,start);
    if (!first.has_value() || !first->unit || (first->unit->kind!=CommonUnitKind::duration
        && first->unit->kind!=CommonUnitKind::measure && first->unit->kind!=CommonUnitKind::money)) return std::nullopt;
    auto result=*first;
    size_t count=1;
    while (result.end<b.size()-1) {
        auto next=common_object(input,b,result.end);
        if (!next.has_value() || !next->unit || next->unit->kind!=first->unit->kind
            || next->unit->dimension!=first->unit->dimension || next->unit->text=="月" || next->unit->text=="日") break;
        result.rendered+=next->rendered;
        result.end=next->end;
        ++count;
    }
    return count>1 ? std::optional<CommonObject>(result) : std::nullopt;
}

RuleApproval approve_common_numeric(const std::string& input, const SafePolicyContext& context,
    const TransformationCandidate& candidate) {
    auto b=unicode_scalar_boundaries(input);
    if (!b.has_value()) return {RuleDecision::error,"",{},{},"invalid_utf8"};
    const auto& span=candidate.semantic_source;
    auto parsed=candidate.rule_id==kCommonNumericSequenceRule ? common_sequence(input,*b,span.start)
        : candidate.rule_id==kCommonNumericRangeRule ? common_range(input,*b,span.start)
        : candidate.subtype=="common_scalar" ? common_scalar(input,*b,span.start)
        : common_object(input,*b,span.start);
    if (!parsed.has_value() || parsed->end!=span.end
        || unicode_scalar_substring(input,*b,span.start,span.end)!=span.text
        || (span.start==0 && context.has_left_neighbor)
        || (span.end==b->size()-1 && context.has_right_neighbor)
        || incomplete_cross_segment_span(context,b->size()-1,span.start,span.end))
        return {RuleDecision::preserve,"",{},{},"not_complete_numeric_object"};
    return {RuleDecision::approve,parsed->rendered,
        {{"shape","complete_number_and_numeric_syntax",candidate.rule_id},
         {"boundary","complete_numeric_object",candidate.rule_id}},
        {{{span.start,span.end,span.text},parsed->rendered}},"complete_numeric_syntax"};
}

std::vector<TransformationCandidate> detect_common_numeric(const std::string& input) {
    const auto b=unicode_scalar_boundaries(input);
    if (!b.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (size_t start=0;start<b->size()-1;++start) {
        const auto initial=unicode_scalar_substring(input,*b,start,start+1);
        if (!common_number_scalar(initial) && initial!="第" && initial!="负" && initial!="-") continue;
        for (int kind=0;kind<4;++kind) {
            auto parsed=kind==0 ? common_object(input,*b,start) : kind==1 ? common_range(input,*b,start) : kind==2 ? common_scalar(input,*b,start) : common_sequence(input,*b,start);
            if (!parsed.has_value()) continue;
            const auto source=unicode_scalar_substring(input,*b,start,parsed->end);
            if (source==parsed->rendered) continue;
            candidates.push_back({"zh-common-"+std::to_string(kind)+"-"+std::to_string(start)+"-"+std::to_string(parsed->end),
                {start,parsed->end,source},"zh-CN","number",kind==1 ? "common_range" : kind==2 ? "common_scalar" : kind==3 ? "common_sequence" : "common_object",
                kind==1 ? kCommonNumericRangeRule : kind==2 ? "zh.number.complete_scalar" : kind==3 ? kCommonNumericSequenceRule : kCommonNumericRule,kCommonNumericRuleVersion});
        }
    }
    return candidates;
}
