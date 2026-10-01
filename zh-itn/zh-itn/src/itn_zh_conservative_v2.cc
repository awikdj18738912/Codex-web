/*
 * itn_zh_conservative_v2.cc - Strict conservative Chinese ITN pipeline.
 */

#include "itn_zh_conservative_v2.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace zh_itn::itn {
namespace {

constexpr const char* kCalendarYearRule = "zh.date.explicit_four_digit_year";
constexpr const char* kCalendarYearRuleVersion = "3";
constexpr const char* kYearRangeRule = "zh.date.explicit_year_range";
constexpr const char* kYearRangeRuleVersion = "1";
constexpr const char* kFullDateRangeRule = "zh.date.explicit_full_date_range";
constexpr const char* kFullDateRangeRuleVersion = "1";
constexpr const char* kSharedYearDateRangeRule =
    "zh.date.explicit_shared_year_date_range";
constexpr const char* kSharedYearDateRangeRuleVersion = "1";
constexpr const char* kSharedMonthDateRangeRule =
    "zh.date.explicit_shared_month_date_range";
constexpr const char* kSharedMonthDateRangeRuleVersion = "1";
constexpr const char* kFullDateSharedMonthRangeRule =
    "zh.date.explicit_full_date_shared_month_range";
constexpr const char* kFullDateSharedMonthRangeRuleVersion = "1";
constexpr const char* kYearMonthRule = "zh.date.explicit_year_month";
constexpr const char* kYearMonthRuleVersion = "1";
constexpr const char* kCrossSegmentMonthRule = "zh.date.contextual_cross_segment_month";
constexpr const char* kCrossSegmentMonthRuleVersion = "1";
constexpr const char* kYearMonthRangeRule = "zh.date.explicit_year_month_range";
constexpr const char* kYearMonthRangeRuleVersion = "2";
constexpr const char* kFullDateRule = "zh.date.explicit_full_date";
constexpr const char* kFullDateRuleVersion = "2";
constexpr const char* kMonthDayRule = "zh.date.explicit_month_day";
constexpr const char* kMonthDayRuleVersion = "2";
constexpr const char* kMonthPeriodRule = "zh.date.explicit_month_period";
constexpr const char* kMonthPeriodRuleVersion = "1";
constexpr const char* kMonthRangeRule = "zh.date.explicit_month_range";
constexpr const char* kMonthRangeRuleVersion = "1";
constexpr const char* kRelativeYearMonthRule = "zh.date.explicit_relative_year_month";
constexpr const char* kRelativeYearMonthRuleVersion = "1";
constexpr const char* kContextualFollowupMonthRule =
    "zh.date.contextual_followup_month";
constexpr const char* kContextualFollowupMonthRuleVersion = "1";
constexpr const char* kRecurringCrossYearMonthRangeRule =
    "zh.date.explicit_recurring_cross_year_month_range";
constexpr const char* kRecurringCrossYearMonthRangeRuleVersion = "1";
constexpr const char* kPercentageRule = "zh.percentage.explicit";
constexpr const char* kPercentageRuleVersion = "1";
constexpr const char* kPercentageRangeRule = "zh.percentage.explicit_range";
constexpr const char* kPercentageRangeRuleVersion = "1";
constexpr const char* kSharedPrefixPercentageRangeRule =
    "zh.percentage.shared_prefix_range";
constexpr const char* kSharedPrefixPercentageRangeRuleVersion = "1";
constexpr const char* kIsolatedDigitGRule = "zh.identifier.isolated_digit_g";
constexpr const char* kIsolatedDigitGRuleVersion = "1";
constexpr const char* kRespiratorRule = "zh.identifier.respirator_standard_n95";
constexpr const char* kRespiratorRuleVersion = "1";
constexpr const char* kAutomotiveStoreRule = "zh.identifier.automotive_4s_store";
constexpr const char* kAutomotiveStoreRuleVersion = "1";
constexpr const char* kPlayStationRule = "zh.identifier.playstation_model";
constexpr const char* kPlayStationRuleVersion = "1";
constexpr const char* kSoftwareVersionRule = "zh.identifier.software_version";
constexpr const char* kSoftwareVersionRuleVersion = "3";
constexpr const char* kTelephoneRule = "zh.identifier.telephone_number";
constexpr const char* kTelephoneRuleVersion = "1";
constexpr const char* kTelephoneListRule = "zh.identifier.telephone_number_list";
constexpr const char* kTelephoneListRuleVersion = "1";
constexpr const char* kExplicitIdentifierRule = "zh.identifier.explicit_digit_sequence";
constexpr const char* kExplicitIdentifierRuleVersion = "8";
constexpr const char* kTrainNumberRule = "zh.identifier.train_number";
constexpr const char* kTrainNumberRuleVersion = "2";
constexpr const char* kFlightNumberRule = "zh.identifier.flight_number";
constexpr const char* kFlightNumberRuleVersion = "1";
constexpr const char* kIpv4AddressRule = "zh.identifier.ipv4_address";
constexpr const char* kIpv4AddressRuleVersion = "1";
constexpr const char* kNetworkPortRule = "zh.identifier.network_port";
constexpr const char* kNetworkPortRuleVersion = "1";
constexpr const char* kPublicServiceNumberRule =
    "zh.identifier.contextual_public_service_number";
constexpr const char* kPublicServiceNumberRuleVersion = "1";
constexpr const char* kClosedProductModelRule = "zh.identifier.closed_product_model";
constexpr const char* kClosedProductModelRuleVersion = "1";
constexpr const char* kContextualProductModelRule =
    "zh.identifier.contextual_product_model";
constexpr const char* kContextualProductModelRuleVersion = "5";
constexpr const char* kLandParcelIdentifierRule =
    "zh.identifier.contextual_land_parcel";
constexpr const char* kLandParcelIdentifierRuleVersion = "1";
constexpr const char* kDisplayResolutionRule = "zh.identifier.display_resolution";
constexpr const char* kDisplayResolutionRuleVersion = "1";
constexpr const char* kExplicitContextualRatioRule = "zh.ratio.explicit_contextual";
constexpr const char* kExplicitContextualRatioRuleVersion = "1";
constexpr const char* kContextualScalarRatioRule = "zh.ratio.contextual_scalar";
constexpr const char* kContextualScalarRatioRuleVersion = "4";
constexpr const char* kContextualSettingValueRule = "zh.number.contextual_setting_value";
constexpr const char* kContextualSettingValueRuleVersion = "1";
constexpr const char* kContextualScoreRule = "zh.score.explicit_contextual";
constexpr const char* kContextualScoreRuleVersion = "2";
constexpr const char* kContextualRankRule = "zh.ordinal.contextual_rank";
constexpr const char* kContextualRankRuleVersion = "1";
constexpr const char* kContextualDocumentPageRule = "zh.ordinal.contextual_document_page";
constexpr const char* kContextualDocumentPageRuleVersion = "1";
constexpr const char* kContextualStructuredOrdinalRule =
    "zh.ordinal.contextual_structured_reference";
constexpr const char* kContextualStructuredOrdinalRuleVersion = "2";
constexpr const char* kContextualAnchoredCardinalRule =
    "zh.number.contextual_anchored_cardinal";
constexpr const char* kContextualAnchoredCardinalRuleVersion = "4";
constexpr const char* kTechnicalIdentifierRule =
    "zh.identifier.contextual_technical_notation";
constexpr const char* kTechnicalIdentifierRuleVersion = "1";
constexpr const char* kStructuredSpokenIdentifierRule =
    "zh.identifier.structured_spoken_locator";
constexpr const char* kStructuredSpokenIdentifierRuleVersion = "1";
constexpr const char* kExplicitGeographicCoordinateRule =
    "zh.measure.explicit_geographic_coordinate";
constexpr const char* kExplicitGeographicCoordinateRuleVersion = "2";
constexpr const char* kCenturyRule = "zh.date.explicit_century";
constexpr const char* kCenturyRuleVersion = "1";
constexpr const char* kClockTimeRule = "zh.time.explicit_clock_time";
constexpr const char* kClockTimeRuleVersion = "2";
constexpr const char* kClockTimeRangeRule = "zh.time.explicit_clock_time_range";
constexpr const char* kClockTimeRangeRuleVersion = "3";
constexpr const char* kContextualClockTimeRule = "zh.time.contextual_unperioded_clock_time";
constexpr const char* kContextualClockTimeRuleVersion = "5";
constexpr const char* kContextualApproximateClockTimeRule =
    "zh.time.contextual_approximate_unperioded_clock_time";
constexpr const char* kContextualApproximateClockTimeRuleVersion = "1";
constexpr const char* kExplicitTimecodeRule = "zh.time.explicit_video_timecode";
constexpr const char* kExplicitTimecodeRuleVersion = "1";
constexpr const char* kClinicalThresholdRule =
    "zh.number.explicit_clinical_threshold";
constexpr const char* kClinicalThresholdRuleVersion = "1";
constexpr const char* kExplicitSignedNumberRule =
    "zh.number.explicit_signed_number";
constexpr const char* kExplicitSignedNumberRuleVersion = "1";
constexpr const char* kCalendarTimestampRule = "zh.time.explicit_calendar_timestamp";
constexpr const char* kCalendarTimestampRuleVersion = "6";
constexpr const char* kCalendarTimestampRangeRule =
    "zh.time.explicit_calendar_timestamp_range";
constexpr const char* kCalendarTimestampRangeRuleVersion = "3";
constexpr const char* kApproximateCalendarTimestampRule =
    "zh.time.explicit_approximate_calendar_timestamp";
constexpr const char* kApproximateCalendarTimestampRuleVersion = "1";
constexpr const char* kContextualMonthDayTimestampRule =
    "zh.time.explicit_contextual_month_day_timestamp";
constexpr const char* kContextualMonthDayTimestampRuleVersion = "2";
constexpr const char* kContextualFullDateTimestampRule =
    "zh.time.explicit_contextual_full_date_timestamp";
constexpr const char* kContextualFullDateTimestampRuleVersion = "2";
constexpr const char* kContextualApproximateMonthDayTimestampRule =
    "zh.time.explicit_contextual_approximate_month_day_timestamp";
constexpr const char* kContextualApproximateMonthDayTimestampRuleVersion = "1";
constexpr const char* kDecimalDurationRule = "zh.duration.explicit_decimal";
constexpr const char* kDecimalDurationRuleVersion = "1";
constexpr const char* kDecimalDurationApproximateRule =
    "zh.duration.explicit_decimal_approximate";
constexpr const char* kDecimalDurationApproximateRuleVersion = "1";
constexpr const char* kDecimalDurationThresholdRule =
    "zh.duration.explicit_decimal_threshold";
constexpr const char* kDecimalDurationThresholdRuleVersion = "1";
constexpr const char* kDecimalMeasureRule = "zh.measure.explicit_decimal";
constexpr const char* kDecimalMeasureRuleVersion = "8";
constexpr const char* kDecimalMeasureApproximateRule =
    "zh.measure.explicit_decimal_approximate";
constexpr const char* kDecimalMeasureApproximateRuleVersion = "2";
constexpr const char* kDecimalMeasureThresholdRule =
    "zh.measure.explicit_decimal_threshold";
constexpr const char* kDecimalMeasureThresholdRuleVersion = "2";
constexpr const char* kDecimalMoneyRule = "zh.money.explicit_decimal";
constexpr const char* kDecimalMoneyRuleVersion = "2";
constexpr const char* kDecimalMoneyApproximateRule =
    "zh.money.explicit_decimal_approximate";
constexpr const char* kDecimalMoneyApproximateRuleVersion = "1";
constexpr const char* kDecimalMoneyThresholdRule = "zh.money.explicit_decimal_threshold";
constexpr const char* kDecimalMoneyThresholdRuleVersion = "1";
constexpr const char* kContextualStockPriceRule = "zh.money.contextual_stock_price";
constexpr const char* kContextualStockPriceRuleVersion = "2";
constexpr const char* kExplicitTemperatureRule = "zh.measure.explicit_temperature";
constexpr const char* kExplicitTemperatureRuleVersion = "3";
constexpr const char* kExplicitTemperatureApproximateRule =
    "zh.measure.explicit_temperature_approximate";
constexpr const char* kExplicitTemperatureApproximateRuleVersion = "1";
constexpr const char* kExplicitTemperatureThresholdRule =
    "zh.measure.explicit_temperature_threshold";
constexpr const char* kExplicitTemperatureThresholdRuleVersion = "1";
constexpr const char* kExplicitTemperatureRangeRule =
    "zh.measure.explicit_temperature_range";
constexpr const char* kExplicitTemperatureRangeRuleVersion = "1";
constexpr const char* kExplicitIntegerMoneyRule = "zh.money.explicit_integer";
constexpr const char* kExplicitIntegerMoneyRuleVersion = "9";
constexpr const char* kExplicitIntegerMoneyThresholdRule =
    "zh.money.explicit_integer_threshold";
constexpr const char* kExplicitIntegerMoneyThresholdRuleVersion = "2";
constexpr const char* kExplicitIntegerMoneyApproximateRule =
    "zh.money.explicit_integer_approximate";
constexpr const char* kExplicitIntegerMoneyApproximateRuleVersion = "1";
constexpr const char* kExplicitIntegerMeasureRule = "zh.measure.explicit_integer";
constexpr const char* kExplicitIntegerMeasureRuleVersion = "16";
constexpr const char* kExplicitIntegerMeasureThresholdRule =
    "zh.measure.explicit_integer_threshold";
constexpr const char* kExplicitIntegerMeasureThresholdRuleVersion = "1";
constexpr const char* kExplicitIntegerMeasureApproximateRule =
    "zh.measure.explicit_integer_approximate";
constexpr const char* kExplicitIntegerMeasureApproximateRuleVersion = "3";
constexpr const char* kContextualApproximateRateDistanceRule =
    "zh.measure.contextual_approximate_rate_distance";
constexpr const char* kContextualApproximateRateDistanceRuleVersion = "1";
constexpr const char* kExplicitIntegerMeasurePeriodicRule =
    "zh.measure.explicit_integer_periodic";
constexpr const char* kExplicitIntegerMeasurePeriodicRuleVersion = "1";
constexpr const char* kExplicitIntegerDurationRule = "zh.duration.explicit_integer";
constexpr const char* kExplicitIntegerDurationRuleVersion = "4";
constexpr const char* kExplicitIntegerDurationThresholdRule =
    "zh.duration.explicit_integer_threshold";
constexpr const char* kExplicitIntegerDurationThresholdRuleVersion = "1";
constexpr const char* kExplicitIntegerDurationApproximateRule =
    "zh.duration.explicit_integer_approximate";
constexpr const char* kExplicitIntegerDurationApproximateRuleVersion = "1";
constexpr const char* kExplicitIntegerDurationPeriodicRule =
    "zh.duration.explicit_integer_periodic";
constexpr const char* kExplicitIntegerDurationPeriodicRuleVersion = "1";
constexpr const char* kExplicitLockPeriodMonthsRule =
    "zh.duration.explicit_lock_period_months";
constexpr const char* kExplicitLockPeriodMonthsRuleVersion = "1";
constexpr const char* kLargeApproximateYearSpanRule =
    "zh.duration.explicit_large_approximate_year_span";
constexpr const char* kLargeApproximateYearSpanRuleVersion = "1";
constexpr const char* kExplicitValidityYearsRule =
    "zh.duration.explicit_validity_years";
constexpr const char* kExplicitValidityYearsRuleVersion = "1";
constexpr const char* kExplicitRenewalApproximateDaysRule =
    "zh.duration.explicit_renewal_approximate_days";
constexpr const char* kExplicitRenewalApproximateDaysRuleVersion = "1";
constexpr const char* kExplicitCompoundDurationRule =
    "zh.duration.explicit_compound_hour_minute";
constexpr const char* kExplicitCompoundDurationRuleVersion = "1";
constexpr const char* kExplicitMinuteSecondDurationRule =
    "zh.duration.explicit_compound_minute_second";
constexpr const char* kExplicitMinuteSecondDurationRuleVersion = "1";
constexpr const char* kExplicitMoneyRangeRule = "zh.money.explicit_range";
constexpr const char* kExplicitMoneyRangeRuleVersion = "2";
constexpr const char* kExplicitMeasureRangeRule = "zh.measure.explicit_range";
constexpr const char* kExplicitMeasureRangeRuleVersion = "4";
constexpr const char* kExplicitDurationRangeRule = "zh.duration.explicit_range";
constexpr const char* kExplicitDurationRangeRuleVersion = "1";

std::optional<std::string> clock_period_at(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
);
bool date_continues_into_minute_clock(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
);
bool contextual_clock_anchor_applies(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
);
std::optional<std::string> parse_software_version(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
);

std::optional<int> spoken_year_digit(const std::string& value) {
    if (value == "零" || value == "〇") return 0;
    if (value == "一") return 1;
    if (value == "二") return 2;
    if (value == "三") return 3;
    if (value == "四") return 4;
    if (value == "五") return 5;
    if (value == "六") return 6;
    if (value == "七") return 7;
    if (value == "八") return 8;
    if (value == "九") return 9;
    return std::nullopt;
}

std::optional<int> spoken_cardinal_digit(const std::string& value) {
    const auto digit = spoken_year_digit(value);
    if (digit.has_value()) return digit;
    if (value == "两") return 2;
    return std::nullopt;
}

std::optional<int> spoken_small_unit(const std::string& value) {
    if (value == "十") return 10;
    if (value == "百") return 100;
    if (value == "千") return 1000;
    return std::nullopt;
}

std::optional<int> parse_spoken_cardinal(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (start >= end) return std::nullopt;
    int total = 0;
    int current = -1;
    int last_unit = 10000;
    bool saw_unit = false;
    bool zero_pending = false;
    bool gap_before_current = false;
    for (size_t index = start; index < end; ++index) {
        const std::string value = unicode_scalar_substring(input, boundaries, index, index + 1);
        const auto digit = spoken_cardinal_digit(value);
        if (digit.has_value()) {
            if (*digit == 0) {
                if (start + 1 == end) return 0;
                if (current >= 0 || zero_pending || total == 0) return std::nullopt;
                zero_pending = true;
                continue;
            }
            if (current >= 0) return std::nullopt;
            current = *digit;
            gap_before_current = zero_pending;
            zero_pending = false;
            continue;
        }
        const auto unit = spoken_small_unit(value);
        if (!unit.has_value() || zero_pending || *unit >= last_unit) return std::nullopt;
        if (current < 0) {
            if (*unit != 10 || total != 0) return std::nullopt;
            current = 1;
        }
        total += current * *unit;
        current = -1;
        gap_before_current = false;
        last_unit = *unit;
        saw_unit = true;
    }
    if (zero_pending) return std::nullopt;
    if (current >= 0) {
        if (saw_unit && last_unit >= 100 && !gap_before_current) return std::nullopt;
        total += current;
    }
    return total;
}

bool valid_calendar_day(int year, int month, int day, bool has_year) {
    if (month < 1 || month > 12 || day < 1) return false;
    static constexpr int days[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int maximum = days[month];
    if (month == 2 && (!has_year
            || (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)))) {
        maximum = 29;
    }
    return day <= maximum;
}

std::string canonical_spoken_calendar_component(int value) {
    static const char* digits[] = {"", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
    if (value >= 1 && value <= 9) return digits[value];
    if (value == 10) return "十";
    if (value >= 11 && value <= 19) return std::string("十") + digits[value - 10];
    if (value == 20) return "二十";
    if (value >= 21 && value <= 29) return std::string("二十") + digits[value - 20];
    if (value == 30) return "三十";
    if (value == 31) return "三十一";
    return "";
}

bool canonical_spoken_calendar_component(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    int value
) {
    return unicode_scalar_substring(input, boundaries, start, end)
        == canonical_spoken_calendar_component(value);
}

bool shared_month_day_continuation(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t end
) {
    const size_t limit = std::min(boundaries.size() - 1, end + 24);
    for (size_t cursor = end; cursor < limit; ++cursor) {
        size_t trigger_length = 0;
        for (const std::string trigger : {
                "上午", "下午", "凌晨", "早上", "中午", "傍晚", "晚上", "夜里", "夜间",
                "即", "至", "到", "和", "及", "与", "或", "、",
            }) {
            const size_t length = trigger.size() == 3 ? 1 : 2;
            if (cursor + length <= limit
                && unicode_scalar_substring(input, boundaries, cursor, cursor + length) == trigger) {
                trigger_length = length;
                break;
            }
        }
        if (trigger_length == 0) continue;
        for (size_t number_start = cursor + trigger_length; number_start < limit; ++number_start) {
            if (!spoken_cardinal_digit(unicode_scalar_substring(
                    input, boundaries, number_start, number_start + 1
                )).has_value()
                && !spoken_small_unit(unicode_scalar_substring(
                    input, boundaries, number_start, number_start + 1
                )).has_value()) {
                continue;
            }
            size_t number_end = number_start + 1;
            while (number_end < limit
                && (spoken_cardinal_digit(unicode_scalar_substring(
                        input, boundaries, number_end, number_end + 1
                    )).has_value()
                    || spoken_small_unit(unicode_scalar_substring(
                        input, boundaries, number_end, number_end + 1
                    )).has_value())) {
                ++number_end;
            }
            if (number_end < limit) {
                const std::string suffix = unicode_scalar_substring(
                    input, boundaries, number_end, number_end + 1
                );
                if (suffix == "日" || suffix == "号") return true;
                if (suffix == "月") break;
            }
            number_start = number_end;
        }
    }
    return false;
}

bool overlaps(size_t start, size_t end, const V2SourceRange& range) {
    return start < range.end && end > range.start;
}

bool ends_with_any(const std::string& value, const std::vector<std::string>& suffixes) {
    for (const auto& suffix : suffixes) {
        if (value.size() >= suffix.size()
            && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

std::optional<std::string> closing_delimiter(const std::string& value) {
    if (value == "《") return "》";
    if (value == "〈") return "〉";
    if (value == "“") return "”";
    if (value == "‘") return "’";
    if (value == "「") return "」";
    if (value == "『") return "』";
    if (value == "【") return "】";
    if (value == "〔") return "〕";
    if (value == "〖") return "〗";
    if (value == "〘") return "〙";
    if (value == "〚") return "〛";
    if (value == "（") return "）";
    if (value == "(") return ")";
    if (value == "［") return "］";
    if (value == "[") return "]";
    if (value == "｛") return "｝";
    if (value == "{") return "}";
    if (value == "\"") return "\"";
    if (value == "'") return "'";
    return std::nullopt;
}

bool direct_speech_quote_opener(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position,
    const std::string& value
) {
    if (value != "“" && value != "「" && value != "『") return false;
    size_t left_end = position;
    while (left_end > 0) {
        const std::string previous = unicode_scalar_substring(
            input, boundaries, left_end - 1, left_end
        );
        if (previous != " " && previous != "\t") break;
        --left_end;
    }
    const size_t left_start = left_end > 12 ? left_end - 12 : 0;
    const std::string left = unicode_scalar_substring(
        input, boundaries, left_start, left_end
    );
    return ends_with_any(left, {
        "说：", "说道：", "表示：", "称：", "指出：", "提到：", "回答：", "问：",
        "写道：", "宣布：", "强调：", "说:", "说道:", "表示:", "称:", "指出:",
        "提到:", "回答:", "问:", "写道:", "宣布:", "强调:",
    });
}

bool inside_protected_delimiters(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position
) {
    std::vector<std::string> expected_closers;
    for (size_t index = 0; index < position; ++index) {
        const std::string value = unicode_scalar_substring(input, boundaries, index, index + 1);
        if (!expected_closers.empty() && value == expected_closers.back()) {
            expected_closers.pop_back();
            continue;
        }
        const auto closer = closing_delimiter(value);
        if (closer.has_value()
            && !direct_speech_quote_opener(input, boundaries, index, value)) {
            expected_closers.push_back(*closer);
        }
    }
    return !expected_closers.empty();
}

bool starts_with_any(const std::string& value, const std::vector<std::string>& prefixes) {
    for (const auto& prefix : prefixes) {
        if (value.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

bool contains_any(const std::string& value, const std::vector<std::string>& needles) {
    for (const auto& needle : needles) {
        if (value.find(needle) != std::string::npos) return true;
    }
    return false;
}

std::string context_window(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    size_t limit
) {
    if (start >= end) return {};
    if (end - start > limit) start = end - limit;
    return unicode_scalar_substring(input, boundaries, start, end);
}

bool clause_boundary_scalar(const std::string& value) {
    return contains_any("，。；：！？!?;:\n\r", {value});
}

std::string left_clause_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position,
    size_t limit
) {
    size_t start = position > limit ? position - limit : 0;
    for (size_t index = position; index > start; --index) {
        if (clause_boundary_scalar(unicode_scalar_substring(
                input, boundaries, index - 1, index
            ))) {
            start = index;
            break;
        }
    }
    return unicode_scalar_substring(input, boundaries, start, position);
}

std::string right_clause_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position,
    size_t limit
) {
    const size_t length = boundaries.size() - 1;
    size_t end = std::min(length, position + limit);
    for (size_t index = position; index < end; ++index) {
        if (clause_boundary_scalar(unicode_scalar_substring(
                input, boundaries, index, index + 1
            ))) {
            end = index;
            break;
        }
    }
    return unicode_scalar_substring(input, boundaries, position, end);
}

bool inline_space_scalar(const std::string& value) {
    return value == " " || value == "\t";
}

std::optional<size_t> skip_limited_inline_spaces(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position,
    size_t limit,
    size_t maximum
) {
    size_t count = 0;
    while (position < limit && inline_space_scalar(unicode_scalar_substring(
            input, boundaries, position, position + 1
        ))) {
        if (++count > maximum) return std::nullopt;
        ++position;
    }
    return position;
}

std::string previous_non_space_scalar(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position
) {
    while (position > 0) {
        const std::string value = unicode_scalar_substring(
            input, boundaries, position - 1, position
        );
        if (!inline_space_scalar(value)) return value;
        --position;
    }
    return {};
}

std::string next_non_space_scalar(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t position
) {
    const size_t length = boundaries.size() - 1;
    while (position < length) {
        const std::string value = unicode_scalar_substring(
            input, boundaries, position, position + 1
        );
        if (!inline_space_scalar(value)) return value;
        ++position;
    }
    return {};
}

size_t structured_identifier_candidate_start(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t identifier_start
) {
    size_t start = identifier_start;
    size_t spaces = 0;
    while (start > 0 && spaces < 2 && inline_space_scalar(unicode_scalar_substring(
            input, boundaries, start - 1, start
        ))) {
        --start;
        ++spaces;
    }
    if (start == identifier_start || start == 0) return start;
    const std::string previous = unicode_scalar_substring(
        input, boundaries, start - 1, start
    );
    return previous.size() == 1
            && std::isalnum(static_cast<unsigned char>(previous[0])) != 0
        ? identifier_start : start;
}

std::optional<size_t> structured_identifier_shape_start(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    return skip_limited_inline_spaces(input, boundaries, start, end, 2);
}

bool literal_scope_before_year(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const auto has_literal_marker = [](const std::string& value) {
        return contains_any(value, {
            "文件名", "文件夹名", "目录名", "密码", "口令", "验证码", "字符串",
            "原文", "原句", "原样", "搜索词", "关键词", "作品名", "书名",
            "名称", "代号", "网名", "昵称", "用户名", "账号", "写的是", "写成",
            "读作", "念作", "叫作", "叫做", "名为", "命名为", "保留原文",
        });
    };
    size_t clause_start = start > 24 ? start - 24 : 0;
    for (size_t index = start; index > clause_start; --index) {
        const std::string value = unicode_scalar_substring(
            input,
            boundaries,
            index - 1,
            index
        );
        if (value == "：" || value == ":") {
            const size_t separator = index - 1;
            size_t label_start = separator > 12 ? separator - 12 : 0;
            for (size_t label_index = separator; label_index > label_start; --label_index) {
                const std::string previous = unicode_scalar_substring(
                    input,
                    boundaries,
                    label_index - 1,
                    label_index
                );
                if (contains_any("，。；！？!?;\n\r", {previous})) {
                    label_start = label_index;
                    break;
                }
            }
            if (has_literal_marker(unicode_scalar_substring(
                    input, boundaries, label_start, separator
                ))) {
                return true;
            }
            clause_start = index;
            break;
        }
        if (contains_any("，。；：！？!?;:\n\r", {value})) {
            clause_start = index;
            break;
        }
    }
    const std::string left = unicode_scalar_substring(input, boundaries, clause_start, start);
    return has_literal_marker(left);
}

bool numeric_scalar(const std::string& value) {
    if (spoken_year_digit(value).has_value()) return true;
    return (value.size() == 1 && value[0] >= '0' && value[0] <= '9')
        || value == "０" || value == "１" || value == "２" || value == "３"
        || value == "４" || value == "５" || value == "６" || value == "７"
        || value == "８" || value == "９" || value == "两" || value == "十";
}

bool range_connector(const std::string& value) {
    return value == "至" || value == "到" || value == "-" || value == "－"
        || value == "—" || value == "~" || value == "～";
}

bool number_continuation(const std::string& value) {
    return numeric_scalar(value) || value == "百" || value == "千" || value == "万"
        || value == "萬" || value == "亿" || value == "億" || value == "兆"
        || value == "壹" || value == "贰" || value == "貳" || value == "叁"
        || value == "參" || value == "肆" || value == "伍" || value == "陆"
        || value == "陸" || value == "柒" || value == "捌" || value == "玖"
        || value == "拾" || value == "佰" || value == "仟" || value == "廿"
        || value == "卅" || value == "卌";
}

bool date_or_number_continuation(const std::string& value) {
    return number_continuation(value) || value == "正" || value == "元"
        || value == "腊" || value == "冬";
}

bool incomplete_cross_segment_span(
    const SafePolicyContext& context,
    size_t input_length,
    size_t start,
    size_t end
) {
    if (end == input_length && !context.right_context.empty()) {
        const auto right_boundaries = unicode_scalar_boundaries(context.right_context);
        if (!right_boundaries.has_value()) return true;
        const std::string first = unicode_scalar_substring(
            context.right_context,
            *right_boundaries,
            0,
            1
        );
        if (date_or_number_continuation(first) || range_connector(first) || first == "代") {
            return true;
        }
        if (first == "的" && right_boundaries->size() > 2) {
            const std::string second = unicode_scalar_substring(
                context.right_context,
                *right_boundaries,
                1,
                2
            );
            if (date_or_number_continuation(second)) return true;
        }
    }
    if (start != 0 || context.left_context.empty()) return false;
    const auto left_boundaries = unicode_scalar_boundaries(context.left_context);
    if (!left_boundaries.has_value()) return true;
    const size_t left_length = left_boundaries->size() - 1;
    if (left_length == 0) return false;
    const std::string last = unicode_scalar_substring(
        context.left_context,
        *left_boundaries,
        left_length - 1,
        left_length
    );
    if (last == "年") return true;
    if (left_length < 2 || !range_connector(last)) return false;
    const std::string before = unicode_scalar_substring(
        context.left_context,
        *left_boundaries,
        left_length - 2,
        left_length - 1
    );
    return number_continuation(before) || before == "年";
}

bool prior_numeric_year_in_range_scope(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    size_t clause_start = 0;
    for (size_t index = start; index > 0; --index) {
        const std::string value = unicode_scalar_substring(
            input,
            boundaries,
            index - 1,
            index
        );
        if (contains_any("，。；：！？!?;:\n\r", {value})) {
            clause_start = index;
            break;
        }
    }
    bool range_scope = false;
    bool prior_numeric_year = false;
    for (size_t index = clause_start; index < start; ++index) {
        const std::string value = unicode_scalar_substring(input, boundaries, index, index + 1);
        if (value == "从" || value == "自") range_scope = true;
        if (value == "年" && index > clause_start) {
            const std::string previous = unicode_scalar_substring(
                input,
                boundaries,
                index - 1,
                index
            );
            if (number_continuation(previous)) prior_numeric_year = true;
        }
    }
    return range_scope && prior_numeric_year;
}

bool incomplete_temporal_span(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (start > 0
        && unicode_scalar_substring(input, boundaries, start - 1, start) == "年") {
        return true;
    }
    if (end < length) {
        const std::string right = unicode_scalar_substring(
            input,
            boundaries,
            end,
            std::min(length, end + 8)
        );
        if (right.find("年代") != std::string::npos) return true;
        const std::string next = unicode_scalar_substring(input, boundaries, end, end + 1);
        bool released_year_suffix = right.rfind("七夕", 0) == 0;
        if (!released_year_suffix && next == "代" && end >= 2) {
            const auto decade_digit = spoken_year_digit(unicode_scalar_substring(
                input, boundaries, end - 2, end - 1
            ));
            released_year_suffix = decade_digit.has_value() && *decade_digit == 0;
        }
        if (!released_year_suffix && next.size() == 1
            && std::isdigit(static_cast<unsigned char>(next[0]))) {
            size_t month_end = end;
            int month = 0;
            while (month_end < length && month_end - end < 2) {
                const std::string digit = unicode_scalar_substring(
                    input, boundaries, month_end, month_end + 1
                );
                if (digit.size() != 1
                    || !std::isdigit(static_cast<unsigned char>(digit[0]))) {
                    break;
                }
                month = month * 10 + digit[0] - '0';
                ++month_end;
            }
            released_year_suffix = month >= 1 && month <= 12 && month_end < length
                && unicode_scalar_substring(input, boundaries, month_end, month_end + 1) == "月";
        }
        if (!released_year_suffix
            && (date_or_number_continuation(next) || range_connector(next) || next == "代")) {
            return true;
        }
        if (next == "的" && end + 1 < length) {
            const std::string after = unicode_scalar_substring(
                input,
                boundaries,
                end + 1,
                end + 2
            );
            if (date_or_number_continuation(after)) return true;
        }
    }
    if (start >= 2) {
        const std::string connector = unicode_scalar_substring(
            input,
            boundaries,
            start - 1,
            start
        );
        const std::string before_connector = unicode_scalar_substring(
            input,
            boundaries,
            start - 2,
            start - 1
        );
        if (range_connector(connector)
            && (number_continuation(before_connector) || before_connector == "年")) {
            return true;
        }
    }
    return false;
}

bool released_cross_segment_year_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    const SafePolicyContext& context,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (!context.right_context.empty()) {
        const auto right = unicode_scalar_boundaries(context.right_context);
        if (!right.has_value()) return false;
        const size_t right_length = right->size() - 1;
        size_t month_mark = 0;
        while (end == length && month_mark < right_length && month_mark < 3
            && unicode_scalar_substring(
                context.right_context, *right, month_mark, month_mark + 1
            ) != "月") {
            ++month_mark;
        }
        if (end == length && month_mark > 0 && month_mark < right_length) {
            const auto month = parse_spoken_cardinal(
                context.right_context, *right, 0, month_mark
            );
            if (month.has_value() && *month >= 1 && *month <= 12
                && canonical_spoken_calendar_component(
                    context.right_context, *right, 0, month_mark, *month
                )) {
                return true;
            }
        }
        if (end + 1 == length
            && unicode_scalar_substring(input, boundaries, end, end + 1) == "至"
            && right_length >= 5) {
            bool four_digits = true;
            for (size_t offset = 0; offset < 4; ++offset) {
                if (!spoken_year_digit(unicode_scalar_substring(
                        context.right_context, *right, offset, offset + 1
                    )).has_value()) {
                    four_digits = false;
                    break;
                }
            }
            if (four_digits && unicode_scalar_substring(
                    context.right_context, *right, 4, 5
                ) == "年") {
                return true;
            }
        }
    }
    if (start == 0 && !context.left_context.empty()) {
        const auto left = unicode_scalar_boundaries(context.left_context);
        if (!left.has_value() || left->size() < 3) return false;
        const size_t left_length = left->size() - 1;
        return left_length >= 2
            && unicode_scalar_substring(
                context.left_context, *left, left_length - 2, left_length
            ) == "年至";
    }
    return false;
}

RuleApproval approve_explicit_year(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const bool released_cross_segment = released_cross_segment_year_context(
        input, *boundaries, context, range.start, range.end
    );
    if (range.end - range.start != 5
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) != "年"
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || (!released_cross_segment
            && incomplete_temporal_span(input, *boundaries, range.start, range.end))
        || prior_numeric_year_in_range_scope(input, *boundaries, range.start)
        || (!released_cross_segment && incomplete_cross_segment_span(
            context,
            boundaries->size() - 1,
            range.start,
            range.end
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "not_explicit_calendar_year"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input,
            *boundaries,
            range.start - 1,
            range.start
        );
        if (spoken_year_digit(previous).has_value()) {
            return {RuleDecision::preserve, "", {}, {}, "numeric_run_left_boundary"};
        }
    }
    int year = 0;
    for (size_t index = range.start; index + 1 < range.end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input,
            *boundaries,
            index,
            index + 1
        ));
        if (!digit.has_value()) {
            return {RuleDecision::preserve, "", {}, {}, "not_spoken_digit_year"};
        }
        year = year * 10 + *digit;
    }
    if (year < 1000 || year > 2999) {
        return {RuleDecision::preserve, "", {}, {}, "year_out_of_strict_range"};
    }
    const std::string left = context_window(input, *boundaries, 0, range.start, 10);
    if ((year < 1900 || year > 2099)
        && !ends_with_any(left, {"公元", "西元"})) {
        return {RuleDecision::preserve, "", {}, {}, "year_outside_modern_context"};
    }
    const std::string digits = unicode_scalar_substring(
        input,
        *boundaries,
        range.start,
        range.end - 1
    );
    const std::string replacement = std::to_string(year);
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "four_spoken_digits_plus_year", kCalendarYearRule},
            {"boundary", "complete_calendar_year", kCalendarYearRule},
            {"context", "complete_digit_sequence_calendar_year", kCalendarYearRule},
        },
        {{{range.start, range.end - 1, digits}, replacement}},
        "explicit_calendar_year",
    };
}

RuleApproval approve_year_month(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || range.end - range.start < 7
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || (range.end == length && context.has_right_neighbor)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_year_month"};
    }
    const size_t year_end = range.start + 4;
    const bool month_part_suffix = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    ) == "份";
    const size_t month_suffix = month_part_suffix ? range.end - 2 : range.end - 1;
    if (unicode_scalar_substring(input, *boundaries, year_end, year_end + 1) != "年"
        || unicode_scalar_substring(input, *boundaries, month_suffix, month_suffix + 1) != "月") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_shape"};
    }
    int year = 0;
    for (size_t index = range.start; index < year_end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, index, index + 1
        ));
        if (!digit.has_value()) {
            return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_year"};
        }
        year = year * 10 + *digit;
    }
    const std::string left = context_window(input, *boundaries, 0, range.start, 10);
    if (year < 1000 || year > 2999
        || ((year < 1900 || year > 2099) && !ends_with_any(left, {"公元", "西元"}))) {
        return {RuleDecision::preserve, "", {}, {}, "year_month_year_out_of_range"};
    }
    const bool possessive = year_end + 1 < month_suffix
        && unicode_scalar_substring(input, *boundaries, year_end + 1, year_end + 2) == "的";
    const size_t month_start = year_end + 1 + (possessive ? 1 : 0);
    const size_t month_end = month_suffix;
    const auto month = parse_spoken_cardinal(
        input, *boundaries, month_start, month_end
    );
    if (!month.has_value() || *month < 1 || *month > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, month_start, month_end, *month
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_month"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年") {
            return {RuleDecision::preserve, "", {}, {}, "year_month_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next)
            || next == "份" || next == "日" || next == "号" || next == "季"
            || next == "代") {
            return {RuleDecision::preserve, "", {}, {}, "year_month_continues_right"};
        }
    }
    if (range.end == length && !context.right_context.empty()) {
        const auto right_boundaries = unicode_scalar_boundaries(context.right_context);
        if (!right_boundaries.has_value() || right_boundaries->size() < 2) {
            return {RuleDecision::preserve, "", {}, {}, "invalid_cross_segment_context"};
        }
        const std::string next = unicode_scalar_substring(
            context.right_context, *right_boundaries, 0, 1
        );
        if (next == "份" || next == "日" || next == "号" || next == "季"
            || next == "代") {
            return {RuleDecision::preserve, "", {}, {}, "cross_segment_year_month_continuation"};
        }
    }
    const std::string replacement = std::to_string(year) + "年"
        + (possessive ? "的" : "") + std::to_string(*month) + "月"
        + (month_part_suffix ? "份" : "");
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "four_digit_year_month", kYearMonthRule},
            {"boundary", "complete_year_month", kYearMonthRule},
            {"context", "explicit_year_and_month_units", kYearMonthRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_year_month",
    };
}

RuleApproval approve_cross_segment_month(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    const auto left = unicode_scalar_boundaries(context.left_context);
    if (!boundaries.has_value() || !left.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t left_length = left->size() - 1;
    if (range.start != 0 || range.end < 2
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) != "月"
        || left_length < 1
        || unicode_scalar_substring(
            context.left_context, *left, left_length - 1, left_length
        ) != "年"
        || inside_protected_delimiters(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_cross_segment_month"};
    }
    const auto month = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end - 1
    );
    if (!month.has_value() || *month < 1 || *month > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start, range.end - 1, *month
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_cross_segment_month"};
    }
    const std::string replacement = std::to_string(*month) + "月";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "complete_spoken_month", kCrossSegmentMonthRule},
            {"boundary", "complete_cross_segment_year_month", kCrossSegmentMonthRule},
            {"context", "adjacent_left_calendar_year", kCrossSegmentMonthRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "contextual_cross_segment_month",
    };
}

RuleApproval approve_year_month_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_year_month_range"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年") {
            return {RuleDecision::preserve, "", {}, {}, "year_month_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next)
            || next == "份" || next == "日" || next == "号" || next == "季"
            || next == "代") {
            return {RuleDecision::preserve, "", {}, {}, "year_month_range_continues_right"};
        }
    }
    const size_t year_end = range.start + 4;
    if (year_end + 5 > range.end
        || unicode_scalar_substring(input, *boundaries, year_end, year_end + 1) != "年"
        || unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) != "月") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_range_shape"};
    }
    int year = 0;
    for (size_t index = range.start; index < year_end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, index, index + 1
        ));
        if (!digit.has_value()) {
            return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_range_year"};
        }
        year = year * 10 + *digit;
    }
    if (year < 1900 || year > 2099) {
        return {RuleDecision::preserve, "", {}, {}, "year_month_range_year_out_of_range"};
    }
    size_t first_month_mark = year_end + 1;
    while (first_month_mark < range.end
        && first_month_mark - (year_end + 1) <= 2
        && unicode_scalar_substring(
            input, *boundaries, first_month_mark, first_month_mark + 1
        ) != "月") {
        ++first_month_mark;
    }
    if (first_month_mark >= range.end || first_month_mark == year_end + 1
        || unicode_scalar_substring(
            input, *boundaries, first_month_mark, first_month_mark + 1
        ) != "月") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_first_range_month"};
    }
    const size_t connector_index = first_month_mark + 1;
    if (connector_index + 2 >= range.end) {
        return {RuleDecision::preserve, "", {}, {}, "missing_second_range_month"};
    }
    const std::string connector = unicode_scalar_substring(
        input, *boundaries, connector_index, connector_index + 1
    );
    if (connector != "到" && connector != "至") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_range_connector"};
    }
    const size_t second_endpoint_start = connector_index + 1;
    const bool has_second_year = second_endpoint_start + 5 <= range.end
        && unicode_scalar_substring(
            input, *boundaries, second_endpoint_start + 4, second_endpoint_start + 5
        ) == "年"
        && spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, second_endpoint_start, second_endpoint_start + 1
        )).has_value()
        && spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, second_endpoint_start + 1, second_endpoint_start + 2
        )).has_value()
        && spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, second_endpoint_start + 2, second_endpoint_start + 3
        )).has_value()
        && spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, second_endpoint_start + 3, second_endpoint_start + 4
        )).has_value();
    const size_t second_month_start = second_endpoint_start + (has_second_year ? 5 : 0);
    const size_t second_month_end = range.end - 1;
    const auto first_month = parse_spoken_cardinal(
        input, *boundaries, year_end + 1, first_month_mark
    );
    const auto second_month = parse_spoken_cardinal(
        input, *boundaries, second_month_start, second_month_end
    );
    int second_year = year;
    if (has_second_year) {
        second_year = 0;
        for (size_t index = second_endpoint_start; index < second_endpoint_start + 4; ++index) {
            const auto digit = spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, index, index + 1
            ));
            if (!digit.has_value()) {
                return {RuleDecision::preserve, "", {}, {}, "invalid_second_range_year"};
            }
            second_year = second_year * 10 + *digit;
        }
    }
    if (!first_month.has_value() || !second_month.has_value()
        || *first_month < 1 || *first_month > 12
        || *second_month < 1 || *second_month > 12
        || (has_second_year
            ? (second_year <= year || second_year > 2099)
            : *first_month >= *second_month)
        || !canonical_spoken_calendar_component(
            input, *boundaries, year_end + 1, first_month_mark, *first_month
        )
        || !canonical_spoken_calendar_component(
            input, *boundaries, second_month_start, second_month_end, *second_month
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_year_month_range_month"};
    }
    const std::string replacement = std::to_string(year) + "年"
        + std::to_string(*first_month) + "月" + connector
        + (has_second_year ? std::to_string(second_year) + "年" : "")
        + std::to_string(*second_month) + "月";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "four_digit_year_two_complete_months", kYearMonthRangeRule},
            {"boundary", "complete_year_month_range", kYearMonthRangeRule},
            {"context", "explicit_year_month_units_and_range_connector", kYearMonthRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_year_month_range",
    };
}

RuleApproval approve_year_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (range.end - range.start != 11
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_year_range"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年") {
            return {RuleDecision::preserve, "", {}, {}, "year_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next) || next == "代") {
            return {RuleDecision::preserve, "", {}, {}, "year_range_continues_right"};
        }
    }
    const std::string first_unit = unicode_scalar_substring(
        input, *boundaries, range.start + 4, range.start + 5
    );
    const std::string connector = unicode_scalar_substring(
        input, *boundaries, range.start + 5, range.start + 6
    );
    const std::string second_unit = unicode_scalar_substring(
        input, *boundaries, range.start + 10, range.start + 11
    );
    if (first_unit != "年" || (connector != "至" && connector != "到")
        || second_unit != "年") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_year_range_shape"};
    }
    int first_year = 0;
    int second_year = 0;
    for (size_t offset = 0; offset < 4; ++offset) {
        const auto first_digit = spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, range.start + offset, range.start + offset + 1
        ));
        const auto second_digit = spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, range.start + 6 + offset, range.start + 7 + offset
        ));
        if (!first_digit.has_value() || !second_digit.has_value()) {
            return {RuleDecision::preserve, "", {}, {}, "invalid_year_range_year"};
        }
        first_year = first_year * 10 + *first_digit;
        second_year = second_year * 10 + *second_digit;
    }
    if (first_year < 1900 || first_year > 2099
        || second_year < 1900 || second_year > 2099) {
        return {RuleDecision::preserve, "", {}, {}, "year_range_out_of_range"};
    }
    const std::string replacement = std::to_string(first_year) + "年" + connector
        + std::to_string(second_year) + "年";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "two_four_digit_years_with_range_connector", kYearRangeRule},
            {"boundary", "complete_year_range", kYearRangeRule},
            {"context", "explicit_year_units_and_range_connector", kYearRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_year_range",
    };
}

RuleApproval approve_full_date(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
);

RuleApproval approve_full_date_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_full_date_range"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年") {
            return {RuleDecision::preserve, "", {}, {}, "full_date_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 2)
        );
        if (date_or_number_continuation(next)
            || (range_connector(next) && !starts_with_any(right, {"到期"}))
            || next == "代") {
            return {RuleDecision::preserve, "", {}, {}, "full_date_range_continues_right"};
        }
    }
    size_t connector_index = range.end;
    std::string connector;
    for (size_t index = range.start; index < range.end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, index, index + 1
        );
        if (value != "到" && value != "至") continue;
        if (connector_index != range.end) {
            return {RuleDecision::preserve, "", {}, {}, "multiple_date_range_connectors"};
        }
        connector_index = index;
        connector = value;
    }
    if (connector_index == range.start || connector_index + 1 >= range.end) {
        return {RuleDecision::preserve, "", {}, {}, "missing_full_date_range_endpoint"};
    }
    const std::string first_text = unicode_scalar_substring(
        input, *boundaries, range.start, connector_index
    );
    const std::string second_text = unicode_scalar_substring(
        input, *boundaries, connector_index + 1, range.end
    );
    const size_t first_length = connector_index - range.start;
    const size_t second_length = range.end - connector_index - 1;
    const SafePolicyContext endpoint_context = {"", "", false, false};
    const TransformationCandidate first_candidate = {
        "zh-full-date-range-first",
        {0, first_length, first_text},
        "zh-CN", "date_time", "explicit_full_date",
        kFullDateRule, kFullDateRuleVersion,
    };
    const TransformationCandidate second_candidate = {
        "zh-full-date-range-second",
        {0, second_length, second_text},
        "zh-CN", "date_time", "explicit_full_date",
        kFullDateRule, kFullDateRuleVersion,
    };
    const auto first = approve_full_date(first_text, endpoint_context, first_candidate);
    const auto second = approve_full_date(second_text, endpoint_context, second_candidate);
    if (first.decision != RuleDecision::approve || second.decision != RuleDecision::approve) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_full_date_range_endpoint"};
    }
    const std::string replacement = first.parsed_value + connector + second.parsed_value;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "two_complete_full_dates_with_range_connector", kFullDateRangeRule},
            {"boundary", "complete_full_date_range", kFullDateRangeRule},
            {"context", "explicit_date_units_and_range_connector", kFullDateRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_full_date_range",
    };
}

RuleApproval approve_full_date(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    bool starts_after_cross_segment_range = false;
    if (range.start == 0 && context.has_left_neighbor && !context.left_context.empty()) {
        const auto left_boundaries = unicode_scalar_boundaries(context.left_context);
        if (!left_boundaries.has_value()) {
            return {RuleDecision::error, "", {}, {}, "invalid_left_context_utf8"};
        }
        const size_t left_length = left_boundaries->size() - 1;
        starts_after_cross_segment_range = left_length > 0 && range_connector(
            unicode_scalar_substring(
                context.left_context,
                *left_boundaries,
                left_length - 1,
                left_length
            )
        );
    }
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || starts_after_cross_segment_range
        || incomplete_cross_segment_span(
            context,
            boundaries->size() - 1,
            range.start,
            range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_calendar_date"};
    }
    size_t year_end = range.start;
    while (year_end < range.end && spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, year_end, year_end + 1
        )).has_value()) {
        ++year_end;
    }
    if (year_end - range.start != 4 || year_end >= range.end
        || unicode_scalar_substring(input, *boundaries, year_end, year_end + 1) != "年") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_full_date_year"};
    }
    int year = 0;
    for (size_t index = range.start; index < year_end; ++index) {
        year = year * 10 + *spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, index, index + 1
        ));
    }
    const std::string left = context_window(input, *boundaries, 0, range.start, 10);
    if (year < 1000 || year > 2999
        || ((year < 1900 || year > 2099) && !ends_with_any(left, {"公元", "西元"}))) {
        return {RuleDecision::preserve, "", {}, {}, "full_date_year_out_of_range"};
    }
    const bool has_linking_particle = year_end + 1 < range.end
        && unicode_scalar_substring(input, *boundaries, year_end + 1, year_end + 2) == "的";
    const size_t month_start = year_end + 1 + (has_linking_particle ? 1 : 0);
    size_t month_mark = month_start;
    while (month_mark < range.end
        && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
        ++month_mark;
    }
    if (month_mark == month_start || month_mark >= range.end) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_full_date_month"};
    }
    const auto month = parse_spoken_cardinal(
        input, *boundaries, month_start, month_mark
    );
    const size_t day_start = month_mark + 1;
    if (!month.has_value() || day_start + 1 > range.end
        || !canonical_spoken_calendar_component(
            input, *boundaries, month_start, month_mark, *month
        )
        || unicode_scalar_substring(input, *boundaries, month_start, range.end - 1)
            .find("两") != std::string::npos) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_full_date_month"};
    }
    const std::string suffix = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    );
    if (suffix != "日" && suffix != "号") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_full_date_suffix"};
    }
    const auto day = parse_spoken_cardinal(input, *boundaries, day_start, range.end - 1);
    if (!day.has_value()
        || !canonical_spoken_calendar_component(
            input, *boundaries, day_start, range.end - 1, *day
        )
        || !valid_calendar_day(year, *month, *day, true)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_full_date_day"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "numeric_run_left_boundary"};
        }
    }
    if (shared_month_day_continuation(input, *boundaries, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "shared_month_date_sequence"};
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next)
            || date_continues_into_minute_clock(input, *boundaries, range.end)) {
            return {RuleDecision::preserve, "", {}, {}, "date_continues_right"};
        }
    }
    const std::string replacement = std::to_string(year) + "年"
        + (has_linking_particle ? "的" : "")
        + std::to_string(*month) + "月" + std::to_string(*day) + suffix;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "four_digit_year_month_day", kFullDateRule},
            {"boundary", "complete_year_month_day", kFullDateRule},
            {"context", "explicit_date_units", kFullDateRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_date",
    };
}

RuleApproval approve_month_day(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    bool starts_after_cross_segment_range = false;
    if (range.start == 0 && context.has_left_neighbor && !context.left_context.empty()) {
        const auto left_boundaries = unicode_scalar_boundaries(context.left_context);
        if (!left_boundaries.has_value()) {
            return {RuleDecision::error, "", {}, {}, "invalid_left_context_utf8"};
        }
        const size_t left_length = left_boundaries->size() - 1;
        starts_after_cross_segment_range = left_length > 0 && range_connector(
            unicode_scalar_substring(
                context.left_context,
                *left_boundaries,
                left_length - 1,
                left_length
            )
        );
    }
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || starts_after_cross_segment_range
        || incomplete_cross_segment_span(
            context,
            boundaries->size() - 1,
            range.start,
            range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_month_day"};
    }
    size_t month_mark = range.start;
    while (month_mark < range.end
        && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
        ++month_mark;
    }
    if (month_mark == range.start || month_mark >= range.end) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month"};
    }
    const auto month = parse_spoken_cardinal(input, *boundaries, range.start, month_mark);
    const std::string suffix = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    );
    const auto day = parse_spoken_cardinal(
        input, *boundaries, month_mark + 1, range.end - 1
    );
    if (!month.has_value() || !day.has_value() || (suffix != "日" && suffix != "号")
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start, month_mark, *month
        )
        || !canonical_spoken_calendar_component(
            input, *boundaries, month_mark + 1, range.end - 1, *day
        )
        || unicode_scalar_substring(input, *boundaries, range.start, range.end - 1)
            .find("两") != std::string::npos
        || !valid_calendar_day(0, *month, *day, false)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_day"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const std::string left = unicode_scalar_substring(
            input,
            *boundaries,
            range.start > 2 ? range.start - 2 : 0,
            range.start
        );
        if (previous == "年"
            || (range_connector(previous) && !ends_with_any(left, {"截至"}))
            || ends_with_any(left, {"年的"})
            || number_continuation(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "month_day_continues_left"};
        }
    }
    if (shared_month_day_continuation(input, *boundaries, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "shared_month_date_sequence"};
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input,
            *boundaries,
            range.end,
            std::min(boundaries->size() - 1, range.end + 3)
        );
        if (date_or_number_continuation(next) || range_connector(next)
            || date_continues_into_minute_clock(input, *boundaries, range.end)
            || starts_with_any(right, {"店", "楼", "线"})) {
            return {RuleDecision::preserve, "", {}, {}, "month_day_continues_right"};
        }
    }
    const std::string replacement = std::to_string(*month) + "月"
        + std::to_string(*day) + suffix;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_month_day", kMonthDayRule},
            {"boundary", "complete_month_day", kMonthDayRule},
            {"context", "explicit_date_units", kMonthDayRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_month_day",
    };
}

RuleApproval approve_month_period(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || range.end - range.start < 3
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_month_period"};
    }
    size_t month_mark = range.start;
    while (month_mark < range.end
        && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
        ++month_mark;
    }
    if (month_mark == range.start
        || (month_mark + 2 != range.end && month_mark + 3 != range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_period_shape"};
    }
    const std::string period = unicode_scalar_substring(
        input, *boundaries, month_mark + 1, range.end
    );
    const auto month = parse_spoken_cardinal(
        input, *boundaries, range.start, month_mark
    );
    if (!month.has_value() || *month < 1 || *month > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start, month_mark, *month
        )
        || (period != "上旬" && period != "中旬" && period != "下旬"
            && period != "单月" && period != "首宗" && period != "首批"
            && period != "首个" && period != "首次" && period != "底")) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_period_value"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || contains_any("十百千至到", {previous})) {
            return {RuleDecision::preserve, "", {}, {}, "month_period_continues_left"};
        }
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || contains_any("年月日号", {next})) {
            return {RuleDecision::preserve, "", {}, {}, "month_period_continues_right"};
        }
    }
    const std::string replacement = std::to_string(*month) + "月" + period;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_month_period", kMonthPeriodRule},
            {"boundary", "complete_month_period", kMonthPeriodRule},
            {"context", "explicit_month_and_period_unit", kMonthPeriodRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_month_period",
    };
}

RuleApproval approve_month_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_range"};
    }
    size_t first_mark = range.start;
    while (first_mark < range.end && unicode_scalar_substring(
            input, *boundaries, first_mark, first_mark + 1
        ) != "月") {
        ++first_mark;
    }
    if (first_mark == range.start || first_mark + 3 > range.end) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_range_shape"};
    }
    const std::string connector = unicode_scalar_substring(
        input, *boundaries, first_mark + 1, first_mark + 2
    );
    if (connector != "到" && connector != "至"
        || unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) != "月") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_range_connector"};
    }
    const auto first = parse_spoken_cardinal(
        input, *boundaries, range.start, first_mark
    );
    const auto second = parse_spoken_cardinal(
        input, *boundaries, first_mark + 2, range.end - 1
    );
    if (!first.has_value() || !second.has_value()
        || *first < 1 || *first > 12 || *second < 1 || *second > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start, first_mark, *first
        )
        || !canonical_spoken_calendar_component(
            input, *boundaries, first_mark + 2, range.end - 1, *second
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_month_range_value"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年") {
            return {RuleDecision::preserve, "", {}, {}, "month_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || contains_any("日号份旬季", {next})) {
            return {RuleDecision::preserve, "", {}, {}, "month_range_continues_right"};
        }
    }
    const std::string replacement = std::to_string(*first) + "月" + connector
        + std::to_string(*second) + "月";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "two_spoken_months_with_range_connector", kMonthRangeRule},
            {"boundary", "complete_month_range", kMonthRangeRule},
            {"context", "explicit_month_units_and_range_connector", kMonthRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_month_range",
    };
}

std::optional<std::string> relative_year_anchor(const std::string& value) {
    for (const std::string anchor : {
            "今年", "明年", "去年", "前年", "后年", "次年", "当年", "每年",
        }) {
        if (value == anchor) return anchor;
    }
    return std::nullopt;
}

bool contextual_followup_month_applies(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    if (start >= 2 && unicode_scalar_substring(
            input, boundaries, start - 2, start
        ) == "截至") {
        return true;
    }
    if (start >= 2 && unicode_scalar_substring(
            input, boundaries, start - 2, start
        ) == "此后") {
        return true;
    }
    if (start == 0 || unicode_scalar_substring(
            input, boundaries, start - 1, start
        ) != "在") {
        return false;
    }
    size_t sentence_start = start - 1;
    while (sentence_start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, boundaries, sentence_start - 1, sentence_start
        );
        if (contains_any("。！？；\n", {previous})) break;
        --sentence_start;
    }
    for (size_t index = sentence_start; index + 3 < start; ++index) {
        if (!relative_year_anchor(unicode_scalar_substring(
                input, boundaries, index, index + 2
            )).has_value()) {
            continue;
        }
        size_t month_mark = index + 2;
        while (month_mark < start && month_mark - index <= 4
            && unicode_scalar_substring(
                input, boundaries, month_mark, month_mark + 1
            ) != "月") {
            ++month_mark;
        }
        if (month_mark >= start || month_mark == index + 2) continue;
        const std::string month_text = unicode_scalar_substring(
            input, boundaries, index + 2, month_mark
        );
        const auto spoken = parse_spoken_cardinal(
            input, boundaries, index + 2, month_mark
        );
        int month = 0;
        if (spoken.has_value()) {
            month = *spoken;
        } else if (month_text.size() <= 2
            && std::all_of(month_text.begin(), month_text.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            })) {
            month = std::stoi(month_text);
        }
        if (month >= 1 && month <= 12) return true;
    }
    return false;
}

RuleApproval approve_contextual_followup_month(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (range.end < range.start + 2
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) != "月"
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || !contextual_followup_month_applies(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_followup_month"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || contains_any("日号份旬季", {next})) {
            return {RuleDecision::preserve, "", {}, {}, "followup_month_continues_right"};
        }
    }
    const auto month = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end - 1
    );
    if (!month.has_value() || *month < 1 || *month > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start, range.end - 1, *month
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_followup_month"};
    }
    const std::string replacement = std::to_string(*month) + "月";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "complete_spoken_month", kContextualFollowupMonthRule},
            {"boundary", "complete_contextual_followup_month", kContextualFollowupMonthRule},
            {"context", "explicit_followup_month_anchor", kContextualFollowupMonthRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "contextual_followup_month",
    };
}

RuleApproval approve_relative_year_month(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (range.end - range.start < 4
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_relative_year_month"};
    }
    const auto anchor = relative_year_anchor(unicode_scalar_substring(
        input, *boundaries, range.start, range.start + 2
    ));
    const bool month_part_suffix = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    ) == "份";
    const size_t month_mark = month_part_suffix ? range.end - 2 : range.end - 1;
    const size_t month_start = range.start + 2;
    if (!anchor.has_value() || month_mark <= month_start
        || unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_relative_year_month_shape"};
    }
    const auto month = parse_spoken_cardinal(
        input, *boundaries, month_start, month_mark
    );
    if (!month.has_value() || *month < 1 || *month > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, month_start, month_mark, *month
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_relative_year_month_value"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next)
            || next == "日" || next == "号" || next == "旬" || next == "季") {
            return {RuleDecision::preserve, "", {}, {}, "relative_year_month_continues_right"};
        }
    }
    const std::string replacement = *anchor + std::to_string(*month) + "月"
        + (month_part_suffix ? "份" : "");
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "relative_year_anchor_plus_spoken_month", kRelativeYearMonthRule},
            {"boundary", "complete_relative_year_month", kRelativeYearMonthRule},
            {"context", "released_relative_year_anchor", kRelativeYearMonthRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_relative_calendar_month",
    };
}

RuleApproval approve_recurring_cross_year_month_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || range.end - range.start < 9
        || unicode_scalar_substring(
            input, *boundaries, range.start, range.start + 2
        ) != "每年") {
        return {
            RuleDecision::preserve, "", {}, {},
            "not_complete_recurring_cross_year_month_range",
        };
    }
    size_t first_month_mark = range.start + 2;
    while (first_month_mark < range.end
        && first_month_mark - (range.start + 2) <= 2
        && unicode_scalar_substring(
            input, *boundaries, first_month_mark, first_month_mark + 1
        ) != "月") {
        ++first_month_mark;
    }
    if (first_month_mark >= range.end || first_month_mark == range.start + 2) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_first_recurring_month"};
    }
    size_t connector_index = first_month_mark + 1;
    const bool first_month_part = connector_index < range.end
        && unicode_scalar_substring(
            input, *boundaries, connector_index, connector_index + 1
        ) == "份";
    if (first_month_part) ++connector_index;
    if (connector_index + 4 > range.end) {
        return {RuleDecision::preserve, "", {}, {}, "missing_next_year_month"};
    }
    const std::string connector = unicode_scalar_substring(
        input, *boundaries, connector_index, connector_index + 1
    );
    if ((connector != "到" && connector != "至")
        || unicode_scalar_substring(
            input, *boundaries, connector_index + 1, connector_index + 3
        ) != "次年") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_recurring_range_bridge"};
    }
    const size_t second_month_start = connector_index + 3;
    const bool second_month_part = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    ) == "份";
    const size_t second_month_mark = second_month_part ? range.end - 2 : range.end - 1;
    if (second_month_mark <= second_month_start
        || unicode_scalar_substring(
            input, *boundaries, second_month_mark, second_month_mark + 1
        ) != "月") {
        return {RuleDecision::preserve, "", {}, {}, "invalid_second_recurring_month"};
    }
    const auto first_month = parse_spoken_cardinal(
        input, *boundaries, range.start + 2, first_month_mark
    );
    const auto second_month = parse_spoken_cardinal(
        input, *boundaries, second_month_start, second_month_mark
    );
    if (!first_month.has_value() || !second_month.has_value()
        || *first_month < 1 || *first_month > 12
        || *second_month < 1 || *second_month > 12
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start + 2, first_month_mark, *first_month
        )
        || !canonical_spoken_calendar_component(
            input, *boundaries, second_month_start, second_month_mark, *second_month
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_recurring_month_value"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next)
            || next == "日" || next == "号" || next == "旬" || next == "季") {
            return {RuleDecision::preserve, "", {}, {}, "recurring_month_range_continues_right"};
        }
    }
    const std::string replacement = "每年" + std::to_string(*first_month) + "月"
        + (first_month_part ? "份" : "") + connector + "次年"
        + std::to_string(*second_month) + "月" + (second_month_part ? "份" : "");
    return {
        RuleDecision::approve,
        replacement,
        {
            {
                "shape", "every_year_month_to_next_year_month",
                kRecurringCrossYearMonthRangeRule,
            },
            {
                "boundary", "complete_recurring_cross_year_month_range",
                kRecurringCrossYearMonthRangeRule,
            },
            {
                "context", "explicit_every_year_and_next_year_anchors",
                kRecurringCrossYearMonthRangeRule,
            },
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_recurring_cross_year_month_range",
    };
}

RuleApproval approve_shared_year_date_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_shared_year_date_range"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年") {
            return {RuleDecision::preserve, "", {}, {},
                    "shared_year_date_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 3)
        );
        if (date_or_number_continuation(next)
            || (range_connector(next) && !starts_with_any(right, {"到期"}))
            || next == "代"
            || starts_with_any(
                right,
                {"上午", "下午", "中午", "凌晨", "早上", "清晨", "晚上", "傍晚", "夜里", "夜间"}
            )) {
            return {RuleDecision::preserve, "", {}, {},
                    "shared_year_date_range_continues_right"};
        }
    }
    size_t connector_index = range.end;
    std::string connector;
    for (size_t index = range.start; index < range.end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, index, index + 1
        );
        if (value != "到" && value != "至") continue;
        if (connector_index != range.end) {
            return {RuleDecision::preserve, "", {}, {},
                    "multiple_shared_year_date_range_connectors"};
        }
        connector_index = index;
        connector = value;
    }
    if (connector_index == range.start || connector_index + 1 >= range.end) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_shared_year_date_range_endpoint"};
    }
    const std::string first_text = unicode_scalar_substring(
        input, *boundaries, range.start, connector_index
    );
    const std::string second_text = unicode_scalar_substring(
        input, *boundaries, connector_index + 1, range.end
    );
    const SafePolicyContext endpoint_context = {"", "", false, false};
    const TransformationCandidate first_candidate = {
        "zh-shared-year-date-range-first",
        {0, connector_index - range.start, first_text},
        "zh-CN", "date_time", "explicit_full_date",
        kFullDateRule, kFullDateRuleVersion,
    };
    const TransformationCandidate second_candidate = {
        "zh-shared-year-date-range-second",
        {0, range.end - connector_index - 1, second_text},
        "zh-CN", "date_time", "explicit_month_day",
        kMonthDayRule, kMonthDayRuleVersion,
    };
    const auto first = approve_full_date(first_text, endpoint_context, first_candidate);
    const auto second = approve_month_day(second_text, endpoint_context, second_candidate);
    if (first.decision != RuleDecision::approve || second.decision != RuleDecision::approve) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_year_date_range_endpoint"};
    }
    const auto first_boundaries = unicode_scalar_boundaries(first_text);
    const auto second_boundaries = unicode_scalar_boundaries(second_text);
    if (!first_boundaries.has_value() || !second_boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_endpoint_utf8"};
    }
    const size_t first_length = first_boundaries->size() - 1;
    const size_t second_length = second_boundaries->size() - 1;
    size_t first_month_mark = 5;
    while (first_month_mark < first_length && unicode_scalar_substring(
            first_text, *first_boundaries, first_month_mark, first_month_mark + 1
        ) != "月") {
        ++first_month_mark;
    }
    size_t second_month_mark = 0;
    while (second_month_mark < second_length && unicode_scalar_substring(
            second_text, *second_boundaries, second_month_mark, second_month_mark + 1
        ) != "月") {
        ++second_month_mark;
    }
    const auto first_month = parse_spoken_cardinal(
        first_text, *first_boundaries, 5, first_month_mark
    );
    const auto first_day = parse_spoken_cardinal(
        first_text, *first_boundaries, first_month_mark + 1, first_length - 1
    );
    const auto second_month = parse_spoken_cardinal(
        second_text, *second_boundaries, 0, second_month_mark
    );
    const auto second_day = parse_spoken_cardinal(
        second_text, *second_boundaries, second_month_mark + 1, second_length - 1
    );
    int year = 0;
    for (size_t index = 0; index < 4; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            first_text, *first_boundaries, index, index + 1
        ));
        if (!digit.has_value()) {
            return {RuleDecision::preserve, "", {}, {},
                    "invalid_shared_year_date_range_year"};
        }
        year = year * 10 + *digit;
    }
    if (!first_month.has_value() || !first_day.has_value()
        || !second_month.has_value() || !second_day.has_value()
        || !valid_calendar_day(year, *second_month, *second_day, true)
        || std::pair(*first_month, *first_day) >= std::pair(*second_month, *second_day)) {
        return {RuleDecision::preserve, "", {}, {},
                "shared_year_date_range_not_strictly_ascending"};
    }
    const std::string replacement = first.parsed_value + connector + second.parsed_value;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "full_date_plus_complete_month_day", kSharedYearDateRangeRule},
            {"boundary", "complete_shared_year_date_range", kSharedYearDateRangeRule},
            {"context", "explicit_date_units_and_range_connector", kSharedYearDateRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_shared_year_date_range",
    };
}

RuleApproval approve_shared_month_date_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "not_complete_shared_month_date_range"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年" || range_connector(previous)) {
            return {RuleDecision::preserve, "", {}, {},
                    "shared_month_date_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 3)
        );
        if ((date_or_number_continuation(next) && !starts_with_any(right, {"陆续"}))
            || (range_connector(next) && !starts_with_any(right, {"到期"}))
            || next == "代"
            || starts_with_any(
                right,
                {"上午", "下午", "中午", "凌晨", "早上", "清晨", "晚上", "傍晚", "夜里", "夜间"}
            )) {
            return {RuleDecision::preserve, "", {}, {},
                    "shared_month_date_range_continues_right"};
        }
    }
    size_t month_mark = range.start;
    while (month_mark < range.end && unicode_scalar_substring(
            input, *boundaries, month_mark, month_mark + 1
        ) != "月") {
        ++month_mark;
    }
    if (month_mark == range.start || month_mark >= range.end) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_month_date_range_month"};
    }
    size_t connector_index = range.end;
    std::string connector;
    for (size_t index = month_mark + 1; index < range.end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, index, index + 1
        );
        if (value != "到" && value != "至") continue;
        if (connector_index != range.end) {
            return {RuleDecision::preserve, "", {}, {},
                    "multiple_shared_month_date_range_connectors"};
        }
        connector_index = index;
        connector = value;
    }
    if (connector_index <= month_mark + 2 || connector_index + 2 >= range.end) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_shared_month_date_range_endpoint"};
    }
    const std::string first_suffix = unicode_scalar_substring(
        input, *boundaries, connector_index - 1, connector_index
    );
    const std::string second_suffix = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    );
    if ((first_suffix != "日" && first_suffix != "号")
        || (second_suffix != "日" && second_suffix != "号")) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_month_date_range_suffix"};
    }
    const auto month = parse_spoken_cardinal(
        input, *boundaries, range.start, month_mark
    );
    const auto first_day = parse_spoken_cardinal(
        input, *boundaries, month_mark + 1, connector_index - 1
    );
    const auto second_day = parse_spoken_cardinal(
        input, *boundaries, connector_index + 1, range.end - 1
    );
    if (!month.has_value() || !first_day.has_value() || !second_day.has_value()
        || !canonical_spoken_calendar_component(
            input, *boundaries, range.start, month_mark, *month
        )
        || !canonical_spoken_calendar_component(
            input, *boundaries, month_mark + 1, connector_index - 1, *first_day
        )
        || !canonical_spoken_calendar_component(
            input, *boundaries, connector_index + 1, range.end - 1, *second_day
        )
        || unicode_scalar_substring(input, *boundaries, range.start, range.end)
            .find("两") != std::string::npos
        || !valid_calendar_day(0, *month, *first_day, false)
        || !valid_calendar_day(0, *month, *second_day, false)
        || *first_day >= *second_day) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_month_date_range_value"};
    }
    const std::string replacement = std::to_string(*month) + "月"
        + std::to_string(*first_day) + first_suffix + connector
        + std::to_string(*second_day) + second_suffix;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "complete_month_day_plus_day", kSharedMonthDateRangeRule},
            {"boundary", "complete_shared_month_date_range", kSharedMonthDateRangeRule},
            {"context", "explicit_date_units_and_range_connector", kSharedMonthDateRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_shared_month_date_range",
    };
}

RuleApproval approve_full_date_shared_month_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || literal_scope_before_year(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {},
                "not_complete_full_date_shared_month_range"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || previous == "年" || range_connector(previous)) {
            return {RuleDecision::preserve, "", {}, {},
                    "full_date_shared_month_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 3)
        );
        if (date_or_number_continuation(next)
            || (range_connector(next) && !starts_with_any(right, {"到期"}))
            || starts_with_any(right, {
                "上午", "下午", "中午", "凌晨", "早上", "清晨", "晚上",
                "傍晚", "夜里", "夜间",
            })) {
            return {RuleDecision::preserve, "", {}, {},
                    "full_date_shared_month_range_continues_right"};
        }
    }
    size_t connector_index = range.end;
    std::string connector;
    for (size_t index = range.start; index < range.end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, index, index + 1
        );
        if (value != "到" && value != "至") continue;
        if (connector_index != range.end) {
            return {RuleDecision::preserve, "", {}, {},
                    "multiple_full_date_shared_month_range_connectors"};
        }
        connector_index = index;
        connector = value;
    }
    if (connector_index == range.start || connector_index + 2 >= range.end) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_full_date_shared_month_range_endpoint"};
    }
    const std::string first_text = unicode_scalar_substring(
        input, *boundaries, range.start, connector_index
    );
    const std::string second_text = unicode_scalar_substring(
        input, *boundaries, connector_index + 1, range.end
    );
    const SafePolicyContext endpoint_context = {"", "", false, false};
    const TransformationCandidate first_candidate = {
        "zh-full-date-shared-month-range-first",
        {0, connector_index - range.start, first_text},
        "zh-CN", "date_time", "explicit_full_date",
        kFullDateRule, kFullDateRuleVersion,
    };
    const auto first = approve_full_date(first_text, endpoint_context, first_candidate);
    const auto first_boundaries = unicode_scalar_boundaries(first_text);
    const auto second_boundaries = unicode_scalar_boundaries(second_text);
    if (first.decision != RuleDecision::approve
        || !first_boundaries.has_value() || !second_boundaries.has_value()) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_full_date_shared_month_range_endpoint"};
    }
    const size_t first_length = first_boundaries->size() - 1;
    const size_t second_length = second_boundaries->size() - 1;
    if (first_length < 9 || second_length < 2) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_full_date_shared_month_range_shape"};
    }
    size_t first_month_start = 5;
    if (unicode_scalar_substring(
            first_text, *first_boundaries, first_month_start, first_month_start + 1
        ) == "的") {
        ++first_month_start;
    }
    size_t first_month_mark = first_month_start;
    while (first_month_mark < first_length && unicode_scalar_substring(
            first_text, *first_boundaries, first_month_mark, first_month_mark + 1
        ) != "月") {
        ++first_month_mark;
    }
    const std::string first_suffix = unicode_scalar_substring(
        first_text, *first_boundaries, first_length - 1, first_length
    );
    const std::string second_suffix = unicode_scalar_substring(
        second_text, *second_boundaries, second_length - 1, second_length
    );
    const auto month = parse_spoken_cardinal(
        first_text, *first_boundaries, first_month_start, first_month_mark
    );
    const auto first_day = parse_spoken_cardinal(
        first_text, *first_boundaries, first_month_mark + 1, first_length - 1
    );
    const auto second_day = parse_spoken_cardinal(
        second_text, *second_boundaries, 0, second_length - 1
    );
    int year = 0;
    for (size_t index = 0; index < 4; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            first_text, *first_boundaries, index, index + 1
        ));
        if (!digit.has_value()) {
            return {RuleDecision::preserve, "", {}, {},
                    "invalid_full_date_shared_month_range_year"};
        }
        year = year * 10 + *digit;
    }
    if (first_month_mark >= first_length
        || (first_suffix != "日" && first_suffix != "号")
        || (second_suffix != "日" && second_suffix != "号")
        || !month.has_value() || !first_day.has_value() || !second_day.has_value()
        || !canonical_spoken_calendar_component(
            second_text, *second_boundaries, 0, second_length - 1, *second_day
        )
        || !valid_calendar_day(year, *month, *second_day, true)
        || *first_day >= *second_day) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_full_date_shared_month_range_value"};
    }
    const std::string replacement = first.parsed_value + connector
        + std::to_string(*second_day) + second_suffix;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "full_date_plus_same_month_day", kFullDateSharedMonthRangeRule},
            {"boundary", "complete_full_date_shared_month_range",
             kFullDateSharedMonthRangeRule},
            {"context", "explicit_date_units_and_shared_month_range_connector",
             kFullDateSharedMonthRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_full_date_shared_month_range",
    };
}

std::optional<std::string> parse_percentage_value(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t decimal_mark = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != end) return std::nullopt;
            decimal_mark = index;
        }
    }
    const auto integer = parse_spoken_cardinal(input, boundaries, start, decimal_mark);
    if (!integer.has_value() || *integer > 1000) return std::nullopt;
    std::string value = std::to_string(*integer);
    if (decimal_mark == end) return value;
    if (decimal_mark + 1 == end) return std::nullopt;
    value += ".";
    for (size_t index = decimal_mark + 1; index < end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        value += static_cast<char>('0' + *digit);
    }
    return value;
}

RuleApproval approve_percentage(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || range.end - range.start < 4
        || unicode_scalar_substring(input, *boundaries, range.start, range.start + 3)
            != "百分之"
        || incomplete_cross_segment_span(
            context,
            boundaries->size() - 1,
            range.start,
            range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "not_explicit_percentage"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (previous == "千" || previous == "万") {
            return {RuleDecision::preserve, "", {}, {}, "fraction_prefix_overlap"};
        }
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input,
            *boundaries,
            range.end,
            std::min(boundaries->size() - 1, range.end + 2)
        );
        const std::string suffix = unicode_scalar_substring(
            input,
            *boundaries,
            range.end,
            std::min(boundaries->size() - 1, range.end + 4)
        );
        const bool adjacent_ascii = next.size() == 1
            && std::isalnum(static_cast<unsigned char>(next[0])) != 0;
        if (number_continuation(next) || adjacent_ascii || range_connector(next)
            || next == "点"
            || next == "几" || next == "多"
            || right == "分之" || starts_with_any(suffix, {"个百分点", "百分点"})) {
            return {RuleDecision::preserve, "", {}, {}, "percentage_continues_right"};
        }
    }
    const auto value = parse_percentage_value(
        input, *boundaries, range.start + 3, range.end
    );
    if (!value.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_percentage_value"};
    }
    const std::string replacement = *value + "%";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "percent_prefix_spoken_number", kPercentageRule},
            {"boundary", "complete_percentage", kPercentageRule},
            {"context", "explicit_percentage_marker", kPercentageRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_percentage",
    };
}

std::optional<std::pair<size_t, std::string>> decimal_duration_unit(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (const std::string unit : {"毫秒", "分钟", "小时", "秒钟", "秒"}) {
        const size_t length = unit == "秒" ? 1 : 2;
        if (end >= start + length
            && unicode_scalar_substring(input, boundaries, end - length, end) == unit) {
            return std::make_pair(end - length, unit);
        }
    }
    return std::nullopt;
}

std::optional<std::string> parse_explicit_decimal_duration(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = decimal_duration_unit(input, boundaries, start, end);
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    size_t decimal_mark = unit->first;
    for (size_t index = start; index < unit->first; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != unit->first) return std::nullopt;
            decimal_mark = index;
        }
    }
    if (decimal_mark == unit->first || decimal_mark == start
        || decimal_mark + 1 == unit->first || unit->first - decimal_mark - 1 > 3) {
        return std::nullopt;
    }
    const auto integer = parse_spoken_cardinal(input, boundaries, start, decimal_mark);
    if (!integer.has_value() || *integer > 999) return std::nullopt;
    std::string replacement = std::to_string(*integer) + ".";
    for (size_t index = decimal_mark + 1; index < unit->first; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        replacement += static_cast<char>('0' + *digit);
    }
    return replacement + unit->second;
}

std::optional<std::pair<size_t, std::string>> decimal_measure_unit(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (const std::string unit : {
            "太拉弗洛普斯", "太字节每秒", "米每二次方秒", "克每百毫升", "兆比特每秒", "公里每小时", "平方公里", "平方米", "立方厘米", "毫米汞柱", "毫安时", "米每秒", "立方米", "牛顿米", "吉赫兹",
            "兆赫兹", "千赫兹", "帧每秒", "核神经引擎", "核CPU", "核GPU",
            "吉比特", "吉字节", "兆字节", "太字节", "毫摩尔每升", "纳米", "微米", "微克", "GB", "TB",
            "个百分点", "百分点",
            "万股", "公斤", "千克", "公顷",
            "赫兹", "公里", "厘米", "毫米", "毫升", "毫克", "兆帕", "百帕",
            "伏特", "安培", "瓦特",
            "毫安", "千伏", "千瓦", "兆瓦", "比特", "字节", "米", "吨", "瓦", "伏",
            "个ppm", "个ppb", "分贝", "帧", "倍", "斤", "个", "万", "亿",
        }) {
        const auto unit_boundaries = unicode_scalar_boundaries(unit);
        if (!unit_boundaries.has_value()) continue;
        const size_t length = unit_boundaries->size() - 1;
        if (end >= start + length
            && unicode_scalar_substring(input, boundaries, end - length, end) == unit) {
            return std::make_pair(end - length, unit);
        }
    }
    return std::nullopt;
}

std::string measure_output_unit(const std::string& unit) {
    if (unit == "太拉弗洛普斯") return " TFLOPS";
    if (unit == "个ppm") return " ppm";
    if (unit == "个ppb") return " ppb";
    if (unit == "克每百毫升") return "克/100毫升";
    return unit;
}

std::optional<int> parse_spoken_money_integer(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
);

std::optional<std::string> parse_explicit_decimal_measure(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = decimal_measure_unit(input, boundaries, start, end);
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    size_t decimal_mark = unit->first;
    for (size_t index = start; index < unit->first; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != unit->first) return std::nullopt;
            decimal_mark = index;
        }
    }
    if (decimal_mark == unit->first || decimal_mark == start
        || decimal_mark + 1 == unit->first || unit->first - decimal_mark - 1 > 3) {
        return std::nullopt;
    }
    const auto integer = unit->second == "平方米"
        ? parse_spoken_money_integer(input, boundaries, start, decimal_mark)
        : parse_spoken_cardinal(input, boundaries, start, decimal_mark);
    const int maximum = unit->second == "平方米" ? 99999999
        : unit->second == "万股" ? 9999 : 999;
    if (!integer.has_value() || *integer > maximum) return std::nullopt;
    std::string replacement = std::to_string(*integer) + ".";
    for (size_t index = decimal_mark + 1; index < unit->first; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        replacement += static_cast<char>('0' + *digit);
    }
    return replacement + measure_output_unit(unit->second);
}

std::optional<int64_t> parse_spoken_large_count_integer(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (start >= end) return std::nullopt;
    size_t yi = end;
    size_t wan = end;
    for (size_t index = start; index < end; ++index) {
        const std::string scalar = unicode_scalar_substring(
            input, boundaries, index, index + 1
        );
        if (scalar == "亿") {
            if (yi != end || wan != end) return std::nullopt;
            yi = index;
        } else if (scalar == "万") {
            if (wan != end) return std::nullopt;
            wan = index;
        }
    }
    if (yi == end && wan == end) {
        const auto value = parse_spoken_cardinal(input, boundaries, start, end);
        return value.has_value() ? std::optional<int64_t>(*value) : std::nullopt;
    }
    if (yi != end && (yi == start || (wan != end && wan < yi))) return std::nullopt;

    auto parse_section = [&](size_t section_start, size_t section_end) -> std::optional<int> {
        if (section_start >= section_end) return std::nullopt;
        const std::string first = unicode_scalar_substring(
            input, boundaries, section_start, section_start + 1
        );
        if (first == "零" || first == "〇") ++section_start;
        if (section_start >= section_end) return std::nullopt;
        return parse_spoken_cardinal(input, boundaries, section_start, section_end);
    };

    int64_t total = 0;
    size_t cursor = start;
    if (yi != end) {
        const auto high = parse_section(cursor, yi);
        if (!high.has_value() || *high <= 0 || *high > 9999) return std::nullopt;
        total = static_cast<int64_t>(*high) * 100000000;
        cursor = yi + 1;
    }
    if (wan != end) {
        const auto middle = parse_section(cursor, wan);
        if (!middle.has_value() || *middle <= 0 || *middle >= 10000) return std::nullopt;
        total += static_cast<int64_t>(*middle) * 10000;
        cursor = wan + 1;
    }
    if (cursor < end) {
        const auto low = parse_section(cursor, end);
        if (!low.has_value() || *low >= 10000) return std::nullopt;
        total += *low;
    }
    return total;
}

std::optional<std::string> format_spoken_large_count_integer(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (!parse_spoken_large_count_integer(input, boundaries, start, end).has_value()) {
        return std::nullopt;
    }
    size_t yi = end;
    size_t wan = end;
    for (size_t index = start; index < end; ++index) {
        const std::string scalar = unicode_scalar_substring(
            input, boundaries, index, index + 1
        );
        if (scalar == "亿") yi = index;
        if (scalar == "万") wan = index;
    }
    if (yi == end && wan == end) {
        const auto value = parse_spoken_cardinal(input, boundaries, start, end);
        return value.has_value() ? std::optional<std::string>(std::to_string(*value))
                                 : std::nullopt;
    }
    auto parse_section = [&](size_t section_start, size_t section_end) -> std::optional<int> {
        if (section_start < section_end && contains_any("零〇", {
                unicode_scalar_substring(input, boundaries, section_start, section_start + 1)
            })) {
            ++section_start;
        }
        if (section_start >= section_end) return std::nullopt;
        return parse_spoken_cardinal(input, boundaries, section_start, section_end);
    };
    std::string result;
    size_t cursor = start;
    if (yi != end) {
        const auto high = parse_section(cursor, yi);
        if (!high.has_value()) return std::nullopt;
        result = std::to_string(*high) + "亿";
        cursor = yi + 1;
    }
    if (wan != end) {
        const auto middle = parse_section(cursor, wan);
        if (!middle.has_value()) return std::nullopt;
        result += std::to_string(*middle) + "万";
        cursor = wan + 1;
    }
    if (cursor < end) {
        const auto low = parse_section(cursor, end);
        if (!low.has_value()) return std::nullopt;
        result += std::to_string(*low);
    }
    return result;
}

std::optional<int> parse_spoken_money_integer(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t wan = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "万") {
            if (wan != end) return std::nullopt;
            wan = index;
        }
    }
    if (wan == end) return parse_spoken_cardinal(input, boundaries, start, end);
    if (wan == start) return std::nullopt;
    const auto high = parse_spoken_cardinal(input, boundaries, start, wan);
    if (!high.has_value() || *high <= 0 || *high > 9999) return std::nullopt;
    if (wan + 1 == end) return *high * 10000;
    const std::string first = unicode_scalar_substring(input, boundaries, wan + 1, wan + 2);
    const bool explicit_zero = first == "零" || first == "〇";
    const size_t low_start = explicit_zero ? wan + 2 : wan + 1;
    if (low_start == end) return std::nullopt;
    const auto low = parse_spoken_cardinal(input, boundaries, low_start, end);
    if (!low.has_value() || *low >= 10000) return std::nullopt;
    if (explicit_zero ? *low >= 1000 : *low < 1000) return std::nullopt;
    return *high * 10000 + *low;
}

std::optional<std::string> format_spoken_money_integer(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t wan = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "万") {
            if (wan != end) return std::nullopt;
            wan = index;
        }
    }
    if (wan == end) {
        const auto value = parse_spoken_cardinal(input, boundaries, start, end);
        if (!value.has_value()) return std::nullopt;
        return std::to_string(*value);
    }
    if (wan == start) return std::nullopt;
    const auto high = parse_spoken_cardinal(input, boundaries, start, wan);
    if (!high.has_value() || *high <= 0 || *high > 9999) return std::nullopt;
    std::string result = std::to_string(*high) + "万";
    if (wan + 1 == end) return result;
    const std::string first = unicode_scalar_substring(input, boundaries, wan + 1, wan + 2);
    const bool explicit_zero = first == "零" || first == "〇";
    const size_t low_start = explicit_zero ? wan + 2 : wan + 1;
    if (low_start == end) return std::nullopt;
    const auto low = parse_spoken_cardinal(input, boundaries, low_start, end);
    if (!low.has_value() || *low >= 10000) return std::nullopt;
    if (explicit_zero ? *low >= 1000 : *low < 1000) return std::nullopt;
    return result + std::to_string(*low);
}

std::optional<std::pair<size_t, std::string>> decimal_money_unit(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (const std::string unit : {
            "亿元人民币", "万元人民币", "亿美元", "亿元", "万元", "美元", "港元", "日元", "欧元", "元",
        }) {
        const size_t length = (unit == "亿元人民币" || unit == "万元人民币") ? 5
            : (unit == "亿美元" ? 3 : (unit == "元" ? 1 : 2));
        if (end >= start + length
            && unicode_scalar_substring(input, boundaries, end - length, end) == unit) {
            return std::make_pair(end - length, unit);
        }
    }
    return std::nullopt;
}

std::optional<std::string> parse_explicit_decimal_money(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = decimal_money_unit(input, boundaries, start, end);
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    size_t decimal_mark = unit->first;
    for (size_t index = start; index < unit->first; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != unit->first) return std::nullopt;
            decimal_mark = index;
        }
    }
    if (decimal_mark == unit->first || decimal_mark == start
        || decimal_mark + 1 == unit->first || unit->first - decimal_mark - 1 > 3) {
        return std::nullopt;
    }
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, decimal_mark
    );
    if (!integer.has_value() || *integer > 99999999) return std::nullopt;
    std::string replacement = std::to_string(*integer) + ".";
    for (size_t index = decimal_mark + 1; index < unit->first; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        replacement += static_cast<char>('0' + *digit);
    }
    return replacement + unit->second;
}

std::optional<std::string> parse_contextual_stock_price(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t kuai = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "块") {
            if (kuai != end) return std::nullopt;
            kuai = index;
        }
    }
    if (kuai == end || kuai == start || kuai + 1 >= end) return std::nullopt;
    const auto integer = parse_spoken_cardinal(input, boundaries, start, kuai);
    if (!integer.has_value() || *integer < 1 || *integer > 9999) return std::nullopt;

    int jiao = 0;
    int fen = 0;
    size_t cursor = kuai + 1;
    size_t mao = end;
    for (size_t index = cursor; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "毛") {
            mao = index;
            break;
        }
    }
    if (mao != end) {
        if (mao != cursor + 1) return std::nullopt;
        const auto value = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, cursor, cursor + 1
        ));
        if (!value.has_value()) return std::nullopt;
        jiao = *value;
        cursor = mao + 1;
        const bool explicit_fen = end > cursor
            && unicode_scalar_substring(input, boundaries, end - 1, end) == "分";
        const size_t digit_end = explicit_fen ? end - 1 : end;
        if (digit_end > cursor + 1) return std::nullopt;
        if (cursor < digit_end) {
            const auto value = spoken_year_digit(unicode_scalar_substring(
                input, boundaries, cursor, cursor + 1
            ));
            if (!value.has_value()) return std::nullopt;
            fen = *value;
        }
    } else {
        if (unicode_scalar_substring(input, boundaries, end - 1, end) != "分") {
            return std::nullopt;
        }
        const size_t digit_end = end - 1;
        if (digit_end <= cursor || digit_end > cursor + 2) return std::nullopt;
        int cents = 0;
        for (size_t index = cursor; index < digit_end; ++index) {
            const auto value = spoken_year_digit(unicode_scalar_substring(
                input, boundaries, index, index + 1
            ));
            if (!value.has_value()) return std::nullopt;
            cents = cents * 10 + *value;
        }
        fen = cents;
    }
    if (mao != end && cursor == end) {
        return std::to_string(*integer) + "." + std::to_string(jiao) + "元";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%d.%02d元", *integer, jiao * 10 + fen);
    return std::string(buffer);
}

std::optional<std::string> parse_explicit_integer_money(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = decimal_money_unit(input, boundaries, start, end);
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    for (size_t index = start; index < unit->first; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            return std::nullopt;
        }
    }
    size_t yi = unit->first;
    for (size_t index = start; index < unit->first; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "亿") {
            if (yi != unit->first) return std::nullopt;
            yi = index;
        }
    }
    if (yi != unit->first) {
        if (unit->second != "万元" && unit->second != "万元人民币"
            && unit->second != "元") {
            return std::nullopt;
        }
        const auto high = parse_spoken_cardinal(input, boundaries, start, yi);
        size_t low_start = yi + 1;
        if (low_start < unit->first && contains_any("零〇", {
                unicode_scalar_substring(input, boundaries, low_start, low_start + 1)
            })) {
            ++low_start;
        }
        const auto low = parse_spoken_cardinal(input, boundaries, low_start, unit->first);
        if (!high.has_value() || *high < 1 || *high > 9999
            || !low.has_value() || *low < 1 || *low > 9999) {
            return std::nullopt;
        }
        return std::to_string(*high) + "亿" + std::to_string(*low) + unit->second;
    }
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!integer.has_value() || *integer > 99999999) {
        return std::nullopt;
    }
    const auto formatted = format_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!formatted.has_value()) return std::nullopt;
    return *formatted + measure_output_unit(unit->second);
}

std::optional<std::pair<size_t, std::string>> explicit_integer_measure_unit(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (const std::string unit : {
            "太拉弗洛普斯", "太字节每秒", "米每二次方秒", "克每百毫升", "兆比特每秒", "千比特每秒", "千米每小时", "公里每小时", "平方公里", "平方米", "立方厘米", "毫米汞柱", "毫安时", "立方米", "牛顿米", "吉赫兹",
            "兆赫兹", "千赫兹", "帧每秒", "核神经引擎", "核CPU", "核GPU",
            "吉比特", "吉字节", "兆字节", "太字节", "毫摩尔每升", "纳米", "微米", "微克", "GB", "TB",
            "公斤", "千克", "公顷",
            "赫兹", "多公里", "公里", "千米", "厘米", "毫米", "毫升", "毫克", "兆帕", "百帕",
            "只私募产品", "个代表团", "个基点", "基点", "伏特", "安培", "瓦特",
            "毫安", "千伏", "千瓦", "兆瓦", "比特", "字节", "欧姆", "站台", "吨", "瓦", "伏", "安", "米",
            "名观众", "人次", "个人", "辆", "颗", "件", "批", "人", "箱", "股", "帧", "转", "斤", "步", "多个",
            "个ppm", "个ppb", "分贝", "度", "倍", "例", "届", "角", "点", "亿",
        }) {
        const auto unit_boundaries = unicode_scalar_boundaries(unit);
        if (!unit_boundaries.has_value()) continue;
        const size_t length = unit_boundaries->size() - 1;
        if (end >= start + length
            && unicode_scalar_substring(input, boundaries, end - length, end) == unit) {
            return std::make_pair(end - length, unit);
        }
    }
    return std::nullopt;
}

std::optional<std::string> parse_explicit_integer_measure(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = explicit_integer_measure_unit(input, boundaries, start, end);
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    const bool exact_count = unit->second == "件" || unit->second == "批"
        || unit->second == "人" || unit->second == "箱" || unit->second == "股"
        || unit->second == "例" || unit->second == "个代表团"
        || unit->second == "名观众";
    if (exact_count) {
        const bool approximate_people = unit->second == "人" && unit->first > start
            && unicode_scalar_substring(
                input, boundaries, unit->first - 1, unit->first
            ) == "多";
        const size_t numeric_end = unit->first - (approximate_people ? 1 : 0);
        const std::string number = unicode_scalar_substring(
            input, boundaries, start, numeric_end
        );
        if (unit->second == "人" && numeric_end > start
            && ends_with_any(number, {"万", "亿"})) {
            const std::string magnitude = unicode_scalar_substring(
                input, boundaries, numeric_end - 1, numeric_end
            );
            const std::string prefix = unicode_scalar_substring(
                input, boundaries, start, numeric_end - 1
            );
            if (!contains_any(prefix, {"万", "亿"})) {
                const auto high = parse_spoken_cardinal(
                    input, boundaries, start, numeric_end - 1
                );
                if (high.has_value() && *high >= 1 && *high <= 9999) {
                    return std::to_string(*high) + magnitude
                        + (approximate_people ? "多人" : unit->second);
                }
            }
        }
        const auto integer = parse_spoken_large_count_integer(
            input, boundaries, start, numeric_end
        );
        if (!integer.has_value() || *integer > 999999999999LL) return std::nullopt;
        if (unit->second == "股") {
            const auto formatted = format_spoken_large_count_integer(
                input, boundaries, start, numeric_end
            );
            if (!formatted.has_value()) return std::nullopt;
            return *formatted + unit->second;
        }
        return std::to_string(*integer)
            + (approximate_people ? "多人" : unit->second);
    }
    const std::string number = unicode_scalar_substring(
        input, boundaries, start, unit->first
    );
    if (number.find("亿") != std::string::npos) {
        const auto integer = parse_spoken_large_count_integer(
            input, boundaries, start, unit->first
        );
        const auto formatted = format_spoken_large_count_integer(
            input, boundaries, start, unit->first
        );
        if (!integer.has_value() || !formatted.has_value()) return std::nullopt;
        return *formatted + measure_output_unit(unit->second);
    }
    const auto integer = parse_spoken_money_integer(input, boundaries, start, unit->first);
    if (!integer.has_value() || *integer > 99999999) {
        return std::nullopt;
    }
    const auto formatted = format_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!formatted.has_value()) return std::nullopt;
    if ((unit->second == "辆" || unit->second == "颗")
        && formatted->find("万") != std::string::npos
        && !ends_with_any(*formatted, {"万"})) {
        return std::to_string(*integer) + unit->second;
    }
    if (unit->second == "亿") {
        return std::to_string(*integer) + unit->second;
    }
    return *formatted + measure_output_unit(unit->second);
}

std::optional<std::pair<size_t, std::string>> explicit_periodic_integer_measure_unit(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto released = explicit_integer_measure_unit(input, boundaries, start, end);
    if (released.has_value()) return released;
    for (const std::string unit : {"克", "米", "升"}) {
        if (end > start
            && unicode_scalar_substring(input, boundaries, end - 1, end) == unit) {
            return std::make_pair(end - 1, unit);
        }
    }
    return std::nullopt;
}

std::optional<std::string> parse_explicit_periodic_integer_measure(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = explicit_periodic_integer_measure_unit(
        input, boundaries, start, end
    );
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!integer.has_value() || *integer > 99999999) return std::nullopt;
    const auto formatted = format_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!formatted.has_value()) return std::nullopt;
    return *formatted + unit->second;
}

std::optional<std::pair<size_t, std::string>> explicit_integer_duration_unit(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (const std::string unit : {"毫秒", "分钟", "小时", "秒钟", "秒"}) {
        const size_t length = unit == "秒" ? 1 : 2;
        if (end >= start + length
            && unicode_scalar_substring(input, boundaries, end - length, end) == unit) {
            return std::make_pair(end - length, unit);
        }
    }
    return std::nullopt;
}

std::optional<std::string> parse_explicit_integer_duration(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto unit = explicit_integer_duration_unit(input, boundaries, start, end);
    if (!unit.has_value() || unit->first <= start) return std::nullopt;
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!integer.has_value() || *integer > 99999999) {
        return std::nullopt;
    }
    const auto formatted = format_spoken_money_integer(
        input, boundaries, start, unit->first
    );
    if (!formatted.has_value()) return std::nullopt;
    return *formatted + unit->second;
}

std::optional<std::string> parse_explicit_lock_period_months(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (end < start + 4
        || unicode_scalar_substring(input, boundaries, end - 2, end) != "个月") {
        return std::nullopt;
    }
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, end - 2
    );
    if (!integer.has_value() || *integer < 10 || *integer > 99999999) {
        return std::nullopt;
    }
    const auto formatted = format_spoken_money_integer(
        input, boundaries, start, end - 2
    );
    if (!formatted.has_value()) return std::nullopt;
    return *formatted + "个月";
}

std::optional<std::string> parse_explicit_anchored_year_or_day_duration(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    const std::string& unit,
    int64_t maximum
) {
    if (end <= start + 1
        || unicode_scalar_substring(input, boundaries, end - 1, end) != unit) {
        return std::nullopt;
    }
    const auto integer = parse_spoken_money_integer(input, boundaries, start, end - 1);
    if (!integer.has_value() || *integer < 10 || *integer > maximum) {
        return std::nullopt;
    }
    const auto formatted = format_spoken_money_integer(input, boundaries, start, end - 1);
    if (!formatted.has_value()) return std::nullopt;
    return *formatted + unit;
}

std::optional<std::string> parse_explicit_compound_duration(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (end < start + 6
        || unicode_scalar_substring(input, boundaries, end - 2, end) != "分钟") {
        return std::nullopt;
    }
    const size_t minute_end = end - 2;
    for (size_t hour_end = start + 1; hour_end + 3 <= end; ++hour_end) {
        if (unicode_scalar_substring(
                input, boundaries, hour_end, hour_end + 2
            ) != "小时") {
            continue;
        }
        const size_t minute_start = hour_end + 2;
        if (minute_start >= minute_end) return std::nullopt;
        const auto hours = parse_spoken_money_integer(
            input, boundaries, start, hour_end
        );
        const auto minutes = parse_spoken_money_integer(
            input, boundaries, minute_start, minute_end
        );
        if (!hours.has_value() || *hours < 10 || *hours > 99999999
            || !minutes.has_value() || *minutes < 10 || *minutes > 59) {
            return std::nullopt;
        }
        const auto formatted_hours = format_spoken_money_integer(
            input, boundaries, start, hour_end
        );
        const auto formatted_minutes = format_spoken_money_integer(
            input, boundaries, minute_start, minute_end
        );
        if (!formatted_hours.has_value() || !formatted_minutes.has_value()) {
            return std::nullopt;
        }
        return *formatted_hours + "小时" + *formatted_minutes + "分钟";
    }
    return std::nullopt;
}

std::optional<std::string> parse_explicit_minute_second_duration(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t suffix_length = 0;
    if (end >= start + 2
        && unicode_scalar_substring(input, boundaries, end - 2, end) == "秒钟") {
        suffix_length = 2;
    } else if (end > start
        && unicode_scalar_substring(input, boundaries, end - 1, end) == "秒") {
        suffix_length = 1;
    } else {
        return std::nullopt;
    }
    const size_t second_end = end - suffix_length;
    for (size_t minute_end = start + 1; minute_end + 3 <= end; ++minute_end) {
        if (unicode_scalar_substring(
                input, boundaries, minute_end, minute_end + 2
            ) != "分钟") {
            continue;
        }
        const size_t second_start = minute_end + 2;
        if (second_start >= second_end) return std::nullopt;
        const auto minutes = parse_spoken_money_integer(
            input, boundaries, start, minute_end
        );
        const auto seconds = parse_spoken_money_integer(
            input, boundaries, second_start, second_end
        );
        if (!minutes.has_value() || *minutes < 10 || *minutes > 99999999
            || !seconds.has_value() || *seconds < 10 || *seconds > 59) {
            return std::nullopt;
        }
        const auto formatted_minutes = format_spoken_money_integer(
            input, boundaries, start, minute_end
        );
        const auto formatted_seconds = format_spoken_money_integer(
            input, boundaries, second_start, second_end
        );
        if (!formatted_minutes.has_value() || !formatted_seconds.has_value()) {
            return std::nullopt;
        }
        return *formatted_minutes + "分钟" + *formatted_seconds
            + unicode_scalar_substring(input, boundaries, second_end, end);
    }
    return std::nullopt;
}

bool integer_unit_threshold_applies(const std::string& left, const std::string& right);
bool integer_unit_threshold_uncertainty_applies(
    const std::string& left,
    const std::string& right
);
bool explicit_unit_approximation_applies(const std::string& left, const std::string& right);
bool money_threshold_applies(const std::string& left, const std::string& right);
bool money_non_threshold_uncertainty_applies(
    const std::string& left,
    const std::string& right
);
bool money_approximation_applies(const std::string& left, const std::string& right);
bool money_predicate_connector_applies(const std::string& left);

RuleApproval approve_explicit_decimal_duration(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool approximate_candidate = candidate.rule_id == kDecimalDurationApproximateRule;
    const bool threshold_candidate = candidate.rule_id == kDecimalDurationThresholdRule;
    const std::string left_context = left_clause_context(
        input, *boundaries, range.start, 16
    );
    const std::string right_context = right_clause_context(
        input, *boundaries, range.end, 8
    );
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_duration_context"};
    }
    if (approximate_candidate && (integer_unit_threshold_applies(left_context, right_context)
            || !explicit_unit_approximation_applies(left_context, right_context))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_duration_approximation"};
    }
    if (threshold_candidate && (!integer_unit_threshold_applies(left_context, right_context)
            || integer_unit_threshold_uncertainty_applies(left_context, right_context))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_duration_threshold"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const std::string left = unicode_scalar_substring(
            input, *boundaries, range.start > 2 ? range.start - 2 : 0, range.start
        );
        const bool released_prefix = approximate_candidate
            && (previous == "约" || previous == "近");
        if (number_continuation(previous) || range_connector(previous)
            || (contains_any("点分秒钟约近", {previous}) && !released_prefix)
            || (previous == "时" && !ends_with_any(left, {"耗时", "用时"}))) {
            return {RuleDecision::preserve, "", {}, {}, "duration_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 2)
        );
        const bool released_suffix = approximate_candidate
            && starts_with_any(right, {"左右", "上下", "前后"});
        const bool released_threshold_suffix = threshold_candidate
            && starts_with_any(right, {"以上", "以下", "以内", "之内"});
        if (number_continuation(next) || range_connector(next)
            || contains_any("点时分秒钟半", {next})
            || (!released_suffix && !released_threshold_suffix && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "duration_continues_right"};
        }
    }
    const auto replacement = parse_explicit_decimal_duration(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_decimal_duration"};
    }
    const char* rule = approximate_candidate
        ? kDecimalDurationApproximateRule
        : threshold_candidate ? kDecimalDurationThresholdRule : kDecimalDurationRule;
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_decimal_plus_duration_unit", rule},
            {"boundary", approximate_candidate
                ? "complete_approximate_decimal_duration"
                : threshold_candidate
                    ? "complete_threshold_decimal_duration" : "complete_decimal_duration", rule},
            {"context", approximate_candidate
                ? "explicit_chinese_duration_approximation"
                : threshold_candidate
                    ? "explicit_chinese_duration_threshold" : "explicit_duration_unit", rule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        approximate_candidate
            ? "explicit_decimal_duration_approximation"
            : threshold_candidate
                ? "explicit_decimal_duration_threshold" : "explicit_decimal_duration",
    };
}

RuleApproval approve_explicit_decimal_measure(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool approximate_candidate = candidate.rule_id == kDecimalMeasureApproximateRule;
    const bool threshold_candidate = candidate.rule_id == kDecimalMeasureThresholdRule;
    const std::string left_context = left_clause_context(
        input, *boundaries, range.start, 16
    );
    const std::string right_context = right_clause_context(
        input, *boundaries, range.end, 8
    );
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_measure_context"};
    }
    const auto measure_unit = decimal_measure_unit(
        input, *boundaries, range.start, range.end
    );
    if (approximate_candidate && measure_unit.has_value()
        && measure_unit->second == "毫米汞柱") {
        return {RuleDecision::preserve, "", {}, {},
                "natural_approximate_blood_pressure_measure"};
    }
    if (approximate_candidate && (integer_unit_threshold_applies(left_context, right_context)
            || !explicit_unit_approximation_applies(left_context, right_context))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_measure_approximation"};
    }
    if (threshold_candidate && (!integer_unit_threshold_applies(left_context, right_context)
            || integer_unit_threshold_uncertainty_applies(left_context, right_context))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_measure_threshold"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool released_prefix = approximate_candidate
            && (previous == "约" || previous == "近");
        const bool released_threshold_prefix = threshold_candidate
            && ends_with_any(left_context, {
                "至少", "至多", "最多", "最少", "最低", "最高", "不到", "不足",
                "超过", "不超过", "不少于", "不低于", "高于", "低于",
            });
        if ((number_continuation(previous) || range_connector(previous))
                && !released_threshold_prefix
            || (contains_any("点约近", {previous}) && !released_prefix)) {
            return {RuleDecision::preserve, "", {}, {}, "measure_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 2)
        );
        const bool released_suffix = approximate_candidate
            && starts_with_any(right, {"左右", "上下", "前后"});
        const bool released_threshold_suffix = threshold_candidate
            && starts_with_any(right, {"以上", "以下", "以内", "之内"});
        if (number_continuation(next) || range_connector(next) || next == "钟"
            || contains_any("点每/／", {next})
            || (!released_suffix && !released_threshold_suffix && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "measure_continues_right"};
        }
    }
    const auto replacement = parse_explicit_decimal_measure(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_decimal_measure"};
    }
    const char* rule = approximate_candidate
        ? kDecimalMeasureApproximateRule
        : threshold_candidate ? kDecimalMeasureThresholdRule : kDecimalMeasureRule;
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_decimal_plus_measure_unit", rule},
            {"boundary", approximate_candidate
                ? "complete_approximate_decimal_measure"
                : threshold_candidate
                    ? "complete_threshold_decimal_measure" : "complete_decimal_measure", rule},
            {"context", approximate_candidate
                ? "explicit_chinese_measure_approximation"
                : threshold_candidate
                    ? "explicit_chinese_measure_threshold" : "released_chinese_measure_unit", rule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        approximate_candidate
            ? "explicit_decimal_measure_approximation"
            : threshold_candidate
                ? "explicit_decimal_measure_threshold" : "explicit_decimal_measure",
    };
}

RuleApproval approve_explicit_decimal_money(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool approximate_candidate = candidate.rule_id == kDecimalMoneyApproximateRule;
    const bool threshold_candidate = candidate.rule_id == kDecimalMoneyThresholdRule;
    const std::string left_context = left_clause_context(
        input, *boundaries, range.start, 16
    );
    const std::string right_context = right_clause_context(
        input, *boundaries, range.end, 8
    );
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_money_context"};
    }
    if (approximate_candidate && (money_threshold_applies(left_context, right_context)
            || !money_approximation_applies(left_context, right_context))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_money_approximation"};
    }
    if (threshold_candidate && (!money_threshold_applies(left_context, right_context)
            || money_non_threshold_uncertainty_applies(left_context, right_context))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_decimal_money_threshold"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool released_prefix = approximate_candidate
            && (previous == "约" || previous == "近");
        const bool released_constraint = approximate_candidate || threshold_candidate;
        if (number_continuation(previous)
            || (range_connector(previous) && !released_constraint
                && !money_predicate_connector_applies(left_context))
            || (contains_any("点约近", {previous}) && !released_prefix)
            || (!released_constraint && ends_with_any(left_context, {
                "大约", "约为", "大约在", "约在", "将近", "接近", "约合", "约合人民币",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "money_continues_left"};
        }
        const auto unit = decimal_money_unit(
            input, *boundaries, range.start, range.end
        );
        const bool rmb_unit = unit.has_value() && (unit->second == "元"
            || unit->second == "万元" || unit->second == "亿元"
            || unit->second == "万元人民币" || unit->second == "亿元人民币");
        if (!rmb_unit && ends_with_any(left_context, {"人民币", "约合人民币"})) {
            return {RuleDecision::preserve, "", {}, {}, "conflicting_currency_context"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 3)
        );
        const bool released_suffix = (approximate_candidate
                && starts_with_any(right, {"左右", "上下", "前后"}))
            || (threshold_candidate
                && starts_with_any(right, {
                    "以上", "以下", "以内", "之内", "及以上", "及以下",
                }))
            || (starts_with_any(right, {"人民币"})
                && contains_any(left_context, {"兑换"}));
        if (number_continuation(next) || range_connector(next) || next == "点" || next == "钱"
            || (!released_suffix && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几", "起",
                "人民币",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "money_continues_right"};
        }
    }
    const auto replacement = parse_explicit_decimal_money(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_decimal_money"};
    }
    const char* rule = approximate_candidate
        ? kDecimalMoneyApproximateRule
        : threshold_candidate ? kDecimalMoneyThresholdRule : kDecimalMoneyRule;
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_decimal_plus_currency_unit", rule},
            {"boundary", approximate_candidate
                ? "complete_approximate_decimal_money"
                : threshold_candidate
                    ? "complete_threshold_decimal_money" : "complete_decimal_money", rule},
            {"context", approximate_candidate
                ? "explicit_chinese_money_approximation"
                : threshold_candidate
                    ? "explicit_chinese_money_threshold"
                    : "released_chinese_currency_unit", rule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        approximate_candidate
            ? "explicit_decimal_money_approximation"
            : threshold_candidate
                ? "explicit_decimal_money_threshold" : "explicit_decimal_money",
    };
}

bool contextual_stock_price_anchor(const std::string& left) {
    return contains_any(left, {
        "股票", "股价", "开盘价", "收盘价", "收盘报", "最高涨到", "最低跌到",
        "商品原价", "原价", "活动价", "订单总额",
    });
}

RuleApproval approve_contextual_stock_price(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = left_clause_context(input, *boundaries, range.start, 24);
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || !contextual_stock_price_anchor(left)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_stock_price_context"};
    }
    if (range.start > 0 && number_continuation(unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "stock_price_continues_left"};
    }
    if (range.end < length && number_continuation(unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "stock_price_continues_right"};
    }
    const auto replacement = parse_contextual_stock_price(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_stock_price"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_yuan_jiao_fen_price", kContextualStockPriceRule},
            {"boundary", "complete_colloquial_stock_price", kContextualStockPriceRule},
            {"context", "explicit_stock_price_anchor", kContextualStockPriceRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "contextual_stock_price",
    };
}

std::optional<std::string> parse_explicit_temperature(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (start >= end) return std::nullopt;
    size_t number_start = start;
    size_t number_end = end;
    std::string prefix;
    std::string suffix;
    if (end >= start + 2) {
        const std::string marker = unicode_scalar_substring(
            input, boundaries, start, start + 2
        );
        if (marker == "摄氏" || marker == "华氏") {
            prefix = marker;
            number_start += 2;
        } else if (marker == "零下" || marker == "零上") {
            prefix = marker;
            number_start += 2;
        }
    }
    if ((prefix == "摄氏" || prefix == "华氏") && end >= number_start + 2) {
        const std::string modifier = unicode_scalar_substring(
            input, boundaries, number_start, number_start + 2
        );
        if (modifier == "零下" || modifier == "零上") {
            prefix += modifier;
            number_start += 2;
        }
    }
    for (const std::string unit : {"摄氏度", "华氏度"}) {
        if (end >= number_start + 3
            && unicode_scalar_substring(input, boundaries, end - 3, end) == unit) {
            suffix = unit;
            number_end = end - 3;
            break;
        }
    }
    if (!suffix.empty() && (prefix.find("摄氏") != std::string::npos
            || prefix.find("华氏") != std::string::npos)) {
        return std::nullopt;
    }
    if (suffix.empty() && end > number_start
        && unicode_scalar_substring(input, boundaries, end - 1, end) == "度") {
        suffix = "度";
        number_end = end - 1;
    }
    if (number_start >= number_end) {
        return std::nullopt;
    }
    size_t decimal_mark = number_end;
    for (size_t index = number_start; index < number_end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != number_end) return std::nullopt;
            decimal_mark = index;
        }
    }
    if (decimal_mark != number_end
        && (decimal_mark == number_start || decimal_mark + 1 == number_end
            || number_end - decimal_mark - 1 > 3)) {
        return std::nullopt;
    }
    const auto value = parse_percentage_value(
        input, boundaries, number_start, number_end
    );
    if (!value.has_value()) return std::nullopt;
    return prefix + *value + suffix;
}

RuleApproval approve_explicit_temperature(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool approximate_candidate =
        candidate.rule_id == kExplicitTemperatureApproximateRule;
    const bool threshold_candidate =
        candidate.rule_id == kExplicitTemperatureThresholdRule;
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 8);
    const bool bare_degree = ends_with_any(range.text, {"度"})
        && !ends_with_any(range.text, {"摄氏度", "华氏度"});
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_temperature_context"};
    }
    if (bare_degree && !contains_any(left, {
            "温度", "气温", "室温", "体温", "水温", "油温", "炉温", "冷库", "空调",
        })) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_bare_degree_temperature_anchor"};
    }
    if (approximate_candidate && (integer_unit_threshold_applies(left, right)
            || !explicit_unit_approximation_applies(left, right))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_temperature_approximation"};
    }
    if (threshold_candidate && (!integer_unit_threshold_applies(left, right)
            || integer_unit_threshold_uncertainty_applies(left, right))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_temperature_threshold"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool released_bare_degree_setpoint = bare_degree && previous == "到"
            && contains_any(left, {"空调", "温度"});
        if (number_continuation(previous)
            || (range_connector(previous) && !released_bare_degree_setpoint)
            || (!approximate_candidate && contains_any("点约近", {previous}))
            || (!approximate_candidate && !threshold_candidate
                && ends_with_any(left, {"大约", "约为", "约在", "将近", "接近"}))) {
            return {RuleDecision::preserve, "", {}, {}, "temperature_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || range_connector(next) || next == "点"
            || (!approximate_candidate && !threshold_candidate && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几", "之间",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "temperature_continues_right"};
        }
    }
    const auto replacement = parse_explicit_temperature(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_explicit_temperature"};
    }
    size_t digit_count = 0;
    for (const char value : *replacement) {
        if (value >= '0' && value <= '9') ++digit_count;
    }
    if (digit_count == 1 && replacement->find('.') == std::string::npos) {
        return {RuleDecision::preserve, "", {}, {}, "natural_single_digit_integer"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_number_with_explicit_temperature_marker", candidate.rule_id},
            {
                "boundary",
                approximate_candidate
                    ? "complete_approximate_temperature"
                    : threshold_candidate
                        ? "complete_threshold_temperature"
                        : "complete_exact_temperature",
                candidate.rule_id,
            },
            {
                "context",
                approximate_candidate
                    ? "explicit_chinese_temperature_approximation"
                    : threshold_candidate
                        ? "explicit_chinese_temperature_threshold"
                        : "explicit_chinese_temperature_marker",
                candidate.rule_id,
            },
        },
        {{{range.start, range.end, range.text}, *replacement}},
        approximate_candidate
            ? "explicit_temperature_approximation"
            : threshold_candidate ? "explicit_temperature_threshold" : "explicit_temperature",
    };
}

std::optional<std::string> parse_explicit_temperature_range(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t connector = end;
    std::string bridge;
    for (size_t index = start; index < end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, boundaries, index, index + 1
        );
        if (value != "到" && value != "至") continue;
        if (connector != end) return std::nullopt;
        connector = index;
        bridge = value;
    }
    if (connector == start || connector + 1 >= end) return std::nullopt;
    const std::string second_source = unicode_scalar_substring(
        input, boundaries, connector + 1, end
    );
    std::string unit;
    if (ends_with_any(second_source, {"摄氏度"})) unit = "摄氏度";
    else if (ends_with_any(second_source, {"华氏度"})) unit = "华氏度";
    else if (ends_with_any(second_source, {"度"})) unit = "度";
    else return std::nullopt;
    const auto second_boundaries = unicode_scalar_boundaries(second_source);
    if (!second_boundaries.has_value()) return std::nullopt;
    const auto second = parse_explicit_temperature(
        second_source, *second_boundaries, 0, second_boundaries->size() - 1
    );
    if (!second.has_value()) return std::nullopt;

    const std::string first_source = unicode_scalar_substring(
        input, boundaries, start, connector
    );
    const bool first_has_unit = ends_with_any(first_source, {"摄氏度", "华氏度"});
    if (first_has_unit && !ends_with_any(first_source, {unit})) return std::nullopt;
    const std::string first_parse_source = first_has_unit
        ? first_source : first_source + unit;
    const auto first_boundaries = unicode_scalar_boundaries(first_parse_source);
    if (!first_boundaries.has_value()) return std::nullopt;
    auto first = parse_explicit_temperature(
        first_parse_source, *first_boundaries, 0, first_boundaries->size() - 1
    );
    if (!first.has_value()) return std::nullopt;
    if (!first_has_unit) first->resize(first->size() - unit.size());

    size_t digits = 0;
    bool decimal = false;
    for (const char value : *first + *second) {
        if (value >= '0' && value <= '9') ++digits;
        if (value == '.') decimal = true;
    }
    if (digits < 3 && !decimal) return std::nullopt;
    return *first + bridge + *second;
}

RuleApproval approve_explicit_temperature_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool bare_degree = ends_with_any(range.text, {"度"})
        && !ends_with_any(range.text, {"摄氏度", "华氏度"});
    const std::string left = left_clause_context(input, *boundaries, range.start, 20);
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_temperature_range_context"};
    }
    if (bare_degree && !contains_any(left, {
            "温度", "气温", "室温", "体温", "水温", "油温", "炉温", "冷库",
        })) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_bare_degree_temperature_anchor"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || range_connector(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "temperature_range_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || range_connector(next)
            || contains_any("点度", {next})) {
            return {RuleDecision::preserve, "", {}, {}, "temperature_range_continues_right"};
        }
    }
    const auto replacement = parse_explicit_temperature_range(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_temperature_range"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {
                "shape", "two_temperature_values_with_shared_or_repeated_scale",
                kExplicitTemperatureRangeRule,
            },
            {"boundary", "complete_temperature_range", kExplicitTemperatureRangeRule},
            {
                "context", bare_degree
                    ? "explicit_temperature_anchor_and_shared_degree"
                    : "explicit_temperature_scale_and_range_connector",
                kExplicitTemperatureRangeRule,
            },
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_temperature_range",
    };
}

bool explicit_integer_money_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    const std::string& unit
) {
    if (unit != "元") return true;
    const std::string left = left_clause_context(input, boundaries, start, 16);
    const std::string right = right_clause_context(input, boundaries, end, 12);
    return contains_any(left + right, {
        "人民币", "价格", "总价", "楼面价", "每平方米", "每平米", "金额", "总额", "成交额",
        "收费", "售价", "收入", "营收", "罚款", "罚金", "实付", "净流入",
        "门票", "票价", "成人票", "儿童票", "工资", "月薪", "薪酬", "赔偿",
        "费用", "预算", "红包", "每人", "每月", "每生", "花费", "支付", "付款", "存入",
        "银行", "现金", "优惠券", "券", "股价", "均价", "单价", "原价", "现价",
        "缴存额", "亏损", "盈利", "利润", "成本", "消费", "补助", "奖励", "押金",
        "融资", "估值", "市值", "财富", "票房", "价值", "采购", "订单", "合同",
        "违约金", "首付", "每股收益", "两融余额", "兑换", "每吨",
        "退款", "赔付", "售票", "购买", "购票", "报收", "基本养老金", "养老金",
        "退休金", "花了", "买了", "买了个", "/股", "/位",
    });
}

bool explicit_integer_meter_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    const std::string& number
) {
    if (ends_with_any(number, {"千"})) return false;
    if (start > 0
        && unicode_scalar_substring(input, boundaries, start - 1, start) == "米") {
        return false;
    }
    const std::string left = left_clause_context(input, boundaries, start, 16);
    const std::string right = right_clause_context(input, boundaries, end, 8);
    return contains_any(left + right, {
        "累计爬升", "累积爬升", "海拔", "标高", "高度", "限高", "长度", "距离",
        "全程", "路程", "距您", "距离您", "向东", "向西", "向南", "向北",
    });
}

bool integer_money_constraint_applies(const std::string& left) {
    static const std::vector<std::string> constraints = {
        "大约", "大概", "约为", "大约在", "约在", "将近", "接近",
        "约合", "约合人民币", "至少", "至多", "最多", "最少", "最低",
        "最高", "不到", "不足", "不过", "超过", "不超过", "不少于", "不低于",
        "高于", "低于", "应该", "好像", "可能", "也许", "估计", "预计",
        "差不多", "几乎",
    };
    static const std::vector<std::string> attributive_superlatives = {
        "最多", "最少", "最低", "最高",
    };
    static const std::string attributive_marker = "的";
    for (const auto& constraint : constraints) {
        size_t position = 0;
        while ((position = left.find(constraint, position)) != std::string::npos) {
            const size_t after = position + constraint.size();
            const bool intervening_attributive = contains_any(
                    constraint, attributive_superlatives
                )
                && left.compare(after, attributive_marker.size(), attributive_marker) == 0
                && after + attributive_marker.size() < left.size();
            if (!intervening_attributive) return true;
            position = after;
        }
    }
    return false;
}

bool money_threshold_applies(const std::string& left, const std::string& right) {
    static const std::vector<std::string> thresholds = {
        "至少", "至多", "最多", "最少", "最低", "最高", "不到", "不足", "不过",
        "超过", "不超过", "不少于", "不低于", "高于", "低于",
    };
    static const std::vector<std::string> money_predicates = {
        "支付", "支付了", "可支付", "需支付", "需要支付", "只能支付", "应支付",
        "缴纳", "缴纳了", "可缴纳", "需缴纳", "支出", "投入", "花费", "赔偿",
        "罚款", "预算为", "金额为", "价格为", "成本为", "收入为", "为", "达",
        "达到",
    };
    for (const auto& threshold : thresholds) {
        size_t position = 0;
        while ((position = left.find(threshold, position)) != std::string::npos) {
            const std::string tail = left.substr(position + threshold.size());
            if (tail.empty() || std::find(
                    money_predicates.begin(), money_predicates.end(), tail
                ) != money_predicates.end()) {
                return true;
            }
            position += threshold.size();
        }
    }
    return starts_with_any(right, {
        "以上", "以下", "以内", "之内", "及以上", "及以下",
    });
}

bool money_approximation_applies(const std::string& left, const std::string& right) {
    return explicit_unit_approximation_applies(left, right)
        || ends_with_any(left, {"约合", "约合人民币"});
}

bool money_non_threshold_uncertainty_applies(
    const std::string& left,
    const std::string& right
) {
    return money_approximation_applies(left, right)
        || contains_any(left, {
            "应该", "好像", "可能", "也许", "估计", "预计", "差不多", "几乎",
        })
        || starts_with_any(right, {"多", "余", "几", "吧", "差不多"});
}

bool money_predicate_connector_applies(const std::string& left) {
    return ends_with_any(left, {
        "达到", "冲到", "涨到", "跌到", "降到", "升到", "增至", "涨至", "跌至",
        "降至", "升至", "甚至",
    });
}

bool integer_unit_threshold_applies(
    const std::string& left,
    const std::string& right
) {
    return ends_with_any(left, {
            "至少", "至多", "最多", "最少", "最低", "最高", "不到", "不足",
            "超过", "不超过", "不少于", "不低于", "高于", "低于",
        })
        || starts_with_any(right, {"以上", "以下", "以内", "之内"});
}

bool integer_unit_threshold_uncertainty_applies(
    const std::string& left,
    const std::string& right
) {
    if (contains_any(left, {
            "大约", "大概", "约为", "大约在", "约在", "将近", "接近", "应该",
            "好像", "可能", "也许", "估计", "预计", "差不多", "几乎",
        }) || starts_with_any(right, {
            "多", "左右", "上下", "前后", "余", "几", "吧", "差不多",
        })) {
        return true;
    }
    for (const auto& suffix : {"以上", "以下", "以内", "之内"}) {
        if (starts_with_any(right, {suffix})
            && starts_with_any(right.substr(std::strlen(suffix)), {
                "多", "左右", "上下", "前后", "余", "几", "吧", "差不多",
            })) {
            return true;
        }
    }
    return false;
}

bool explicit_unit_approximation_applies(
    const std::string& left,
    const std::string& right
) {
    const bool bare_approximately = ends_with_any(left, {"约"})
        && !ends_with_any(left, {
            "预约", "合约", "条约", "契约", "违约", "公约", "制约", "节约",
            "简约", "特约", "相约",
        });
    return bare_approximately
        || ends_with_any(left, {
            "大约", "大概", "约为", "大约在", "约在", "将近", "接近",
        })
        || starts_with_any(right, {"左右", "上下", "前后", "多", "开外"});
}

bool contextual_infix_distance_approximation(
    const std::string& left,
    const std::string& unit
) {
    return unit == "多公里" && contains_any(left, {
        "续航", "里程", "距离", "路程", "行驶", "路段", "全程",
    });
}

bool contextual_approximate_rate_distance(
    const std::string& left,
    const std::string& unit
) {
    if (unit != "米" && unit != "公里" && unit != "千米") return false;
    return ends_with_any(left, {
        "每秒约", "每秒大约", "每秒大概",
        "每秒跑约", "每秒跑大约", "每秒跑大概",
        "每秒能跑约", "每秒能跑大约", "每秒能跑大概",
        "每秒只能跑约", "每秒只能跑大约", "每秒只能跑大概",
        "每秒可跑约", "每秒可跑大约", "每秒可跑大概",
        "每秒移动约", "每秒移动大约", "每秒移动大概",
        "每秒传播约", "每秒传播大约", "每秒传播大概",
        "每秒行进约", "每秒行进大约", "每秒行进大概",
        "每秒前进约", "每秒前进大约", "每秒前进大概",
        "每秒可达约", "每秒可达大约", "每秒可达大概",
        "每秒达到约", "每秒达到大约", "每秒达到大概",
    });
}

bool contextual_battery_capacity_degree(
    const std::string& left,
    const std::string& unit
) {
    return unit == "度" && contains_any(left, {
        "电池容量", "电池电量", "电量", "满电", "充入", "充电量",
    });
}

bool explicit_periodic_unit_applies(const std::string& left) {
    return ends_with_any(left, {"每", "每隔", "每秒", "每分钟", "每小时"});
}

bool explicit_periodic_unit_is_uncertain(const std::string& left) {
    static const std::vector<std::string> markers = {
        "大约", "大概", "约", "将近", "接近", "至少", "至多", "最多", "最少",
        "不到", "不足", "超过", "不超过", "不少于", "不低于", "高于", "低于",
    };
    for (const auto& marker : markers) {
        if (ends_with_any(left, {marker + "每", marker + "每隔"})) return true;
    }
    return false;
}

RuleApproval approve_explicit_integer_money(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool threshold_candidate = candidate.rule_id == kExplicitIntegerMoneyThresholdRule;
    const bool approximate_candidate = candidate.rule_id == kExplicitIntegerMoneyApproximateRule;
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 8);
    const auto unit = decimal_money_unit(
        input, *boundaries, range.start, range.end
    );
    if (!unit.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || (!threshold_candidate && !approximate_candidate && !explicit_integer_money_context(
                input, *boundaries, range.start, range.end, unit->second
            ))
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_money_context"};
    }
    if (approximate_candidate && (money_threshold_applies(left, right)
            || !money_approximation_applies(left, right))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_money_approximation"};
    }
    if (threshold_candidate && (!money_threshold_applies(left, right)
            || money_non_threshold_uncertainty_applies(left, right))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_money_threshold"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool released_prefix = approximate_candidate
            && (previous == "约" || previous == "近");
        const bool released_constraint = threshold_candidate || approximate_candidate;
        const bool released_planned_total = ends_with_any(left, {
            "发放总额", "预计发放总额",
        });
        if (number_continuation(previous)
            || (!released_constraint && range_connector(previous)
                && !money_predicate_connector_applies(left))
            || (contains_any("点约近几多儿啊呃嗯元", {previous})
                && !released_prefix
                && !(released_constraint && ends_with_any(left, {"原价最多"})))
            || (!released_constraint && !released_planned_total
                && integer_money_constraint_applies(left))) {
            return {RuleDecision::preserve, "", {}, {}, "integer_money_continues_left"};
        }
        const bool rmb_unit = unit->second == "元" || unit->second == "万元"
            || unit->second == "亿元" || unit->second == "亿元人民币";
        if (!rmb_unit && ends_with_any(left, {"人民币"})) {
            return {RuleDecision::preserve, "", {}, {}, "conflicting_currency_context"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const bool released_fraction_suffix = unit->second == "元"
            && starts_with_any(right, {
                "一角", "二角", "三角", "四角", "五角",
                "六角", "七角", "八角", "九角", "1角", "2角", "3角", "4角", "5角",
                "6角", "7角", "8角", "9角",
            });
        const bool released_following_payment_word = unit->second == "元"
            && starts_with_any(right, {"一缴", "一交"});
        const bool released_suffix = (approximate_candidate
                && (starts_with_any(right, {"左右", "上下", "前后"})
                    || (starts_with_any(right, {"多"})
                        && contains_any(left, {"两融余额", "账户余额", "余额"}))))
            || (threshold_candidate
                && starts_with_any(right, {
                    "以上", "以下", "以内", "之内", "及以上", "及以下",
                }));
        if ((number_continuation(next) && !released_fraction_suffix
                && !released_following_payment_word)
            || range_connector(next) || next == "点" || next == "钱"
            || (!released_suffix && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
                "起", "人民币", "之间", "吧", "之多", "级别", "价位", "差不多",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "integer_money_continues_right"};
        }
        if (contains_any(right, {"吧", "好像", "大概", "可能", "也许"})) {
            return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_money_suffix"};
        }
        if (unit->second == "元" && starts_with_any(right, {
                "旦", "年", "件", "素", "音", "组", "论", "化", "函数", "方程",
                "关系", "结构", "系统", "模型", "分析", "运算", "数据", "宇宙", "裤业",
                "单位", "编码",
            })) {
            return {RuleDecision::preserve, "", {}, {}, "non_money_yuan_lexeme"};
        }
        if (unit->second == "元" && starts_with_any(right, {"每人"})) {
            const std::string rate_left = left_clause_context(
                input, *boundaries, range.start, 16
            );
            if (contains_any(rate_left, {"每人"})) {
                return {RuleDecision::preserve, "", {}, {}, "repeated_money_rate_context"};
            }
        }
    }
    const auto integer = parse_spoken_money_integer(
        input, *boundaries, range.start, unit->first
    );
    const bool released_exchange_unit = integer.has_value() && *integer == 1
        && unit->second == "美元" && starts_with_any(right, {"兑换"});
    if (integer.has_value() && *integer < 10 && !released_exchange_unit) {
        return {RuleDecision::preserve, "", {}, {}, "natural_single_digit_integer"};
    }
    auto replacement = parse_explicit_integer_money(
        input, *boundaries, range.start, range.end
    );
    if (replacement.has_value() && integer.has_value() && unit->second == "元"
        && contains_any(left, {"楼面价", "每平方米", "每平米"})) {
        replacement = std::to_string(*integer) + "元";
    }
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_integer_money"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_integer_plus_currency_unit", candidate.rule_id},
            {
                "boundary",
                approximate_candidate
                    ? "complete_approximate_integer_money"
                    : threshold_candidate
                    ? "complete_threshold_integer_money"
                    : "complete_exact_integer_money",
                candidate.rule_id,
            },
            {
                "context",
                approximate_candidate
                    ? "explicit_chinese_money_approximation"
                    : threshold_candidate
                    ? "explicit_chinese_money_threshold"
                    : "released_chinese_currency_unit",
                candidate.rule_id,
            },
        },
        {{{range.start, range.end, range.text}, *replacement}},
        approximate_candidate
            ? "explicit_integer_money_approximation"
            : threshold_candidate ? "explicit_integer_money_threshold" : "explicit_integer_money",
    };
}

RuleApproval approve_explicit_integer_measure(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool threshold_candidate =
        candidate.rule_id == kExplicitIntegerMeasureThresholdRule;
    const bool approximate_candidate =
        candidate.rule_id == kExplicitIntegerMeasureApproximateRule
        || candidate.rule_id == kContextualApproximateRateDistanceRule;
    const bool rate_distance_candidate =
        candidate.rule_id == kContextualApproximateRateDistanceRule;
    const bool periodic_candidate =
        candidate.rule_id == kExplicitIntegerMeasurePeriodicRule;
    const auto unit = periodic_candidate
        ? explicit_periodic_integer_measure_unit(
            input, *boundaries, range.start, range.end
        )
        : explicit_integer_measure_unit(input, *boundaries, range.start, range.end);
    if (!unit.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_measure_context"};
    }
    const bool exact_count = unit->second == "件" || unit->second == "批"
        || unit->second == "人" || unit->second == "箱" || unit->second == "股"
        || unit->second == "例" || unit->second == "个代表团"
        || unit->second == "名观众";
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 8);
    if (unit->second == "多个" && !contains_any(
            unicode_scalar_substring(input, *boundaries, range.start, unit->first),
            {"千", "万", "亿"})) {
        return {RuleDecision::preserve, "", {}, {}, "unanchored_approximate_count"};
    }
    const bool approximate_person_count = unit->second == "人"
        && unit->first > range.start
        && unicode_scalar_substring(
            input, *boundaries, unit->first - 1, unit->first
        ) == "多";
    if (approximate_person_count && !ends_with_any(left, {
            "现场有", "现场共有", "共有", "一共有", "总共有", "累计有",
            "到场有", "出席有", "参会有",
        })) {
        return {RuleDecision::preserve, "", {}, {}, "unanchored_approximate_person_count"};
    }
    const bool released_money_fraction = unit->second == "角"
        && ends_with_any(left, {"元"})
        && contains_any(left, {"原价", "活动价", "报价", "金额", "总额", "最多"});
    if (approximate_candidate && unit->second == "毫米汞柱") {
        return {RuleDecision::preserve, "", {}, {},
                "natural_approximate_blood_pressure_measure"};
    }
    const bool shorthand_count_threshold = threshold_candidate
        && (unit->second == "辆" || unit->second == "颗")
        && ends_with_any(left, {"超"});
    const bool explicit_human_traffic_threshold = threshold_candidate
        && unit->second == "人次" && ends_with_any(left, {"超过"});
    if (threshold_candidate && ((!integer_unit_threshold_applies(left, right)
            && !shorthand_count_threshold)
            || (integer_unit_threshold_uncertainty_applies(left, right)
                && !explicit_human_traffic_threshold))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_measure_threshold"};
    }
    const bool infix_distance_approximation = contextual_infix_distance_approximation(
        left, unit->second
    );
    const bool rate_distance_approximation = contextual_approximate_rate_distance(
        left, unit->second
    );
    if (rate_distance_candidate && !rate_distance_approximation) {
        return {RuleDecision::preserve, "", {}, {},
                "uncertain_approximate_rate_distance"};
    }
    if (unit->second == "多公里" && !infix_distance_approximation) {
        return {RuleDecision::preserve, "", {}, {},
                "uncertain_infix_distance_approximation"};
    }
    if (approximate_candidate && (integer_unit_threshold_applies(left, right)
            || (!approximate_person_count
                && !explicit_unit_approximation_applies(left, right)
                && !infix_distance_approximation))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_measure_approximation"};
    }
    if (periodic_candidate && (!explicit_periodic_unit_applies(left)
            || explicit_periodic_unit_is_uncertain(left))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_measure_period"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool released_periodic_prefix = periodic_candidate
            && (previous == "每" || previous == "隔");
        const bool released_threshold_prefix = threshold_candidate
            && (previous == "超" || integer_unit_threshold_applies(left, right));
        const bool released_approximate_prefix = approximate_candidate
            && (previous == "约" || previous == "近");
        const bool released_platform_prefix = unit->second == "站台"
            && previous == "到";
        if ((number_continuation(previous) && !released_approximate_prefix
                && !released_money_fraction)
            || (exact_count && previous == "乘")
            || (!threshold_candidate && range_connector(previous)
                && !released_platform_prefix && !released_approximate_prefix
                && !released_money_fraction)
            || (!approximate_candidate && !released_periodic_prefix
                && !released_threshold_prefix
                && contains_any("点约近几多儿啊呃嗯每/／超", {previous}))
            || (!threshold_candidate && !approximate_candidate && !periodic_candidate
                && !released_money_fraction
                && contains_any(left, {
                "大约", "大概", "约为", "大约在", "约在", "将近", "接近",
                "至少", "至多", "最多", "最少", "最低", "最高", "不到", "不足",
                "超过", "不超过", "不少于", "不低于", "高于", "低于", "应该",
                "好像", "可能", "也许", "估计", "预计", "差不多", "几乎",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "integer_measure_continues_left"};
        }
    }
    const std::string number = unicode_scalar_substring(
        input, *boundaries, range.start, unit->first
    );
    if (unit->second == "角" && !ends_with_any(left, {"元"})) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_money_fraction_context"};
    }
    if (unit->second == "角" && starts_with_any(right, {"形", "函数"})) {
        return {RuleDecision::preserve, "", {}, {}, "non_money_angle_lexeme"};
    }
    if (unit->second == "例" && !contains_any(left, {
            "病例", "新增", "确诊", "感染", "死亡",
        })) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_statistical_case_context"};
    }
    if (unit->second == "届" && (range.start == 0
            || unicode_scalar_substring(
                input, *boundaries, range.start - 1, range.start
            ) != "第")) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_ordinal_session_context"};
    }
    if (unit->second == "点" && !contains_any(left, {
            "指数报", "指数收报", "指数为",
        })) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_market_index_context"};
    }
    if (unit->second == "亿" && !contains_any(left, {"成交额", "交易额"})) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_financial_magnitude_context"};
    }
    if (unit->second == "安" && !contains_any(left, {
            "电流", "额定电流", "工作电流",
        })) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_electric_current_context"};
    }
    if (unit->second == "度" && !contextual_battery_capacity_degree(left, unit->second)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_capacity_degree_context"};
    }
    if (unit->second == "米" && !periodic_candidate && !rate_distance_candidate
        && (threshold_candidate || !explicit_integer_meter_context(
                input, *boundaries, range.start, range.end, number
            ))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_meter_context"};
    }
    if (unit->second == "公里" && number == "一" && contains_any(left, {"最后"})) {
        return {RuleDecision::preserve, "", {}, {}, "fixed_last_mile_expression"};
    }
    if (unit->second == "倍" && !contains_any(left, {
            "带宽相差", "容量相差", "性能提升", "算力提升", "速度提升",
        })) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_multiple_context"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const bool count_followed_by_noun = unit->second == "件"
            && starts_with_any(right, {"零件"});
        const bool audience_followed_by_predicate = unit->second == "名观众"
            && starts_with_any(right, {"到场"});
        const bool meter_followed_by_turn = unit->second == "米"
            && starts_with_any(right, {"三角掉头", "三角调头"});
        if ((number_continuation(next) && !count_followed_by_noun
                && !meter_followed_by_turn) || (range_connector(next)
                && !audience_followed_by_predicate)
            || (exact_count && next == "乘")
            || (unit->second == "人" && starts_with_any(right, {"工", "民币"}))
            || contains_any("点每/／", {next})
            || (!threshold_candidate && !approximate_candidate && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
                "起", "之间", "吧", "之多", "级别", "差不多",
            }))
            || contains_any(right, {"吧", "好像", "大概", "可能", "也许"})) {
            return {RuleDecision::preserve, "", {}, {}, "integer_measure_continues_right"};
        }
        if (unit->second == "例" && starts_with_any(right, {"外"})) {
            return {RuleDecision::preserve, "", {}, {}, "non_count_case_lexeme"};
        }
        if ((unit->second == "毫米" && starts_with_any(right, {"汞柱"}))
            || (unit->second == "吨" && starts_with_any(right, {"位", "公里"}))
            || (unit->second == "股" && starts_with_any(right, {
                "东", "份", "票", "权", "价", "市",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "incomplete_compound_measure_unit"};
        }
    }
    const auto integer = parse_spoken_money_integer(
        input, *boundaries, range.start, unit->first
    );
    const bool explicit_storage_capacity = contains_any(
        unit->second, {
            "吉比特", "吉字节", "兆字节", "太字节", "太字节每秒", "太拉弗洛普斯",
            "个ppm", "个ppb",
        }
    );
    const bool explicit_process_node = unit->second == "纳米";
    const bool explicit_money_fraction = unit->second == "角";
    if (integer.has_value() && *integer < 10
        && !explicit_storage_capacity && !explicit_process_node
        && !explicit_money_fraction) {
        return {RuleDecision::preserve, "", {}, {}, "natural_single_digit_integer"};
    }
    const auto replacement = periodic_candidate
        ? parse_explicit_periodic_integer_measure(
            input, *boundaries, range.start, range.end
        )
        : parse_explicit_integer_measure(input, *boundaries, range.start, range.end);
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_integer_measure"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_integer_plus_measure_unit", candidate.rule_id},
            {
                "boundary",
                threshold_candidate
                    ? "complete_threshold_integer_measure"
                    : periodic_candidate
                    ? "complete_periodic_integer_measure"
                    : approximate_candidate
                        ? (rate_distance_candidate
                            ? "complete_contextual_approximate_rate_distance"
                            : "complete_approximate_integer_measure")
                        : "complete_exact_integer_measure",
                candidate.rule_id,
            },
            {
                "context",
                threshold_candidate
                    ? "explicit_chinese_measure_threshold"
                    : periodic_candidate
                    ? "explicit_chinese_measure_period"
                    : approximate_candidate
                        ? (rate_distance_candidate
                            ? "explicit_rate_and_approximation_anchor"
                            : "explicit_chinese_measure_approximation")
                        : "released_chinese_measure_unit",
                candidate.rule_id,
            },
        },
        {{{range.start, range.end, range.text}, *replacement}},
        threshold_candidate
            ? "explicit_integer_measure_threshold"
            : periodic_candidate
            ? "explicit_integer_measure_periodic"
            : approximate_candidate
                ? (rate_distance_candidate
                    ? "contextual_approximate_rate_distance"
                    : "explicit_integer_measure_approximation")
                : "explicit_integer_measure",
    };
}

RuleApproval approve_explicit_integer_duration(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool threshold_candidate =
        candidate.rule_id == kExplicitIntegerDurationThresholdRule;
    const bool approximate_candidate =
        candidate.rule_id == kExplicitIntegerDurationApproximateRule;
    const bool periodic_candidate =
        candidate.rule_id == kExplicitIntegerDurationPeriodicRule;
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 8);
    const bool released_exact_duration_context = !threshold_candidate
        && !approximate_candidate && !periodic_candidate
        && ends_with_any(left, {"预计运行", "补时", "耗时", "用时"});
    const auto unit = explicit_integer_duration_unit(
        input, *boundaries, range.start, range.end
    );
    if (!unit.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_duration_context"};
    }
    if (threshold_candidate && (!integer_unit_threshold_applies(left, right)
            || integer_unit_threshold_uncertainty_applies(left, right))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_duration_threshold"};
    }
    if (approximate_candidate && (integer_unit_threshold_applies(left, right)
            || !explicit_unit_approximation_applies(left, right))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_duration_approximation"};
    }
    if (periodic_candidate && (!explicit_periodic_unit_applies(left)
            || explicit_periodic_unit_is_uncertain(left))) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_integer_duration_period"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool adjacent_ascii = previous.size() == 1
            && std::isalnum(static_cast<unsigned char>(previous[0])) != 0;
        const bool released_periodic_prefix = periodic_candidate
            && (previous == "每" || previous == "隔");
        if (adjacent_ascii || number_continuation(previous)
            || (!threshold_candidate && range_connector(previous))
            || (!approximate_candidate && !released_periodic_prefix
                && !released_exact_duration_context
                && contains_any("点约近几多儿啊呃嗯每/／时分秒钟", {previous}))
            || (!threshold_candidate && !approximate_candidate && !periodic_candidate
                && !released_exact_duration_context
                && contains_any(left, {
                "大约", "大概", "约为", "大约在", "约在", "将近", "接近",
                "至少", "至多", "最多", "最少", "最低", "最高", "不到", "不足",
                "超过", "不超过", "不少于", "不低于", "高于", "低于", "长过",
                "短于", "应该", "好像", "可能", "也许", "估计", "预计",
                "差不多", "几乎",
            }))) {
            return {RuleDecision::preserve, "", {}, {}, "integer_duration_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || range_connector(next)
            || contains_any("点每/／钟", {next})
            || (!threshold_candidate && !approximate_candidate && starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
                "起", "之间", "吧", "之多", "级别", "差不多", "内", "之内",
            }))
            || contains_any(right, {"吧", "好像", "大概", "可能", "也许"})) {
            return {RuleDecision::preserve, "", {}, {}, "integer_duration_continues_right"};
        }
    }
    const auto integer = parse_spoken_money_integer(
        input, *boundaries, range.start, unit->first
    );
    if (integer.has_value() && *integer < 10) {
        return {RuleDecision::preserve, "", {}, {}, "natural_single_digit_integer"};
    }
    const auto replacement = parse_explicit_integer_duration(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_integer_duration"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_integer_plus_duration_unit", candidate.rule_id},
            {
                "boundary",
                threshold_candidate
                    ? "complete_threshold_integer_duration"
                    : periodic_candidate
                    ? "complete_periodic_integer_duration"
                    : approximate_candidate
                        ? "complete_approximate_integer_duration"
                        : "complete_exact_integer_duration",
                candidate.rule_id,
            },
            {
                "context",
                threshold_candidate
                    ? "explicit_chinese_duration_threshold"
                    : periodic_candidate
                    ? "explicit_chinese_duration_period"
                    : approximate_candidate
                        ? "explicit_chinese_duration_approximation"
                        : "released_chinese_duration_unit",
                candidate.rule_id,
            },
        },
        {{{range.start, range.end, range.text}, *replacement}},
        threshold_candidate
            ? "explicit_integer_duration_threshold"
            : periodic_candidate
            ? "explicit_integer_duration_periodic"
            : approximate_candidate
                ? "explicit_integer_duration_approximation"
                : "explicit_integer_duration",
    };
}

RuleApproval approve_explicit_lock_period_months(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (range.start < 3
        || unicode_scalar_substring(input, *boundaries, range.start - 3, range.start)
            != "锁定期"
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_lock_period_months"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || range_connector(next)) {
            return {RuleDecision::preserve, "", {}, {}, "lock_period_months_continues_right"};
        }
    }
    const auto replacement = parse_explicit_lock_period_months(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_lock_period_months"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_integer_plus_month_duration", kExplicitLockPeriodMonthsRule},
            {"boundary", "complete_lock_period_month_duration", kExplicitLockPeriodMonthsRule},
            {"context", "explicit_lock_period_anchor", kExplicitLockPeriodMonthsRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_lock_period_months",
    };
}

RuleApproval approve_large_approximate_year_span(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (range.end - range.start < 4
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || unicode_scalar_substring(input, *boundaries, range.end - 2, range.end) != "余年"
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_large_year_span_context"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "large_year_span_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (date_or_number_continuation(next) || range_connector(next)) {
            return {RuleDecision::preserve, "", {}, {}, "large_year_span_continues_right"};
        }
    }
    const size_t number_end = range.end - 2;
    const auto value = parse_spoken_money_integer(
        input, *boundaries, range.start, number_end
    );
    const auto formatted = format_spoken_money_integer(
        input, *boundaries, range.start, number_end
    );
    if (!value.has_value() || *value < 100 || *value > 99999999
        || !formatted.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_large_approximate_year_span"};
    }
    const std::string replacement = *formatted + "余年";
    return {
        RuleDecision::approve,
        replacement,
        {
            {
                "shape", "large_spoken_integer_plus_yu_year",
                kLargeApproximateYearSpanRule,
            },
            {
                "boundary", "complete_large_approximate_year_span",
                kLargeApproximateYearSpanRule,
            },
            {
                "context", "explicit_large_approximate_year_unit",
                kLargeApproximateYearSpanRule,
            },
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_large_approximate_year_span",
    };
}

RuleApproval approve_explicit_anchored_year_or_day_duration(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool validity_years = candidate.rule_id == kExplicitValidityYearsRule;
    const std::string unit = validity_years ? "年" : "天";
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 12);
    const bool anchored = validity_years
        ? ends_with_any(left, {"有效期为", "有效期是", "有效期限为", "有效期限是"})
        : ends_with_any(left, {
            "期满前大概", "期满前大约", "期满前约",
            "到期前大概", "到期前大约", "到期前约",
        }) && starts_with_any(right, {
            "可申请续签", "可以申请续签", "可申请续期", "可以申请续期",
        });
    if (!anchored
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_anchored_year_or_day_duration"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || range_connector(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "anchored_duration_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || range_connector(next)) {
            return {RuleDecision::preserve, "", {}, {}, "anchored_duration_continues_right"};
        }
    }
    const auto replacement = parse_explicit_anchored_year_or_day_duration(
        input, *boundaries, range.start, range.end, unit,
        validity_years ? 999 : 999999
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_anchored_year_or_day_duration"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", validity_years
                ? "spoken_integer_plus_validity_year"
                : "spoken_integer_plus_renewal_day", candidate.rule_id},
            {"boundary", "complete_anchored_year_or_day_duration", candidate.rule_id},
            {"context", validity_years
                ? "explicit_validity_period_anchor"
                : "explicit_renewal_application_anchor", candidate.rule_id},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        validity_years
            ? "explicit_validity_years"
            : "explicit_renewal_approximate_days",
    };
}

RuleApproval approve_explicit_compound_duration(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const bool minute_second_candidate =
        candidate.rule_id == kExplicitMinuteSecondDurationRule;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_compound_duration_context"};
    }
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const bool adjacent_ascii = previous.size() == 1
            && std::isalnum(static_cast<unsigned char>(previous[0])) != 0;
        if (adjacent_ascii || number_continuation(previous) || range_connector(previous)
            || contains_any("点约近几多儿啊呃嗯每/／时分秒钟", {previous})
            || contains_any(left, {
                "大约", "大概", "约为", "大约在", "约在", "将近", "接近",
                "至少", "至多", "最多", "最少", "最低", "最高", "不到", "不足",
                "超过", "不超过", "不少于", "不低于", "高于", "低于", "长过",
                "短于", "应该", "好像", "可能", "也许", "估计", "预计",
                "差不多", "几乎",
            })) {
            return {RuleDecision::preserve, "", {}, {}, "compound_duration_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = right_clause_context(
            input, *boundaries, range.end, 8
        );
        if (number_continuation(next) || range_connector(next)
            || contains_any("点每/／钟", {next})
            || starts_with_any(right, {
                "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
                "起", "之间", "吧", "之多", "级别", "差不多", "内", "之内",
            })
            || contains_any(right, {"吧", "好像", "大概", "可能", "也许"})) {
            return {RuleDecision::preserve, "", {}, {}, "compound_duration_continues_right"};
        }
    }
    const auto replacement = minute_second_candidate
        ? parse_explicit_minute_second_duration(
            input, *boundaries, range.start, range.end
        )
        : parse_explicit_compound_duration(
            input, *boundaries, range.start, range.end
        );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_compound_duration"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {
                "shape",
                minute_second_candidate
                    ? "spoken_minutes_plus_seconds" : "spoken_hours_plus_minutes",
                candidate.rule_id,
            },
            {"boundary", "complete_exact_compound_duration", candidate.rule_id},
            {
                "context",
                minute_second_candidate
                    ? "explicit_minute_and_second_units" : "explicit_hour_and_minute_units",
                candidate.rule_id,
            },
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_compound_duration",
    };
}

bool ambiguous_digit_g_quantity_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    const std::string left = unicode_scalar_substring(
        input, boundaries, start > 6 ? start - 6 : 0, start
    );
    const std::string right = unicode_scalar_substring(
        input, boundaries, end, std::min(length, end + 3)
    );
    return ends_with_any(left, {
        "重", "约重", "重约", "重达", "净重", "净重约", "重量", "重量为",
        "重量是", "重量约", "称重", "称重为", "称重是", "净含量",
    }) || starts_with_any(right, {"赫兹", "加速度", "重力"});
}

std::optional<size_t> spaced_digit_g_end(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const size_t length = boundaries.size() - 1;
    if (start >= length || !spoken_year_digit(unicode_scalar_substring(
            input, boundaries, start, start + 1
        )).has_value()) {
        return std::nullopt;
    }
    const auto marker = skip_limited_inline_spaces(
        input, boundaries, start + 1, length, 2
    );
    if (!marker.has_value() || *marker >= length) return std::nullopt;
    const std::string value = unicode_scalar_substring(
        input, boundaries, *marker, *marker + 1
    );
    return value == "G" || value == "g"
        ? std::optional<size_t>(*marker + 1) : std::nullopt;
}

std::optional<size_t> spaced_respirator_end(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const size_t length = boundaries.size() - 1;
    if (start >= length || !contains_any("Nn", {
            unicode_scalar_substring(input, boundaries, start, start + 1)
        })) {
        return std::nullopt;
    }
    const auto nine = skip_limited_inline_spaces(
        input, boundaries, start + 1, length, 2
    );
    if (!nine.has_value() || *nine >= length
        || unicode_scalar_substring(input, boundaries, *nine, *nine + 1) != "九") {
        return std::nullopt;
    }
    const auto five = skip_limited_inline_spaces(
        input, boundaries, *nine + 1, length, 2
    );
    if (!five.has_value() || *five >= length
        || unicode_scalar_substring(input, boundaries, *five, *five + 1) != "五") {
        return std::nullopt;
    }
    return *five + 1;
}

std::optional<size_t> spaced_automotive_store_end(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const size_t length = boundaries.size() - 1;
    if (start >= length
        || unicode_scalar_substring(input, boundaries, start, start + 1) != "四") {
        return std::nullopt;
    }
    const auto marker = skip_limited_inline_spaces(
        input, boundaries, start + 1, length, 2
    );
    if (!marker.has_value() || *marker >= length
        || !contains_any("Ss", {
            unicode_scalar_substring(input, boundaries, *marker, *marker + 1)
        })) {
        return std::nullopt;
    }
    const auto store = skip_limited_inline_spaces(
        input, boundaries, *marker + 1, length, 2
    );
    if (!store.has_value() || *store >= length
        || unicode_scalar_substring(input, boundaries, *store, *store + 1) != "店") {
        return std::nullopt;
    }
    return *store + 1;
}

std::optional<size_t> spaced_playstation_end(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const size_t length = boundaries.size() - 1;
    if (start >= length || !contains_any("Pp", {
            unicode_scalar_substring(input, boundaries, start, start + 1)
        })) {
        return std::nullopt;
    }
    const auto marker = skip_limited_inline_spaces(
        input, boundaries, start + 1, length, 2
    );
    if (!marker.has_value() || *marker >= length || !contains_any("Ss", {
            unicode_scalar_substring(input, boundaries, *marker, *marker + 1)
        })) {
        return std::nullopt;
    }
    const auto two = skip_limited_inline_spaces(
        input, boundaries, *marker + 1, length, 2
    );
    if (!two.has_value() || *two >= length
        || unicode_scalar_substring(input, boundaries, *two, *two + 1) != "二") {
        return std::nullopt;
    }
    return *two + 1;
}

RuleApproval approve_isolated_digit_g(
    const std::string& input,
    const SafePolicyContext&,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto shape_start = structured_identifier_shape_start(
        input, *boundaries, range.start, range.end
    );
    const auto expected_end = shape_start.has_value()
        ? spaced_digit_g_end(input, *boundaries, *shape_start) : std::nullopt;
    if (!shape_start.has_value() || !expected_end.has_value() || *expected_end != range.end
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "not_isolated_digit_g"};
    }
    const auto digit = spoken_year_digit(unicode_scalar_substring(
        input, *boundaries, *shape_start, *shape_start + 1
    ));
    const std::string marker = unicode_scalar_substring(
        input, *boundaries, range.end - 1, range.end
    );
    if (!digit.has_value() || *digit < 2 || *digit > 6 || (marker != "G" && marker != "g")) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_isolated_digit_g"};
    }
    if (range.start > 0) {
        const std::string previous = previous_non_space_scalar(
            input, *boundaries, range.start
        );
        if (number_continuation(previous) || previous == "幺" || previous == "点"
            || (previous.size() == 1 && std::isalnum(static_cast<unsigned char>(previous[0])))) {
            return {RuleDecision::preserve, "", {}, {}, "identifier_continues_left"};
        }
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = next_non_space_scalar(input, *boundaries, range.end);
        if (number_continuation(next) || next == "幺" || next == "点"
            || (next.size() == 1 && std::isalnum(static_cast<unsigned char>(next[0])))) {
            return {RuleDecision::preserve, "", {}, {}, "identifier_continues_right"};
        }
    }
    if (ambiguous_digit_g_quantity_context(input, *boundaries, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "ambiguous_digit_g_quantity_context"};
    }
    const std::string replacement = std::to_string(*digit) + "G";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_digit_plus_latin_g", kIsolatedDigitGRule},
            {"boundary", "complete_isolated_digit_g", kIsolatedDigitGRule},
            {"context", "non_quantity_mixed_script_context", kIsolatedDigitGRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "standard_isolated_digit_g_identifier",
    };
}

bool ascii_alnum_scalar(const std::string& value) {
    return value.size() == 1
        && std::isalnum(static_cast<unsigned char>(value[0])) != 0;
}

RuleApproval approve_respirator_standard(
    const std::string& input,
    const SafePolicyContext&,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto shape_start = structured_identifier_shape_start(
        input, *boundaries, range.start, range.end
    );
    const auto expected_end = shape_start.has_value()
        ? spaced_respirator_end(input, *boundaries, *shape_start) : std::nullopt;
    if (!shape_start.has_value() || !expected_end.has_value() || *expected_end != range.end
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_respirator_identifier"};
    }
    if (!contains_any(unicode_scalar_substring(
            input,
            *boundaries,
            range.start > 12 ? range.start - 12 : 0,
            std::min(boundaries->size() - 1, range.end + 12)
        ), {"口罩", "呼吸器", "呼吸防护", "医用防护", "佩戴", "戴上"})) {
        return {RuleDecision::preserve, "", {}, {}, "missing_respirator_context"};
    }
    if (range.start > 0 && ascii_alnum_scalar(previous_non_space_scalar(
            input, *boundaries, range.start
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "identifier_continues_left"};
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = next_non_space_scalar(input, *boundaries, range.end);
        if (ascii_alnum_scalar(next) || number_continuation(next)) {
            return {RuleDecision::preserve, "", {}, {}, "identifier_continues_right"};
        }
    }
    return {
        RuleDecision::approve,
        "N95",
        {
            {"shape", "latin_n_spoken_nine_five", kRespiratorRule},
            {"boundary", "complete_n95_identifier", kRespiratorRule},
            {"context", "respirator_context", kRespiratorRule},
        },
        {{{range.start, range.end, range.text}, "N95"}},
        "standard_respirator_identifier",
    };
}

RuleApproval approve_automotive_store(
    const std::string& input,
    const SafePolicyContext&,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto shape_start = structured_identifier_shape_start(
        input, *boundaries, range.start, range.end
    );
    const auto expected_end = spaced_automotive_store_end(
        input, *boundaries, shape_start.value_or(range.end)
    );
    if (!shape_start.has_value() || !expected_end.has_value() || *expected_end != range.end
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_automotive_store_identifier"};
    }
    if (range.start > 0 && ascii_alnum_scalar(previous_non_space_scalar(
            input, *boundaries, range.start
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "identifier_continues_left"};
    }
    if (range.end < boundaries->size() - 1 && ascii_alnum_scalar(next_non_space_scalar(
            input, *boundaries, range.end
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "identifier_continues_right"};
    }
    return {
        RuleDecision::approve,
        "4S店",
        {
            {"shape", "spoken_four_s_store", kAutomotiveStoreRule},
            {"boundary", "complete_4s_store_term", kAutomotiveStoreRule},
            {"context", "automotive_store_lexeme", kAutomotiveStoreRule},
        },
        {{{range.start, range.end, range.text}, "4S店"}},
        "standard_automotive_store_identifier",
    };
}

bool playstation_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const std::string context = unicode_scalar_substring(
        input,
        boundaries,
        start > 16 ? start - 16 : 0,
        std::min(boundaries.size() - 1, end + 16)
    );
    return contains_any(context, {
        "索尼", "SONY", "Sony", "PlayStation", "PLAYSTATION", "游戏机", "主机",
    });
}

RuleApproval approve_playstation_model(
    const std::string& input,
    const SafePolicyContext&,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto shape_start = structured_identifier_shape_start(
        input, *boundaries, range.start, range.end
    );
    const auto expected_end = shape_start.has_value()
        ? spaced_playstation_end(input, *boundaries, *shape_start) : std::nullopt;
    if (!shape_start.has_value() || !expected_end.has_value() || *expected_end != range.end
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || !playstation_context(input, *boundaries, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_playstation_identifier"};
    }
    if (range.start > 0 && ascii_alnum_scalar(previous_non_space_scalar(
            input, *boundaries, range.start
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "identifier_continues_left"};
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = next_non_space_scalar(input, *boundaries, range.end);
        if (ascii_alnum_scalar(next)) {
            return {RuleDecision::preserve, "", {}, {}, "identifier_continues_right"};
        }
        if (number_continuation(next)) {
            const std::string right = unicode_scalar_substring(
                input,
                *boundaries,
                range.end,
                std::min(boundaries->size() - 1, range.end + 12)
            );
            if (next != "两" || !contains_any(right, {"台", "订单"})) {
                return {RuleDecision::preserve, "", {}, {}, "ambiguous_numeric_suffix"};
            }
        }
    }
    return {
        RuleDecision::approve,
        "PS2",
        {
            {"shape", "latin_ps_spoken_two", kPlayStationRule},
            {"boundary", "complete_playstation_model", kPlayStationRule},
            {"context", "sony_playstation_context", kPlayStationRule},
        },
        {{{range.start, range.end, range.text}, "PS2"}},
        "standard_playstation_model",
    };
}

std::optional<std::string> closed_product_model_replacement(
    const std::string& value
) {
    if (value == "波音七三七八零零客机") return "波音737-800客机";
    if (value == "波音七三七八零零飞机") return "波音737-800飞机";
    if (value == "F杠二十二战斗机" || value == "f杠二十二战斗机") {
        return "F-22战斗机";
    }
    if (value == "F杠二十二隐形战斗机" || value == "f杠二十二隐形战斗机") {
        return "F-22隐形战斗机";
    }
    if (value == "米二直升机") return "米2直升机";
    if (value == "米二直升飞机") return "米2直升飞机";
    return std::nullopt;
}

RuleApproval approve_closed_product_model(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_closed_product_model"};
    }
    const auto replacement = closed_product_model_replacement(range.text);
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "unsupported_closed_product_model"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "closed_spoken_product_model", kClosedProductModelRule},
            {"boundary", "complete_product_model_and_type", kClosedProductModelRule},
            {"context", "closed_product_entity_and_type", kClosedProductModelRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_closed_product_model",
    };
}

bool integer_money_number_scalar(const std::string& value);

std::optional<std::string> contextual_product_model_replacement(
    const std::string& value
) {
    const auto boundaries = unicode_scalar_boundaries(value);
    if (!boundaries.has_value()) return std::nullopt;
    const size_t length = boundaries->size() - 1;
    size_t prefix_end = 0;
    std::string prefix;
    while (prefix_end < length) {
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, prefix_end, prefix_end + 1
        );
        if (scalar.size() != 1
            || !std::isupper(static_cast<unsigned char>(scalar[0]))) {
            break;
        }
        prefix += scalar;
        ++prefix_end;
    }
    if (prefix.empty() || prefix == "N" || prefix == "PS"
        || prefix.size() > 4 || prefix_end >= length) {
        return std::nullopt;
    }
    bool separated = false;
    size_t number_start = prefix_end;
    if (unicode_scalar_substring(
            value, *boundaries, number_start, number_start + 1
        ) == " ") {
        separated = true;
        ++number_start;
    }
    size_t number_end = number_start;
    while (number_end < length) {
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, number_end, number_end + 1
        );
        if (!integer_money_number_scalar(scalar)
            && scalar != "幺" && scalar != "洞") {
            break;
        }
        ++number_end;
    }
    if (number_end <= number_start) return std::nullopt;
    std::string suffix;
    for (size_t index = number_end; index < length; ++index) {
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, index, index + 1
        );
        if (scalar.size() != 1
            || !std::isalpha(static_cast<unsigned char>(scalar[0]))) {
            return std::nullopt;
        }
        suffix += scalar;
    }
    const auto number = parse_spoken_money_integer(
        value, *boundaries, number_start, number_end
    );
    std::string normalized_suffix = suffix;
    const bool named_suffix = suffix.size() > 1
        && std::isupper(static_cast<unsigned char>(suffix[0]))
        && std::islower(static_cast<unsigned char>(suffix[1]));
    if (named_suffix || suffix == "AI") {
        normalized_suffix = " " + suffix;
    } else if (suffix.size() > 2 && ends_with_any(suffix, {"AI"})) {
        normalized_suffix = suffix.substr(0, suffix.size() - 2) + " AI";
    }
    const std::string normalized_prefix = prefix + (separated ? " " : "");
    if (number.has_value() && *number <= 9999) {
        return normalized_prefix + std::to_string(*number) + normalized_suffix;
    }
    if (number_end - number_start > 8) return std::nullopt;
    std::string digits;
    for (size_t index = number_start; index < number_end; ++index) {
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, index, index + 1
        );
        const auto digit = scalar == "幺" ? std::optional<int>{1}
            : scalar == "洞" ? std::optional<int>{0} : spoken_year_digit(scalar);
        if (!digit.has_value()) return std::nullopt;
        digits += static_cast<char>('0' + *digit);
    }
    return normalized_prefix + digits + normalized_suffix;
}

bool contextual_product_model_applies(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const std::string left = context_window(input, boundaries, 0, start, 24);
    const std::string right = unicode_scalar_substring(
        input, boundaries, end, std::min(boundaries.size() - 1, end + 32)
    );
    const bool numbered_test_device = starts_with_any(right, {
        "号测试机", "号测试设备",
    });
    if (!numbered_test_device && starts_with_any(right, {
            "点", "年", "月", "日", "号", "个", "次", "种", "款", "台", "套", "组", "类", "项",
            "章", "节", "万元", "亿元", "元", "个月", "小时", "分钟",
        })) {
        return false;
    }
    return numbered_test_device || starts_with_any(right, {
            "车型", "车系", "芯片", "处理器", "主机", "型号", "系列", "机型",
            "服务器芯片", "加速器", "提升", "最高提升", "Ultra", "统一内存",
        })
        || contains_any(right, {
            "车型", "车系", "芯片", "处理器", "换代", "上市交付", "累计下线",
            "高速", "国道", "枢纽", "公交",
        })
        || contains_any(left, {
            "车型", "车系", "芯片", "处理器", "主机", "型号", "机型", "高速", "国道", "枢纽", "公交",
        });
}

RuleApproval approve_contextual_product_model(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || !contextual_product_model_applies(
            input, *boundaries, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_product_model_context"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (previous.size() == 1
            && std::isalnum(static_cast<unsigned char>(previous[0]))) {
            return {RuleDecision::preserve, "", {}, {}, "product_model_continues_left"};
        }
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string right = right_clause_context(
            input, *boundaries, range.end, 8
        );
        if (number_continuation(next)
            || (next.size() == 1
                && std::isalnum(static_cast<unsigned char>(next[0]))
                && !starts_with_any(right, {"Ultra"}))) {
            return {RuleDecision::preserve, "", {}, {}, "product_model_continues_right"};
        }
    }
    const auto replacement = contextual_product_model_replacement(range.text);
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_product_model"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "latin_prefix_spoken_numeric_model", kContextualProductModelRule},
            {"boundary", "complete_contextual_product_model", kContextualProductModelRule},
            {"context", "explicit_product_type_or_lifecycle_anchor", kContextualProductModelRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_contextual_product_model",
    };
}

std::optional<std::string> parse_spoken_identifier_digits(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    size_t minimum,
    size_t maximum
);
std::optional<int> spoken_telephone_digit(const std::string& value);

std::optional<std::string> land_parcel_identifier_replacement(const std::string& value) {
    const auto boundaries = unicode_scalar_boundaries(value);
    if (!boundaries.has_value()) return std::nullopt;
    const size_t length = boundaries->size() - 1;
    size_t cursor = 0;
    std::string replacement;
    while (cursor < length && cursor < 4) {
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, cursor, cursor + 1
        );
        if (scalar.size() != 1
            || !std::isupper(static_cast<unsigned char>(scalar[0]))) {
            break;
        }
        replacement += scalar;
        ++cursor;
    }
    if (replacement.empty() || cursor >= length) return std::nullopt;
    size_t group_count = 0;
    while (cursor < length) {
        const size_t group_start = cursor;
        while (cursor < length && spoken_telephone_digit(unicode_scalar_substring(
                value, *boundaries, cursor, cursor + 1
            )).has_value()) {
            ++cursor;
        }
        const auto digits = parse_spoken_identifier_digits(
            value, *boundaries, group_start, cursor, 2, 8
        );
        if (!digits.has_value()) return std::nullopt;
        replacement += group_count == 0 ? *digits : "-" + *digits;
        ++group_count;
        if (cursor == length) break;
        if (unicode_scalar_substring(value, *boundaries, cursor, cursor + 1) != "杠") {
            return std::nullopt;
        }
        ++cursor;
        if (cursor == length) return std::nullopt;
    }
    if (group_count < 2 || group_count > 4) return std::nullopt;
    return replacement;
}

bool land_parcel_identifier_applies(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t end
) {
    const std::string right = unicode_scalar_substring(
        input, boundaries, end, std::min(boundaries.size() - 1, end + 3)
    );
    return starts_with_any(right, {"地块", "宗地"});
}

RuleApproval approve_land_parcel_identifier(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = land_parcel_identifier_replacement(range.text);
    if (!replacement.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !land_parcel_identifier_applies(input, *boundaries, range.end)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_land_parcel_identifier"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (previous.size() == 1
            && std::isalnum(static_cast<unsigned char>(previous[0]))) {
            return {RuleDecision::preserve, "", {}, {}, "land_parcel_continues_left"};
        }
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "latin_prefix_grouped_spoken_digits", kLandParcelIdentifierRule},
            {"boundary", "complete_grouped_land_parcel_identifier", kLandParcelIdentifierRule},
            {"context", "explicit_land_parcel_suffix", kLandParcelIdentifierRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_land_parcel_identifier",
    };
}

bool explicit_display_resolution_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const std::string left = context_window(input, boundaries, 0, start, 12);
    return ends_with_any(left, {
        "分辨率", "分辨率为", "分辨率是", "分辨率达", "分辨率达到",
        "分辨率高达", "分辨率：", "分辨率:",
    });
}

std::optional<std::string> parse_display_resolution(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (size_t connector = start + 1; connector + 1 < end; ++connector) {
        if (unicode_scalar_substring(
                input, boundaries, connector, connector + 1
            ) != "乘") {
            continue;
        }
        const auto width = parse_spoken_money_integer(
            input, boundaries, start, connector
        );
        const auto height = parse_spoken_money_integer(
            input, boundaries, connector + 1, end
        );
        if (!width.has_value() || !height.has_value()
            || *width < 100 || *width > 99999 || *height < 100 || *height > 99999) {
            return std::nullopt;
        }
        const auto formatted_width = format_spoken_money_integer(
            input, boundaries, start, connector
        );
        const auto formatted_height = format_spoken_money_integer(
            input, boundaries, connector + 1, end
        );
        if (!formatted_width.has_value() || !formatted_height.has_value()) {
            return std::nullopt;
        }
        return *formatted_width + "×" + *formatted_height;
    }
    return std::nullopt;
}

RuleApproval approve_display_resolution(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || !explicit_display_resolution_context(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_display_resolution_context"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || next == "乘" || next == "点") {
            return {RuleDecision::preserve, "", {}, {}, "display_resolution_continues_right"};
        }
    }
    const auto replacement = parse_display_resolution(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_display_resolution"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "two_spoken_resolution_dimensions", kDisplayResolutionRule},
            {"boundary", "complete_display_resolution", kDisplayResolutionRule},
            {"context", "explicit_resolution_anchor", kDisplayResolutionRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_display_resolution",
    };
}

bool spoken_decimal_scalar(const std::string& value) {
    return spoken_cardinal_digit(value).has_value()
        || spoken_small_unit(value).has_value() || value == "点";
}

bool explicit_ratio_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    size_t cue_end = start;
    while (cue_end > 0 && contains_any("是为：: \t", {unicode_scalar_substring(
            input, boundaries, cue_end - 1, cue_end
        )})) {
        --cue_end;
    }
    const std::string left = context_window(input, boundaries, 0, cue_end, 16);
    return ends_with_any(left, {
        "比分", "画面比例", "屏幕比例", "图像比例", "宽高比", "长宽比",
        "稀释比例", "配比", "赔率", "比例", "按照",
    });
}

std::optional<std::string> parse_contextual_ratio(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t connector = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) != "比") continue;
        if (connector != end) return std::nullopt;
        connector = index;
    }
    if (connector == start || connector + 1 >= end) return std::nullopt;
    const auto left = parse_percentage_value(input, boundaries, start, connector);
    const auto right = parse_percentage_value(input, boundaries, connector + 1, end);
    if (!left.has_value() || !right.has_value()) return std::nullopt;
    return *left + "比" + *right;
}

RuleApproval approve_explicit_contextual_ratio(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto replacement = parse_contextual_ratio(
        input, *boundaries, range.start, range.end
    );
    const std::string next = range.end < boundaries->size() - 1
        ? unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) : "";
    if (!replacement.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !explicit_ratio_anchor(input, *boundaries, range.start)
        || (!next.empty() && (next == "比" || number_continuation(next) || next == "点"
            || contains_any("元人名位个次岁年月日号", {next})))
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_contextual_ratio"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "two_spoken_ratio_operands", kExplicitContextualRatioRule},
            {"boundary", "complete_contextual_ratio", kExplicitContextualRatioRule},
            {"context", "explicit_ratio_or_score_anchor", kExplicitContextualRatioRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_contextual_ratio",
    };
}

bool contextual_scalar_ratio_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    size_t cue_end = start;
    while (cue_end > 0 && contains_any("是为：: \t", {unicode_scalar_substring(
            input, boundaries, cue_end - 1, cue_end
        )})) {
        --cue_end;
    }
    const std::string left = context_window(input, boundaries, 0, cue_end, 12);
    return ends_with_any(left, {
        "容积率", "平均值", "平均偏差", "平均偏差约", "允许误差", "允许误差为正负",
        "最低值", "最低值是负", "最高值", "最高值是正", "最小值", "最大值",
        "声道格式", "pH值",
    });
}

std::optional<std::string> parse_contextual_scalar_value(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t decimal_mark = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != end) return std::nullopt;
            decimal_mark = index;
        }
    }
    if (decimal_mark == start || decimal_mark == end
        || decimal_mark + 1 == end || end - decimal_mark - 1 > 3) {
        return std::nullopt;
    }
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, decimal_mark
    );
    if (!integer.has_value()) return std::nullopt;
    std::string result = std::to_string(*integer) + ".";
    for (size_t index = decimal_mark + 1; index < end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        result += static_cast<char>('0' + *digit);
    }
    return result;
}

RuleApproval approve_contextual_scalar_ratio(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = parse_contextual_scalar_value(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value() || range.text.find("点") == std::string::npos
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !contextual_scalar_ratio_anchor(input, *boundaries, range.start)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_contextual_scalar_ratio"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || next == "点") {
            return {RuleDecision::preserve, "", {}, {}, "scalar_ratio_continues_right"};
        }
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_decimal_scalar", kContextualScalarRatioRule},
            {"boundary", "complete_contextual_scalar", kContextualScalarRatioRule},
            {"context", "explicit_dimensionless_ratio_anchor", kContextualScalarRatioRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_contextual_scalar_ratio",
    };
}

bool contextual_setting_value_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const std::string left = left_clause_context(input, boundaries, start, 20);
    return ends_with_any(left, {
        "音量调到", "音量调整到", "音量设为", "音量设置为",
    });
}

RuleApproval approve_contextual_setting_value(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto value = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end
    );
    if (!value.has_value() || *value < 0 || *value > 100
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !contextual_setting_value_anchor(input, *boundaries, range.start)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_contextual_setting_value"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || next == "点") {
            return {RuleDecision::preserve, "", {}, {}, "setting_value_continues_right"};
        }
    }
    const std::string replacement = std::to_string(*value);
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_integer_setting_value", kContextualSettingValueRule},
            {"boundary", "complete_contextual_setting_value", kContextualSettingValueRule},
            {"context", "explicit_device_setting_anchor", kContextualSettingValueRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "standard_contextual_setting_value",
    };
}

bool contextual_score_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const std::string left = context_window(input, boundaries, 0, start, 16);
    return ends_with_any(left, {
        "语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
        "考了", "考到", "考得", "考试得", "成绩为", "成绩是", "得分为", "得分是",
        "满分", "录取分数线为", "录取分数线是",
        "总分为", "总分是", "平均分为", "平均分是",
        "选手得到", "选手拿到", "选手砍下", "得到", "拿到", "砍下", "贡献",
    });
}

RuleApproval approve_contextual_score(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = parse_percentage_value(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || range.end >= length
        || unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) != "分"
        || !contextual_score_anchor(input, *boundaries, range.start)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end + 1)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_contextual_score"};
    }
    if (range.end + 1 < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end + 1, range.end + 2
        );
        if (next == "钟" || next == "之" || number_continuation(next) || next == "点") {
            return {RuleDecision::preserve, "", {}, {}, "score_unit_continues_right"};
        }
    }
    const double numeric = std::stod(*replacement);
    if (numeric < 1.0 || numeric > 1000.0
        || (range.text.find("点") == std::string::npos && numeric < 10.0)) {
        return {RuleDecision::preserve, "", {}, {}, "score_outside_conservative_range"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_number_plus_score_unit", kContextualScoreRule},
            {"boundary", "complete_contextual_score", kContextualScoreRule},
            {"context", "explicit_academic_or_sports_score_anchor", kContextualScoreRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_contextual_score",
    };
}

bool contextual_rank_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t ordinal_start
) {
    const std::string left = left_clause_context(input, boundaries, ordinal_start, 20);
    return contains_any(left, {
        "排名", "名次", "排在", "位列", "名列", "获得", "取得", "拿到",
    });
}

RuleApproval approve_contextual_rank(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value() || *replacement < 10 || *replacement > 9999
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || range.start == 0 || range.end >= length
        || unicode_scalar_substring(input, *boundaries, range.start - 1, range.start) != "第"
        || unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) != "名"
        || !contextual_rank_anchor(input, *boundaries, range.start - 1)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start - 1, range.end + 1)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_contextual_rank"};
    }
    const std::string formatted = std::to_string(*replacement);
    return {
        RuleDecision::approve,
        formatted,
        {
            {"shape", "spoken_ordinal_plus_rank_unit", kContextualRankRule},
            {"boundary", "complete_contextual_rank", kContextualRankRule},
            {"context", "explicit_ranking_anchor", kContextualRankRule},
        },
        {{{range.start, range.end, range.text}, formatted}},
        "standard_contextual_rank",
    };
}

bool contextual_document_page_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t ordinal_start
) {
    const std::string left = left_clause_context(input, boundaries, ordinal_start, 24);
    return contains_any(left, {
        "试卷", "文件", "报告", "文档", "手册", "说明书", "合同", "附件",
        "论文", "课本", "教材", "书籍", "书中", "请查看", "参见", "详见",
        "翻到", "打开",
    });
}

RuleApproval approve_contextual_document_page(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value() || *replacement < 10 || *replacement > 9999
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || range.start == 0 || range.end >= length
        || unicode_scalar_substring(input, *boundaries, range.start - 1, range.start) != "第"
        || unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) != "页"
        || !contextual_document_page_anchor(input, *boundaries, range.start - 1)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start - 1, range.end + 1)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_contextual_document_page"};
    }
    const std::string formatted = std::to_string(*replacement);
    return {
        RuleDecision::approve,
        formatted,
        {
            {"shape", "spoken_ordinal_plus_page_unit", kContextualDocumentPageRule},
            {"boundary", "complete_contextual_document_page", kContextualDocumentPageRule},
            {"context", "explicit_document_anchor", kContextualDocumentPageRule},
        },
        {{{range.start, range.end, range.text}, formatted}},
        "standard_contextual_document_page",
    };
}

bool contextual_structured_ordinal_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t ordinal_start,
    size_t number_end,
    const std::string& unit
) {
    const std::string left = left_clause_context(input, boundaries, ordinal_start, 24);
    const std::string right = right_clause_context(input, boundaries, number_end + 1, 8);
    if (unit == "组") {
        return contains_any(left, {"实验", "测量", "样品", "数据"})
            || starts_with_any(right, {"数据", "样品", "测量"});
    }
    if (unit == "层") {
        return contains_any(left, {
            "房源", "房屋", "楼盘", "楼层", "位于", "所在",
        });
    }
    if (unit == "首") {
        return contains_any(left, {"播放列表", "歌单"})
            && starts_with_any(right, {"歌", "歌曲"});
    }
    return false;
}

RuleApproval approve_contextual_structured_ordinal(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto value = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end
    );
    const std::string unit = range.end < length
        ? unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) : "";
    if (!value.has_value() || *value < 1 || *value > 9999
        || range.start == 0 || range.end >= length
        || unicode_scalar_substring(input, *boundaries, range.start - 1, range.start) != "第"
        || !contains_any(unit, {"组", "层", "首"})
        || !contextual_structured_ordinal_anchor(
            input, *boundaries, range.start - 1, range.end, unit
        )
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, length, range.start - 1, range.end + 1
        )) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_structured_ordinal"};
    }
    const std::string replacement = std::to_string(*value);
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_ordinal_plus_structural_unit", kContextualStructuredOrdinalRule},
            {"boundary", "complete_structured_ordinal", kContextualStructuredOrdinalRule},
            {"context", "explicit_structured_reference_anchor", kContextualStructuredOrdinalRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "standard_contextual_structured_ordinal",
    };
}

RuleApproval approve_contextual_anchored_cardinal(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto value = parse_spoken_large_count_integer(
        input, *boundaries, range.start, range.end
    );
    const auto formatted = format_spoken_large_count_integer(
        input, *boundaries, range.start, range.end
    );
    const std::string left = left_clause_context(input, *boundaries, range.start, 28);
    const std::string right = right_clause_context(input, *boundaries, range.end, 14);
    const bool preceded_by_ordinal = range.start > 0
        && unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        ) == "第";
    const bool exact_counter = contains_any(input, {"计数器"})
        && (contains_any(left, {"跳到"}) || starts_with_any(right, {"跳到"}));
    const bool elapsed_days = contains_any(left, {"持续"})
        && starts_with_any(right, {"天多"});
    const bool building_number = starts_with_any(right, {"号楼"})
        && ((preceded_by_ordinal && contains_any(left, {"楼盘", "房源", "位于"}))
            || contains_any(left, {"地址", "收件地址"}));
    const bool address_unit = starts_with_any(right, {"单元"})
        && contains_any(left, {"地址", "收件地址", "号楼", "栋"});
    const bool train_carriage = starts_with_any(right, {"车"})
        && contains_any(left, {"座位", "车票", "列车"});
    const std::string next = range.end < length
        ? unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) : "";
    const bool seat_code = next.size() == 1
        && std::isupper(static_cast<unsigned char>(next[0]))
        && contains_any(left, {"座位", "车厢"});
    const bool subway_line = (starts_with_any(right, {"号线"})
            && contains_any(left, {"地铁"}))
        || starts_with_any(right, {"号线地铁站"});
    const bool property_term = starts_with_any(right, {"年"})
        && contains_any(left, {"产权年限"});
    const bool standing = preceded_by_ordinal && contains_any(left, {"暂列"});
    const bool approximate_shots = ends_with_any(left, {"将近"})
        && contains_any(left, {"射门"}) && starts_with_any(right, {"次"});
    const bool approximate_score = ends_with_any(left, {"将近"})
        && contains_any(left, {"录取线", "录取分数线"})
        && starts_with_any(right, {"分"});
    const bool implied_revenue = ends_with_any(left, {"营收不足"})
        && (ends_with_any(range.text, {"万"}) || starts_with_any(right, {"万"}));
    const bool threshold_mixed_money = ends_with_any(left, {"原价最多"})
        && starts_with_any(right, {"元"});
    const bool threshold_money_fraction = contains_any(left, {"原价最多"})
        && ends_with_any(left, {"元"}) && starts_with_any(right, {"角"});
    if (!value.has_value() || !formatted.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || (!exact_counter && !elapsed_days && !building_number && !address_unit
            && !train_carriage && !seat_code && !subway_line && !property_term && !standing
            && !approximate_shots && !approximate_score && !implied_revenue
            && !threshold_mixed_money && !threshold_money_fraction)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_anchored_cardinal"};
    }
    if ((elapsed_days && *value < 10)
        || (standing && *value < 10) || (approximate_shots && *value < 10)
        || (approximate_score && *value < 10) || (implied_revenue && *value < 10)
        || (property_term && *value < 10)) {
        return {RuleDecision::preserve, "", {}, {}, "natural_single_digit_cardinal"};
    }
    const std::string replacement = (threshold_mixed_money || implied_revenue)
        ? *formatted : std::to_string(*value);
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_cardinal", kContextualAnchoredCardinalRule},
            {"boundary", "complete_contextual_cardinal", kContextualAnchoredCardinalRule},
            {"context", "explicit_released_cardinal_anchor", kContextualAnchoredCardinalRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "contextual_anchored_cardinal",
    };
}

RuleApproval approve_technical_identifier(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_technical_identifier"};
    }
    std::string replacement;
    if (range.text == "H. 点二六四" || range.text == "H.点二六四"
        || range.text == "H点二六四") {
        replacement = "H.264";
    } else if (range.text == "WiFi六E") {
        replacement = "WiFi 6E";
    } else {
        const auto parse_prefixed_version = [&](const std::string& prefix)
                -> std::optional<std::string> {
            const auto prefix_boundaries = unicode_scalar_boundaries(prefix);
            if (!prefix_boundaries.has_value()) return std::nullopt;
            const size_t prefix_length = prefix_boundaries->size() - 1;
            if (range.start + prefix_length >= range.end
                || unicode_scalar_substring(
                    input, *boundaries, range.start, range.start + prefix_length
                ) != prefix) {
                return std::nullopt;
            }
            const std::string left = left_clause_context(
                input, *boundaries, range.start, 24
            );
            if (!contains_any(left, {
                    "项目使用", "使用", "采用", "开发环境", "运行环境", "工具链",
                })) {
                return std::nullopt;
            }
            const auto version = parse_software_version(
                input, *boundaries, range.start + prefix_length, range.end
            );
            if (!version.has_value()) return std::nullopt;
            return prefix + " " + *version;
        };
        if (const auto version = parse_prefixed_version("Python"); version.has_value()) {
            replacement = *version;
        } else if (const auto version = parse_prefixed_version("CUDA"); version.has_value()) {
            replacement = *version;
        } else if (starts_with_any(range.text, {"C加加"})) {
            const size_t prefix_length = 3;
            const std::string left = left_clause_context(
                input, *boundaries, range.start, 24
            );
            if (!contains_any(left, {
                    "项目使用", "使用", "采用", "开发环境", "运行环境", "工具链",
                })) {
                return {RuleDecision::preserve, "", {}, {},
                        "missing_technical_version_context"};
            }
            const auto standard = parse_spoken_cardinal(
                input, *boundaries, range.start + prefix_length, range.end
            );
            const bool supported_standard = standard.has_value()
                && (*standard == 11 || *standard == 14 || *standard == 17
                    || *standard == 20 || *standard == 23 || *standard == 26);
            if (!supported_standard) {
                return {RuleDecision::preserve, "", {}, {},
                        "unsupported_technical_identifier"};
            }
            replacement = "C++" + std::to_string(*standard);
        } else {
            return {RuleDecision::preserve, "", {}, {},
                    "unsupported_technical_identifier"};
        }
    }
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "closed_technical_notation", kTechnicalIdentifierRule},
            {"boundary", "complete_technical_identifier", kTechnicalIdentifierRule},
            {"context", "standard_codec_or_wireless_notation", kTechnicalIdentifierRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "contextual_technical_notation",
    };
}

std::optional<char> structured_identifier_digit(const std::string& scalar) {
    if (scalar == "零" || scalar == "〇" || scalar == "洞"
        || scalar == "栋" || scalar == "动") return '0';
    if (scalar == "一" || scalar == "幺") return '1';
    if (scalar == "二" || scalar == "两") return '2';
    if (scalar == "三") return '3';
    if (scalar == "四") return '4';
    if (scalar == "五") return '5';
    if (scalar == "六") return '6';
    if (scalar == "七") return '7';
    if (scalar == "八") return '8';
    if (scalar == "九") return '9';
    return std::nullopt;
}

char lowercase_ascii(char value) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
}

std::optional<std::string> parse_spoken_domain(const std::string& value) {
    const auto boundaries = unicode_scalar_boundaries(value);
    if (!boundaries.has_value()) return std::nullopt;
    std::string output;
    size_t dots = 0;
    bool label_has_value = false;
    for (size_t index = 0; index + 1 < boundaries->size(); ++index) {
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, index, index + 1
        );
        if (scalar == "点") {
            if (!label_has_value) return std::nullopt;
            output.push_back('.');
            ++dots;
            label_has_value = false;
            continue;
        }
        const auto digit = structured_identifier_digit(scalar);
        if (digit.has_value()) {
            output.push_back(*digit);
            label_has_value = true;
            continue;
        }
        if (scalar.size() == 1 && (std::isalnum(
                static_cast<unsigned char>(scalar[0])) || scalar[0] == '-')) {
            output.push_back(lowercase_ascii(scalar[0]));
            label_has_value = scalar[0] != '-';
            continue;
        }
        return std::nullopt;
    }
    if (!label_has_value || dots < 1 || output.back() == '-') return std::nullopt;
    const size_t suffix = output.rfind('.');
    if (suffix == std::string::npos || suffix + 3 > output.size()) return std::nullopt;
    for (size_t index = suffix + 1; index < output.size(); ++index) {
        if (!std::isalpha(static_cast<unsigned char>(output[index]))) return std::nullopt;
    }
    return output;
}

std::optional<std::string> parse_spoken_url(const std::string& value) {
    const std::string marker = "冒号双斜杠";
    const size_t marker_position = value.find(marker);
    if (marker_position == std::string::npos) return std::nullopt;
    std::string scheme = value.substr(0, marker_position);
    std::transform(scheme.begin(), scheme.end(), scheme.begin(), lowercase_ascii);
    if (scheme != "http" && scheme != "https") return std::nullopt;
    const std::string remainder = value.substr(marker_position + marker.size());
    const auto boundaries = unicode_scalar_boundaries(remainder);
    if (!boundaries.has_value()) return std::nullopt;
    std::string output = scheme + "://";
    size_t dots = 0;
    size_t slashes = 0;
    bool component_has_value = false;
    for (size_t index = 0; index + 1 < boundaries->size(); ++index) {
        const std::string scalar = unicode_scalar_substring(
            remainder, *boundaries, index, index + 1
        );
        if (scalar == "点") {
            if (!component_has_value || slashes > 0) return std::nullopt;
            output.push_back('.');
            ++dots;
            component_has_value = false;
            continue;
        }
        if (index + 2 < boundaries->size() && unicode_scalar_substring(
                remainder, *boundaries, index, index + 2
            ) == "斜杠") {
            if (!component_has_value) return std::nullopt;
            output.push_back('/');
            ++slashes;
            component_has_value = false;
            ++index;
            continue;
        }
        const auto digit = structured_identifier_digit(scalar);
        if (digit.has_value()) {
            output.push_back(*digit);
            component_has_value = true;
            continue;
        }
        if (scalar.size() == 1 && (std::isalnum(
                static_cast<unsigned char>(scalar[0])) || scalar[0] == '-'
                || scalar[0] == '_')) {
            output.push_back(lowercase_ascii(scalar[0]));
            component_has_value = true;
            continue;
        }
        return std::nullopt;
    }
    if (!component_has_value || dots < 1 || slashes == 0) return std::nullopt;
    return output;
}

std::optional<std::string> parse_spoken_windows_path(const std::string& value) {
    const auto boundaries = unicode_scalar_boundaries(value);
    if (!boundaries.has_value() || boundaries->size() < 6) return std::nullopt;
    const size_t length = boundaries->size() - 1;
    const std::string drive = unicode_scalar_substring(value, *boundaries, 0, 1);
    if (drive.size() != 1 || !std::isalpha(static_cast<unsigned char>(drive[0]))) {
        return std::nullopt;
    }
    size_t cursor = 1;
    while (cursor < length && unicode_scalar_substring(
            value, *boundaries, cursor, cursor + 1
        ) == " ") ++cursor;
    if (cursor + 4 > length || unicode_scalar_substring(
            value, *boundaries, cursor, cursor + 4
        ) != "盘反斜杠") return std::nullopt;
    cursor += 4;
    std::string output;
    output.push_back(static_cast<char>(std::toupper(
        static_cast<unsigned char>(drive[0])
    )));
    output += ":\\";
    size_t separators = 1;
    bool component_has_value = false;
    while (cursor < length) {
        if (cursor + 3 <= length && unicode_scalar_substring(
                value, *boundaries, cursor, cursor + 3
            ) == "反斜杠") {
            while (!output.empty() && output.back() == ' ') output.pop_back();
            if (!component_has_value) return std::nullopt;
            output.push_back('\\');
            ++separators;
            component_has_value = false;
            cursor += 3;
            continue;
        }
        const std::string scalar = unicode_scalar_substring(
            value, *boundaries, cursor, cursor + 1
        );
        if (scalar == "点" || scalar == "杠") {
            while (!output.empty() && output.back() == ' ') output.pop_back();
            if (!component_has_value) return std::nullopt;
            output.push_back(scalar == "点" ? '.' : '-');
            component_has_value = false;
            ++cursor;
            continue;
        }
        const auto digit = structured_identifier_digit(scalar);
        if (digit.has_value()) {
            output.push_back(*digit);
            component_has_value = true;
            ++cursor;
            continue;
        }
        if (scalar == " ") {
            if (component_has_value && !output.empty() && output.back() != ' ') {
                output.push_back(' ');
            }
            ++cursor;
            continue;
        }
        if (scalar.size() == 1 && (std::isalnum(
                static_cast<unsigned char>(scalar[0])) || scalar[0] == '_'
                || scalar[0] == '-')) {
            output.push_back(scalar[0]);
            component_has_value = true;
            ++cursor;
            continue;
        }
        return std::nullopt;
    }
    while (!output.empty() && output.back() == ' ') output.pop_back();
    if (!component_has_value || separators < 3 || output.size() < 5
        || output.substr(output.size() - 4) != ".log") return std::nullopt;
    return output;
}

RuleApproval approve_structured_spoken_identifier(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    std::string left = left_clause_context(input, *boundaries, range.start, 20);
    while (!left.empty() && left.back() == ' ') left.pop_back();
    const bool path_anchor = range.text.find("盘反斜杠") != std::string::npos
        && ends_with_any(left, {"日志文件位于", "文件位于", "路径是", "路径为"});
    const bool url_anchor = range.text.find("冒号双斜杠") != std::string::npos
        && ends_with_any(left, {"接口地址是", "接口地址为", "网址是", "网址为", "链接是", "链接为"});
    const bool domain_anchor = !path_anchor && !url_anchor
        && ends_with_any(left, {"网站域名是", "网站域名为", "域名是", "域名为"});
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || (!path_anchor && !url_anchor && !domain_anchor)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_structured_locator"};
    }
    std::optional<std::string> replacement;
    if (path_anchor) {
        replacement = parse_spoken_windows_path(range.text);
    } else if (url_anchor) {
        replacement = parse_spoken_url(range.text);
    } else {
        replacement = parse_spoken_domain(range.text);
    }
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "unsupported_structured_locator"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "strict_spoken_structured_locator", kStructuredSpokenIdentifierRule},
            {"boundary", "complete_structured_locator", kStructuredSpokenIdentifierRule},
            {"context", "explicit_domain_url_or_path_anchor", kStructuredSpokenIdentifierRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "structured_spoken_locator",
    };
}

std::optional<std::string> parse_explicit_geographic_coordinate(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (end < start + 4) return std::nullopt;
    const std::string anchor = unicode_scalar_substring(input, boundaries, start, start + 2);
    const bool latitude = anchor == "北纬" || anchor == "南纬" || anchor == "纬度";
    const bool longitude = anchor == "东经" || anchor == "西经" || anchor == "经度";
    if (!latitude && !longitude) return std::nullopt;
    size_t degree_mark = start + 2;
    while (degree_mark < end && unicode_scalar_substring(
            input, boundaries, degree_mark, degree_mark + 1
        ) != "度") {
        ++degree_mark;
    }
    if (degree_mark == start + 2 || degree_mark >= end) return std::nullopt;
    const auto degree = parse_percentage_value(
        input, boundaries, start + 2, degree_mark
    );
    if (!degree.has_value()) return std::nullopt;
    const double degree_value = std::stod(*degree);
    if (degree_value < 0.0 || (latitude && degree_value > 90.0)
        || (longitude && degree_value > 180.0)) {
        return std::nullopt;
    }
    std::string replacement = anchor + *degree + "度";
    size_t cursor = degree_mark + 1;
    if (cursor == end) return replacement;
    size_t minute_mark = cursor;
    while (minute_mark < end && unicode_scalar_substring(
            input, boundaries, minute_mark, minute_mark + 1
        ) != "分") {
        ++minute_mark;
    }
    if (minute_mark == cursor || minute_mark >= end) return std::nullopt;
    const auto minute = parse_percentage_value(input, boundaries, cursor, minute_mark);
    if (!minute.has_value() || std::stod(*minute) >= 60.0) return std::nullopt;
    replacement += *minute + "分";
    cursor = minute_mark + 1;
    if (cursor == end) return replacement;
    if (unicode_scalar_substring(input, boundaries, end - 1, end) != "秒") {
        return std::nullopt;
    }
    const auto second = parse_percentage_value(input, boundaries, cursor, end - 1);
    if (!second.has_value() || std::stod(*second) >= 60.0) return std::nullopt;
    return replacement + *second + "秒";
}

RuleApproval approve_explicit_geographic_coordinate(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto replacement = parse_explicit_geographic_coordinate(
        input, *boundaries, range.start, range.end
    );
    const std::string next = range.end < boundaries->size() - 1
        ? unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) : "";
    if (!replacement.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || (!next.empty() && (number_continuation(next) || next == "点"))
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_geographic_coordinate"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "directional_spoken_degree_value",
             kExplicitGeographicCoordinateRule},
            {"boundary", "complete_geographic_coordinate",
             kExplicitGeographicCoordinateRule},
            {"context", "explicit_latitude_or_longitude_anchor",
             kExplicitGeographicCoordinateRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_geographic_coordinate",
    };
}

bool version_component_scalar(const std::string& value) {
    return spoken_cardinal_digit(value).has_value() || spoken_small_unit(value).has_value();
}

std::optional<std::string> parse_version_component(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (start >= end) return std::nullopt;
    bool digit_sequence = true;
    std::string digits;
    for (size_t index = start; index < end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) {
            digit_sequence = false;
            break;
        }
        digits += static_cast<char>('0' + *digit);
    }
    if (digit_sequence) return digits;
    const auto value = parse_spoken_cardinal(input, boundaries, start, end);
    if (!value.has_value() || *value > 999) return std::nullopt;
    return std::to_string(*value);
}

std::optional<std::string> parse_software_version(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    std::vector<std::string> components;
    size_t component_start = start;
    for (size_t index = start; index <= end; ++index) {
        if (index != end && unicode_scalar_substring(
                input, boundaries, index, index + 1
            ) != "点") {
            continue;
        }
        const auto component = parse_version_component(
            input, boundaries, component_start, index
        );
        if (!component.has_value()) return std::nullopt;
        components.push_back(*component);
        component_start = index + 1;
    }
    if (components.size() < 2 || components.size() > 5) return std::nullopt;
    std::string output;
    for (size_t index = 0; index < components.size(); ++index) {
        if (index > 0) output += ".";
        output += components[index];
    }
    return output;
}

bool explicit_software_version_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const std::string left = context_window(input, boundaries, 0, start, 12);
    const std::string right = unicode_scalar_substring(
        input, boundaries, end, std::min(boundaries.size() - 1, end + 8)
    );
    const bool prefixed_version = start > 0
        && contains_any("Vv", {unicode_scalar_substring(
            input, boundaries, start - 1, start
        )})
        && ends_with_any(context_window(input, boundaries, 0, start - 1, 12), {
            "版本", "版本为", "版本是", "版本号", "版本号为", "版本号是",
            "软件版本", "固件版本", "应用版本", "加载版本",
        });
    return prefixed_version || ends_with_any(left, {
        "版本", "版本为", "版本是", "版本号", "版本号为", "版本号是",
        "软件版本", "软件版本为", "软件版本是", "固件版本", "固件版本为",
        "固件版本是", "应用版本", "应用版本为", "应用版本是",
    }) || starts_with_any(right, {"版本", "版本号", "构建版本", "构建号"});
}

RuleApproval approve_software_version(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_software_version"};
    }
    if (!explicit_software_version_context(
            input, *boundaries, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "missing_software_version_context"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (previous == "点" || number_continuation(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "version_continues_left"};
        }
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (next == "点" || number_continuation(next)) {
            return {RuleDecision::preserve, "", {}, {}, "version_continues_right"};
        }
    }
    const auto replacement = parse_software_version(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_version_components"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_dotted_version", kSoftwareVersionRule},
            {"boundary", "complete_version_components", kSoftwareVersionRule},
            {"context", "explicit_software_version_context", kSoftwareVersionRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_software_version",
    };
}

std::optional<std::string> parse_explicit_video_timecode(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t hour_mark = end;
    size_t minute_mark = end;
    if (end <= start || unicode_scalar_substring(
            input, boundaries, end - 1, end
        ) != "秒") {
        return std::nullopt;
    }
    for (size_t index = start; index + 1 < end; ++index) {
        const std::string scalar = unicode_scalar_substring(
            input, boundaries, index, index + 1
        );
        if (scalar == "小" && index + 1 < end
            && unicode_scalar_substring(input, boundaries, index + 1, index + 2) == "时") {
            if (hour_mark != end) return std::nullopt;
            hour_mark = index;
        } else if (scalar == "分") {
            if (minute_mark != end) return std::nullopt;
            minute_mark = index;
        }
    }
    if (hour_mark == start || hour_mark + 2 >= minute_mark
        || minute_mark + 1 >= end - 1) {
        return std::nullopt;
    }
    const auto hour_text = parse_version_component(input, boundaries, start, hour_mark);
    const auto minute_text = parse_version_component(
        input, boundaries, hour_mark + 2, minute_mark
    );
    const auto second_text = parse_version_component(
        input, boundaries, minute_mark + 1, end - 1
    );
    if (!hour_text.has_value() || !minute_text.has_value() || !second_text.has_value()) {
        return std::nullopt;
    }
    const int hour = std::stoi(*hour_text);
    const int minute = std::stoi(*minute_text);
    const int second = std::stoi(*second_text);
    if (hour < 0 || hour > 99 || minute < 0 || minute > 59
        || second < 0 || second > 59) {
        return std::nullopt;
    }
    char output[9];
    std::snprintf(output, sizeof(output), "%02d:%02d:%02d", hour, minute, second);
    return std::string(output);
}

RuleApproval approve_explicit_video_timecode(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const auto replacement = parse_explicit_video_timecode(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()
        || !ends_with_any(left, {"视频时间码是", "视频时间码为", "时间码是", "时间码为"})
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_video_timecode"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_hour_minute_second_timecode", kExplicitTimecodeRule},
            {"boundary", "complete_video_timecode", kExplicitTimecodeRule},
            {"context", "explicit_video_timecode_anchor", kExplicitTimecodeRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_video_timecode",
    };
}

RuleApproval approve_clinical_threshold(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = left_clause_context(input, *boundaries, range.start, 20);
    const auto replacement = parse_contextual_scalar_value(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()
        || !ends_with_any(left, {"空腹血糖必须小于", "空腹血糖应小于"})
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_clinical_threshold"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_decimal_clinical_threshold", kClinicalThresholdRule},
            {"boundary", "complete_clinical_threshold_value", kClinicalThresholdRule},
            {"context", "explicit_fasting_glucose_threshold", kClinicalThresholdRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_clinical_threshold_value",
    };
}

std::optional<int> spoken_telephone_digit(const std::string& value) {
    const auto digit = spoken_year_digit(value);
    if (digit.has_value()) return digit;
    if (value == "幺") return 1;
    if (value == "洞") return 0;
    if (value == "拐") return 7;
    if (value == "勾") return 9;
    return std::nullopt;
}

bool explicit_telephone_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t cue_end = start;
    while (cue_end > 0) {
        const std::string previous = unicode_scalar_substring(
            input, boundaries, cue_end - 1, cue_end
        );
        if (!contains_any("是为：: \t", {previous})) break;
        --cue_end;
    }
    const std::string left = context_window(input, boundaries, 0, cue_end, 20);
    const std::string right = unicode_scalar_substring(
        input,
        boundaries,
        end,
        std::min(boundaries.size() - 1, end + 10)
    );
    return ends_with_any(left, {"电话", "号码", "热线", "拨打", "呼叫", "致电"})
        || starts_with_any(right, {"打这个电话", "打电话", "电话", "号码", "热线"});
}

bool valid_domestic_telephone_shape(const std::string& digits) {
    const size_t length = digits.size();
    if (length >= 3 && length <= 9) return true;
    if (length == 10) {
        return digits.rfind("0", 0) == 0 || digits.rfind("400", 0) == 0
            || digits.rfind("800", 0) == 0;
    }
    if (length == 11) return digits[0] == '0' || digits[0] == '1';
    return false;
}

std::optional<std::string> parse_telephone_candidate_digits(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    std::string digits;
    bool recovered_boundary = false;
    for (size_t index = start; index < end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, boundaries, index, index + 1
        );
        const auto digit = spoken_telephone_digit(value);
        if (digit.has_value()) {
            digits += static_cast<char>('0' + *digit);
            continue;
        }
        if (recovered_boundary && contains_any(" \t\n\r", {value})) {
            continue;
        }
        if (value != "。" || recovered_boundary || digits.empty() || index + 1 >= end) {
            return std::nullopt;
        }
        recovered_boundary = true;
    }
    if (recovered_boundary && (digits.size() != 10 || digits.rfind("400", 0) != 0)) {
        return std::nullopt;
    }
    return digits;
}

bool telephone_list_continues_right(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (end >= length) return false;
    const std::string connector = unicode_scalar_substring(
        input, boundaries, end, end + 1
    );
    if (!contains_any("或和及与、,，加", {connector})) return false;
    size_t cursor = end + 1;
    while (cursor < length && contains_any(" \t", {unicode_scalar_substring(
            input, boundaries, cursor, cursor + 1
        )})) {
        ++cursor;
    }
    return cursor < length && spoken_telephone_digit(unicode_scalar_substring(
        input, boundaries, cursor, cursor + 1
    )).has_value();
}

RuleApproval approve_telephone_number(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )
        || !explicit_telephone_context(input, *boundaries, range.start, range.end)
        || telephone_list_continues_right(input, *boundaries, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_telephone_context"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (ascii_alnum_scalar(previous) || number_continuation(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "telephone_continues_left"};
        }
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (ascii_alnum_scalar(next) || number_continuation(next) || next == "点"
            || contains_any("年月日号", {next})) {
            return {RuleDecision::preserve, "", {}, {}, "telephone_continues_right"};
        }
    }
    const auto replacement = parse_telephone_candidate_digits(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_telephone_digits"};
    }
    if (!valid_domestic_telephone_shape(*replacement)) {
        return {RuleDecision::preserve, "", {}, {}, "unsupported_telephone_shape"};
    }
    const bool recovered_boundary = range.text.find("。") != std::string::npos;
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", recovered_boundary
                ? "spoken_400_service_number_with_asr_boundary"
                : "spoken_digit_sequence", kTelephoneRule},
            {"boundary", "complete_telephone_number", kTelephoneRule},
            {"context", "explicit_telephone_context", kTelephoneRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_telephone_identifier",
    };
}

bool contextual_public_service_number(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    const std::string& digits
) {
    const std::string left = context_window(input, boundaries, 0, start, 24);
    const std::string right = unicode_scalar_substring(
        input, boundaries, end, std::min(boundaries.size() - 1, end + 16)
    );
    if (digits == "12306") {
        return starts_with_any(right, {"网站", "客服中心", "客服", "候补购票"})
            || (contains_any(left, {"铁路", "铁道"})
                && starts_with_any(right, {"技术部", "服务平台"}));
    }
    if (digits == "12315") {
        return contains_any(left, {"市场监管", "消费者", "消费维权"})
            || starts_with_any(right, {"投诉举报专线", "投诉举报", "投诉", "举报"});
    }
    if (digits == "12345") {
        return contains_any(left, {"政务", "市民服务", "政府服务"})
            || starts_with_any(right, {"政务服务热线", "服务热线", "市场热线"});
    }
    if (digits == "12122") {
        return contains_any(left, {"高速", "交警", "交通", "陕西"})
            && starts_with_any(right, {"微信小程序", "救援", "报警", "热线"});
    }
    const bool carrier_role = starts_with_any(
        right, {"发送", "短信", "客服", "营业厅", "服务平台"}
    );
    if (digits == "10000") {
        return contains_any(left, {"中国电信", "电信客服"}) && carrier_role;
    }
    if (digits == "10010") {
        return contains_any(left, {"中国联通", "联通客服"}) && carrier_role;
    }
    if (digits == "10086") {
        return contains_any(left, {"中国移动", "移动客服"}) && carrier_role;
    }
    if (digits == "110" || digits == "119" || digits == "120" || digits == "122") {
        return ends_with_any(left, {"打了", "打给", "报警", "急救", "火警", "求助"});
    }
    return false;
}

RuleApproval approve_contextual_public_service_number(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto replacement = parse_telephone_candidate_digits(
        input, *boundaries, range.start, range.end
    );
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !replacement.has_value()
        || explicit_telephone_context(input, *boundaries, range.start, range.end)
        || !contextual_public_service_number(
            input, *boundaries, range.start, range.end, *replacement
        )
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_public_service_number"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (ascii_alnum_scalar(previous) || number_continuation(previous)) {
            return {RuleDecision::preserve, "", {}, {}, "service_number_continues_left"};
        }
    }
    if (range.end < boundaries->size() - 1) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (ascii_alnum_scalar(next) || number_continuation(next) || next == "点"
            || contains_any("年月日号分次个元名位人辆家台项岁", {next})) {
            return {RuleDecision::preserve, "", {}, {}, "service_number_continues_right"};
        }
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_public_service_digit_sequence", kPublicServiceNumberRule},
            {"boundary", "complete_public_service_number", kPublicServiceNumberRule},
            {"context", "closed_public_service_entity_or_role", kPublicServiceNumberRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_public_service_number",
    };
}

std::optional<std::string> parse_telephone_number_list(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    std::string contiguous_digits;
    for (size_t index = start; index < end; ++index) {
        const auto digit = spoken_telephone_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) {
            contiguous_digits.clear();
            break;
        }
        contiguous_digits += static_cast<char>('0' + *digit);
    }
    if (contiguous_digits == "119120110") return "119、120、110";

    std::string output;
    size_t cursor = start;
    size_t count = 0;
    while (cursor < end) {
        const size_t number_start = cursor;
        std::string digits;
        while (cursor < end) {
            const auto digit = spoken_telephone_digit(unicode_scalar_substring(
                input, boundaries, cursor, cursor + 1
            ));
            if (!digit.has_value()) break;
            digits += static_cast<char>('0' + *digit);
            ++cursor;
        }
        if (cursor == number_start || !valid_domestic_telephone_shape(digits)) {
            return std::nullopt;
        }
        output += digits;
        ++count;
        if (cursor == end) break;
        while (cursor < end && contains_any(" \t", {unicode_scalar_substring(
                input, boundaries, cursor, cursor + 1
            )})) {
            output += unicode_scalar_substring(input, boundaries, cursor, cursor + 1);
            ++cursor;
        }
        if (cursor >= end) return std::nullopt;
        const std::string connector = unicode_scalar_substring(
            input, boundaries, cursor, cursor + 1
        );
        if (!contains_any("或和及与、,，", {connector})) return std::nullopt;
        output += connector;
        ++cursor;
        while (cursor < end && contains_any(" \t", {unicode_scalar_substring(
                input, boundaries, cursor, cursor + 1
            )})) {
            output += unicode_scalar_substring(input, boundaries, cursor, cursor + 1);
            ++cursor;
        }
    }
    return count >= 2 ? std::optional<std::string>(output) : std::nullopt;
}

bool emergency_telephone_list_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    const std::string& replacement
) {
    if (replacement != "119、120、110") return false;
    const std::string left = context_window(input, boundaries, 0, start, 8);
    return ends_with_any(left, {"他给", "她给", "它给"});
}

RuleApproval approve_telephone_number_list(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_telephone_list_context"};
    }
    const auto replacement = parse_telephone_number_list(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_telephone_list_shape"};
    }
    if (!explicit_telephone_context(input, *boundaries, range.start, range.end)
        && !emergency_telephone_list_context(
            input, *boundaries, range.start, *replacement
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_telephone_list_context"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_telephone_number_list", kTelephoneListRule},
            {"boundary", "complete_telephone_number_list", kTelephoneListRule},
            {"context", "shared_explicit_telephone_context", kTelephoneListRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_telephone_identifier_list",
    };
}

std::optional<std::string> parse_spoken_identifier_digits(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    size_t minimum,
    size_t maximum
) {
    if (end <= start || end - start < minimum || end - start > maximum) {
        return std::nullopt;
    }
    std::string digits;
    for (size_t index = start; index < end; ++index) {
        const auto digit = spoken_telephone_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        digits += static_cast<char>('0' + *digit);
    }
    return digits;
}

bool explicit_identifier_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    size_t cue_end = start;
    while (cue_end > 0 && contains_any("是为：: \t", {unicode_scalar_substring(
            input, boundaries, cue_end - 1, cue_end
        )})) {
        --cue_end;
    }
    const std::string left = context_window(input, boundaries, 0, cue_end, 20);
    return ends_with_any(left, {
        "编号", "订单号", "取件码", "包裹单号", "快递单号", "运单号",
        "验证码", "邮政编码", "邮编", "工号", "学号",
        "合同号", "序列号", "设备号", "代码", "区号", "分机号", "分机",
    });
}

bool room_identifier_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (start == 0 || end >= length || end - start < 2 || end - start > 4) {
        return false;
    }
    const std::string previous = unicode_scalar_substring(
        input, boundaries, start - 1, start
    );
    const std::string right = unicode_scalar_substring(
        input, boundaries, end, std::min(length, end + 2)
    );
    if (starts_with_any(right, {"号房"})) return true;
    const std::string left = left_clause_context(input, boundaries, start, 20);
    return starts_with_any(right, {"室"})
        && (contains_any("座楼层栋", {previous})
            || contains_any(left, {"房子", "房间", "楼", "层", "栋", "座"}));
}

bool street_number_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (start == 0 || end >= length) return false;
    const std::string left = left_clause_context(input, boundaries, start, 24);
    const std::string right = right_clause_context(input, boundaries, end, 6);
    return starts_with_any(right, {"号"})
        && !starts_with_any(right, {"号文件", "号文", "号公告", "号命令"})
        && !ends_with_any(left, {"铁路", "公路", "线路", "道路"})
        && (ends_with_any(left, {"路", "街", "巷", "弄", "大道"})
            || contains_any(left, {"地址"}));
}

bool facility_number_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (end >= length) return false;
    const std::string right = right_clause_context(input, boundaries, end, 6);
    return starts_with_any(right, {
        "号驿站", "号快递柜", "号柜台", "号窗口",
    });
}

bool contextual_technical_model_number_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (start == 0 || end >= length || end - start < 3 || end - start > 8) {
        return false;
    }
    const std::string left = context_window(input, boundaries, 0, start, 20);
    const std::string right = right_clause_context(input, boundaries, end, 24);
    size_t dots = 0;
    size_t uppercase = 0;
    for (const unsigned char ch : left) {
        if (ch == '.') ++dots;
        if (std::isupper(ch)) ++uppercase;
    }
    const std::string next = unicode_scalar_substring(input, boundaries, end, end + 1);
    return dots >= 2 && uppercase >= 2
        && next.size() == 1
        && std::isupper(static_cast<unsigned char>(next[0]))
        && contains_any(right, {"服务器芯片", "芯片", "处理器", "加速器"});
}

RuleApproval approve_explicit_identifier(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    auto replacement = parse_spoken_identifier_digits(
        input, *boundaries, range.start, range.end, 2, 24
    );
    const bool standard_identifier = range.end - range.start >= 3
        && explicit_identifier_anchor(input, *boundaries, range.start);
    const bool room_identifier = room_identifier_anchor(
        input, *boundaries, range.start, range.end
    );
    const bool street_number = street_number_anchor(
        input, *boundaries, range.start, range.end
    );
    const bool facility_number = facility_number_anchor(
        input, *boundaries, range.start, range.end
    );
    const bool technical_model_number = contextual_technical_model_number_anchor(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value() && (street_number || facility_number)) {
        const auto integer = parse_spoken_money_integer(
            input, *boundaries, range.start, range.end
        );
        if (integer.has_value() && *integer >= 10 && *integer <= 99999) {
            replacement = std::to_string(*integer);
        }
    }
    if (!replacement.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || (!standard_identifier && !room_identifier && !street_number && !facility_number
            && !technical_model_number)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_explicit_identifier"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_digit_sequence", kExplicitIdentifierRule},
            {"boundary", "complete_explicit_identifier", kExplicitIdentifierRule},
            {"context", "closed_identifier_anchor", kExplicitIdentifierRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_explicit_identifier",
    };
}

std::optional<std::string> train_number_replacement(const std::string& value) {
    const auto boundaries = unicode_scalar_boundaries(value);
    if (!boundaries.has_value()) return std::nullopt;
    const size_t length = boundaries->size() - 1;
    if (length < 5) return std::nullopt;
    const std::string prefix = unicode_scalar_substring(value, *boundaries, 0, 1);
    if (!contains_any(prefix, {"G", "D", "C", "Z", "T", "K", "第"})) {
        return std::nullopt;
    }
    size_t suffix_start = length;
    std::string suffix;
    for (const std::string candidate : {"次列车", "车次"}) {
        const auto suffix_boundaries = unicode_scalar_boundaries(candidate);
        if (!suffix_boundaries.has_value()) continue;
        const size_t suffix_length = suffix_boundaries->size() - 1;
        if (length >= suffix_length
            && unicode_scalar_substring(
                value, *boundaries, length - suffix_length, length
            ) == candidate) {
            suffix_start = length - suffix_length;
            suffix = candidate;
            break;
        }
    }
    if (suffix.empty()) return std::nullopt;
    for (size_t index = 1; index < suffix_start; ++index) {
        if (!contains_any(unicode_scalar_substring(
                value, *boundaries, index, index + 1
            ), {"零", "〇", "一", "二", "三", "四", "五", "六", "七", "八", "九", "幺"})) {
            return std::nullopt;
        }
    }
    const auto digits = parse_spoken_identifier_digits(
        value, *boundaries, 1, suffix_start, 2, 6
    );
    if (!digits.has_value()) return std::nullopt;
    return prefix + *digits + suffix;
}

RuleApproval approve_train_number(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = train_number_replacement(range.text);
    const bool continues_left = range.start > 0 && ascii_alnum_scalar(
        previous_non_space_scalar(input, *boundaries, range.start)
    );
    const std::string next = range.end < length
        ? next_non_space_scalar(input, *boundaries, range.end) : "";
    if (!replacement.has_value() || continues_left || ascii_alnum_scalar(next)
        || number_continuation(next)
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_train_number"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "train_prefix_plus_spoken_digits", kTrainNumberRule},
            {"boundary", "complete_train_number", kTrainNumberRule},
            {"context", "train_number_suffix", kTrainNumberRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_train_number",
    };
}

std::optional<std::string> flight_prefix_scalar(const std::string& scalar) {
    if (scalar.size() == 1) {
        const auto ch = static_cast<unsigned char>(scalar[0]);
        if (std::isupper(ch) != 0 || std::isdigit(ch) != 0) return scalar;
    }
    const auto digit = spoken_year_digit(scalar);
    if (digit.has_value()) return std::to_string(*digit);
    return std::nullopt;
}

std::optional<std::string> flight_number_replacement(const std::string& value) {
    const auto boundaries = unicode_scalar_boundaries(value);
    if (!boundaries.has_value()) return std::nullopt;
    const size_t length = boundaries->size() - 1;
    if (length < 4 || length > 8) return std::nullopt;
    std::string prefix;
    bool has_letter = false;
    for (size_t index = 0; index < 2; ++index) {
        const std::string scalar = unicode_scalar_substring(value, *boundaries, index, index + 1);
        const auto normalized = flight_prefix_scalar(scalar);
        if (!normalized.has_value()) return std::nullopt;
        has_letter = has_letter || (scalar.size() == 1
            && std::isupper(static_cast<unsigned char>(scalar[0])) != 0);
        prefix += *normalized;
    }
    if (!has_letter) return std::nullopt;
    for (size_t index = 2; index < length; ++index) {
        if (!contains_any(unicode_scalar_substring(
                value, *boundaries, index, index + 1
            ), {"零", "〇", "一", "二", "三", "四", "五", "六", "七", "八", "九", "幺"})) {
            return std::nullopt;
        }
    }
    const auto digits = parse_spoken_identifier_digits(value, *boundaries, 2, length, 2, 6);
    if (!digits.has_value()) return std::nullopt;
    return prefix + *digits;
}

bool explicit_flight_number_context(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const std::string left = context_window(input, boundaries, 0, start, 12);
    const std::string right = right_clause_context(input, boundaries, end, 4);
    return ends_with_any(left, {"航班号为", "航班号是", "航班号"})
        || starts_with_any(right, {"航班", "这个航班"});
}

RuleApproval approve_flight_number(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto replacement = flight_number_replacement(range.text);
    const std::string previous = range.start > 0
        ? previous_non_space_scalar(input, *boundaries, range.start) : "";
    const bool continues_left = ascii_alnum_scalar(previous) || number_continuation(previous);
    const std::string next = range.end < length
        ? next_non_space_scalar(input, *boundaries, range.end) : "";
    if (!replacement.has_value() || continues_left
        || (ascii_alnum_scalar(next) && next != "航") || number_continuation(next)
        || !explicit_flight_number_context(input, *boundaries, range.start, range.end)
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_flight_number"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "iata_prefix_plus_spoken_digits", kFlightNumberRule},
            {"boundary", "complete_flight_number", kFlightNumberRule},
            {"context", "flight_number_anchor_or_suffix", kFlightNumberRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_flight_number",
    };
}

std::optional<std::string> parse_ipv4_component(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto digits = parse_spoken_identifier_digits(
        input, boundaries, start, end, 1, 3
    );
    std::optional<int> value;
    if (digits.has_value()) {
        if (digits->size() > 1 && (*digits)[0] == '0') return std::nullopt;
        value = std::stoi(*digits);
    } else {
        value = parse_spoken_cardinal(input, boundaries, start, end);
    }
    if (!value.has_value() || *value < 0 || *value > 255) return std::nullopt;
    return std::to_string(*value);
}

std::optional<std::string> parse_ipv4_address(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    std::string output;
    size_t component_start = start;
    size_t component_count = 0;
    for (size_t index = start; index <= end; ++index) {
        if (index != end && unicode_scalar_substring(
                input, boundaries, index, index + 1
            ) != "点") {
            continue;
        }
        const auto component = parse_ipv4_component(
            input, boundaries, component_start, index
        );
        if (!component.has_value()) return std::nullopt;
        if (component_count++ > 0) output += ".";
        output += *component;
        component_start = index + 1;
    }
    return component_count == 4 ? std::optional<std::string>(output) : std::nullopt;
}

bool explicit_ipv4_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    size_t cue_end = start;
    while (cue_end > 0 && contains_any("是为：: \t", {unicode_scalar_substring(
            input, boundaries, cue_end - 1, cue_end
        )})) {
        --cue_end;
    }
    const std::string left = context_window(input, boundaries, 0, cue_end, 16);
    return ends_with_any(left, {"IP", "IP地址", "IPv4", "IPv4地址", "服务器地址"});
}

RuleApproval approve_ipv4_address(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto replacement = parse_ipv4_address(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !explicit_ipv4_anchor(input, *boundaries, range.start)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_ipv4_address"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "four_spoken_ipv4_components", kIpv4AddressRule},
            {"boundary", "complete_ipv4_address", kIpv4AddressRule},
            {"context", "explicit_ipv4_anchor", kIpv4AddressRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_ipv4_address",
    };
}

std::optional<std::string> parse_network_port(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto digits = parse_spoken_identifier_digits(
        input, boundaries, start, end, 1, 5
    );
    std::optional<int> value;
    if (digits.has_value()) {
        if (digits->size() > 1 && (*digits)[0] == '0') return std::nullopt;
        value = std::stoi(*digits);
    } else {
        value = parse_spoken_cardinal(input, boundaries, start, end);
    }
    if (!value.has_value() || *value < 1 || *value > 65535) return std::nullopt;
    return std::to_string(*value);
}

bool explicit_network_port_anchor(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    size_t cue_end = start;
    while (cue_end > 0 && contains_any("是为：: \t", {unicode_scalar_substring(
            input, boundaries, cue_end - 1, cue_end
        )})) {
        --cue_end;
    }
    const std::string left = context_window(input, boundaries, 0, cue_end, 12);
    return ends_with_any(left, {"端口", "端口号", "监听端口", "服务端口"});
}

RuleApproval approve_network_port(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto replacement = parse_network_port(
        input, *boundaries, range.start, range.end
    );
    const size_t length = boundaries->size() - 1;
    const bool continues_left = range.start > 0 && (number_continuation(
        unicode_scalar_substring(input, *boundaries, range.start - 1, range.start)
    ) || unicode_scalar_substring(input, *boundaries, range.start - 1, range.start) == "幺");
    const bool continues_right = range.end < length && (number_continuation(
        unicode_scalar_substring(input, *boundaries, range.end, range.end + 1)
    ) || unicode_scalar_substring(input, *boundaries, range.end, range.end + 1) == "幺");
    if (!replacement.has_value()
        || continues_left || continues_right
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !explicit_network_port_anchor(input, *boundaries, range.start)
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, length, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_network_port"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "spoken_network_port", kNetworkPortRule},
            {"boundary", "complete_network_port", kNetworkPortRule},
            {"context", "explicit_network_port_anchor", kNetworkPortRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "standard_network_port",
    };
}

RuleApproval approve_explicit_century(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (range.end <= range.start + 2
        || unicode_scalar_substring(input, *boundaries, range.end - 2, range.end) != "世纪"
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_explicit_century"};
    }
    const auto value = parse_spoken_cardinal(
        input, *boundaries, range.start, range.end - 2
    );
    if (!value.has_value() || *value < 10 || *value > 30) {
        return {RuleDecision::preserve, "", {}, {}, "unsupported_century"};
    }
    const std::string replacement = std::to_string(*value) + "世纪";
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "spoken_cardinal_plus_century", kCenturyRule},
            {"boundary", "complete_century_expression", kCenturyRule},
            {"context", "explicit_century_unit", kCenturyRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_century",
    };
}

std::optional<std::string> clock_period_at(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    static const std::vector<std::string> periods = {
        "凌晨", "清晨", "早上", "上午", "中午", "下午", "傍晚", "晚上", "夜里", "夜间",
    };
    for (const auto& period : periods) {
        if (start + 2 <= boundaries.size() - 1
            && unicode_scalar_substring(input, boundaries, start, start + 2) == period) {
            return period;
        }
    }
    return std::nullopt;
}

bool date_continues_into_minute_clock(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start
) {
    const auto period = clock_period_at(input, boundaries, start);
    if (!period.has_value()) return false;
    const size_t length = boundaries.size() - 1;
    size_t point = start + 2;
    while (point < length && point - (start + 2) <= 3
        && unicode_scalar_substring(input, boundaries, point, point + 1) != "点") {
        if (!spoken_cardinal_digit(unicode_scalar_substring(
                input, boundaries, point, point + 1
            )).has_value()
            && !spoken_small_unit(unicode_scalar_substring(
                input, boundaries, point, point + 1
            )).has_value()) {
            return false;
        }
        ++point;
    }
    if (point == start + 2 || point >= length
        || unicode_scalar_substring(input, boundaries, point, point + 1) != "点") {
        return false;
    }
    size_t minute_end = point + 1;
    while (minute_end < length && minute_end - (point + 1) <= 3
        && (spoken_cardinal_digit(unicode_scalar_substring(
                input, boundaries, minute_end, minute_end + 1
            )).has_value()
            || spoken_small_unit(unicode_scalar_substring(
                input, boundaries, minute_end, minute_end + 1
            )).has_value())) {
        ++minute_end;
    }
    return minute_end > point + 1 && minute_end < length
        && unicode_scalar_substring(input, boundaries, minute_end, minute_end + 1) == "分";
}

bool clock_number_scalar(const std::string& value) {
    return spoken_cardinal_digit(value).has_value() || spoken_small_unit(value).has_value();
}

std::optional<std::pair<int, std::string>> parse_clock_number(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto cardinal = parse_spoken_cardinal(input, boundaries, start, end);
    if (cardinal.has_value()) return std::make_pair(*cardinal, std::to_string(*cardinal));
    if (end - start != 2) return std::nullopt;
    const auto leading = spoken_year_digit(unicode_scalar_substring(
        input, boundaries, start, start + 1
    ));
    const auto digit = spoken_year_digit(unicode_scalar_substring(
        input, boundaries, start + 1, end
    ));
    if (!leading.has_value() || *leading != 0 || !digit.has_value()) return std::nullopt;
    return std::make_pair(*digit, "0" + std::to_string(*digit));
}

std::optional<std::string> parse_explicit_clock_time(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const auto period = clock_period_at(input, boundaries, start);
    if (!period.has_value() || start + 3 >= end) return std::nullopt;
    size_t point = start + 2;
    while (point < end
        && !contains_any("点时", {unicode_scalar_substring(
            input, boundaries, point, point + 1
        )})) {
        if (!clock_number_scalar(unicode_scalar_substring(
                input, boundaries, point, point + 1
            ))) {
            return std::nullopt;
        }
        ++point;
    }
    if (point == start + 2 || point >= end) return std::nullopt;
    const auto hour = parse_clock_number(input, boundaries, start + 2, point);
    if (!hour.has_value() || hour->first > 12) return std::nullopt;

    const std::string marker = unicode_scalar_substring(
        input, boundaries, point, point + 1
    );
    std::string replacement = *period + std::to_string(hour->first) + marker;
    size_t cursor = point + 1;
    if (cursor == end) return replacement;
    if (marker != "点") return std::nullopt;
    const std::string suffix = unicode_scalar_substring(input, boundaries, cursor, end);
    if (suffix == "整" || suffix == "半" || suffix == "一刻" || suffix == "三刻") {
        return replacement + suffix;
    }

    size_t minute_end = cursor;
    while (minute_end < end && clock_number_scalar(unicode_scalar_substring(
            input, boundaries, minute_end, minute_end + 1
        ))) {
        ++minute_end;
    }
    if (minute_end == cursor || minute_end >= end
        || unicode_scalar_substring(input, boundaries, minute_end, minute_end + 1) != "分") {
        return std::nullopt;
    }
    const auto minute = parse_clock_number(input, boundaries, cursor, minute_end);
    if (!minute.has_value() || minute->first > 59) return std::nullopt;
    replacement += minute->second + "分";
    cursor = minute_end + 1;
    if (cursor == end) return replacement;

    size_t second_end = cursor;
    while (second_end < end && clock_number_scalar(unicode_scalar_substring(
            input, boundaries, second_end, second_end + 1
        ))) {
        ++second_end;
    }
    if (second_end == cursor || second_end + 1 != end
        || unicode_scalar_substring(input, boundaries, second_end, end) != "秒") {
        return std::nullopt;
    }
    const auto second = parse_clock_number(input, boundaries, cursor, second_end);
    if (!second.has_value() || second->first > 59) return std::nullopt;
    return replacement + second->second + "秒";
}

std::optional<std::string> parse_unperioded_exact_clock_time(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (start + 1 >= end) return std::nullopt;
    size_t marker = start;
    while (marker < end) {
        const std::string value = unicode_scalar_substring(
            input, boundaries, marker, marker + 1
        );
        if (value == "点" || value == "时") break;
        if (!clock_number_scalar(unicode_scalar_substring(
                input, boundaries, marker, marker + 1
            ))) {
            return std::nullopt;
        }
        ++marker;
    }
    if (marker == start || marker >= end) return std::nullopt;
    const std::string marker_text = unicode_scalar_substring(
        input, boundaries, marker, marker + 1
    );
    const auto hour = parse_clock_number(input, boundaries, start, marker);
    if (!hour.has_value() || hour->first > 23) return std::nullopt;
    if (marker + 1 == end) {
        return std::to_string(hour->first) + marker_text;
    }
    if (marker_text != "点") return std::nullopt;
    const std::string suffix = unicode_scalar_substring(
        input, boundaries, marker + 1, end
    );
    if (suffix == "整" || suffix == "半" || suffix == "一刻" || suffix == "三刻") {
        return std::to_string(hour->first) + marker_text + suffix;
    }
    if (marker + 2 >= end) return std::nullopt;
    size_t minute_end = marker + 1;
    while (minute_end < end && clock_number_scalar(unicode_scalar_substring(
            input, boundaries, minute_end, minute_end + 1
        ))) {
        ++minute_end;
    }
    if (minute_end == marker + 1 || minute_end >= end
        || unicode_scalar_substring(input, boundaries, minute_end, minute_end + 1) != "分") {
        return std::nullopt;
    }
    const auto minute = parse_clock_number(input, boundaries, marker + 1, minute_end);
    if (!minute.has_value() || minute->first > 59) {
        return std::nullopt;
    }
    std::string replacement = std::to_string(hour->first) + "点" + minute->second + "分";
    size_t cursor = minute_end + 1;
    if (cursor == end) return replacement;
    size_t second_end = cursor;
    while (second_end < end && clock_number_scalar(unicode_scalar_substring(
            input, boundaries, second_end, second_end + 1
        ))) {
        ++second_end;
    }
    if (second_end == cursor || second_end + 1 != end
        || unicode_scalar_substring(input, boundaries, second_end, end) != "秒") {
        return std::nullopt;
    }
    const auto second = parse_clock_number(input, boundaries, cursor, second_end);
    if (!second.has_value() || second->first > 59) return std::nullopt;
    return replacement + second->second + "秒";
}

std::optional<std::string> parse_clock_range_endpoint(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end,
    bool allow_24
) {
    if (start >= end) return std::nullopt;
    size_t point = start;
    while (point < end && !contains_any("点时", {unicode_scalar_substring(
            input, boundaries, point, point + 1
        )})) {
        if (!clock_number_scalar(unicode_scalar_substring(
                input, boundaries, point, point + 1
            ))) {
            return std::nullopt;
        }
        ++point;
    }
    if (point == start || point >= end) return std::nullopt;
    const auto hour = parse_clock_number(input, boundaries, start, point);
    const int maximum_hour = allow_24 ? 24 : 23;
    if (!hour.has_value() || hour->first > maximum_hour) return std::nullopt;
    const std::string marker = unicode_scalar_substring(
        input, boundaries, point, point + 1
    );
    std::string replacement = std::to_string(hour->first) + marker;
    size_t cursor = point + 1;
    if (cursor == end) return replacement;
    if (marker != "点") return std::nullopt;
    const std::string suffix = unicode_scalar_substring(input, boundaries, cursor, end);
    if (suffix == "整" || suffix == "半" || suffix == "一刻" || suffix == "三刻") {
        return replacement + suffix;
    }
    size_t minute_end = cursor;
    while (minute_end < end && clock_number_scalar(unicode_scalar_substring(
            input, boundaries, minute_end, minute_end + 1
        ))) {
        ++minute_end;
    }
    if (minute_end == cursor || minute_end + 1 != end
        || unicode_scalar_substring(input, boundaries, minute_end, end) != "分") {
        return std::nullopt;
    }
    const auto minute = parse_clock_number(input, boundaries, cursor, minute_end);
    if (!minute.has_value() || minute->first > 59
        || (hour->first == 24 && minute->first != 0)) {
        return std::nullopt;
    }
    return replacement + minute->second + "分";
}

std::optional<std::string> parse_explicit_clock_time_range(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t connector = end;
    for (size_t index = start; index < end; ++index) {
        if (contains_any("到至", {unicode_scalar_substring(
                input, boundaries, index, index + 1
            )})) {
            if (connector != end) return std::nullopt;
            connector = index;
        }
    }
    if (connector == start || connector + 1 >= end || connector == end) {
        return std::nullopt;
    }
    const auto first_period = clock_period_at(input, boundaries, start);
    std::optional<std::string> first;
    if (first_period.has_value()) {
        first = parse_explicit_clock_time(input, boundaries, start, connector);
    } else {
        first = parse_clock_range_endpoint(
            input, boundaries, start, connector, true
        );
    }
    if (!first.has_value()) return std::nullopt;
    const auto second_period = clock_period_at(input, boundaries, connector + 1);
    const auto second = second_period.has_value()
        ? parse_explicit_clock_time(input, boundaries, connector + 1, end)
        : parse_clock_range_endpoint(
            input, boundaries, connector + 1, end, !first_period.has_value()
        );
    if (!second.has_value() || *first == *second) return std::nullopt;
    return *first + unicode_scalar_substring(
        input, boundaries, connector, connector + 1
    ) + *second;
}

RuleApproval approve_explicit_clock_time_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const auto replacement = parse_explicit_clock_time_range(
        input, *boundaries, range.start, range.end
    );
    const bool explicit_period = clock_period_at(input, *boundaries, range.start).has_value();
    const bool contextual = contextual_clock_anchor_applies(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value() || (!explicit_period && !contextual)
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_clock_time_range"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "two_spoken_clock_endpoints", kClockTimeRangeRule},
            {"boundary", "complete_clock_time_range", kClockTimeRangeRule},
            {"context", "explicit_clock_range_context", kClockTimeRangeRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_clock_time_range",
    };
}

bool clock_boundary_is_uncertain(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const size_t length = boundaries.size() - 1;
    if (start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, boundaries, start - 1, start
        );
        if (contains_any("至到~～—-－", {previous})) return true;
    }
    if (end < length) {
        const std::string next = unicode_scalar_substring(input, boundaries, end, end + 1);
        const std::string right = unicode_scalar_substring(
            input, boundaries, end, std::min(length, end + 2)
        );
        if (date_or_number_continuation(next) || contains_any("点分秒刻整半", {next})
            || starts_with_any(right, {
                "多", "左右", "前后", "上下", "许", "来", "余", "几", "至", "到", "~", "～", "—", "-", "－",
            })) {
            return true;
        }
    }
    return false;
}

RuleApproval approve_explicit_clock_time(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == boundaries->size() - 1 && context.has_right_neighbor)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )
        || clock_boundary_is_uncertain(
            input, *boundaries, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_clock_time_context"};
    }
    const auto replacement = parse_explicit_clock_time(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_clock_time"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "period_spoken_clock_time", kClockTimeRule},
            {"boundary", "complete_exact_clock_time", kClockTimeRule},
            {"context", "explicit_chinese_period", kClockTimeRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_clock_time",
    };
}

bool contextual_clock_anchor_applies(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const std::string left = left_clause_context(input, boundaries, start, 16);
    const std::string right = right_clause_context(input, boundaries, end, 10);
    const bool event_anchor = ends_with_any(left, {
        "定于", "安排在", "时间为", "时刻为", "开会时间", "开始时间",
        "发车时间", "起飞时间", "截止时间", "营业时间", "改到", "每日", "期间",
        "当前时间是", "当前时间为", "闹钟设在",
    }) || starts_with_any(right, {
        "开会", "开始", "开幕", "出发", "到达", "进行", "举行", "截止",
        "结束", "启动", "上课", "下课", "起飞", "发车", "到站", "营业",
        "提醒", "设置闹钟", "设定闹钟", "的闹钟", "的提醒",
        "实施交通管控", "实施交通管制",
    });
    const bool departure_anchor = starts_with_any(right, {"从"})
        && contains_any(right, {"出发"});
    return event_anchor || departure_anchor;
}

bool contextual_approximate_clock_anchor_applies(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    const std::string left = left_clause_context(input, boundaries, start, 20);
    const std::string right = right_clause_context(input, boundaries, end, 14);
    const bool suffix = starts_with_any(right, {"左右"});
    const std::string after_suffix = suffix ? right.substr(std::strlen("左右")) : "";
    const bool suffix_event = starts_with_any(after_suffix, {
        "开会", "开始", "开幕", "出发", "到达", "进行", "举行", "截止",
        "结束", "上课", "下课", "起飞", "发车", "营业",
    });
    const bool suffix_with_anchor = suffix
        && (suffix_event || contextual_clock_anchor_applies(
            input, boundaries, start, end
        ));
    const bool prefix_with_anchor = ends_with_any(left, {
        "定于大约", "定于约", "安排在大约", "安排在约",
        "时间约为", "时刻约为", "时间大约在", "时刻大约在",
        "开会时间约为", "开始时间约为", "发车时间约为", "起飞时间约为",
        "截止时间约为", "营业时间约为",
    });
    return suffix_with_anchor || prefix_with_anchor;
}

RuleApproval approve_contextual_clock_time(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 10);
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || (clock_boundary_is_uncertain(input, *boundaries, range.start, range.end)
            && !ends_with_any(left, {"改到"}) && !starts_with_any(right, {"到站"}))
        || !contextual_clock_anchor_applies(
            input, *boundaries, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_contextual_clock_time"};
    }
    const auto replacement = parse_unperioded_exact_clock_time(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_contextual_clock_time"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "unperioded_spoken_clock_with_minute", kContextualClockTimeRule},
            {"boundary", "complete_contextual_exact_clock_time", kContextualClockTimeRule},
            {"context", "explicit_clock_event_context", kContextualClockTimeRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "contextual_exact_clock_time",
    };
}

RuleApproval approve_contextual_approximate_clock_time(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || !contextual_approximate_clock_anchor_applies(
            input, *boundaries, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {},
                "uncertain_contextual_approximate_clock_time"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        if (number_continuation(previous) || range_connector(previous)) {
            return {RuleDecision::preserve, "", {}, {},
                    "contextual_approximate_clock_continues_left"};
        }
    }
    const auto replacement = parse_unperioded_exact_clock_time(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_contextual_approximate_clock_time"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "unperioded_spoken_clock_with_minute",
             kContextualApproximateClockTimeRule},
            {"boundary", "complete_contextual_approximate_clock_time",
             kContextualApproximateClockTimeRule},
            {"context", "explicit_clock_event_and_chinese_approximation",
             kContextualApproximateClockTimeRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "contextual_approximate_clock_time",
    };
}

RuleApproval approve_explicit_calendar_timestamp(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_calendar_timestamp"};
    }
    if (range.start > 0) {
        const std::string previous = unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        );
        const std::string left = unicode_scalar_substring(
            input,
            *boundaries,
            range.start > 2 ? range.start - 2 : 0,
            range.start
        );
        const bool leading_until_timestamp = previous == "至"
            && (range.start == 1 || contains_any(unicode_scalar_substring(
                    input, *boundaries, range.start - 2, range.start - 1
                ), {"。", "！", "？", ";", "；", "\n", "\r"}));
        const std::string prefix = unicode_scalar_substring(
            input, *boundaries, 0, range.start - 1
        );
        const bool released_range_endpoint = previous == "到"
            && contains_any(prefix, {"从"});
        if (range_connector(previous)
            && !ends_with_any(left, {"截至"})
            && !leading_until_timestamp && !released_range_endpoint) {
            return {RuleDecision::preserve, "", {}, {}, "calendar_timestamp_continues_left"};
        }
    }
    size_t clock_start = range.start;
    bool contextual_clock = false;
    while (clock_start < range.end
        && !clock_period_at(input, *boundaries, clock_start).has_value()) {
        ++clock_start;
    }
    if (clock_start == range.end) {
        for (size_t start = range.start + 1; start < range.end; ++start) {
            if (!parse_unperioded_exact_clock_time(
                    input, *boundaries, start, range.end
                ).has_value()
                || !contextual_clock_anchor_applies(
                    input, *boundaries, start, range.end
                )) {
                continue;
            }
            clock_start = start;
            contextual_clock = true;
            break;
        }
    }
    if (clock_start == range.start || clock_start >= range.end) {
        return {RuleDecision::preserve, "", {}, {}, "missing_calendar_clock_time"};
    }
    const std::string date_text = unicode_scalar_substring(
        input, *boundaries, range.start, clock_start
    );
    const bool has_year = date_text.find("年") != std::string::npos;
    const SafePolicyContext component_context = {"", "", false, false};
    const TransformationCandidate date_candidate = {
        "zh-calendar-timestamp-date",
        {0, clock_start - range.start, date_text},
        "zh-CN",
        "date_time",
        has_year ? "explicit_full_date" : "explicit_month_day",
        has_year ? kFullDateRule : kMonthDayRule,
        has_year ? kFullDateRuleVersion : kMonthDayRuleVersion,
    };
    const TransformationCandidate clock_candidate = {
        "zh-calendar-timestamp-clock",
        {
            clock_start,
            range.end,
            unicode_scalar_substring(input, *boundaries, clock_start, range.end),
        },
        "zh-CN",
        "date_time",
        contextual_clock ? "contextual_unperioded_clock_time" : "explicit_clock_time",
        contextual_clock ? kContextualClockTimeRule : kClockTimeRule,
        contextual_clock ? kContextualClockTimeRuleVersion : kClockTimeRuleVersion,
    };
    const auto date = has_year
        ? approve_full_date(date_text, component_context, date_candidate)
        : approve_month_day(date_text, component_context, date_candidate);
    const auto clock = contextual_clock
        ? approve_contextual_clock_time(input, context, clock_candidate)
        : approve_explicit_clock_time(input, context, clock_candidate);
    if (date.decision != RuleDecision::approve || clock.decision != RuleDecision::approve) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_calendar_timestamp_component"};
    }
    const std::string replacement = date.parsed_value + clock.parsed_value;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", contextual_clock
                ? "calendar_date_plus_contextual_clock"
                : "calendar_date_plus_period_clock", kCalendarTimestampRule},
            {"boundary", "complete_exact_calendar_timestamp", kCalendarTimestampRule},
            {"context", contextual_clock
                ? "explicit_date_and_clock_event_anchor"
                : "explicit_date_and_chinese_period", kCalendarTimestampRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_timestamp",
    };
}

RuleApproval approve_explicit_calendar_timestamp_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_calendar_timestamp_range"};
    }
    size_t connector = range.start;
    for (size_t index = range.start + 1; index + 1 < range.end; ++index) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, index, index + 1
        );
        if (value == "到" || value == "至") {
            if (connector != range.start) {
                return {RuleDecision::preserve, "", {}, {},
                        "ambiguous_calendar_timestamp_range_connector"};
            }
            connector = index;
        }
    }
    if (connector == range.start) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_calendar_timestamp_range_connector"};
    }
    const std::string first_text = unicode_scalar_substring(
        input, *boundaries, range.start, connector
    );
    const std::string second_text = unicode_scalar_substring(
        input, *boundaries, connector + 1, range.end
    );
    const auto first_boundaries = unicode_scalar_boundaries(first_text);
    const auto second_boundaries = unicode_scalar_boundaries(second_text);
    if (!first_boundaries.has_value() || !second_boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const SafePolicyContext component_context = {"", "", false, false};
    const TransformationCandidate first_candidate = {
        "zh-calendar-timestamp-range-first",
        {0, first_boundaries->size() - 1, first_text},
        "zh-CN", "date_time", "explicit_calendar_timestamp",
        kCalendarTimestampRule, kCalendarTimestampRuleVersion,
    };
    const TransformationCandidate second_candidate = {
        "zh-calendar-timestamp-range-second",
        {0, second_boundaries->size() - 1, second_text},
        "zh-CN", "date_time", "explicit_calendar_timestamp",
        kCalendarTimestampRule, kCalendarTimestampRuleVersion,
    };
    auto first = approve_explicit_calendar_timestamp(
        first_text, component_context, first_candidate
    );
    auto second = approve_explicit_calendar_timestamp(
        second_text, component_context, second_candidate
    );
    const auto parse_contextual_component = [&](const std::string& value) {
        RuleApproval preserved = {
            RuleDecision::preserve, "", {}, {}, "invalid_contextual_timestamp_component",
        };
        const auto component_boundaries = unicode_scalar_boundaries(value);
        if (!component_boundaries.has_value()) return preserved;
        const size_t component_length = component_boundaries->size() - 1;
        size_t date_end = 0;
        while (date_end < component_length) {
            const std::string scalar = unicode_scalar_substring(
                value, *component_boundaries, date_end, date_end + 1
            );
            ++date_end;
            if (scalar == "日" || scalar == "号") break;
        }
        if (date_end == 0 || date_end >= component_length) return preserved;
        const std::string date_text = unicode_scalar_substring(
            value, *component_boundaries, 0, date_end
        );
        const bool has_year = date_text.find("年") != std::string::npos;
        const TransformationCandidate date_candidate = {
            "zh-calendar-timestamp-range-contextual-date",
            {0, date_end, date_text}, "zh-CN", "date_time",
            has_year ? "explicit_full_date" : "explicit_month_day",
            has_year ? kFullDateRule : kMonthDayRule,
            has_year ? kFullDateRuleVersion : kMonthDayRuleVersion,
        };
        const auto date = has_year
            ? approve_full_date(date_text, component_context, date_candidate)
            : approve_month_day(date_text, component_context, date_candidate);
        const auto clock = parse_unperioded_exact_clock_time(
            value, *component_boundaries, date_end, component_length
        );
        if (date.decision != RuleDecision::approve || !clock.has_value()) return preserved;
        return RuleApproval{
            RuleDecision::approve, date.parsed_value + *clock, {}, {},
            "valid_contextual_timestamp_component",
        };
    };
    const auto parse_clock_component = [&](const std::string& value) {
        RuleApproval preserved = {
            RuleDecision::preserve, "", {}, {}, "invalid_clock_timestamp_component",
        };
        const auto component_boundaries = unicode_scalar_boundaries(value);
        if (!component_boundaries.has_value()) return preserved;
        const auto clock = parse_unperioded_exact_clock_time(
            value, *component_boundaries, 0, component_boundaries->size() - 1
        );
        if (!clock.has_value()) return preserved;
        return RuleApproval{
            RuleDecision::approve, *clock, {}, {}, "valid_clock_timestamp_component",
        };
    };
    if (first.decision != RuleDecision::approve) first = parse_contextual_component(first_text);
    if (second.decision != RuleDecision::approve) second = parse_contextual_component(second_text);
    if (second.decision != RuleDecision::approve && first_text.find("年") != std::string::npos) {
        second = parse_clock_component(second_text);
    }
    if (first.decision != RuleDecision::approve || second.decision != RuleDecision::approve
        || first.parsed_value == second.parsed_value) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_calendar_timestamp_range_component"};
    }
    const std::string connector_text = unicode_scalar_substring(
        input, *boundaries, connector, connector + 1
    );
    const std::string replacement = first.parsed_value + connector_text + second.parsed_value;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "two_calendar_timestamps_with_range_connector",
             kCalendarTimestampRangeRule},
            {"boundary", "complete_calendar_timestamp_range",
             kCalendarTimestampRangeRule},
            {"context", "two_explicit_dates_period_clocks_and_range_connector",
             kCalendarTimestampRangeRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_calendar_timestamp_range",
    };
}

RuleApproval approve_explicit_approximate_calendar_timestamp(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(
            context, boundaries->size() - 1, range.start, range.end
        )) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_approximate_calendar_timestamp"};
    }
    if (range.start > 0 && range_connector(unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        ))) {
        return {RuleDecision::preserve, "", {}, {},
                "approximate_calendar_timestamp_range_boundary"};
    }
    size_t suffix_length = 0;
    std::string suffix;
    if (range.end >= range.start + 2
        && unicode_scalar_substring(input, *boundaries, range.end - 2, range.end)
            == "左右") {
        suffix_length = 2;
        suffix = "左右";
    } else if (range.end > range.start
        && unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) == "许") {
        suffix_length = 1;
        suffix = "许";
    } else {
        return {RuleDecision::preserve, "", {}, {},
                "unsupported_calendar_timestamp_approximation"};
    }
    const size_t core_end = range.end - suffix_length;
    const std::string core = unicode_scalar_substring(
        input, *boundaries, range.start, core_end
    );
    const auto core_boundaries = unicode_scalar_boundaries(core);
    if (!core_boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const TransformationCandidate core_candidate = {
        "zh-approximate-calendar-timestamp-core",
        {0, core_boundaries->size() - 1, core},
        "zh-CN", "date_time", "explicit_calendar_timestamp",
        kCalendarTimestampRule, kCalendarTimestampRuleVersion,
    };
    const SafePolicyContext component_context = {"", "", false, false};
    const auto approved = approve_explicit_calendar_timestamp(
        core, component_context, core_candidate
    );
    if (approved.decision != RuleDecision::approve) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_approximate_calendar_timestamp_component"};
    }
    const std::string replacement = approved.parsed_value + suffix;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "calendar_date_plus_period_clock_approximation",
             kApproximateCalendarTimestampRule},
            {"boundary", "complete_approximate_calendar_timestamp",
             kApproximateCalendarTimestampRule},
            {"context", "explicit_date_period_and_chinese_approximation",
             kApproximateCalendarTimestampRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_approximate_calendar_timestamp",
    };
}

RuleApproval approve_contextual_month_day_timestamp(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = context_window(input, *boundaries, 0, range.start, 8);
    const bool explicit_anchor = ends_with_any(left, {"时间", "截至", "从", "持续到"});
    const bool continues_in_right_context = range.end == length
        && starts_with_any(context.right_context, {
            "多", "左右", "前后", "上下", "许", "来", "余", "几",
            "至", "到", "~", "～", "—", "-", "－",
        });
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !explicit_anchor
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || continues_in_right_context
        || clock_boundary_is_uncertain(input, *boundaries, 0, range.end)) {
        return {RuleDecision::preserve, "", {}, {},
                "uncertain_contextual_timestamp_context"};
    }
    size_t date_end = range.start;
    while (date_end < range.end) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, date_end, date_end + 1
        );
        ++date_end;
        if (value == "日" || value == "号") break;
    }
    if (date_end <= range.start || date_end >= range.end) {
        return {RuleDecision::preserve, "", {}, {}, "missing_contextual_timestamp_date"};
    }
    const std::string date_text = unicode_scalar_substring(
        input, *boundaries, range.start, date_end
    );
    const SafePolicyContext component_context = {"", "", false, false};
    const TransformationCandidate date_candidate = {
        "zh-contextual-timestamp-date",
        {0, date_end - range.start, date_text},
        "zh-CN", "date_time", "explicit_month_day",
        kMonthDayRule, kMonthDayRuleVersion,
    };
    const auto date = approve_month_day(date_text, component_context, date_candidate);
    const auto clock = parse_unperioded_exact_clock_time(
        input, *boundaries, date_end, range.end
    );
    if (date.decision != RuleDecision::approve || !clock.has_value()) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_contextual_timestamp_component"};
    }
    const std::string replacement = date.parsed_value + *clock;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "month_day_plus_unperioded_exact_clock",
             kContextualMonthDayTimestampRule},
            {"boundary", "complete_exact_contextual_timestamp",
             kContextualMonthDayTimestampRule},
            {"context", "explicit_time_anchor",
             kContextualMonthDayTimestampRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_contextual_month_day_timestamp",
    };
}

RuleApproval approve_contextual_full_date_timestamp(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = context_window(input, *boundaries, 0, range.start, 8);
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !ends_with_any(left, {"时间", "截至", "从", "持续到"})
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || clock_boundary_is_uncertain(input, *boundaries, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {},
                "uncertain_contextual_full_date_timestamp_context"};
    }
    size_t date_end = range.start;
    while (date_end < range.end) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, date_end, date_end + 1
        );
        ++date_end;
        if (value == "日" || value == "号") break;
    }
    if (date_end <= range.start || date_end >= range.end) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_contextual_full_date_timestamp_date"};
    }
    const std::string date_text = unicode_scalar_substring(
        input, *boundaries, range.start, date_end
    );
    const SafePolicyContext component_context = {"", "", false, false};
    const TransformationCandidate date_candidate = {
        "zh-contextual-full-date-timestamp-date",
        {0, date_end - range.start, date_text},
        "zh-CN", "date_time", "explicit_full_date",
        kFullDateRule, kFullDateRuleVersion,
    };
    const auto date = approve_full_date(date_text, component_context, date_candidate);
    const auto clock = parse_unperioded_exact_clock_time(
        input, *boundaries, date_end, range.end
    );
    if (date.decision != RuleDecision::approve || !clock.has_value()) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_contextual_full_date_timestamp_component"};
    }
    const std::string replacement = date.parsed_value + *clock;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "full_date_plus_unperioded_exact_clock",
             kContextualFullDateTimestampRule},
            {"boundary", "complete_exact_contextual_full_date_timestamp",
             kContextualFullDateTimestampRule},
            {"context", "explicit_time_anchor",
             kContextualFullDateTimestampRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_contextual_full_date_timestamp",
    };
}

RuleApproval approve_contextual_approximate_month_day_timestamp(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = context_window(input, *boundaries, 0, range.start, 8);
    size_t suffix_length = 0;
    std::string suffix;
    if (range.end >= range.start + 2) {
        const std::string tail = unicode_scalar_substring(
            input, *boundaries, range.end - 2, range.end
        );
        if (tail == "左右" || tail == "前后") {
            suffix_length = 2;
            suffix = tail;
        }
    }
    if (suffix_length == 0 && range.end > range.start
        && unicode_scalar_substring(input, *boundaries, range.end - 1, range.end) == "许") {
        suffix_length = 1;
        suffix = "许";
    }
    if (suffix_length == 0 || range.end < range.start + suffix_length + 4
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || !ends_with_any(left, {"时间"})
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || clock_boundary_is_uncertain(input, *boundaries, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {},
                "uncertain_contextual_approximate_timestamp_context"};
    }
    const size_t core_end = range.end - suffix_length;
    size_t date_end = range.start;
    while (date_end < core_end) {
        const std::string value = unicode_scalar_substring(
            input, *boundaries, date_end, date_end + 1
        );
        ++date_end;
        if (value == "日" || value == "号") break;
    }
    if (date_end <= range.start || date_end >= core_end) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_contextual_approximate_timestamp_date"};
    }
    const std::string date_text = unicode_scalar_substring(
        input, *boundaries, range.start, date_end
    );
    const SafePolicyContext component_context = {"", "", false, false};
    const TransformationCandidate date_candidate = {
        "zh-contextual-approximate-timestamp-date",
        {0, date_end - range.start, date_text},
        "zh-CN", "date_time", "explicit_month_day",
        kMonthDayRule, kMonthDayRuleVersion,
    };
    const auto date = approve_month_day(date_text, component_context, date_candidate);
    const auto clock = parse_unperioded_exact_clock_time(
        input, *boundaries, date_end, core_end
    );
    if (date.decision != RuleDecision::approve || !clock.has_value()) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_contextual_approximate_timestamp_component"};
    }
    const std::string replacement = date.parsed_value + *clock + suffix;
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "month_day_plus_unperioded_clock_and_approximation",
             kContextualApproximateMonthDayTimestampRule},
            {"boundary", "complete_approximate_contextual_timestamp",
             kContextualApproximateMonthDayTimestampRule},
            {"context", "explicit_time_anchor_and_approximation",
             kContextualApproximateMonthDayTimestampRule},
        },
        {{{range.start, range.end, range.text}, replacement}},
        "explicit_contextual_approximate_month_day_timestamp",
    };
}

std::vector<TransformationCandidate> detect_full_dates(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 8 <= length; ++start) {
        bool four_digits = true;
        for (size_t offset = 0; offset < 4; ++offset) {
            if (!spoken_year_digit(unicode_scalar_substring(
                    input, *boundaries, start + offset, start + offset + 1
                )).has_value()) {
                four_digits = false;
                break;
            }
        }
        if (!four_digits || unicode_scalar_substring(
                input, *boundaries, start + 4, start + 5
            ) != "年") {
            continue;
        }
        const bool has_linking_particle = start + 5 < length
            && unicode_scalar_substring(input, *boundaries, start + 5, start + 6) == "的";
        const size_t month_start = start + 5 + (has_linking_particle ? 1 : 0);
        if (month_start >= length) continue;
        const std::string month_first = unicode_scalar_substring(
            input, *boundaries, month_start, month_start + 1
        );
        if (!spoken_cardinal_digit(month_first).has_value() && month_first != "十") {
            continue;
        }
        size_t month_mark = month_start;
        while (month_mark < length && month_mark - month_start <= 2
            && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
            ++month_mark;
        }
        if (month_mark >= length || month_mark == month_start
            || unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
            continue;
        }
        size_t suffix = month_mark + 1;
        while (suffix < length && suffix - (month_mark + 1) <= 3) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, suffix, suffix + 1
            );
            if (value == "日" || value == "号") break;
            ++suffix;
        }
        if (suffix >= length || suffix == month_mark + 1) continue;
        const std::string value = unicode_scalar_substring(
            input, *boundaries, suffix, suffix + 1
        );
        if (value != "日" && value != "号") continue;
        const size_t end = suffix + 1;
        candidates.push_back({
            "zh-full-date-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_full_date",
            kFullDateRule, kFullDateRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_full_date_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& full_dates
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (size_t index = 0; index + 1 < full_dates.size(); ++index) {
        const auto& first = full_dates[index].semantic_source;
        const auto& second = full_dates[index + 1].semantic_source;
        if (first.end + 1 != second.start) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, first.end, second.start
        );
        if (connector != "到" && connector != "至") continue;
        candidates.push_back({
            "zh-full-date-range-" + std::to_string(first.start) + "-"
                + std::to_string(second.end),
            {
                first.start,
                second.end,
                unicode_scalar_substring(input, *boundaries, first.start, second.end),
            },
            "zh-CN", "date_time", "explicit_full_date_range",
            kFullDateRangeRule, kFullDateRangeRuleVersion,
        });
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_year_months(
    const std::string& input,
    const std::vector<TransformationCandidate>& exclusions
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 7 <= length; ++start) {
        bool four_digits = true;
        for (size_t offset = 0; offset < 4; ++offset) {
            if (!spoken_year_digit(unicode_scalar_substring(
                    input, *boundaries, start + offset, start + offset + 1
                )).has_value()) {
                four_digits = false;
                break;
            }
        }
        if (!four_digits || unicode_scalar_substring(
                input, *boundaries, start + 4, start + 5
            ) != "年") {
            continue;
        }
        size_t month_start = start + 5;
        if (month_start < length
            && unicode_scalar_substring(input, *boundaries, month_start, month_start + 1) == "的") {
            ++month_start;
        }
        if (month_start >= length) continue;
        const std::string month_first = unicode_scalar_substring(
            input, *boundaries, month_start, month_start + 1
        );
        if (!spoken_cardinal_digit(month_first).has_value() && month_first != "十") {
            continue;
        }
        size_t month_mark = month_start;
        while (month_mark < length && month_mark - month_start <= 2
            && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
            ++month_mark;
        }
        if (month_mark >= length || month_mark == month_start
            || month_mark - month_start > 2
            || unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
            continue;
        }
        size_t end = month_mark + 1;
        if (end < length
            && unicode_scalar_substring(input, *boundaries, end, end + 1) == "份") {
            ++end;
        }
        if (std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& item) {
                return overlaps(start, end, item.semantic_source);
            })) {
            start = end - 1;
            continue;
        }
        candidates.push_back({
            "zh-year-month-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_year_month",
            kYearMonthRule, kYearMonthRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_cross_segment_month(
    const std::string& input,
    const std::string& left_context
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    const auto left = unicode_scalar_boundaries(left_context);
    if (!boundaries.has_value() || !left.has_value() || left->size() < 2) return {};
    const size_t length = boundaries->size() - 1;
    const size_t left_length = left->size() - 1;
    if (unicode_scalar_substring(
            left_context, *left, left_length - 1, left_length
        ) != "年") {
        return {};
    }
    size_t month_mark = 0;
    while (month_mark < length && month_mark < 3
        && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
        ++month_mark;
    }
    if (month_mark == 0 || month_mark >= length) return {};
    const size_t end = month_mark + 1;
    if (end < length) {
        const std::string next = unicode_scalar_substring(input, *boundaries, end, end + 1);
        if (date_or_number_continuation(next) || next == "日" || next == "号" || next == "份") {
            return {};
        }
    }
    const auto month = parse_spoken_cardinal(input, *boundaries, 0, month_mark);
    if (!month.has_value() || *month < 1 || *month > 12
        || !canonical_spoken_calendar_component(input, *boundaries, 0, month_mark, *month)) {
        return {};
    }
    return {{
        "zh-cross-segment-month-0-" + std::to_string(end),
        {0, end, unicode_scalar_substring(input, *boundaries, 0, end)},
        "zh-CN", "date_time", "contextual_cross_segment_month",
        kCrossSegmentMonthRule, kCrossSegmentMonthRuleVersion,
    }};
}

std::vector<TransformationCandidate> detect_year_month_ranges(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 5 < length; ++start) {
        bool four_digits = true;
        for (size_t offset = 0; offset < 4; ++offset) {
            if (!spoken_year_digit(unicode_scalar_substring(
                    input, *boundaries, start + offset, start + offset + 1
                )).has_value()) {
                four_digits = false;
                break;
            }
        }
        if (!four_digits || unicode_scalar_substring(
                input, *boundaries, start + 4, start + 5
            ) != "年") {
            continue;
        }
        size_t first_month_mark = start + 5;
        while (first_month_mark < length && first_month_mark - (start + 5) <= 2
            && unicode_scalar_substring(
                input, *boundaries, first_month_mark, first_month_mark + 1
            ) != "月") {
            ++first_month_mark;
        }
        if (first_month_mark >= length || first_month_mark == start + 5
            || first_month_mark - (start + 5) > 2
            || unicode_scalar_substring(
                input, *boundaries, first_month_mark, first_month_mark + 1
            ) != "月") {
            continue;
        }
        const size_t connector = first_month_mark + 1;
        const std::string connector_text = connector < length
            ? unicode_scalar_substring(input, *boundaries, connector, connector + 1)
            : "";
        if (connector_text != "到" && connector_text != "至") {
            continue;
        }
        const size_t second_endpoint_start = connector + 1;
        const bool has_second_year = second_endpoint_start + 5 <= length
            && unicode_scalar_substring(
                input, *boundaries, second_endpoint_start + 4, second_endpoint_start + 5
            ) == "年"
            && spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, second_endpoint_start, second_endpoint_start + 1
            )).has_value()
            && spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, second_endpoint_start + 1, second_endpoint_start + 2
            )).has_value()
            && spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, second_endpoint_start + 2, second_endpoint_start + 3
            )).has_value()
            && spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, second_endpoint_start + 3, second_endpoint_start + 4
            )).has_value();
        const size_t second_month_start = second_endpoint_start
            + (has_second_year ? 5 : 0);
        size_t second_month_mark = second_month_start;
        while (second_month_mark < length
            && second_month_mark - second_month_start <= 2
            && unicode_scalar_substring(
                input, *boundaries, second_month_mark, second_month_mark + 1
            ) != "月") {
            ++second_month_mark;
        }
        if (second_month_mark >= length || second_month_mark == second_month_start
            || second_month_mark - second_month_start > 2
            || unicode_scalar_substring(
                input, *boundaries, second_month_mark, second_month_mark + 1
            ) != "月") {
            continue;
        }
        const size_t end = second_month_mark + 1;
        candidates.push_back({
            "zh-year-month-range-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_year_month_range",
            kYearMonthRangeRule, kYearMonthRangeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_year_ranges(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 11 <= length; ++start) {
        bool valid = unicode_scalar_substring(
            input, *boundaries, start + 4, start + 5
        ) == "年" && unicode_scalar_substring(
            input, *boundaries, start + 10, start + 11
        ) == "年";
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, start + 5, start + 6
        );
        valid = valid && (connector == "至" || connector == "到");
        for (size_t offset = 0; valid && offset < 4; ++offset) {
            valid = spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, start + offset, start + offset + 1
            )).has_value() && spoken_year_digit(unicode_scalar_substring(
                input, *boundaries, start + 6 + offset, start + 7 + offset
            )).has_value();
        }
        if (!valid) continue;
        const size_t end = start + 11;
        candidates.push_back({
            "zh-year-range-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_year_range",
            kYearRangeRule, kYearRangeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_month_days(
    const std::string& input,
    const std::vector<TransformationCandidate>& exclusions
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 3 < length; ++start) {
        if (!spoken_cardinal_digit(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            )).has_value()
            && unicode_scalar_substring(input, *boundaries, start, start + 1) != "十") {
            continue;
        }
        size_t month_mark = start;
        while (month_mark < length && month_mark - start <= 2
            && unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
            ++month_mark;
        }
        if (month_mark >= length || month_mark == start
            || unicode_scalar_substring(input, *boundaries, month_mark, month_mark + 1) != "月") {
            continue;
        }
        size_t suffix = month_mark + 1;
        while (suffix < length && suffix - (month_mark + 1) <= 3) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, suffix, suffix + 1
            );
            if (value == "日" || value == "号") break;
            ++suffix;
        }
        if (suffix >= length || suffix == month_mark + 1) continue;
        const std::string suffix_value = unicode_scalar_substring(
            input, *boundaries, suffix, suffix + 1
        );
        if (suffix_value != "日" && suffix_value != "号") continue;
        const size_t end = suffix + 1;
        if (std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& item) {
                return overlaps(start, end, item.semantic_source);
            })) {
            start = end - 1;
            continue;
        }
        candidates.push_back({
            "zh-month-day-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_month_day",
            kMonthDayRule, kMonthDayRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_month_periods(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 2 < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value() && first != "十") continue;
        size_t month_mark = start;
        while (month_mark < length && month_mark - start <= 2
            && unicode_scalar_substring(
                input, *boundaries, month_mark, month_mark + 1
            ) != "月") {
            ++month_mark;
        }
        if (month_mark >= length || month_mark == start || month_mark + 3 > length
            || unicode_scalar_substring(
                input, *boundaries, month_mark, month_mark + 1
            ) != "月") {
            continue;
        }
        const auto month = parse_spoken_cardinal(
            input, *boundaries, start, month_mark
        );
        if (!month.has_value() || *month < 1 || *month > 12
            || !canonical_spoken_calendar_component(
                input, *boundaries, start, month_mark, *month
            )) {
            continue;
        }
        size_t end = month_mark + 2;
        std::string period = unicode_scalar_substring(
            input, *boundaries, month_mark + 1, end
        );
        if (period != "底") {
            if (month_mark + 3 > length) continue;
            end = month_mark + 3;
            period = unicode_scalar_substring(
                input, *boundaries, month_mark + 1, end
            );
            if (period != "上旬" && period != "中旬" && period != "下旬"
                && period != "单月" && period != "首宗" && period != "首批"
                && period != "首个" && period != "首次") {
                continue;
            }
        }
        candidates.push_back({
            "zh-month-period-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_month_period",
            kMonthPeriodRule, kMonthPeriodRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_month_ranges(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 5 <= length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value() && first != "十") continue;
        size_t first_mark = start;
        while (first_mark < length && first_mark - start <= 2
            && unicode_scalar_substring(
                input, *boundaries, first_mark, first_mark + 1
            ) != "月") {
            ++first_mark;
        }
        if (first_mark >= length || first_mark == start) continue;
        if (first_mark + 3 >= length) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, first_mark + 1, first_mark + 2
        );
        if (connector != "到" && connector != "至") continue;
        size_t second_mark = first_mark + 2;
        while (second_mark < length && second_mark - (first_mark + 2) <= 2
            && unicode_scalar_substring(
                input, *boundaries, second_mark, second_mark + 1
            ) != "月") {
            ++second_mark;
        }
        if (second_mark >= length || second_mark == first_mark + 2) continue;
        const auto first_month = parse_spoken_cardinal(
            input, *boundaries, start, first_mark
        );
        const auto second_month = parse_spoken_cardinal(
            input, *boundaries, first_mark + 2, second_mark
        );
        if (!first_month.has_value() || !second_month.has_value()
            || *first_month < 1 || *first_month > 12
            || *second_month < 1 || *second_month > 12) {
            continue;
        }
        const size_t end = second_mark + 1;
        candidates.push_back({
            "zh-month-range-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_month_range",
            kMonthRangeRule, kMonthRangeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_recurring_cross_year_month_ranges(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 9 <= length; ++start) {
        if (unicode_scalar_substring(input, *boundaries, start, start + 2) != "每年") {
            continue;
        }
        size_t first_month_mark = start + 2;
        while (first_month_mark < length && first_month_mark - (start + 2) <= 2
            && unicode_scalar_substring(
                input, *boundaries, first_month_mark, first_month_mark + 1
            ) != "月") {
            ++first_month_mark;
        }
        if (first_month_mark >= length || first_month_mark == start + 2
            || unicode_scalar_substring(
                input, *boundaries, first_month_mark, first_month_mark + 1
            ) != "月") {
            continue;
        }
        size_t connector_index = first_month_mark + 1;
        if (connector_index < length && unicode_scalar_substring(
                input, *boundaries, connector_index, connector_index + 1
            ) == "份") {
            ++connector_index;
        }
        if (connector_index + 4 > length) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, connector_index, connector_index + 1
        );
        if ((connector != "到" && connector != "至")
            || unicode_scalar_substring(
                input, *boundaries, connector_index + 1, connector_index + 3
            ) != "次年") {
            continue;
        }
        const size_t second_month_start = connector_index + 3;
        size_t second_month_mark = second_month_start;
        while (second_month_mark < length
            && second_month_mark - second_month_start <= 2
            && unicode_scalar_substring(
                input, *boundaries, second_month_mark, second_month_mark + 1
            ) != "月") {
            ++second_month_mark;
        }
        if (second_month_mark >= length || second_month_mark == second_month_start
            || unicode_scalar_substring(
                input, *boundaries, second_month_mark, second_month_mark + 1
            ) != "月") {
            continue;
        }
        size_t end = second_month_mark + 1;
        if (end < length && unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ) == "份") {
            ++end;
        }
        candidates.push_back({
            "zh-recurring-cross-year-month-range-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_recurring_cross_year_month_range",
            kRecurringCrossYearMonthRangeRule,
            kRecurringCrossYearMonthRangeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_relative_year_months(
    const std::string& input,
    const std::vector<TransformationCandidate>& exclusions
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 4 <= length; ++start) {
        if (!relative_year_anchor(unicode_scalar_substring(
                input, *boundaries, start, start + 2
            )).has_value()) {
            continue;
        }
        const size_t month_start = start + 2;
        size_t month_mark = month_start;
        while (month_mark < length && month_mark - month_start <= 2
            && unicode_scalar_substring(
                input, *boundaries, month_mark, month_mark + 1
            ) != "月") {
            ++month_mark;
        }
        if (month_mark >= length || month_mark == month_start
            || unicode_scalar_substring(
                input, *boundaries, month_mark, month_mark + 1
            ) != "月") {
            continue;
        }
        const auto month = parse_spoken_cardinal(
            input, *boundaries, month_start, month_mark
        );
        if (!month.has_value() || *month < 1 || *month > 12
            || !canonical_spoken_calendar_component(
                input, *boundaries, month_start, month_mark, *month
            )) {
            continue;
        }
        size_t end = month_mark + 1;
        if (end < length && unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ) == "份") {
            ++end;
        }
        if (std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& exclusion) {
                return overlaps(start, end, exclusion.semantic_source);
            })) {
            start = end - 1;
            continue;
        }
        if (end < length) {
            const std::string next = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (date_or_number_continuation(next) || range_connector(next)
                || contains_any("日号旬季上中下", {next})) {
                continue;
            }
        }
        candidates.push_back({
            "zh-relative-year-month-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_relative_year_month",
            kRelativeYearMonthRule, kRelativeYearMonthRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_followup_months(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 2 <= length; ++start) {
        if (!contextual_followup_month_applies(input, *boundaries, start)) continue;
        size_t month_mark = start;
        while (month_mark < length && month_mark - start <= 2
            && unicode_scalar_substring(
                input, *boundaries, month_mark, month_mark + 1
            ) != "月") {
            ++month_mark;
        }
        if (month_mark >= length || month_mark == start) continue;
        const size_t end = month_mark + 1;
        if (end < length) {
            const std::string next = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (date_or_number_continuation(next) || contains_any("日号份旬季", {next})) {
                continue;
            }
        }
        const auto month = parse_spoken_cardinal(
            input, *boundaries, start, month_mark
        );
        if (!month.has_value() || *month < 1 || *month > 12
            || !canonical_spoken_calendar_component(
                input, *boundaries, start, month_mark, *month
            )) {
            continue;
        }
        candidates.push_back({
            "zh-contextual-followup-month-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "contextual_followup_month",
            kContextualFollowupMonthRule, kContextualFollowupMonthRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_shared_year_date_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& full_dates,
    const std::vector<TransformationCandidate>& month_days
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (const auto& full_date : full_dates) {
        const auto& first = full_date.semantic_source;
        const auto second = std::find_if(
            month_days.begin(),
            month_days.end(),
            [&](const auto& month_day) {
                return month_day.semantic_source.start == first.end + 1;
            }
        );
        if (second == month_days.end()) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, first.end, first.end + 1
        );
        if (connector != "到" && connector != "至") continue;
        const size_t end = second->semantic_source.end;
        candidates.push_back({
            "zh-shared-year-date-range-" + std::to_string(first.start) + "-"
                + std::to_string(end),
            {
                first.start,
                end,
                unicode_scalar_substring(input, *boundaries, first.start, end),
            },
            "zh-CN", "date_time", "explicit_shared_year_date_range",
            kSharedYearDateRangeRule, kSharedYearDateRangeRuleVersion,
        });
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_shared_month_date_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& month_days
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (const auto& month_day : month_days) {
        const auto& first = month_day.semantic_source;
        if (first.end + 2 >= length) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, first.end, first.end + 1
        );
        if (connector != "到" && connector != "至") continue;
        size_t suffix = first.end + 1;
        while (suffix < length && suffix - (first.end + 1) <= 3) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, suffix, suffix + 1
            );
            if (value == "日" || value == "号") break;
            if (!spoken_cardinal_digit(value).has_value() && value != "十") break;
            ++suffix;
        }
        if (suffix >= length || suffix == first.end + 1) continue;
        const std::string suffix_value = unicode_scalar_substring(
            input, *boundaries, suffix, suffix + 1
        );
        if (suffix_value != "日" && suffix_value != "号") continue;
        const size_t end = suffix + 1;
        candidates.push_back({
            "zh-shared-month-date-range-" + std::to_string(first.start) + "-"
                + std::to_string(end),
            {
                first.start,
                end,
                unicode_scalar_substring(input, *boundaries, first.start, end),
            },
            "zh-CN", "date_time", "explicit_shared_month_date_range",
            kSharedMonthDateRangeRule, kSharedMonthDateRangeRuleVersion,
        });
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_full_date_shared_month_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& full_dates
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (const auto& full_date : full_dates) {
        const auto& first = full_date.semantic_source;
        if (first.end + 2 >= length) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, first.end, first.end + 1
        );
        if (connector != "到" && connector != "至") continue;
        size_t suffix = first.end + 1;
        while (suffix < length && suffix - (first.end + 1) <= 3) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, suffix, suffix + 1
            );
            if (value == "日" || value == "号") break;
            if (!spoken_cardinal_digit(value).has_value() && value != "十") break;
            ++suffix;
        }
        if (suffix >= length || suffix == first.end + 1) continue;
        const std::string suffix_value = unicode_scalar_substring(
            input, *boundaries, suffix, suffix + 1
        );
        if (suffix_value != "日" && suffix_value != "号") continue;
        const size_t end = suffix + 1;
        candidates.push_back({
            "zh-full-date-shared-month-range-" + std::to_string(first.start) + "-"
                + std::to_string(end),
            {
                first.start,
                end,
                unicode_scalar_substring(input, *boundaries, first.start, end),
            },
            "zh-CN", "date_time", "explicit_full_date_shared_month_range",
            kFullDateSharedMonthRangeRule, kFullDateSharedMonthRangeRuleVersion,
        });
    }
    return candidates;
}

bool percentage_number_scalar(const std::string& value) {
    return spoken_cardinal_digit(value).has_value() || spoken_small_unit(value).has_value()
        || value == "点";
}

std::vector<TransformationCandidate> detect_percentages(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 3 < length; ++start) {
        if (unicode_scalar_substring(input, *boundaries, start, start + 3) != "百分之") {
            continue;
        }
        size_t end = start + 3;
        while (end < length && percentage_number_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (end == start + 3) continue;
        candidates.push_back({
            "zh-percentage-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "percentage", "explicit_percentage",
            kPercentageRule, kPercentageRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

bool percentage_range_bridge(const std::string& bridge) {
    return bridge == "到" || bridge == "至" || bridge == "-" || bridge == "－"
        || bridge == "—" || bridge == "~" || bridge == "～"
        || bridge == "升至" || bridge == "降至"
        || (ends_with_any(bridge, {"到", "至"}) && contains_any(bridge, {
            "增长", "增加", "上涨", "上升", "下降", "减少", "缩短", "延长",
            "提高", "降低", "扩大", "收窄", "调整", "变为", "改为",
        }));
}

std::vector<TransformationCandidate> detect_percentage_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& percentages
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (size_t index = 1; index < percentages.size(); ++index) {
        const auto& first = percentages[index - 1].semantic_source;
        const auto& second = percentages[index].semantic_source;
        if (second.start <= first.end || second.start - first.end > 12) continue;
        const std::string bridge = unicode_scalar_substring(
            input, *boundaries, first.end, second.start
        );
        if (!percentage_range_bridge(bridge)) continue;
        candidates.push_back({
            "zh-percentage-range-" + std::to_string(first.start) + "-"
                + std::to_string(second.end),
            {
                first.start,
                second.end,
                unicode_scalar_substring(input, *boundaries, first.start, second.end),
            },
            "zh-CN", "percentage", "explicit_percentage_range",
            kPercentageRangeRule, kPercentageRangeRuleVersion,
        });
    }
    return candidates;
}

RuleApproval approve_percentage_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto percentages = detect_percentages(input);
    std::vector<TransformationCandidate> members;
    std::copy_if(
        percentages.begin(), percentages.end(), std::back_inserter(members),
        [&](const auto& member) {
            return candidate.semantic_source.start <= member.semantic_source.start
                && candidate.semantic_source.end >= member.semantic_source.end;
        }
    );
    if (members.size() != 2) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_percentage_range_members"};
    }
    const std::string bridge = unicode_scalar_substring(
        input,
        *boundaries,
        members[0].semantic_source.end,
        members[1].semantic_source.start
    );
    if (!percentage_range_bridge(bridge)) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_percentage_range_bridge"};
    }
    const auto parse_endpoint = [&](const TransformationCandidate& member, bool first) {
        const auto& range = member.semantic_source;
        if (inside_protected_delimiters(input, *boundaries, range.start)
            || incomplete_cross_segment_span(
                context, boundaries->size() - 1, range.start, range.end
            )) {
            return std::optional<std::string>();
        }
        if (range.start > 0) {
            const std::string previous = unicode_scalar_substring(
                input, *boundaries, range.start - 1, range.start
            );
            if (previous == "千" || previous == "万") return std::optional<std::string>();
        }
        if (!first && range.end < boundaries->size() - 1) {
            const std::string next = unicode_scalar_substring(
                input, *boundaries, range.end, range.end + 1
            );
            const std::string right = unicode_scalar_substring(
                input,
                *boundaries,
                range.end,
                std::min(boundaries->size() - 1, range.end + 2)
            );
            const std::string suffix = unicode_scalar_substring(
                input,
                *boundaries,
                range.end,
                std::min(boundaries->size() - 1, range.end + 4)
            );
            const bool adjacent_ascii = next.size() == 1
                && std::isalnum(static_cast<unsigned char>(next[0])) != 0;
            if (number_continuation(next) || adjacent_ascii || range_connector(next)
                || next == "点" || next == "几" || next == "多" || right == "分之"
                || starts_with_any(suffix, {"个百分点", "百分点"})) {
                return std::optional<std::string>();
            }
        }
        return parse_percentage_value(input, *boundaries, range.start + 3, range.end);
    };
    const auto first_value = parse_endpoint(members[0], true);
    const auto second_value = parse_endpoint(members[1], false);
    if (!first_value.has_value() || !second_value.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_percentage_range_endpoint"};
    }
    const std::string first_replacement = *first_value + "%";
    const std::string second_replacement = *second_value + "%";
    std::vector<AtomicEdit> edits = {
        {{
            members[0].semantic_source.start,
            members[0].semantic_source.end,
            members[0].semantic_source.text,
        }, first_replacement},
        {{
            members[1].semantic_source.start,
            members[1].semantic_source.end,
            members[1].semantic_source.text,
        }, second_replacement},
    };
    return {
        RuleDecision::approve,
        first_replacement + bridge + second_replacement,
        {
            {"shape", "two_explicit_percentages_with_range_bridge", kPercentageRangeRule},
            {"boundary", "complete_percentage_range", kPercentageRangeRule},
            {"context", "explicit_percentage_markers_and_range_relation", kPercentageRangeRule},
        },
        std::move(edits),
        "explicit_percentage_range",
    };
}

bool shared_prefix_percentage_range_bridge(const std::string& bridge) {
    return bridge == "到" || bridge == "至" || bridge == "-" || bridge == "－"
        || bridge == "—" || bridge == "~" || bridge == "～";
}

std::vector<TransformationCandidate> detect_shared_prefix_percentage_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& percentages
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (const auto& percentage : percentages) {
        const auto& first = percentage.semantic_source;
        if (first.end >= length) continue;
        const std::string bridge = unicode_scalar_substring(
            input, *boundaries, first.end, first.end + 1
        );
        if (!shared_prefix_percentage_range_bridge(bridge)) continue;
        size_t second_end = first.end + 1;
        while (second_end < length && percentage_number_scalar(unicode_scalar_substring(
                input, *boundaries, second_end, second_end + 1
            ))) {
            ++second_end;
        }
        if (second_end == first.end + 1
            || !parse_percentage_value(
                input, *boundaries, first.end + 1, second_end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-percentage-shared-prefix-range-" + std::to_string(first.start) + "-"
                + std::to_string(second_end),
            {
                first.start,
                second_end,
                unicode_scalar_substring(input, *boundaries, first.start, second_end),
            },
            "zh-CN", "percentage", "shared_prefix_percentage_range",
            kSharedPrefixPercentageRangeRule, kSharedPrefixPercentageRangeRuleVersion,
        });
    }
    return candidates;
}

RuleApproval approve_shared_prefix_percentage_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_prefix_percentage_range"};
    }
    const auto percentages = detect_percentages(input);
    const auto first = std::find_if(
        percentages.begin(), percentages.end(), [&](const auto& percentage) {
            return percentage.semantic_source.start == range.start
                && percentage.semantic_source.end < range.end;
        }
    );
    if (first == percentages.end()) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_shared_prefix_percentage_endpoint"};
    }
    const size_t bridge_start = first->semantic_source.end;
    const size_t second_start = bridge_start + 1;
    if (second_start >= range.end) {
        return {RuleDecision::preserve, "", {}, {},
                "missing_shared_prefix_percentage_endpoint"};
    }
    const std::string bridge = unicode_scalar_substring(
        input, *boundaries, bridge_start, second_start
    );
    if (!shared_prefix_percentage_range_bridge(bridge)) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_prefix_percentage_bridge"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        const std::string suffix = unicode_scalar_substring(
            input, *boundaries, range.end, std::min(length, range.end + 4)
        );
        const bool adjacent_ascii = next.size() == 1
            && std::isalnum(static_cast<unsigned char>(next[0])) != 0;
        if (number_continuation(next) || adjacent_ascii || range_connector(next)
            || next == "点" || next == "几" || next == "多"
            || starts_with_any(suffix, {"个百分点", "百分点"})) {
            return {RuleDecision::preserve, "", {}, {},
                    "shared_prefix_percentage_range_continues_right"};
        }
    }
    const auto first_value = parse_percentage_value(
        input, *boundaries, range.start + 3, bridge_start
    );
    const auto second_value = parse_percentage_value(
        input, *boundaries, second_start, range.end
    );
    if (!first_value.has_value() || !second_value.has_value()) {
        return {RuleDecision::preserve, "", {}, {},
                "invalid_shared_prefix_percentage_endpoint"};
    }
    const std::string first_replacement = *first_value + "%";
    const std::string second_replacement = *second_value + "%";
    return {
        RuleDecision::approve,
        first_replacement + bridge + second_replacement,
        {
            {"shape", "explicit_percentage_plus_shared_numeric_endpoint",
             kSharedPrefixPercentageRangeRule},
            {"boundary", "complete_shared_prefix_percentage_range",
             kSharedPrefixPercentageRangeRule},
            {"context", "single_percentage_prefix_scopes_both_range_endpoints",
             kSharedPrefixPercentageRangeRule},
        },
        {
            {{range.start, bridge_start, unicode_scalar_substring(
                input, *boundaries, range.start, bridge_start
            )}, first_replacement},
            {{second_start, range.end, unicode_scalar_substring(
                input, *boundaries, second_start, range.end
            )}, second_replacement},
        },
        "shared_prefix_percentage_range",
    };
}

bool decimal_duration_number_scalar(const std::string& value) {
    return spoken_cardinal_digit(value).has_value() || spoken_small_unit(value).has_value()
        || value == "点";
}

bool decimal_measure_number_scalar(const std::string& value) {
    return decimal_duration_number_scalar(value) || value == "万";
}

std::vector<TransformationCandidate> detect_explicit_decimal_durations(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && decimal_duration_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        size_t end = number_end;
        for (const size_t unit_length : {size_t{2}, size_t{1}}) {
            if (number_end + unit_length > length) continue;
            const std::string unit = unicode_scalar_substring(
                input, *boundaries, number_end, number_end + unit_length
            );
            if (unit == "毫秒" || unit == "分钟" || unit == "小时"
                || unit == "秒钟" || unit == "秒") {
                end = number_end + unit_length;
                break;
            }
        }
        if (end == number_end || !parse_explicit_decimal_duration(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        const bool threshold = integer_unit_threshold_applies(left, right);
        const bool approximate = !threshold
            && explicit_unit_approximation_applies(left, right);
        candidates.push_back({
            std::string(approximate ? "zh-decimal-duration-approximate-"
                : threshold ? "zh-decimal-duration-threshold-" : "zh-decimal-duration-")
                + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", approximate
                ? "explicit_decimal_duration_approximation"
                : threshold
                    ? "explicit_decimal_duration_threshold" : "explicit_decimal_duration",
            approximate ? kDecimalDurationApproximateRule
                : threshold ? kDecimalDurationThresholdRule : kDecimalDurationRule,
            approximate ? kDecimalDurationApproximateRuleVersion
                : threshold ? kDecimalDurationThresholdRuleVersion
                    : kDecimalDurationRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_decimal_measures(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        bool seen_decimal_mark = false;
        while (number_end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, number_end, number_end + 1
            );
            const bool unit_prefix = ((scalar == "万" || scalar == "亿")
                    && seen_decimal_mark)
                || (scalar == "万" && number_end + 1 < length
                    && unicode_scalar_substring(
                        input, *boundaries, number_end + 1, number_end + 2
                    ) == "股");
            if (!decimal_measure_number_scalar(scalar) || unit_prefix) break;
            if (scalar == "点") seen_decimal_mark = true;
            ++number_end;
        }
        size_t end = number_end;
        std::vector<size_t> unit_starts;
        if (number_end > start && unicode_scalar_substring(
                input, *boundaries, number_end - 1, number_end
            ) == "千") {
            unit_starts.push_back(number_end - 1);
        }
        if (number_end > start && unicode_scalar_substring(
                input, *boundaries, number_end - 1, number_end
            ) == "百") {
            unit_starts.push_back(number_end - 1);
        }
        if (number_end > start && unicode_scalar_substring(
            input, *boundaries, number_end - 1, number_end
            ) == "万") {
            unit_starts.push_back(number_end - 1);
        }
        if (number_end > start && unicode_scalar_substring(
            input, *boundaries, number_end - 1, number_end
            ) == "亿") {
            unit_starts.push_back(number_end - 1);
        }
        unit_starts.push_back(number_end);
        for (const size_t unit_start : unit_starts) {
            for (const size_t unit_length : {size_t{6}, size_t{5}, size_t{4}, size_t{3}, size_t{2}, size_t{1}}) {
                if (unit_start + unit_length > length) continue;
                const std::string unit = unicode_scalar_substring(
                    input, *boundaries, unit_start, unit_start + unit_length
                );
                if (unit == "太拉弗洛普斯" || unit == "太字节每秒"
                    || unit == "米每二次方秒" || unit == "克每百毫升"
                    || unit == "立方厘米" || unit == "牛顿米"
                    || unit == "个ppm" || unit == "个ppb" || unit == "分贝"
                    || unit == "兆比特每秒" || unit == "公里每小时" || unit == "平方公里" || unit == "平方米" || unit == "毫米汞柱" || unit == "毫安时"
                    || unit == "米每秒" || unit == "立方米" || unit == "吉赫兹"
                    || unit == "兆赫兹" || unit == "千赫兹" || unit == "公斤"
                    || unit == "千克" || unit == "公顷" || unit == "个百分点"
                    || unit == "百分点" || unit == "万股" || unit == "赫兹"
                    || unit == "公里" || unit == "厘米" || unit == "毫米" || unit == "毫升"
                    || unit == "毫克" || unit == "兆帕" || unit == "百帕" || unit == "帧"
                    || unit == "倍"
                    || unit == "帧每秒" || unit == "核神经引擎" || unit == "核CPU"
                    || unit == "核GPU" || unit == "吉比特" || unit == "吉字节"
                    || unit == "兆字节" || unit == "太字节" || unit == "毫摩尔每升"
                    || unit == "纳米" || unit == "微米" || unit == "微克"
                    || unit == "GB" || unit == "TB" || unit == "伏特"
                    || unit == "安培" || unit == "瓦特" || unit == "毫安" || unit == "千伏"
                    || unit == "千瓦" || unit == "兆瓦" || unit == "比特" || unit == "字节"
                    || unit == "米" || unit == "吨" || unit == "瓦" || unit == "伏"
                    || unit == "斤") {
                    end = unit_start + unit_length;
                    break;
                }
            }
            if (end != number_end) break;
        }
        if (end == number_end) {
            for (const size_t unit_start : unit_starts) {
                for (const size_t unit_length : {size_t{6}, size_t{5}, size_t{4}, size_t{3}, size_t{2}, size_t{1}}) {
                    if (unit_start + unit_length > length) continue;
                    const auto unit = decimal_measure_unit(
                        input, *boundaries, start, unit_start + unit_length
                    );
                    if (unit.has_value() && unit->first == unit_start) {
                        end = unit_start + unit_length;
                        break;
                    }
                }
                if (end != number_end) break;
            }
        }
        if (end == number_end || !parse_explicit_decimal_measure(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        const bool threshold = integer_unit_threshold_applies(left, right);
        const bool approximate = !threshold
            && explicit_unit_approximation_applies(left, right);
        candidates.push_back({
            std::string(approximate ? "zh-decimal-measure-approximate-"
                : threshold ? "zh-decimal-measure-threshold-" : "zh-decimal-measure-")
                + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "measure", approximate
                ? "explicit_decimal_measure_approximation"
                : threshold
                    ? "explicit_decimal_measure_threshold" : "explicit_decimal_measure",
            approximate ? kDecimalMeasureApproximateRule
                : threshold ? kDecimalMeasureThresholdRule : kDecimalMeasureRule,
            approximate ? kDecimalMeasureApproximateRuleVersion
                : threshold ? kDecimalMeasureThresholdRuleVersion
                    : kDecimalMeasureRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

bool decimal_money_number_scalar(const std::string& value) {
    return decimal_duration_number_scalar(value) || value == "万";
}

std::vector<TransformationCandidate> detect_explicit_decimal_money(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && decimal_money_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        size_t end = number_end;
        std::vector<size_t> unit_starts;
        if (number_end > start && unicode_scalar_substring(
                input, *boundaries, number_end - 1, number_end
            ) == "万") {
            unit_starts.push_back(number_end - 1);
        }
        unit_starts.push_back(number_end);
        for (const size_t unit_start : unit_starts) {
            for (const size_t unit_length : {size_t{5}, size_t{3}, size_t{2}, size_t{1}}) {
                if (unit_start + unit_length > length) continue;
                const std::string unit = unicode_scalar_substring(
                    input, *boundaries, unit_start, unit_start + unit_length
                );
                if (unit == "亿元人民币" || unit == "万元人民币" || unit == "亿美元"
                    || unit == "亿元" || unit == "万元" || unit == "美元"
                    || unit == "港元" || unit == "日元" || unit == "欧元" || unit == "元") {
                    end = unit_start + unit_length;
                    break;
                }
            }
            if (end != number_end) break;
        }
        if (end == number_end || !parse_explicit_decimal_money(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        const bool threshold = money_threshold_applies(left, right)
            && !money_non_threshold_uncertainty_applies(left, right);
        const bool approximate = !threshold && money_approximation_applies(left, right);
        candidates.push_back({
            std::string(approximate
                ? "zh-decimal-money-approximate-"
                : threshold ? "zh-decimal-money-threshold-" : "zh-decimal-money-")
                + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "money", approximate
                ? "explicit_decimal_money_approximation"
                : threshold ? "explicit_decimal_money_threshold" : "explicit_decimal_money",
            approximate
                ? kDecimalMoneyApproximateRule
                : threshold ? kDecimalMoneyThresholdRule : kDecimalMoneyRule,
            approximate
                ? kDecimalMoneyApproximateRuleVersion
                : threshold ? kDecimalMoneyThresholdRuleVersion : kDecimalMoneyRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_stock_prices(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t kuai = start;
        while (kuai < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, kuai, kuai + 1
            );
            if (scalar == "块") break;
            if (!spoken_cardinal_digit(scalar).has_value()
                && !spoken_small_unit(scalar).has_value()) {
                break;
            }
            ++kuai;
        }
        if (kuai == start || kuai >= length
            || unicode_scalar_substring(input, *boundaries, kuai, kuai + 1) != "块") {
            continue;
        }
        size_t end = kuai + 1;
        const size_t maximum = std::min(length, kuai + 5);
        for (size_t candidate_end = kuai + 2; candidate_end <= maximum; ++candidate_end) {
            if (parse_contextual_stock_price(
                    input, *boundaries, start, candidate_end
                ).has_value()) {
                end = candidate_end;
            }
        }
        const std::string left = left_clause_context(input, *boundaries, start, 24);
        if (end == kuai + 1 || !contextual_stock_price_anchor(left)) continue;
        candidates.push_back({
            std::string("zh-contextual-stock-price-") + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "money", "contextual_stock_price",
            kContextualStockPriceRule, kContextualStockPriceRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_temperatures(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        size_t number_start = start;
        std::string prefix;
        if (start + 2 <= length) {
            const std::string marker = unicode_scalar_substring(
                input, *boundaries, start, start + 2
            );
            if (marker == "摄氏" || marker == "华氏"
                || marker == "零下" || marker == "零上") {
                prefix = marker;
                number_start += 2;
            }
        }
        if ((prefix == "摄氏" || prefix == "华氏") && number_start + 2 <= length) {
            const std::string modifier = unicode_scalar_substring(
                input, *boundaries, number_start, number_start + 2
            );
            if (modifier == "零下" || modifier == "零上") number_start += 2;
        }
        if (prefix.empty()) {
            const std::string first = unicode_scalar_substring(
                input, *boundaries, start, start + 1
            );
            if (!spoken_cardinal_digit(first).has_value()
                && !spoken_small_unit(first).has_value()) {
                continue;
            }
        }
        size_t number_end = number_start;
        while (number_end < length && decimal_duration_number_scalar(
                unicode_scalar_substring(input, *boundaries, number_end, number_end + 1)
            )) {
            ++number_end;
        }
        if (number_end == number_start) continue;
        size_t end = number_end;
        if (prefix == "摄氏" || prefix == "华氏") {
            if (number_end < length
                && unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                ) == "度") {
                end = number_end + 1;
            }
        } else {
            if (number_end + 3 <= length) {
                const std::string unit = unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 3
                );
                if (unit == "摄氏度" || unit == "华氏度") end = number_end + 3;
            }
            if (end == number_end && !prefix.empty() && number_end < length
                && unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                ) == "度") {
                end = number_end + 1;
            }
            if (end == number_end && prefix.empty() && number_end < length
                && unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                ) == "度") {
                const std::string left = left_clause_context(
                    input, *boundaries, start, 16
                );
                if (contains_any(left, {
                        "温度", "气温", "室温", "体温", "水温", "油温", "炉温",
                        "冷库", "空调",
                    })) {
                    end = number_end + 1;
                }
            }
        }
        if (end == number_end || !parse_explicit_temperature(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        const bool threshold_context = integer_unit_threshold_applies(left, right);
        const bool threshold = threshold_context
            && !integer_unit_threshold_uncertainty_applies(left, right);
        const bool approximate = !threshold_context
            && explicit_unit_approximation_applies(left, right);
        candidates.push_back({
            std::string(approximate ? "zh-temperature-approximate-"
                : threshold ? "zh-temperature-threshold-" : "zh-temperature-")
                + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "measure", approximate
                ? "explicit_temperature_approximation"
                : threshold ? "explicit_temperature_threshold" : "explicit_temperature",
            approximate
                ? kExplicitTemperatureApproximateRule
                : threshold ? kExplicitTemperatureThresholdRule : kExplicitTemperatureRule,
            approximate
                ? kExplicitTemperatureApproximateRuleVersion
                : threshold
                    ? kExplicitTemperatureThresholdRuleVersion
                    : kExplicitTemperatureRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_temperature_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& temperatures
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (const auto& second : temperatures) {
        if (second.semantic_source.start == 0) continue;
        const size_t connector = second.semantic_source.start - 1;
        const std::string bridge = unicode_scalar_substring(
            input, *boundaries, connector, connector + 1
        );
        if (bridge != "到" && bridge != "至") continue;
        size_t start = connector;
        const auto first = std::find_if(
            temperatures.begin(), temperatures.end(), [&](const auto& endpoint) {
                return endpoint.semantic_source.end == connector;
            }
        );
        if (first != temperatures.end()) {
            start = first->semantic_source.start;
        } else {
            while (start > 0 && decimal_duration_number_scalar(
                    unicode_scalar_substring(input, *boundaries, start - 1, start)
                )) {
                --start;
            }
            if (start >= 2) {
                const std::string sign = unicode_scalar_substring(
                    input, *boundaries, start - 2, start
                );
                if (sign == "零下" || sign == "零上") start -= 2;
            }
        }
        if (start == connector) continue;
        const size_t end = second.semantic_source.end;
        if (!parse_explicit_temperature_range(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-temperature-range-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "measure", "explicit_temperature_range",
            kExplicitTemperatureRangeRule, kExplicitTemperatureRangeRuleVersion,
        });
    }
    const size_t length = boundaries->size() - 1;
    for (size_t connector = 0; connector < length; ++connector) {
        const std::string bridge = unicode_scalar_substring(
            input, *boundaries, connector, connector + 1
        );
        if (bridge != "到" && bridge != "至") continue;
        size_t second_number_start = connector + 1;
        if (second_number_start + 2 <= length) {
            const std::string sign = unicode_scalar_substring(
                input, *boundaries, second_number_start, second_number_start + 2
            );
            if (sign == "零下" || sign == "零上") second_number_start += 2;
        }
        size_t second_number_end = second_number_start;
        while (second_number_end < length && decimal_duration_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, second_number_end, second_number_end + 1
                )
            )) {
            ++second_number_end;
        }
        if (second_number_end == second_number_start || second_number_end >= length
            || unicode_scalar_substring(
                input, *boundaries, second_number_end, second_number_end + 1
            ) != "度") {
            continue;
        }
        size_t start = connector;
        while (start > 0 && decimal_duration_number_scalar(
                unicode_scalar_substring(input, *boundaries, start - 1, start)
            )) {
            --start;
        }
        if (start >= 2) {
            const std::string sign = unicode_scalar_substring(
                input, *boundaries, start - 2, start
            );
            if (sign == "零下" || sign == "零上") start -= 2;
        }
        const size_t end = second_number_end + 1;
        if (start == connector || !parse_explicit_temperature_range(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 20);
        if (!contains_any(left, {
                "温度", "气温", "室温", "体温", "水温", "油温", "炉温", "冷库",
            })) {
            continue;
        }
        candidates.push_back({
            "zh-temperature-range-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "measure", "explicit_temperature_range",
            kExplicitTemperatureRangeRule, kExplicitTemperatureRangeRuleVersion,
        });
    }
    return candidates;
}

bool integer_money_number_scalar(const std::string& value) {
    return spoken_cardinal_digit(value).has_value() || spoken_small_unit(value).has_value()
        || value == "万" || value == "亿";
}

bool integer_measure_number_scalar(const std::string& value) {
    return integer_money_number_scalar(value) || value == "亿";
}

bool explicit_signed_number_anchor_applies(const std::string& left) {
    return ends_with_any(left, {
        "计算结果是", "计算结果为", "结果是", "结果为",
        "测量结果是", "测量结果为", "数值是", "数值为",
        "读数为", "测得值为", "温度是", "温度为", "误差为", "差值为",
    });
}

std::optional<std::string> parse_explicit_signed_number(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (start >= end
        || unicode_scalar_substring(input, boundaries, start, start + 1) != "负") {
        return std::nullopt;
    }
    const size_t number_start = start + 1;
    size_t decimal_mark = end;
    for (size_t index = number_start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != end) return std::nullopt;
            decimal_mark = index;
        }
    }
    if (decimal_mark == number_start) return std::nullopt;
    const auto integer = parse_spoken_money_integer(
        input, boundaries, number_start, decimal_mark
    );
    if (!integer.has_value() || *integer > 99999999) return std::nullopt;
    const auto formatted_integer = format_spoken_money_integer(
        input, boundaries, number_start, decimal_mark
    );
    if (!formatted_integer.has_value()) return std::nullopt;
    if (decimal_mark == end) return "-" + *formatted_integer;
    if (decimal_mark + 1 >= end || end - decimal_mark - 1 > 3) {
        return std::nullopt;
    }
    std::string replacement = "-" + std::to_string(*integer) + ".";
    for (size_t index = decimal_mark + 1; index < end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        replacement += static_cast<char>('0' + *digit);
    }
    return replacement;
}

RuleApproval approve_explicit_signed_number(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const std::string left = left_clause_context(input, *boundaries, range.start, 18);
    if (unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)
        || !explicit_signed_number_anchor_applies(left)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_signed_number_context"};
    }
    if (range.end < length) {
        const std::string next = unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        );
        if (number_continuation(next) || next == "点") {
            return {RuleDecision::preserve, "", {}, {}, "signed_number_continues_right"};
        }
    }
    const auto replacement = parse_explicit_signed_number(
        input, *boundaries, range.start, range.end
    );
    if (!replacement.has_value()) {
        return {RuleDecision::preserve, "", {}, {}, "invalid_signed_number"};
    }
    return {
        RuleDecision::approve,
        *replacement,
        {
            {"shape", "explicit_negative_spoken_cardinal", kExplicitSignedNumberRule},
            {"boundary", "complete_signed_integer_or_decimal", kExplicitSignedNumberRule},
            {"context", "explicit_numeric_result_anchor", kExplicitSignedNumberRule},
        },
        {{{range.start, range.end, range.text}, *replacement}},
        "explicit_signed_number",
    };
}

std::vector<TransformationCandidate> detect_explicit_signed_numbers(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (unicode_scalar_substring(input, *boundaries, start, start + 1) != "负") {
            continue;
        }
        size_t end = start + 1;
        while (end < length && integer_money_number_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (end == start + 1) continue;
        if (end < length && unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ) == "点") {
            ++end;
            const size_t fraction_start = end;
            while (end < length && spoken_year_digit(unicode_scalar_substring(
                    input, *boundaries, end, end + 1
                )).has_value()) {
                ++end;
            }
            if (end == fraction_start || end - fraction_start > 3) continue;
        }
        if (end < length) {
            const std::string next = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (number_continuation(next) || next == "点") continue;
        }
        if (!explicit_signed_number_anchor_applies(
                left_clause_context(input, *boundaries, start, 18)
            )
            || inside_protected_delimiters(input, *boundaries, start)
            || !parse_explicit_signed_number(input, *boundaries, start, end).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-signed-number-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "number", "explicit_signed_number",
            kExplicitSignedNumberRule, kExplicitSignedNumberRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_integer_money(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, number_end, number_end + 1
            );
            if (!integer_money_number_scalar(scalar)) break;
            if (scalar == "亿" && number_end + 1 < length
                && unicode_scalar_substring(
                    input, *boundaries, number_end + 1, number_end + 2
                ) == "元") {
                break;
            }
            ++number_end;
        }
        size_t end = number_end;
        for (const size_t unit_length : {size_t{5}, size_t{3}, size_t{2}, size_t{1}}) {
            if (number_end + unit_length > length) continue;
            const std::string unit = unicode_scalar_substring(
                input, *boundaries, number_end, number_end + unit_length
            );
            if (unit == "亿元人民币" || unit == "亿美元" || unit == "亿元"
                || unit == "万元" || unit == "美元" || unit == "港元"
                || unit == "日元" || unit == "欧元" || unit == "元") {
                end = number_end + unit_length;
                break;
            }
        }
        if (end == number_end || !parse_explicit_integer_money(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        const bool threshold = money_threshold_applies(left, right)
            && !money_non_threshold_uncertainty_applies(left, right);
        const bool approximate = !threshold && money_approximation_applies(left, right);
        candidates.push_back({
            std::string(approximate
                ? "zh-integer-money-approximate-"
                : threshold ? "zh-integer-money-threshold-" : "zh-integer-money-")
                + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "money",
            approximate
                ? "explicit_integer_money_approximation"
                : threshold ? "explicit_integer_money_threshold" : "explicit_integer_money",
            approximate
                ? kExplicitIntegerMoneyApproximateRule
                : threshold ? kExplicitIntegerMoneyThresholdRule : kExplicitIntegerMoneyRule,
            approximate
                ? kExplicitIntegerMoneyApproximateRuleVersion
                : threshold
                ? kExplicitIntegerMoneyThresholdRuleVersion
                : kExplicitIntegerMoneyRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_integer_measures(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (start > 0 && contains_any(
                unicode_scalar_substring(input, *boundaries, start - 1, start),
                {"点", "."})) {
            continue;
        }
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && integer_measure_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        size_t end = number_end;
        std::vector<size_t> unit_starts;
        if (number_end > start && unicode_scalar_substring(
                input, *boundaries, number_end - 1, number_end
            ) == "千") {
            unit_starts.push_back(number_end - 1);
        }
        if (number_end > start && unicode_scalar_substring(
                input, *boundaries, number_end - 1, number_end
            ) == "百") {
            unit_starts.push_back(number_end - 1);
        }
        if (number_end > start && unicode_scalar_substring(
                input, *boundaries, number_end - 1, number_end
            ) == "亿") {
            unit_starts.push_back(number_end - 1);
        }
        unit_starts.push_back(number_end);
        std::string detected_unit;
        bool approximate_person_count = false;
        for (const size_t unit_start : unit_starts) {
            for (const size_t unit_length : {size_t{6}, size_t{5}, size_t{4}, size_t{3}, size_t{2}, size_t{1}}) {
                if (unit_start + unit_length > length) continue;
                const std::string unit = unicode_scalar_substring(
                    input, *boundaries, unit_start, unit_start + unit_length
                );
                if (unit == "太拉弗洛普斯" || unit == "太字节每秒"
                    || unit == "米每二次方秒" || unit == "克每百毫升"
                    || unit == "立方厘米" || unit == "牛顿米"
                    || unit == "个ppm" || unit == "个ppb" || unit == "分贝"
                    || unit == "兆比特每秒" || unit == "千米每小时" || unit == "公里每小时"
                    || unit == "平方公里" || unit == "平方米"
                    || unit == "毫米汞柱"
                    || unit == "毫安时" || unit == "立方米"
                    || unit == "吉赫兹" || unit == "兆赫兹" || unit == "千赫兹"
                    || unit == "千比特每秒"
                    || unit == "公斤" || unit == "千克" || unit == "公顷"
                    || unit == "赫兹" || unit == "公里" || unit == "千米" || unit == "厘米"
                    || unit == "毫米" || unit == "毫升" || unit == "毫克"
                    || unit == "兆帕" || unit == "百帕" || unit == "只私募产品"
                    || unit == "个代表团" || unit == "人次"
                    || unit == "个基点" || unit == "基点"
                    || unit == "吨"
                    || unit == "帧每秒" || unit == "核神经引擎" || unit == "核CPU"
                    || unit == "核GPU" || unit == "吉比特" || unit == "吉字节"
                    || unit == "兆字节" || unit == "太字节" || unit == "毫摩尔每升"
                    || unit == "纳米" || unit == "微米" || unit == "微克"
                    || unit == "GB" || unit == "TB" || unit == "伏特"
                    || unit == "安培" || unit == "瓦特" || unit == "毫安" || unit == "千伏"
                    || unit == "千瓦" || unit == "兆瓦" || unit == "比特" || unit == "字节"
                    || unit == "欧姆" || unit == "站台"
                    || unit == "瓦" || unit == "伏" || unit == "安" || unit == "辆" || unit == "颗"
                    || unit == "件" || unit == "批" || unit == "人" || unit == "箱"
                    || unit == "股"
                    || unit == "帧" || unit == "转" || unit == "步"
                    || unit == "克" || unit == "米" || unit == "升" || unit == "斤"
                    || unit == "多公里" || unit == "度" || unit == "倍"
                    || unit == "名观众" || unit == "例" || unit == "届" || unit == "角"
                    || unit == "点" || unit == "亿") {
                    end = unit_start + unit_length;
                    detected_unit = unit;
                    break;
                }
            }
            if (end != number_end) break;
        }
        if (detected_unit.empty()) {
            for (const size_t unit_length : {size_t{6}, size_t{5}, size_t{4}, size_t{3}, size_t{2}, size_t{1}}) {
                if (number_end + unit_length > length) continue;
                const auto unit = explicit_integer_measure_unit(
                    input, *boundaries, start, number_end + unit_length
                );
                if (unit.has_value() && unit->first == number_end) {
                    end = number_end + unit_length;
                    detected_unit = unit->second;
                    break;
                }
            }
        }
        if (detected_unit.empty() && number_end + 2 <= length
            && unicode_scalar_substring(
                input, *boundaries, number_end, number_end + 2
            ) == "多人") {
            end = number_end + 2;
            detected_unit = "人";
            approximate_person_count = true;
        }
        if (detected_unit.empty()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        if (approximate_person_count && !ends_with_any(left, {
                "现场有", "现场共有", "共有", "一共有", "总共有", "累计有",
                "到场有", "出席有", "参会有",
            })) {
            continue;
        }
        if ((detected_unit == "角" && (!ends_with_any(left, {"元"})
                || starts_with_any(right, {"形", "函数"})))
            || (detected_unit == "例" && !contains_any(left, {
                "病例", "新增", "确诊", "感染", "死亡",
            }))
            || (detected_unit == "届" && (start == 0
                || unicode_scalar_substring(input, *boundaries, start - 1, start) != "第"))
            || (detected_unit == "点" && !contains_any(left, {
                "指数报", "指数收报", "指数为",
            }))
            || (detected_unit == "亿" && !contains_any(left, {"成交额", "交易额"}))
            || (detected_unit == "安" && !contains_any(left, {"电流"}))) {
            continue;
        }
        const bool threshold_context = integer_unit_threshold_applies(left, right)
            || ((detected_unit == "辆" || detected_unit == "颗")
                && ends_with_any(left, {"超"}));
        const bool explicit_human_traffic_threshold = detected_unit == "人次"
            && ends_with_any(left, {"超过"});
        const bool threshold = threshold_context
            && (!integer_unit_threshold_uncertainty_applies(left, right)
                || explicit_human_traffic_threshold);
        const bool rate_distance_approximation = !threshold_context
            && contextual_approximate_rate_distance(left, detected_unit);
        const bool approximate = !threshold_context && !rate_distance_approximation
            && (approximate_person_count
                || explicit_unit_approximation_applies(left, right)
                || contextual_infix_distance_approximation(left, detected_unit));
        if (detected_unit == "千米" && !approximate) {
            continue;
        }
        const bool periodic = !threshold_context && !approximate
            && !rate_distance_approximation
            && explicit_periodic_unit_applies(left);
        const std::string number = unicode_scalar_substring(
            input, *boundaries, start, number_end
        );
        const bool range_bound_meter = detected_unit == "米" && start >= 2
            && range_connector(unicode_scalar_substring(
                input, *boundaries, start - 1, start
            ))
            && decimal_money_number_scalar(unicode_scalar_substring(
                input, *boundaries, start - 2, start - 1
            ));
        const bool strongly_anchored_meter = detected_unit == "米"
            && !threshold && !periodic
            && (range_bound_meter
                || explicit_integer_meter_context(input, *boundaries, start, end, number));
        const bool periodic_only_unit = detected_unit == "克" || detected_unit == "升"
            || (detected_unit == "米" && !strongly_anchored_meter);
        const bool unsupported_degree = detected_unit == "度"
            && !contextual_battery_capacity_degree(left, detected_unit);
        const bool unsupported_infix_distance = detected_unit == "多公里"
            && !contextual_infix_distance_approximation(left, detected_unit);
        if (unsupported_degree || unsupported_infix_distance
            || (periodic_only_unit && !periodic && !rate_distance_approximation)
            || !(periodic
                ? parse_explicit_periodic_integer_measure(
                    input, *boundaries, start, end
                )
                : parse_explicit_integer_measure(input, *boundaries, start, end)).has_value()) {
            continue;
        }
        const char* candidate_prefix = threshold
            ? "zh-integer-measure-threshold-"
            : rate_distance_approximation ? "zh-rate-distance-approximate-"
            : approximate ? "zh-integer-measure-approximate-"
            : periodic ? "zh-integer-measure-periodic-" : "zh-integer-measure-";
        const char* subtype = threshold
            ? "explicit_integer_measure_threshold"
            : rate_distance_approximation ? "contextual_approximate_rate_distance"
            : approximate ? "explicit_integer_measure_approximation"
            : periodic ? "explicit_integer_measure_periodic" : "explicit_integer_measure";
        const char* rule_id = threshold
            ? kExplicitIntegerMeasureThresholdRule
            : rate_distance_approximation ? kContextualApproximateRateDistanceRule
            : approximate ? kExplicitIntegerMeasureApproximateRule
            : periodic ? kExplicitIntegerMeasurePeriodicRule : kExplicitIntegerMeasureRule;
        const char* rule_version = threshold
            ? kExplicitIntegerMeasureThresholdRuleVersion
            : rate_distance_approximation
                ? kContextualApproximateRateDistanceRuleVersion
            : approximate
                ? kExplicitIntegerMeasureApproximateRuleVersion
                : periodic
                ? kExplicitIntegerMeasurePeriodicRuleVersion
                : kExplicitIntegerMeasureRuleVersion;
        candidates.push_back({
            std::string(candidate_prefix) + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "measure", subtype, rule_id, rule_version,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_integer_durations(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && integer_money_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        size_t end = number_end;
        for (const size_t unit_length : {size_t{2}, size_t{1}}) {
            if (number_end + unit_length > length) continue;
            const std::string unit = unicode_scalar_substring(
                input, *boundaries, number_end, number_end + unit_length
            );
            if (unit == "毫秒" || unit == "分钟" || unit == "小时"
                || unit == "秒钟" || unit == "秒") {
                end = number_end + unit_length;
                break;
            }
        }
        if (end == number_end || !parse_explicit_integer_duration(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 8);
        const bool threshold_context = integer_unit_threshold_applies(left, right);
        const bool threshold = threshold_context
            && !integer_unit_threshold_uncertainty_applies(left, right);
        const bool approximate = !threshold_context
            && explicit_unit_approximation_applies(left, right);
        const bool periodic = !threshold_context && !approximate
            && explicit_periodic_unit_applies(left);
        const char* candidate_prefix = threshold
            ? "zh-integer-duration-threshold-"
            : approximate ? "zh-integer-duration-approximate-"
            : periodic ? "zh-integer-duration-periodic-" : "zh-integer-duration-";
        const char* subtype = threshold
            ? "explicit_integer_duration_threshold"
            : approximate ? "explicit_integer_duration_approximation"
            : periodic ? "explicit_integer_duration_periodic" : "explicit_integer_duration";
        const char* rule_id = threshold
            ? kExplicitIntegerDurationThresholdRule
            : approximate ? kExplicitIntegerDurationApproximateRule
            : periodic ? kExplicitIntegerDurationPeriodicRule : kExplicitIntegerDurationRule;
        const char* rule_version = threshold
            ? kExplicitIntegerDurationThresholdRuleVersion
            : approximate
                ? kExplicitIntegerDurationApproximateRuleVersion
                : periodic
                ? kExplicitIntegerDurationPeriodicRuleVersion
                : kExplicitIntegerDurationRuleVersion;
        candidates.push_back({
            std::string(candidate_prefix) + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", subtype, rule_id, rule_version,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_lock_period_months(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 3; start < length; ++start) {
        if (unicode_scalar_substring(input, *boundaries, start - 3, start) != "锁定期") {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && integer_money_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        const size_t end = number_end + 2;
        if (end > length
            || unicode_scalar_substring(input, *boundaries, number_end, end) != "个月"
            || !parse_explicit_lock_period_months(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-lock-period-months-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_lock_period_months",
            kExplicitLockPeriodMonthsRule, kExplicitLockPeriodMonthsRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_large_approximate_year_spans(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 4 <= length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && integer_money_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        if (number_end + 2 > length || unicode_scalar_substring(
                input, *boundaries, number_end, number_end + 2
            ) != "余年") {
            continue;
        }
        const auto value = parse_spoken_money_integer(
            input, *boundaries, start, number_end
        );
        if (!value.has_value() || *value < 100 || *value > 99999999) continue;
        const size_t end = number_end + 2;
        candidates.push_back({
            "zh-large-approximate-year-span-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_large_approximate_year_span",
            kLargeApproximateYearSpanRule, kLargeApproximateYearSpanRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_anchored_year_or_day_durations(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t number_end = start;
        while (number_end < length && integer_money_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, number_end, number_end + 1
                )
            )) {
            ++number_end;
        }
        if (number_end >= length) continue;
        const std::string unit = unicode_scalar_substring(
            input, *boundaries, number_end, number_end + 1
        );
        if (unit != "年" && unit != "天") continue;
        const size_t end = number_end + 1;
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        const std::string right = right_clause_context(input, *boundaries, end, 12);
        const bool validity_years = unit == "年" && ends_with_any(left, {
            "有效期为", "有效期是", "有效期限为", "有效期限是",
        });
        const bool renewal_days = unit == "天" && ends_with_any(left, {
            "期满前大概", "期满前大约", "期满前约",
            "到期前大概", "到期前大约", "到期前约",
        }) && starts_with_any(right, {
            "可申请续签", "可以申请续签", "可申请续期", "可以申请续期",
        });
        if (!validity_years && !renewal_days) continue;
        if (!parse_explicit_anchored_year_or_day_duration(
                input, *boundaries, start, end, unit,
                validity_years ? 999 : 999999
            ).has_value()) {
            continue;
        }
        const char* rule_id = validity_years
            ? kExplicitValidityYearsRule : kExplicitRenewalApproximateDaysRule;
        const char* rule_version = validity_years
            ? kExplicitValidityYearsRuleVersion
            : kExplicitRenewalApproximateDaysRuleVersion;
        const char* subtype = validity_years
            ? "explicit_validity_years" : "explicit_renewal_approximate_days";
        candidates.push_back({
            std::string(validity_years ? "zh-validity-years-" : "zh-renewal-days-")
                + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", subtype, rule_id, rule_version,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_compound_durations(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const auto components = detect_explicit_integer_durations(input);
    std::vector<TransformationCandidate> candidates;
    for (size_t index = 0; index + 1 < components.size(); ++index) {
        const auto& first = components[index].semantic_source;
        const auto& second = components[index + 1].semantic_source;
        if (first.end != second.start) {
            continue;
        }
        const bool hour_minute = ends_with_any(first.text, {"小时"})
            && ends_with_any(second.text, {"分钟"});
        const bool minute_second = ends_with_any(first.text, {"分钟"})
            && ends_with_any(second.text, {"秒", "秒钟"});
        if (!hour_minute && !minute_second) {
            continue;
        }
        const size_t start = first.start;
        const size_t end = second.end;
        const auto replacement = minute_second
            ? parse_explicit_minute_second_duration(input, *boundaries, start, end)
            : parse_explicit_compound_duration(input, *boundaries, start, end);
        if (!replacement.has_value()) continue;
        const char* rule_id = minute_second
            ? kExplicitMinuteSecondDurationRule : kExplicitCompoundDurationRule;
        const char* rule_version = minute_second
            ? kExplicitMinuteSecondDurationRuleVersion : kExplicitCompoundDurationRuleVersion;
        candidates.push_back({
            "zh-compound-duration-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_compound_duration",
            rule_id, rule_version,
        });
        ++index;
    }
    return candidates;
}

enum class ExplicitUnitRangeKind {
    money,
    measure,
    duration,
};

struct ExplicitUnitRangeMatch {
    size_t start;
    size_t end;
    size_t first_number_start;
    size_t first_number_end;
    size_t second_number_start;
    size_t second_number_end;
    std::string bridge;
    std::string unit;
    std::string first_replacement;
    std::string second_replacement;
};

std::optional<std::pair<size_t, std::string>> explicit_range_unit(
    ExplicitUnitRangeKind kind,
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    if (kind == ExplicitUnitRangeKind::money) {
        return decimal_money_unit(input, boundaries, start, end);
    }
    if (kind == ExplicitUnitRangeKind::measure) {
        return explicit_integer_measure_unit(input, boundaries, start, end);
    }
    return explicit_integer_duration_unit(input, boundaries, start, end);
}

std::optional<std::string> format_explicit_range_number(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    size_t decimal_mark = end;
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            if (decimal_mark != end) return std::nullopt;
            decimal_mark = index;
        }
    }
    const auto integer = parse_spoken_money_integer(
        input, boundaries, start, decimal_mark
    );
    if (!integer.has_value() || *integer > 99999999) return std::nullopt;
    if (decimal_mark == end) {
        return format_spoken_money_integer(input, boundaries, start, end);
    }
    if (decimal_mark == start || decimal_mark + 1 == end
        || end - decimal_mark - 1 > 3) {
        return std::nullopt;
    }
    std::string replacement = std::to_string(*integer) + ".";
    for (size_t index = decimal_mark + 1; index < end; ++index) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, boundaries, index, index + 1
        ));
        if (!digit.has_value()) return std::nullopt;
        replacement += static_cast<char>('0' + *digit);
    }
    return replacement;
}

bool natural_single_digit_range_endpoint(
    const std::string& input,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    for (size_t index = start; index < end; ++index) {
        if (unicode_scalar_substring(input, boundaries, index, index + 1) == "点") {
            return false;
        }
    }
    const auto integer = parse_spoken_money_integer(input, boundaries, start, end);
    return integer.has_value() && *integer < 10;
}

std::vector<TransformationCandidate> explicit_range_endpoints(
    ExplicitUnitRangeKind kind,
    const std::string& input
) {
    std::vector<TransformationCandidate> endpoints;
    if (kind == ExplicitUnitRangeKind::money) {
        endpoints = detect_explicit_decimal_money(input);
        const auto integers = detect_explicit_integer_money(input);
        endpoints.insert(endpoints.end(), integers.begin(), integers.end());
    } else if (kind == ExplicitUnitRangeKind::measure) {
        endpoints = detect_explicit_decimal_measures(input);
        const auto integers = detect_explicit_integer_measures(input);
        endpoints.insert(endpoints.end(), integers.begin(), integers.end());
    } else {
        endpoints = detect_explicit_decimal_durations(input);
        const auto integers = detect_explicit_integer_durations(input);
        endpoints.insert(endpoints.end(), integers.begin(), integers.end());
    }
    std::sort(endpoints.begin(), endpoints.end(), [](const auto& left, const auto& right) {
        return left.semantic_source.start < right.semantic_source.start
            || (left.semantic_source.start == right.semantic_source.start
                && left.semantic_source.end < right.semantic_source.end);
    });
    endpoints.erase(
        std::remove_if(endpoints.begin(), endpoints.end(), [&](const auto& endpoint) {
            return std::any_of(endpoints.begin(), endpoints.end(), [&](const auto& other) {
                return other.semantic_source.start <= endpoint.semantic_source.start
                    && other.semantic_source.end >= endpoint.semantic_source.end
                    && (other.semantic_source.start < endpoint.semantic_source.start
                        || other.semantic_source.end > endpoint.semantic_source.end);
            });
        }),
        endpoints.end()
    );
    return endpoints;
}

std::vector<ExplicitUnitRangeMatch> find_explicit_unit_ranges(
    ExplicitUnitRangeKind kind,
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const auto endpoints = explicit_range_endpoints(kind, input);
    std::vector<ExplicitUnitRangeMatch> matches;
    for (size_t second_index = 0; second_index < endpoints.size(); ++second_index) {
        const auto& second = endpoints[second_index].semantic_source;
        const auto second_unit = explicit_range_unit(
            kind, input, *boundaries, second.start, second.end
        );
        if (!second_unit.has_value() || second_unit->first <= second.start) continue;
        const auto second_replacement = format_explicit_range_number(
            input, *boundaries, second.start, second_unit->first
        );
        if (!second_replacement.has_value()) continue;

        bool found_full_endpoint = false;
        for (size_t first_index = second_index; first_index > 0; --first_index) {
            const auto& first = endpoints[first_index - 1].semantic_source;
            if (first.end > second.start) continue;
            if (second.start - first.end > 12) break;
            const auto first_unit = explicit_range_unit(
                kind, input, *boundaries, first.start, first.end
            );
            if (!first_unit.has_value() || first_unit->second != second_unit->second) continue;
            const std::string bridge = unicode_scalar_substring(
                input, *boundaries, first.end, second.start
            );
            if (!percentage_range_bridge(bridge)) continue;
            const auto first_replacement = format_explicit_range_number(
                input, *boundaries, first.start, first_unit->first
            );
            if (!first_replacement.has_value()) continue;
            matches.push_back({
                first.start,
                second.end,
                first.start,
                first_unit->first,
                second.start,
                second_unit->first,
                bridge,
                second_unit->second,
                *first_replacement,
                *second_replacement,
            });
            found_full_endpoint = true;
            break;
        }
        if (found_full_endpoint || second.start == 0) continue;
        const size_t connector = second.start - 1;
        const std::string bridge = unicode_scalar_substring(
            input, *boundaries, connector, second.start
        );
        if (!range_connector(bridge)) continue;
        size_t first_start = connector;
        while (first_start > 0 && decimal_money_number_scalar(
                unicode_scalar_substring(
                    input, *boundaries, first_start - 1, first_start
                )
            )) {
            --first_start;
        }
        if (first_start == connector) continue;
        const auto first_replacement = format_explicit_range_number(
            input, *boundaries, first_start, connector
        );
        if (!first_replacement.has_value()) continue;
        matches.push_back({
            first_start,
            second.end,
            first_start,
            connector,
            second.start,
            second_unit->first,
            bridge,
            second_unit->second,
            *first_replacement,
            *second_replacement,
        });
    }
    return matches;
}

std::vector<TransformationCandidate> detect_explicit_unit_ranges(
    ExplicitUnitRangeKind kind,
    const std::string& input,
    const char* category,
    const char* subtype,
    const char* rule_id,
    const char* rule_version
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (const auto& match : find_explicit_unit_ranges(kind, input)) {
        candidates.push_back({
            std::string(rule_id) + "-" + std::to_string(match.start) + "-"
                + std::to_string(match.end),
            {
                match.start,
                match.end,
                unicode_scalar_substring(input, *boundaries, match.start, match.end),
            },
            "zh-CN", category, subtype, rule_id, rule_version,
        });
    }
    return candidates;
}

RuleApproval approve_explicit_unit_range(
    ExplicitUnitRangeKind kind,
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) {
        return {RuleDecision::error, "", {}, {}, "invalid_utf8"};
    }
    const auto& range = candidate.semantic_source;
    const size_t length = boundaries->size() - 1;
    const auto matches = find_explicit_unit_ranges(kind, input);
    const auto match = std::find_if(matches.begin(), matches.end(), [&](const auto& value) {
        return value.start == range.start && value.end == range.end;
    });
    if (match == matches.end()
        || unicode_scalar_substring(input, *boundaries, range.start, range.end) != range.text
        || inside_protected_delimiters(input, *boundaries, range.start)
        || (range.start == 0 && context.has_left_neighbor)
        || (range.end == length && context.has_right_neighbor)
        || incomplete_cross_segment_span(context, length, range.start, range.end)) {
        return {RuleDecision::preserve, "", {}, {}, "uncertain_explicit_unit_range"};
    }
    const std::string left = left_clause_context(input, *boundaries, range.start, 16);
    const std::string right = right_clause_context(input, *boundaries, range.end, 8);
    const bool released_approximate_distance_range = kind == ExplicitUnitRangeKind::measure
        && (match->unit == "公里" || match->unit == "千米")
        && contains_any(left, {"路程", "距离", "全程"});
    if ((contains_any(left, {
            "大约", "大概", "约为", "约在", "将近", "接近", "差不多",
        }) && !released_approximate_distance_range)
        || integer_unit_threshold_applies(left, right)
        || ends_with_any(left, {"百分之", "千分之"})
        || starts_with_any(right, {
            "多", "左右", "上下", "前后", "以内", "以上", "以下", "余", "几",
        })) {
        return {RuleDecision::preserve, "", {}, {}, "approximate_explicit_unit_range"};
    }
    if ((range.start > 0 && range_connector(unicode_scalar_substring(
            input, *boundaries, range.start - 1, range.start
        )) && !money_predicate_connector_applies(left))
        || (range.end < length && range_connector(unicode_scalar_substring(
            input, *boundaries, range.end, range.end + 1
        )))) {
        return {RuleDecision::preserve, "", {}, {}, "chained_explicit_unit_range"};
    }
    const bool released_single_digit_range = (kind == ExplicitUnitRangeKind::measure
            && (match->unit == "米" || match->unit == "个人"))
        || contains_any(left, {
        "距地铁站", "距离", "补贴金额", "成交额", "交易额",
    });
    if (!released_single_digit_range && (natural_single_digit_range_endpoint(
            input, *boundaries, match->first_number_start, match->first_number_end
        )
        || natural_single_digit_range_endpoint(
            input, *boundaries, match->second_number_start, match->second_number_end
        ))) {
        return {RuleDecision::preserve, "", {}, {}, "natural_single_digit_range_endpoint"};
    }
    const std::string middle = unicode_scalar_substring(
        input, *boundaries, match->first_number_end, match->second_number_start
    );
    const std::string suffix = unicode_scalar_substring(
        input, *boundaries, match->second_number_end, range.end
    );
    const std::string replacement = match->first_replacement + middle
        + match->second_replacement + suffix;
    std::vector<AtomicEdit> edits = {
        {{
            match->first_number_start,
            match->first_number_end,
            unicode_scalar_substring(
                input, *boundaries, match->first_number_start, match->first_number_end
            ),
        }, match->first_replacement},
        {{
            match->second_number_start,
            match->second_number_end,
            unicode_scalar_substring(
                input, *boundaries, match->second_number_start, match->second_number_end
            ),
        }, match->second_replacement},
    };
    return {
        RuleDecision::approve,
        replacement,
        {
            {"shape", "two_spoken_numbers_with_explicit_unit_range", candidate.rule_id},
            {"boundary", "complete_explicit_unit_range", candidate.rule_id},
            {"context", "released_unit_and_range_relation", candidate.rule_id},
        },
        std::move(edits),
        "explicit_unit_range",
    };
}

RuleApproval approve_explicit_money_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    return approve_explicit_unit_range(
        ExplicitUnitRangeKind::money, input, context, candidate
    );
}

RuleApproval approve_explicit_measure_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    return approve_explicit_unit_range(
        ExplicitUnitRangeKind::measure, input, context, candidate
    );
}

RuleApproval approve_explicit_duration_range(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
) {
    return approve_explicit_unit_range(
        ExplicitUnitRangeKind::duration, input, context, candidate
    );
}

std::vector<TransformationCandidate> detect_isolated_digit_g_identifiers(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 1 < length; ++start) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input, *boundaries, start, start + 1
        ));
        const auto matched_end = spaced_digit_g_end(input, *boundaries, start);
        if (!digit.has_value() || *digit < 2 || !matched_end.has_value()) continue;
        const size_t end = *matched_end;
        const size_t candidate_start = structured_identifier_candidate_start(
            input, *boundaries, start
        );
        if (start > 0) {
            const std::string previous = previous_non_space_scalar(
                input, *boundaries, start
            );
            if (number_continuation(previous) || previous == "幺" || previous == "点"
                || ascii_alnum_scalar(previous)) {
                continue;
            }
        }
        if (end < length) {
            const std::string next = next_non_space_scalar(input, *boundaries, end);
            if (number_continuation(next) || next == "幺" || next == "点"
                || ascii_alnum_scalar(next)) {
                continue;
            }
        }
        if (ambiguous_digit_g_quantity_context(input, *boundaries, start, end)) continue;
        candidates.push_back({
            "zh-isolated-digit-g-" + std::to_string(candidate_start) + "-"
                + std::to_string(end),
            {candidate_start, end,
             unicode_scalar_substring(input, *boundaries, candidate_start, end)},
            "zh-CN", "identifier", "isolated_digit_g",
            kIsolatedDigitGRule, kIsolatedDigitGRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_respirator_standards(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 2 < length; ++start) {
        const auto matched_end = spaced_respirator_end(input, *boundaries, start);
        if (!matched_end.has_value()) continue;
        const size_t end = *matched_end;
        const size_t candidate_start = structured_identifier_candidate_start(
            input, *boundaries, start
        );
        const std::string value = unicode_scalar_substring(
            input, *boundaries, candidate_start, end
        );
        candidates.push_back({
            "zh-respirator-n95-" + std::to_string(candidate_start) + "-"
                + std::to_string(end),
            {candidate_start, end, value},
            "zh-CN", "identifier", "respirator_standard_n95",
            kRespiratorRule, kRespiratorRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_automotive_stores(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 2 < length; ++start) {
        const auto matched_end = spaced_automotive_store_end(
            input, *boundaries, start
        );
        if (!matched_end.has_value()) continue;
        const size_t end = *matched_end;
        const size_t candidate_start = structured_identifier_candidate_start(
            input, *boundaries, start
        );
        const std::string value = unicode_scalar_substring(
            input, *boundaries, candidate_start, end
        );
        candidates.push_back({
            "zh-automotive-4s-store-" + std::to_string(candidate_start) + "-"
                + std::to_string(end),
            {candidate_start, end, value},
            "zh-CN", "identifier", "automotive_4s_store",
            kAutomotiveStoreRule, kAutomotiveStoreRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_playstation_models(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 2 < length; ++start) {
        const auto matched_end = spaced_playstation_end(input, *boundaries, start);
        if (!matched_end.has_value()) continue;
        const size_t end = *matched_end;
        const size_t candidate_start = structured_identifier_candidate_start(
            input, *boundaries, start
        );
        const std::string value = unicode_scalar_substring(
            input, *boundaries, candidate_start, end
        );
        candidates.push_back({
            "zh-playstation-2-" + std::to_string(candidate_start) + "-"
                + std::to_string(end),
            {candidate_start, end, value},
            "zh-CN", "identifier", "playstation_model",
            kPlayStationRule, kPlayStationRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_closed_product_models(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    const std::vector<std::string> forms = {
        "波音七三七八零零客机", "波音七三七八零零飞机",
        "F杠二十二战斗机", "f杠二十二战斗机",
        "F杠二十二隐形战斗机", "f杠二十二隐形战斗机",
        "米二直升机", "米二直升飞机",
    };
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        for (const auto& form : forms) {
            const auto form_boundaries = unicode_scalar_boundaries(form);
            if (!form_boundaries.has_value()) continue;
            const size_t form_length = form_boundaries->size() - 1;
            if (start + form_length > length
                || unicode_scalar_substring(
                    input, *boundaries, start, start + form_length
                ) != form) {
                continue;
            }
            const size_t end = start + form_length;
            if (!inside_protected_delimiters(input, *boundaries, start)) {
                candidates.push_back({
                    "zh-closed-product-model-" + std::to_string(start) + "-"
                        + std::to_string(end),
                    {start, end, form},
                    "zh-CN", "identifier", "closed_product_model",
                    kClosedProductModelRule, kClosedProductModelRuleVersion,
                });
            }
            start = end - 1;
            break;
        }
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_product_models(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (first.size() != 1
            || !std::isupper(static_cast<unsigned char>(first[0]))) {
            continue;
        }
        if (start > 0) {
            const std::string previous = unicode_scalar_substring(
                input, *boundaries, start - 1, start
            );
            if (previous.size() == 1
                && std::isalnum(static_cast<unsigned char>(previous[0]))) {
                continue;
            }
        }
        size_t prefix_end = start;
        while (prefix_end < length && prefix_end - start < 4) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, prefix_end, prefix_end + 1
            );
            if (scalar.size() != 1
                || !std::isupper(static_cast<unsigned char>(scalar[0]))) {
                break;
            }
            ++prefix_end;
        }
        if (prefix_end == start || prefix_end >= length) continue;
        size_t number_start = prefix_end;
        if (unicode_scalar_substring(
                input, *boundaries, number_start, number_start + 1
            ) == " ") {
            ++number_start;
        }
        size_t end = number_start;
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!integer_money_number_scalar(scalar) && scalar != "幺" && scalar != "洞") break;
            ++end;
        }
        if (end == number_start) continue;
        if (end < length) {
            size_t latin_end = end;
            while (latin_end < length) {
                const std::string scalar = unicode_scalar_substring(
                    input, *boundaries, latin_end, latin_end + 1
                );
                if (scalar.size() != 1
                    || !std::isalpha(static_cast<unsigned char>(scalar[0]))) {
                    break;
                }
                ++latin_end;
            }
            const std::string after_latin = right_clause_context(
                input, *boundaries, latin_end, 8
            );
            if (latin_end > end && starts_with_any(
                    after_latin, {"服务器芯片", "加速器"}
                )) {
                end = latin_end;
            } else {
                const std::string suffix = unicode_scalar_substring(
                    input, *boundaries, end, end + 1
                );
                const std::string right = right_clause_context(
                    input, *boundaries, end, 8
                );
                if (!starts_with_any(right, {"Ultra"}) && suffix.size() == 1
                    && std::isupper(static_cast<unsigned char>(suffix[0]))) {
                    ++end;
                }
            }
        }
        const std::string value = unicode_scalar_substring(
            input, *boundaries, start, end
        );
        if (!contextual_product_model_replacement(value).has_value()
            || !contextual_product_model_applies(input, *boundaries, start, end)) {
            continue;
        }
        candidates.push_back({
            "zh-contextual-product-model-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, value},
            "zh-CN", "identifier", "contextual_product_model",
            kContextualProductModelRule, kContextualProductModelRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_land_parcel_identifiers(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (first.size() != 1
            || !std::isupper(static_cast<unsigned char>(first[0]))) {
            continue;
        }
        size_t end = start;
        while (end < length && end - start < 4) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (scalar.size() != 1
                || !std::isupper(static_cast<unsigned char>(scalar[0]))) {
                break;
            }
            ++end;
        }
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_telephone_digit(scalar).has_value() && scalar != "杠") break;
            ++end;
        }
        const std::string value = unicode_scalar_substring(
            input, *boundaries, start, end
        );
        if (!land_parcel_identifier_replacement(value).has_value()
            || !land_parcel_identifier_applies(input, *boundaries, end)) {
            continue;
        }
        candidates.push_back({
            "zh-land-parcel-identifier-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, value},
            "zh-CN", "identifier", "contextual_land_parcel",
            kLandParcelIdentifierRule, kLandParcelIdentifierRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_display_resolutions(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if ((!spoken_cardinal_digit(first).has_value()
                && !spoken_small_unit(first).has_value())
            || !explicit_display_resolution_context(input, *boundaries, start)) {
            continue;
        }
        size_t cursor = start;
        while (cursor < length && integer_money_number_scalar(unicode_scalar_substring(
                input, *boundaries, cursor, cursor + 1
            ))) {
            ++cursor;
        }
        if (cursor == start || cursor >= length
            || unicode_scalar_substring(input, *boundaries, cursor, cursor + 1) != "乘") {
            continue;
        }
        ++cursor;
        const size_t second_start = cursor;
        while (cursor < length && integer_money_number_scalar(unicode_scalar_substring(
                input, *boundaries, cursor, cursor + 1
            ))) {
            ++cursor;
        }
        if (cursor == second_start || !parse_display_resolution(
                input, *boundaries, start, cursor
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-display-resolution-" + std::to_string(start) + "-" + std::to_string(cursor),
            {start, cursor, unicode_scalar_substring(input, *boundaries, start, cursor)},
            "zh-CN", "identifier", "display_resolution",
            kDisplayResolutionRule, kDisplayResolutionRuleVersion,
        });
        start = cursor - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_train_numbers(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string prefix = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!contains_any(prefix, {"G", "D", "C", "Z", "T", "K", "第"})) continue;
        size_t digit_end = start + 1;
        while (digit_end < length && contains_any(unicode_scalar_substring(
                input, *boundaries, digit_end, digit_end + 1
            ), {"零", "〇", "一", "二", "三", "四", "五", "六", "七", "八", "九", "幺"})) {
            ++digit_end;
        }
        const size_t digit_count = digit_end - start - 1;
        if (digit_count < 2 || digit_count > 6) continue;
        size_t end = digit_end;
        for (const std::string suffix : {"次列车", "车次"}) {
            const auto suffix_boundaries = unicode_scalar_boundaries(suffix);
            if (!suffix_boundaries.has_value()) continue;
            const size_t suffix_length = suffix_boundaries->size() - 1;
            if (digit_end + suffix_length <= length
                && unicode_scalar_substring(
                    input, *boundaries, digit_end, digit_end + suffix_length
                ) == suffix) {
                end = digit_end + suffix_length;
                break;
            }
        }
        if (end == digit_end) continue;
        const bool continues_left = start > 0 && ascii_alnum_scalar(
            previous_non_space_scalar(input, *boundaries, start)
        );
        const std::string next = end < length
            ? next_non_space_scalar(input, *boundaries, end) : "";
        if (continues_left || ascii_alnum_scalar(next) || number_continuation(next)
            || inside_protected_delimiters(input, *boundaries, start)) {
            continue;
        }
        candidates.push_back({
            "zh-train-number-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "identifier", "train_number",
            kTrainNumberRule, kTrainNumberRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_flight_numbers(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 4 <= length; ++start) {
        bool valid_prefix = true;
        bool has_letter = false;
        for (size_t index = start; index < start + 2; ++index) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, index, index + 1
            );
            if (!flight_prefix_scalar(scalar).has_value()) {
                valid_prefix = false;
                break;
            }
            has_letter = has_letter || (scalar.size() == 1
                && std::isupper(static_cast<unsigned char>(scalar[0])) != 0);
        }
        if (!valid_prefix || !has_letter) continue;
        size_t end = start + 2;
        while (end < length && contains_any(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ), {"零", "〇", "一", "二", "三", "四", "五", "六", "七", "八", "九", "幺"})) {
            ++end;
        }
        const size_t digit_count = end - start - 2;
        if (digit_count < 2 || digit_count > 6
            || !explicit_flight_number_context(input, *boundaries, start, end)) {
            continue;
        }
        const std::string previous = start > 0
            ? previous_non_space_scalar(input, *boundaries, start) : "";
        const bool continues_left = ascii_alnum_scalar(previous)
            || number_continuation(previous);
        const std::string next = end < length
            ? next_non_space_scalar(input, *boundaries, end) : "";
        if (continues_left || (ascii_alnum_scalar(next) && next != "航")
            || number_continuation(next)
            || inside_protected_delimiters(input, *boundaries, start)) {
            continue;
        }
        candidates.push_back({
            "zh-flight-number-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "identifier", "flight_number",
            kFlightNumberRule, kFlightNumberRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_contextual_ratios(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_decimal_scalar(first) || first == "点"
            || !explicit_ratio_anchor(input, *boundaries, start)) {
            continue;
        }
        size_t connector = start;
        while (connector < length && spoken_decimal_scalar(unicode_scalar_substring(
                input, *boundaries, connector, connector + 1
            ))) {
            ++connector;
        }
        if (connector >= length || unicode_scalar_substring(
                input, *boundaries, connector, connector + 1
            ) != "比") {
            continue;
        }
        size_t end = connector + 1;
        while (end < length && spoken_decimal_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (!parse_contextual_ratio(input, *boundaries, start, end).has_value()) continue;
        candidates.push_back({
            "zh-contextual-ratio-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "ratio", "explicit_contextual_ratio",
            kExplicitContextualRatioRule, kExplicitContextualRatioRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_scalar_ratios(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_decimal_scalar(first) || first == "点"
            || !contextual_scalar_ratio_anchor(input, *boundaries, start)) {
            continue;
        }
        size_t end = start;
        bool saw_decimal_mark = false;
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_decimal_scalar(scalar) && scalar != "万") break;
            saw_decimal_mark = saw_decimal_mark || scalar == "点";
            ++end;
        }
        if (!saw_decimal_mark || !parse_contextual_scalar_value(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-contextual-scalar-ratio-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "ratio", "contextual_scalar",
            kContextualScalarRatioRule, kContextualScalarRatioRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_setting_values(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (!contextual_setting_value_anchor(input, *boundaries, start)) continue;
        size_t end = start;
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_cardinal_digit(scalar).has_value()
                && !spoken_small_unit(scalar).has_value()) {
                break;
            }
            ++end;
        }
        const auto value = parse_spoken_cardinal(input, *boundaries, start, end);
        if (end == start || !value.has_value() || *value < 0 || *value > 100) continue;
        candidates.push_back({
            "zh-contextual-setting-value-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "number", "contextual_setting_value",
            kContextualSettingValueRule, kContextualSettingValueRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_scores(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_decimal_scalar(first) || first == "点"
            || !contextual_score_anchor(input, *boundaries, start)) {
            continue;
        }
        size_t end = start;
        while (end < length && spoken_decimal_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (end >= length
            || unicode_scalar_substring(input, *boundaries, end, end + 1) != "分"
            || !parse_percentage_value(input, *boundaries, start, end).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-contextual-score-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "number", "contextual_score",
            kContextualScoreRule, kContextualScoreRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_ranks(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t ordinal_start = 0; ordinal_start + 3 < length; ++ordinal_start) {
        if (unicode_scalar_substring(
                input, *boundaries, ordinal_start, ordinal_start + 1
            ) != "第" || !contextual_rank_anchor(input, *boundaries, ordinal_start)) {
            continue;
        }
        const size_t start = ordinal_start + 1;
        size_t end = start;
        while (end < length) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_cardinal_digit(value).has_value()
                && !spoken_small_unit(value).has_value()) {
                break;
            }
            ++end;
        }
        const auto rank = parse_spoken_cardinal(input, *boundaries, start, end);
        if (!rank.has_value() || *rank < 10 || *rank > 9999 || end >= length
            || unicode_scalar_substring(input, *boundaries, end, end + 1) != "名") {
            continue;
        }
        candidates.push_back({
            "zh-contextual-rank-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "ordinal", "contextual_rank",
            kContextualRankRule, kContextualRankRuleVersion,
        });
        ordinal_start = end;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_document_pages(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t ordinal_start = 0; ordinal_start + 3 < length; ++ordinal_start) {
        if (unicode_scalar_substring(
                input, *boundaries, ordinal_start, ordinal_start + 1
            ) != "第" || !contextual_document_page_anchor(input, *boundaries, ordinal_start)) {
            continue;
        }
        const size_t start = ordinal_start + 1;
        size_t end = start;
        while (end < length) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_cardinal_digit(value).has_value()
                && !spoken_small_unit(value).has_value()) {
                break;
            }
            ++end;
        }
        const auto page = parse_spoken_cardinal(input, *boundaries, start, end);
        if (!page.has_value() || *page < 10 || *page > 9999 || end >= length
            || unicode_scalar_substring(input, *boundaries, end, end + 1) != "页") {
            continue;
        }
        candidates.push_back({
            "zh-contextual-document-page-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "ordinal", "contextual_document_page",
            kContextualDocumentPageRule, kContextualDocumentPageRuleVersion,
        });
        ordinal_start = end;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_structured_ordinals(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t ordinal_start = 0; ordinal_start + 3 <= length; ++ordinal_start) {
        if (unicode_scalar_substring(
                input, *boundaries, ordinal_start, ordinal_start + 1
            ) != "第") {
            continue;
        }
        const size_t start = ordinal_start + 1;
        size_t end = start;
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_cardinal_digit(scalar).has_value()
                && !spoken_small_unit(scalar).has_value()) {
                break;
            }
            ++end;
        }
        if (end == start || end >= length) continue;
        const std::string unit = unicode_scalar_substring(
            input, *boundaries, end, end + 1
        );
        const auto value = parse_spoken_cardinal(input, *boundaries, start, end);
        if (!value.has_value() || *value < 1 || *value > 9999
            || !contains_any(unit, {"组", "层", "首"})
            || !contextual_structured_ordinal_anchor(
                input, *boundaries, ordinal_start, end, unit
            )) {
            continue;
        }
        candidates.push_back({
            "zh-contextual-structured-ordinal-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "ordinal", "contextual_structured_reference",
            kContextualStructuredOrdinalRule, kContextualStructuredOrdinalRuleVersion,
        });
        ordinal_start = end;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_anchored_cardinals(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!integer_measure_number_scalar(first)) continue;
        if (start > 0 && integer_measure_number_scalar(unicode_scalar_substring(
                input, *boundaries, start - 1, start
            ))) {
            continue;
        }
        size_t end = start + 1;
        while (end < length && integer_measure_number_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (!parse_spoken_large_count_integer(
                input, *boundaries, start, end
            ).has_value()) {
            start = end - 1;
            continue;
        }
        TransformationCandidate candidate = {
            "zh-contextual-anchored-cardinal-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "number", "contextual_anchored_cardinal",
            kContextualAnchoredCardinalRule, kContextualAnchoredCardinalRuleVersion,
        };
        const SafePolicyContext detection_context = {"", "", false, false};
        if (approve_contextual_anchored_cardinal(
                input, detection_context, candidate
            ).decision == RuleDecision::approve) {
            candidates.push_back(std::move(candidate));
        }
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_technical_identifiers(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        bool matched = false;
        for (const std::string& value : {std::string("H. 点二六四"),
                std::string("H.点二六四"), std::string("H点二六四"),
                std::string("WiFi六E")}) {
            const auto value_boundaries = unicode_scalar_boundaries(value);
            if (!value_boundaries.has_value()) continue;
            const size_t value_length = value_boundaries->size() - 1;
            if (start + value_length > length
                || unicode_scalar_substring(
                    input, *boundaries, start, start + value_length
                ) != value) {
                continue;
            }
            const size_t end = start + value_length;
            candidates.push_back({
                "zh-technical-identifier-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, value}, "zh-CN", "identifier",
                "contextual_technical_notation",
                kTechnicalIdentifierRule, kTechnicalIdentifierRuleVersion,
            });
            start = end - 1;
            matched = true;
            break;
        }
        if (matched) continue;
        for (const std::string& prefix : {
                std::string("C加加"), std::string("Python"), std::string("CUDA")}) {
            const auto prefix_boundaries = unicode_scalar_boundaries(prefix);
            if (!prefix_boundaries.has_value()) continue;
            const size_t prefix_length = prefix_boundaries->size() - 1;
            if (start + prefix_length >= length
                || unicode_scalar_substring(
                    input, *boundaries, start, start + prefix_length
                ) != prefix) {
                continue;
            }
            size_t end = start + prefix_length;
            while (end < length) {
                const std::string scalar = unicode_scalar_substring(
                    input, *boundaries, end, end + 1
                );
                if (!version_component_scalar(scalar) && scalar != "点") break;
                ++end;
            }
            if (end == start + prefix_length) continue;
            TransformationCandidate candidate = {
                "zh-technical-identifier-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "contextual_technical_notation",
                kTechnicalIdentifierRule, kTechnicalIdentifierRuleVersion,
            };
            const SafePolicyContext detection_context = {"", "", false, false};
            if (approve_technical_identifier(
                    input, detection_context, candidate
                ).decision == RuleDecision::approve) {
                candidates.push_back(std::move(candidate));
                start = end - 1;
                break;
            }
        }
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_structured_spoken_identifiers(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    std::vector<size_t> starts;
    const auto add_after_anchor = [&](size_t anchor_start, const std::string& anchor,
                                      bool allow_spaces) {
        const auto anchor_boundaries = unicode_scalar_boundaries(anchor);
        if (!anchor_boundaries.has_value()) return;
        const size_t anchor_length = anchor_boundaries->size() - 1;
        if (anchor_start + anchor_length > length
            || unicode_scalar_substring(
                input, *boundaries, anchor_start, anchor_start + anchor_length
            ) != anchor) return;
        size_t start = anchor_start + anchor_length;
        while (start < length && unicode_scalar_substring(
                input, *boundaries, start, start + 1
            ) == " ") ++start;
        if (start >= length || std::find(starts.begin(), starts.end(), start) != starts.end()) {
            return;
        }
        size_t end = start;
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (clause_boundary_scalar(scalar) || (!allow_spaces && scalar == " ")) break;
            ++end;
        }
        while (end > start && unicode_scalar_substring(
                input, *boundaries, end - 1, end
            ) == " ") --end;
        if (end <= start) return;
        const std::string value = unicode_scalar_substring(input, *boundaries, start, end);
        std::optional<std::string> parsed;
        if (value.find("盘反斜杠") != std::string::npos) {
            parsed = parse_spoken_windows_path(value);
        } else if (value.find("冒号双斜杠") != std::string::npos) {
            parsed = parse_spoken_url(value);
        } else {
            parsed = parse_spoken_domain(value);
        }
        if (!parsed.has_value()) return;
        starts.push_back(start);
        candidates.push_back({
            "zh-structured-spoken-locator-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, value}, "zh-CN", "identifier", "structured_spoken_locator",
            kStructuredSpokenIdentifierRule, kStructuredSpokenIdentifierRuleVersion,
        });
    };
    for (size_t index = 0; index < length; ++index) {
        for (const std::string& anchor : {
                std::string("网站域名是"), std::string("网站域名为"),
                std::string("域名是"), std::string("域名为"),
            }) add_after_anchor(index, anchor, false);
        for (const std::string& anchor : {
                std::string("接口地址是"), std::string("接口地址为"),
                std::string("网址是"), std::string("网址为"),
                std::string("链接是"), std::string("链接为"),
            }) add_after_anchor(index, anchor, false);
        for (const std::string& anchor : {
                std::string("日志文件位于"), std::string("文件位于"),
                std::string("路径是"), std::string("路径为"),
            }) add_after_anchor(index, anchor, true);
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_geographic_coordinates(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 3 < length; ++start) {
        const std::string anchor = unicode_scalar_substring(
            input, *boundaries, start, start + 2
        );
        if (anchor != "北纬" && anchor != "南纬" && anchor != "纬度"
            && anchor != "东经" && anchor != "西经" && anchor != "经度") {
            continue;
        }
        size_t end = start + 2;
        while (end < length && spoken_decimal_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (end >= length || unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ) != "度") {
            continue;
        }
        ++end;
        size_t component_end = end;
        while (component_end < length && spoken_decimal_scalar(
                unicode_scalar_substring(
                    input, *boundaries, component_end, component_end + 1
                )
            )) {
            ++component_end;
        }
        if (component_end > end && component_end < length
            && unicode_scalar_substring(
                input, *boundaries, component_end, component_end + 1
            ) == "分") {
            end = component_end + 1;
            component_end = end;
            while (component_end < length && spoken_decimal_scalar(
                    unicode_scalar_substring(
                        input, *boundaries, component_end, component_end + 1
                    )
                )) {
                ++component_end;
            }
            if (component_end > end && component_end < length
                && unicode_scalar_substring(
                    input, *boundaries, component_end, component_end + 1
                ) == "秒") {
                end = component_end + 1;
            }
        }
        if (!parse_explicit_geographic_coordinate(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-geographic-coordinate-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "measure", "explicit_geographic_coordinate",
            kExplicitGeographicCoordinateRule, kExplicitGeographicCoordinateRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_software_versions(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!version_component_scalar(first)) continue;
        if (start > 0) {
            const std::string previous = unicode_scalar_substring(
                input, *boundaries, start - 1, start
            );
            if (version_component_scalar(previous) || previous == "点") continue;
        }
        size_t cursor = start;
        size_t component_count = 0;
        bool complete = true;
        while (component_count < 5) {
            const size_t component_start = cursor;
            while (cursor < length && version_component_scalar(unicode_scalar_substring(
                    input, *boundaries, cursor, cursor + 1
                ))) {
                ++cursor;
            }
            if (cursor == component_start) {
                complete = false;
                break;
            }
            ++component_count;
            if (cursor >= length || unicode_scalar_substring(
                    input, *boundaries, cursor, cursor + 1
                ) != "点") {
                break;
            }
            ++cursor;
        }
        if (!complete || component_count < 2 || cursor > length
            || unicode_scalar_substring(input, *boundaries, cursor - 1, cursor) == "点") {
            continue;
        }
        if (!explicit_software_version_context(
                input, *boundaries, start, cursor
            )) {
            continue;
        }
        candidates.push_back({
            "zh-software-version-" + std::to_string(start) + "-" + std::to_string(cursor),
            {start, cursor, unicode_scalar_substring(input, *boundaries, start, cursor)},
            "zh-CN", "identifier", "software_version",
            kSoftwareVersionRule, kSoftwareVersionRuleVersion,
        });
        start = cursor - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_video_timecodes(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string left = left_clause_context(input, *boundaries, start, 16);
        if (!ends_with_any(left, {
                "视频时间码是", "视频时间码为", "时间码是", "时间码为",
            })) {
            continue;
        }
        size_t end = start;
        while (end < length) {
            const std::string scalar = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!version_component_scalar(scalar)
                && scalar != "小" && scalar != "时" && scalar != "分" && scalar != "秒") {
                break;
            }
            ++end;
            if (scalar == "秒") break;
        }
        if (!parse_explicit_video_timecode(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-video-timecode-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_video_timecode",
            kExplicitTimecodeRule, kExplicitTimecodeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_clinical_thresholds(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string left = left_clause_context(input, *boundaries, start, 20);
        if (!ends_with_any(left, {"空腹血糖必须小于", "空腹血糖应小于"})) continue;
        size_t end = start;
        while (end < length && spoken_decimal_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        if (end == start || !parse_contextual_scalar_value(
                input, *boundaries, start, end
            ).has_value()) {
            continue;
        }
        candidates.push_back({
            "zh-clinical-threshold-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "number", "explicit_clinical_threshold",
            kClinicalThresholdRule, kClinicalThresholdRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_telephone_numbers(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (!spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            )).has_value()) {
            continue;
        }
        size_t end = start + 1;
        while (end < length && spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            )).has_value()) {
            ++end;
        }
        auto parsed_digits = parse_telephone_candidate_digits(
            input, *boundaries, start, end
        );
        if (parsed_digits.has_value() && parsed_digits->rfind("400", 0) == 0
            && end + 1 < length && unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ) == "。") {
            size_t continuation_end = end + 1;
            while (continuation_end < length && contains_any(
                    " \t\n\r",
                    {unicode_scalar_substring(
                        input, *boundaries, continuation_end, continuation_end + 1
                    )}
                )) {
                ++continuation_end;
            }
            while (continuation_end < length
                && spoken_telephone_digit(unicode_scalar_substring(
                    input, *boundaries, continuation_end, continuation_end + 1
                )).has_value()) {
                ++continuation_end;
            }
            const auto recovered = parse_telephone_candidate_digits(
                input, *boundaries, start, continuation_end
            );
            if (recovered.has_value() && recovered->size() == 10) {
                end = continuation_end;
                parsed_digits = recovered;
            }
        }
        bool complete_boundary = true;
        if (start > 0) {
            const std::string previous = unicode_scalar_substring(
                input, *boundaries, start - 1, start
            );
            complete_boundary = !ascii_alnum_scalar(previous)
                && !number_continuation(previous);
        }
        if (complete_boundary && end < length) {
            const std::string next = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            complete_boundary = !ascii_alnum_scalar(next) && !number_continuation(next)
                && next != "点" && !contains_any("年月日号", {next});
        }
        if (end - start >= 3 && parsed_digits.has_value()
            && valid_domestic_telephone_shape(*parsed_digits)
            && complete_boundary
            && explicit_telephone_context(input, *boundaries, start, end)
            && !telephone_list_continues_right(input, *boundaries, end)
            && !inside_protected_delimiters(input, *boundaries, start)) {
            candidates.push_back({
                "zh-telephone-number-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "explicit_telephone_number",
                kTelephoneRule, kTelephoneRuleVersion,
            });
        }
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_public_service_numbers(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (!spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            )).has_value()) {
            continue;
        }
        size_t end = start + 1;
        while (end < length && spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            )).has_value()) {
            ++end;
        }
        const auto digits = parse_telephone_candidate_digits(
            input, *boundaries, start, end
        );
        bool complete_boundary = digits.has_value();
        if (complete_boundary && start > 0) {
            const std::string previous = unicode_scalar_substring(
                input, *boundaries, start - 1, start
            );
            complete_boundary = !ascii_alnum_scalar(previous)
                && !number_continuation(previous);
        }
        if (complete_boundary && end < length) {
            const std::string next = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            complete_boundary = !ascii_alnum_scalar(next) && !number_continuation(next)
                && next != "点"
                && !contains_any("年月日号分次个元名位人辆家台项岁", {next});
        }
        if (complete_boundary && !explicit_telephone_context(
                input, *boundaries, start, end
            ) && contextual_public_service_number(
                input, *boundaries, start, end, *digits
            ) && !inside_protected_delimiters(input, *boundaries, start)) {
            candidates.push_back({
                "zh-public-service-number-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "contextual_public_service_number",
                kPublicServiceNumberRule, kPublicServiceNumberRuleVersion,
            });
        }
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_telephone_number_lists(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (!spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            )).has_value()) {
            continue;
        }
        size_t cursor = start;
        size_t list_end = start;
        bool valid = true;
        while (cursor < length) {
            const size_t number_start = cursor;
            std::string digits;
            while (cursor < length) {
                const auto digit = spoken_telephone_digit(unicode_scalar_substring(
                    input, *boundaries, cursor, cursor + 1
                ));
                if (!digit.has_value()) break;
                digits += static_cast<char>('0' + *digit);
                ++cursor;
            }
            if (cursor == number_start || !valid_domestic_telephone_shape(digits)) {
                valid = false;
                break;
            }
            list_end = cursor;
            size_t connector = cursor;
            while (connector < length && contains_any(" \t", {unicode_scalar_substring(
                    input, *boundaries, connector, connector + 1
                )})) {
                ++connector;
            }
            if (connector >= length || !contains_any("或和及与、,，", {
                    unicode_scalar_substring(input, *boundaries, connector, connector + 1)
                })) {
                break;
            }
            size_t next = connector + 1;
            while (next < length && contains_any(" \t", {unicode_scalar_substring(
                    input, *boundaries, next, next + 1
                )})) {
                ++next;
            }
            if (next >= length || !spoken_telephone_digit(unicode_scalar_substring(
                    input, *boundaries, next, next + 1
                )).has_value()) {
                break;
            }
            cursor = next;
        }
        const auto replacement = valid ? parse_telephone_number_list(
            input, *boundaries, start, list_end
        ) : std::nullopt;
        const bool explicit_context = explicit_telephone_context(
            input, *boundaries, start, list_end
        );
        const bool emergency_context = replacement.has_value()
            && emergency_telephone_list_context(
                input, *boundaries, start, *replacement
            );
        if (replacement.has_value() && (explicit_context || emergency_context)) {
            candidates.push_back({
                "zh-telephone-number-list-" + std::to_string(start) + "-"
                    + std::to_string(list_end),
                {start, list_end,
                 unicode_scalar_substring(input, *boundaries, start, list_end)},
                "zh-CN", "identifier", "explicit_telephone_number_list",
                kTelephoneListRule, kTelephoneListRuleVersion,
            });
            start = list_end - 1;
        }
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_identifiers(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (!spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            )).has_value()) {
            continue;
        }
        size_t end = start;
        while (end < length && spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            )).has_value()) {
            ++end;
        }
        if (parse_spoken_identifier_digits(
                input, *boundaries, start, end, 2, 24
            ).has_value()
            && ((end - start >= 3 && (explicit_identifier_anchor(
                     input, *boundaries, start
                 ) || contextual_technical_model_number_anchor(
                     input, *boundaries, start, end
                ) || street_number_anchor(input, *boundaries, start, end)
                    || facility_number_anchor(input, *boundaries, start, end)))
                || room_identifier_anchor(input, *boundaries, start, end))
            && !inside_protected_delimiters(input, *boundaries, start)) {
            candidates.push_back({
                "zh-explicit-identifier-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "explicit_digit_sequence",
                kExplicitIdentifierRule, kExplicitIdentifierRuleVersion,
            });
        }
        start = end - 1;
    }
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_cardinal_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t end = start;
        while (end < length && integer_money_number_scalar(unicode_scalar_substring(
                input, *boundaries, end, end + 1
            ))) {
            ++end;
        }
        const auto integer = parse_spoken_money_integer(input, *boundaries, start, end);
        if (integer.has_value() && *integer >= 10 && *integer <= 99999
            && (street_number_anchor(input, *boundaries, start, end)
                || facility_number_anchor(input, *boundaries, start, end))
            && !inside_protected_delimiters(input, *boundaries, start)) {
            candidates.push_back({
                "zh-explicit-identifier-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "explicit_digit_sequence",
                kExplicitIdentifierRule, kExplicitIdentifierRuleVersion,
            });
        }
        if (end > start) start = end - 1;
    }
    return candidates;
}

bool ipv4_number_scalar(const std::string& value) {
    return spoken_telephone_digit(value).has_value()
        || spoken_small_unit(value).has_value() || value == "点";
}

std::vector<TransformationCandidate> detect_ipv4_addresses(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        if (!spoken_telephone_digit(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            )).has_value()) {
            continue;
        }
        size_t end = start;
        while (end < length && end - start <= 24 && ipv4_number_scalar(
                unicode_scalar_substring(input, *boundaries, end, end + 1)
            )) {
            ++end;
        }
        const auto replacement = parse_ipv4_address(
            input, *boundaries, start, end
        );
        if (replacement.has_value() && explicit_ipv4_anchor(input, *boundaries, start)
            && !inside_protected_delimiters(input, *boundaries, start)) {
            candidates.push_back({
                "zh-ipv4-address-" + std::to_string(start) + "-" + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "ipv4_address",
                kIpv4AddressRule, kIpv4AddressRuleVersion,
            });
        }
        if (end > start) start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_network_ports(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        const std::string first = unicode_scalar_substring(
            input, *boundaries, start, start + 1
        );
        if (!spoken_telephone_digit(first).has_value()
            && !spoken_small_unit(first).has_value()) {
            continue;
        }
        size_t end = start;
        while (end < length && end - start <= 8) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, end, end + 1
            );
            if (!spoken_telephone_digit(value).has_value()
                && !spoken_small_unit(value).has_value()) {
                break;
            }
            ++end;
        }
        if (parse_network_port(input, *boundaries, start, end).has_value()
            && explicit_network_port_anchor(input, *boundaries, start)
            && !inside_protected_delimiters(input, *boundaries, start)) {
            candidates.push_back({
                "zh-network-port-" + std::to_string(start) + "-" + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "identifier", "network_port",
                kNetworkPortRule, kNetworkPortRuleVersion,
            });
        }
        if (end > start) start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_centuries(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 3 <= length; ++start) {
        size_t unit = start;
        while (unit < length && unit - start <= 4) {
            if (unit + 2 <= length && unicode_scalar_substring(
                    input, *boundaries, unit, unit + 2
                ) == "世纪") {
                break;
            }
            const std::string value = unicode_scalar_substring(
                input, *boundaries, unit, unit + 1
            );
            if (!spoken_cardinal_digit(value).has_value()
                && !spoken_small_unit(value).has_value()) {
                break;
            }
            ++unit;
        }
        if (unit <= start || unit + 2 > length || unicode_scalar_substring(
                input, *boundaries, unit, unit + 2
            ) != "世纪") {
            continue;
        }
        const auto value = parse_spoken_cardinal(input, *boundaries, start, unit);
        if (!value.has_value() || *value < 10 || *value > 30) continue;
        const size_t end = unit + 2;
        candidates.push_back({
            "zh-explicit-century-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_century",
            kCenturyRule, kCenturyRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_clock_time_ranges(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start < length; ++start) {
        for (size_t end = start + 5; end <= std::min(length, start + 28); ++end) {
            const auto replacement = parse_explicit_clock_time_range(
                input, *boundaries, start, end
            );
            if (!replacement.has_value()) continue;
            const bool explicit_period = clock_period_at(
                input, *boundaries, start
            ).has_value();
            if (!explicit_period && !contextual_clock_anchor_applies(
                    input, *boundaries, start, end
                )) {
                continue;
            }
            candidates.push_back({
                "zh-clock-time-range-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "date_time", "explicit_clock_time_range",
                kClockTimeRangeRule, kClockTimeRangeRuleVersion,
            });
            start = end - 1;
            break;
        }
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_clock_times(const std::string& input) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 3 < length; ++start) {
        if (!clock_period_at(input, *boundaries, start).has_value()) continue;
        size_t point = start + 2;
        while (point < length && point - (start + 2) <= 3
            && !contains_any("点时", {unicode_scalar_substring(
                input, *boundaries, point, point + 1
            )})) {
            if (!clock_number_scalar(unicode_scalar_substring(
                    input, *boundaries, point, point + 1
                ))) {
                break;
            }
            ++point;
        }
        if (point == start + 2 || point >= length
            || !contains_any("点时", {unicode_scalar_substring(
                input, *boundaries, point, point + 1
            )})) {
            continue;
        }
        size_t end = point + 1;
        if (unicode_scalar_substring(input, *boundaries, point, point + 1) == "点"
            && end < length) {
            const std::string one = unicode_scalar_substring(input, *boundaries, end, end + 1);
            const std::string two = unicode_scalar_substring(
                input, *boundaries, end, std::min(length, end + 2)
            );
            if (one == "整" || one == "半") {
                ++end;
            } else if (two == "一刻" || two == "三刻") {
                end += 2;
            } else {
                size_t minute_end = end;
                while (minute_end < length && minute_end - end <= 3
                    && clock_number_scalar(unicode_scalar_substring(
                        input, *boundaries, minute_end, minute_end + 1
                    ))) {
                    ++minute_end;
                }
                if (minute_end > end && minute_end < length
                    && unicode_scalar_substring(
                        input, *boundaries, minute_end, minute_end + 1
                    ) == "分") {
                    end = minute_end + 1;
                    size_t second_end = end;
                    while (second_end < length && second_end - end <= 3
                        && clock_number_scalar(unicode_scalar_substring(
                            input, *boundaries, second_end, second_end + 1
                        ))) {
                        ++second_end;
                    }
                    if (second_end > end && second_end < length
                        && unicode_scalar_substring(
                            input, *boundaries, second_end, second_end + 1
                        ) == "秒") {
                        end = second_end + 1;
                    }
                }
            }
        }
        const std::string approximation = unicode_scalar_substring(
            input, *boundaries, end, std::min(length, end + 2)
        );
        const bool supported_approximation = approximation == "左右"
            || starts_with_any(approximation, {"许"});
        if (!parse_explicit_clock_time(input, *boundaries, start, end).has_value()
            || (clock_boundary_is_uncertain(input, *boundaries, start, end)
                && !supported_approximation)
            || inside_protected_delimiters(input, *boundaries, start)) {
            continue;
        }
        candidates.push_back({
            "zh-clock-time-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_clock_time",
            kClockTimeRule, kClockTimeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_clock_times(
    const std::string& input
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t start = 0; start + 3 < length; ++start) {
        if (!clock_number_scalar(unicode_scalar_substring(
                input, *boundaries, start, start + 1
            ))) {
            continue;
        }
        size_t point = start;
        while (point < length && point - start <= 3
            && unicode_scalar_substring(input, *boundaries, point, point + 1) != "点") {
            if (!clock_number_scalar(unicode_scalar_substring(
                    input, *boundaries, point, point + 1
                ))) {
                break;
            }
            ++point;
        }
        if (point == start || point >= length
            || unicode_scalar_substring(input, *boundaries, point, point + 1) != "点") {
            continue;
        }
        size_t minute_end = point + 1;
        while (minute_end < length && minute_end - (point + 1) <= 3
            && clock_number_scalar(unicode_scalar_substring(
                input, *boundaries, minute_end, minute_end + 1
            ))) {
            ++minute_end;
        }
        size_t end = minute_end;
        const std::string clock_suffix = unicode_scalar_substring(
            input, *boundaries, point + 1, std::min(length, point + 3)
        );
        if (starts_with_any(clock_suffix, {"一刻", "三刻"})) {
            end = point + 3;
        } else if (starts_with_any(clock_suffix, {"整", "半"})) {
            end = point + 2;
        } else {
            if (minute_end == point + 1 || minute_end >= length
                || unicode_scalar_substring(
                    input, *boundaries, minute_end, minute_end + 1
                ) != "分") {
                continue;
            }
            end = minute_end + 1;
        }
        size_t second_end = end;
        while (second_end < length && second_end - end <= 3
            && clock_number_scalar(unicode_scalar_substring(
                input, *boundaries, second_end, second_end + 1
            ))) {
            ++second_end;
        }
        if (second_end > end && second_end < length
            && unicode_scalar_substring(
                input, *boundaries, second_end, second_end + 1
            ) == "秒") {
            end = second_end + 1;
        }
        if (!parse_unperioded_exact_clock_time(
                input, *boundaries, start, end
            ).has_value() || inside_protected_delimiters(input, *boundaries, start)) {
            continue;
        }
        const bool approximate = contextual_approximate_clock_anchor_applies(
            input, *boundaries, start, end
        );
        if (!approximate && !contextual_clock_anchor_applies(
                input, *boundaries, start, end
            )) {
            continue;
        }
        candidates.push_back({
            std::string(approximate ? "zh-contextual-approximate-clock-time-"
                : "zh-contextual-clock-time-")
                + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", approximate
                ? "contextual_approximate_unperioded_clock_time"
                : "contextual_unperioded_clock_time",
            approximate ? kContextualApproximateClockTimeRule : kContextualClockTimeRule,
            approximate ? kContextualApproximateClockTimeRuleVersion
                : kContextualClockTimeRuleVersion,
        });
        start = end - 1;
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_calendar_timestamps(
    const std::string& input,
    const std::vector<TransformationCandidate>& dates,
    const std::vector<TransformationCandidate>& clock_times
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (const auto& date : dates) {
        for (const auto& clock : clock_times) {
            if (date.semantic_source.end != clock.semantic_source.start) continue;
            const size_t start = date.semantic_source.start;
            const size_t end = clock.semantic_source.end;
            candidates.push_back({
                "zh-calendar-timestamp-" + std::to_string(start) + "-"
                    + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN", "date_time", "explicit_calendar_timestamp",
                kCalendarTimestampRule, kCalendarTimestampRuleVersion,
            });
        }
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_calendar_timestamp_ranges(
    const std::string& input,
    const std::vector<TransformationCandidate>& timestamps
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    std::vector<TransformationCandidate> candidates;
    for (size_t index = 0; index + 1 < timestamps.size(); ++index) {
        const auto& first = timestamps[index].semantic_source;
        const auto& second = timestamps[index + 1].semantic_source;
        if (first.end + 1 != second.start) continue;
        const std::string connector = unicode_scalar_substring(
            input, *boundaries, first.end, second.start
        );
        if (connector != "到" && connector != "至") continue;
        candidates.push_back({
            "zh-calendar-timestamp-range-" + std::to_string(first.start) + "-"
                + std::to_string(second.end),
            {
                first.start,
                second.end,
                unicode_scalar_substring(input, *boundaries, first.start, second.end),
            },
            "zh-CN", "date_time", "explicit_calendar_timestamp_range",
            kCalendarTimestampRangeRule, kCalendarTimestampRangeRuleVersion,
        });
    }
    const size_t length = boundaries->size() - 1;
    for (size_t connector = 1; connector + 1 < length; ++connector) {
        const std::string connector_text = unicode_scalar_substring(
            input, *boundaries, connector, connector + 1
        );
        if (connector_text != "到" && connector_text != "至") continue;
        if (std::any_of(candidates.begin(), candidates.end(), [&](const auto& item) {
                return item.semantic_source.start < connector
                    && item.semantic_source.end > connector;
            })) {
            continue;
        }
        std::optional<size_t> left_start;
        const size_t minimum_start = connector > 32 ? connector - 32 : 0;
        for (size_t start = connector; start-- > minimum_start;) {
            if (start + 5 >= connector) continue;
            bool four_digits = true;
            for (size_t offset = 0; offset < 4; ++offset) {
                if (!spoken_year_digit(unicode_scalar_substring(
                        input, *boundaries, start + offset, start + offset + 1
                    )).has_value()) {
                    four_digits = false;
                    break;
                }
            }
            if (!four_digits || unicode_scalar_substring(
                    input, *boundaries, start + 4, start + 5
                ) != "年") {
                continue;
            }
            const std::string text = unicode_scalar_substring(
                input, *boundaries, start, connector
            );
            if (text.find("月") != std::string::npos
                && (text.find("日") != std::string::npos || text.find("号") != std::string::npos)
                && text.find("点") != std::string::npos
                && ends_with_any(text, {"分", "点"})) {
                left_start = start;
                break;
            }
        }
        if (!left_start.has_value()) continue;
        std::optional<size_t> right_end;
        const size_t maximum_end = std::min(length, connector + 33);
        for (size_t end = connector + 2; end <= maximum_end; ++end) {
            const std::string text = unicode_scalar_substring(
                input, *boundaries, connector + 1, end
            );
            if (end > connector + 6 && text.find("年") != std::string::npos
                && text.find("月") != std::string::npos
                && (text.find("日") != std::string::npos || text.find("号") != std::string::npos)
                && text.find("点") != std::string::npos
                && ends_with_any(text, {"分", "点"})) {
                right_end = end;
                if (ends_with_any(text, {"分"})) break;
            }
        }
        if (!right_end.has_value()) continue;
        candidates.push_back({
            "zh-calendar-timestamp-range-" + std::to_string(*left_start) + "-"
                + std::to_string(*right_end),
            {
                *left_start,
                *right_end,
                unicode_scalar_substring(input, *boundaries, *left_start, *right_end),
            },
            "zh-CN", "date_time", "explicit_calendar_timestamp_range",
            kCalendarTimestampRangeRule, kCalendarTimestampRangeRuleVersion,
        });
    }
    const SafePolicyContext component_context = {"", "", false, false};
    for (size_t connector = 1; connector + 1 < length; ++connector) {
        const std::string connector_text = unicode_scalar_substring(
            input, *boundaries, connector, connector + 1
        );
        if (connector_text != "到" && connector_text != "至") continue;
        if (std::any_of(candidates.begin(), candidates.end(), [&](const auto& item) {
                return item.semantic_source.start < connector
                    && item.semantic_source.end > connector;
            })) {
            continue;
        }
        const size_t minimum_start = connector > 16 ? connector - 16 : 0;
        const size_t maximum_end = std::min(length, connector + 17);
        bool found = false;
        for (size_t start = minimum_start; start < connector && !found; ++start) {
            for (size_t end = connector + 2; end <= maximum_end; ++end) {
                const std::string value = unicode_scalar_substring(
                    input, *boundaries, start, end
                );
                const TransformationCandidate candidate = {
                    "zh-calendar-timestamp-range-" + std::to_string(start) + "-"
                        + std::to_string(end),
                    {start, end, value}, "zh-CN", "date_time",
                    "explicit_calendar_timestamp_range",
                    kCalendarTimestampRangeRule, kCalendarTimestampRangeRuleVersion,
                };
                if (approve_explicit_calendar_timestamp_range(
                        input, component_context, candidate
                    ).decision != RuleDecision::approve) {
                    continue;
                }
                candidates.push_back(candidate);
                found = true;
                break;
            }
        }
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_approximate_calendar_timestamps(
    const std::string& input,
    const std::vector<TransformationCandidate>& timestamps
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (const auto& timestamp : timestamps) {
        if (timestamp.rule_id != kCalendarTimestampRule) continue;
        size_t end = timestamp.semantic_source.end;
        if (end + 2 <= length
            && unicode_scalar_substring(input, *boundaries, end, end + 2) == "左右") {
            end += 2;
        } else if (end < length
            && unicode_scalar_substring(input, *boundaries, end, end + 1) == "许") {
            ++end;
        } else {
            continue;
        }
        const size_t start = timestamp.semantic_source.start;
        candidates.push_back({
            "zh-approximate-calendar-timestamp-" + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", "explicit_approximate_calendar_timestamp",
            kApproximateCalendarTimestampRule,
            kApproximateCalendarTimestampRuleVersion,
        });
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_contextual_month_day_timestamps(
    const std::string& input,
    const std::vector<TransformationCandidate>& dates
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (const auto& date : dates) {
        const bool full_date = date.subtype == "explicit_full_date";
        if (!full_date && date.subtype != "explicit_month_day") continue;
        const size_t start = date.semantic_source.start;
        const size_t clock_start = date.semantic_source.end;
        const std::string left = context_window(input, *boundaries, 0, start, 8);
        const bool explicit_anchor = full_date
            ? ends_with_any(left, {"时间", "截至", "从", "持续到"})
            : ends_with_any(left, {"时间", "截至", "从", "持续到"});
        if (!explicit_anchor || clock_start + 1 >= length) continue;
        size_t marker = clock_start;
        while (marker < length && marker - clock_start <= 3) {
            const std::string value = unicode_scalar_substring(
                input, *boundaries, marker, marker + 1
            );
            if (value == "点" || value == "时") break;
            if (!clock_number_scalar(unicode_scalar_substring(
                    input, *boundaries, marker, marker + 1
                ))) {
                break;
            }
            ++marker;
        }
        if (marker == clock_start || marker >= length) {
            continue;
        }
        const std::string marker_text = unicode_scalar_substring(
            input, *boundaries, marker, marker + 1
        );
        size_t end = marker + 1;
        if (marker_text == "点") {
            size_t minute_end = marker + 1;
            while (minute_end < length && minute_end - (marker + 1) <= 3
                && clock_number_scalar(unicode_scalar_substring(
                    input, *boundaries, minute_end, minute_end + 1
                ))) {
                ++minute_end;
            }
            if (minute_end > marker + 1 && minute_end < length
                && unicode_scalar_substring(
                    input, *boundaries, minute_end, minute_end + 1
                ) == "分") {
                end = minute_end + 1;
                size_t second_end = end;
                while (second_end < length && second_end - end <= 3
                    && clock_number_scalar(unicode_scalar_substring(
                        input, *boundaries, second_end, second_end + 1
                    ))) {
                    ++second_end;
                }
                if (second_end > end && second_end < length
                    && unicode_scalar_substring(
                        input, *boundaries, second_end, second_end + 1
                    ) == "秒") {
                    end = second_end + 1;
                }
            }
        }
        const std::string right = unicode_scalar_substring(
            input, *boundaries, end, std::min(length, end + 2)
        );
        size_t approximation_length = 0;
        if (right == "左右" || right == "前后") {
            approximation_length = 2;
        } else if (end < length
            && unicode_scalar_substring(input, *boundaries, end, end + 1) == "许") {
            approximation_length = 1;
        }
        if (approximation_length > 0) {
            const size_t approximate_end = end + approximation_length;
            if (parse_unperioded_exact_clock_time(
                    input, *boundaries, clock_start, end
                ).has_value()
                && !clock_boundary_is_uncertain(
                    input, *boundaries, 0, approximate_end
                )
                && !inside_protected_delimiters(input, *boundaries, start)) {
                candidates.push_back({
                    "zh-contextual-approximate-month-day-timestamp-"
                        + std::to_string(start) + "-" + std::to_string(approximate_end),
                    {
                        start,
                        approximate_end,
                        unicode_scalar_substring(
                            input, *boundaries, start, approximate_end
                        ),
                    },
                    "zh-CN", "date_time",
                    "explicit_contextual_approximate_month_day_timestamp",
                    kContextualApproximateMonthDayTimestampRule,
                    kContextualApproximateMonthDayTimestampRuleVersion,
                });
            }
            continue;
        }
        if (!parse_unperioded_exact_clock_time(
                input, *boundaries, clock_start, end
            ).has_value()
            || clock_boundary_is_uncertain(input, *boundaries, 0, end)
            || inside_protected_delimiters(input, *boundaries, start)) {
            continue;
        }
        candidates.push_back({
            std::string(full_date ? "zh-contextual-full-date-timestamp-"
                : "zh-contextual-month-day-timestamp-") + std::to_string(start) + "-"
                + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN", "date_time", full_date
                ? "explicit_contextual_full_date_timestamp"
                : "explicit_contextual_month_day_timestamp",
            full_date ? kContextualFullDateTimestampRule : kContextualMonthDayTimestampRule,
            full_date ? kContextualFullDateTimestampRuleVersion
                : kContextualMonthDayTimestampRuleVersion,
        });
    }
    return candidates;
}

std::vector<TransformationCandidate> detect_explicit_years(
    const std::string& input,
    const std::vector<TransformationCandidate>& exclusions
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return {};
    const size_t length = boundaries->size() - 1;
    std::vector<TransformationCandidate> candidates;
    for (size_t index = 0; index < length;) {
        const auto digit = spoken_year_digit(unicode_scalar_substring(
            input,
            *boundaries,
            index,
            index + 1
        ));
        if (!digit.has_value()) {
            ++index;
            continue;
        }
        size_t end = index;
        while (end < length && spoken_year_digit(unicode_scalar_substring(
                input,
                *boundaries,
                end,
                end + 1
            )).has_value()) {
            ++end;
        }
        if (end - index == 4 && end < length
            && unicode_scalar_substring(input, *boundaries, end, end + 1) == "年"
            && !inside_protected_delimiters(input, *boundaries, index)
            && !std::any_of(exclusions.begin(), exclusions.end(), [&](const auto& item) {
                return overlaps(index, end + 1, item.semantic_source);
            })) {
            candidates.push_back({
                "zh-year-" + std::to_string(index) + "-" + std::to_string(end + 1),
                {
                    index,
                    end + 1,
                    unicode_scalar_substring(input, *boundaries, index, end + 1),
                },
                "zh-CN",
                "date_time",
                "explicit_four_digit_year",
                kCalendarYearRule,
                kCalendarYearRuleVersion,
            });
        }
        index = end;
    }
    return candidates;
}

const std::vector<ReleasedRule>& released_rules() {
    static const std::vector<ReleasedRule> rules = {
        {
            kCalendarYearRule, kCalendarYearRuleVersion, "zh-CN",
            "date_time", "explicit_four_digit_year", approve_explicit_year,
        },
        {
            kYearRangeRule, kYearRangeRuleVersion, "zh-CN",
            "date_time", "explicit_year_range", approve_year_range,
        },
        {
            kFullDateRangeRule, kFullDateRangeRuleVersion, "zh-CN",
            "date_time", "explicit_full_date_range", approve_full_date_range,
        },
        {
            kSharedYearDateRangeRule, kSharedYearDateRangeRuleVersion, "zh-CN",
            "date_time", "explicit_shared_year_date_range",
            approve_shared_year_date_range,
        },
        {
            kSharedMonthDateRangeRule, kSharedMonthDateRangeRuleVersion, "zh-CN",
            "date_time", "explicit_shared_month_date_range",
            approve_shared_month_date_range,
        },
        {
            kFullDateSharedMonthRangeRule, kFullDateSharedMonthRangeRuleVersion, "zh-CN",
            "date_time", "explicit_full_date_shared_month_range",
            approve_full_date_shared_month_range,
        },
        {
            kYearMonthRule, kYearMonthRuleVersion, "zh-CN",
            "date_time", "explicit_year_month", approve_year_month,
        },
        {
            kCrossSegmentMonthRule, kCrossSegmentMonthRuleVersion, "zh-CN",
            "date_time", "contextual_cross_segment_month", approve_cross_segment_month,
        },
        {
            kYearMonthRangeRule, kYearMonthRangeRuleVersion, "zh-CN",
            "date_time", "explicit_year_month_range", approve_year_month_range,
        },
        {
            kFullDateRule, kFullDateRuleVersion, "zh-CN",
            "date_time", "explicit_full_date", approve_full_date,
        },
        {
            kMonthDayRule, kMonthDayRuleVersion, "zh-CN",
            "date_time", "explicit_month_day", approve_month_day,
        },
        {
            kMonthPeriodRule, kMonthPeriodRuleVersion, "zh-CN",
            "date_time", "explicit_month_period", approve_month_period,
        },
        {
            kMonthRangeRule, kMonthRangeRuleVersion, "zh-CN",
            "date_time", "explicit_month_range", approve_month_range,
        },
        {
            kRelativeYearMonthRule, kRelativeYearMonthRuleVersion, "zh-CN",
            "date_time", "explicit_relative_year_month", approve_relative_year_month,
        },
        {
            kContextualFollowupMonthRule, kContextualFollowupMonthRuleVersion, "zh-CN",
            "date_time", "contextual_followup_month", approve_contextual_followup_month,
        },
        {
            kRecurringCrossYearMonthRangeRule,
            kRecurringCrossYearMonthRangeRuleVersion,
            "zh-CN", "date_time", "explicit_recurring_cross_year_month_range",
            approve_recurring_cross_year_month_range,
        },
        {
            kPercentageRule, kPercentageRuleVersion, "zh-CN",
            "percentage", "explicit_percentage", approve_percentage,
        },
        {
            kPercentageRangeRule, kPercentageRangeRuleVersion, "zh-CN",
            "percentage", "explicit_percentage_range", approve_percentage_range,
        },
        {
            kSharedPrefixPercentageRangeRule, kSharedPrefixPercentageRangeRuleVersion, "zh-CN",
            "percentage", "shared_prefix_percentage_range",
            approve_shared_prefix_percentage_range,
        },
        {
            kIsolatedDigitGRule, kIsolatedDigitGRuleVersion, "zh-CN",
            "identifier", "isolated_digit_g", approve_isolated_digit_g,
        },
        {
            kRespiratorRule, kRespiratorRuleVersion, "zh-CN",
            "identifier", "respirator_standard_n95", approve_respirator_standard,
        },
        {
            kAutomotiveStoreRule, kAutomotiveStoreRuleVersion, "zh-CN",
            "identifier", "automotive_4s_store", approve_automotive_store,
        },
        {
            kPlayStationRule, kPlayStationRuleVersion, "zh-CN",
            "identifier", "playstation_model", approve_playstation_model,
        },
        {
            kClosedProductModelRule, kClosedProductModelRuleVersion, "zh-CN",
            "identifier", "closed_product_model", approve_closed_product_model,
        },
        {
            kContextualProductModelRule, kContextualProductModelRuleVersion, "zh-CN",
            "identifier", "contextual_product_model", approve_contextual_product_model,
        },
        {
            kLandParcelIdentifierRule, kLandParcelIdentifierRuleVersion, "zh-CN",
            "identifier", "contextual_land_parcel", approve_land_parcel_identifier,
        },
        {
            kDisplayResolutionRule, kDisplayResolutionRuleVersion, "zh-CN",
            "identifier", "display_resolution", approve_display_resolution,
        },
        {
            kExplicitContextualRatioRule, kExplicitContextualRatioRuleVersion, "zh-CN",
            "ratio", "explicit_contextual_ratio", approve_explicit_contextual_ratio,
        },
        {
            kContextualScalarRatioRule, kContextualScalarRatioRuleVersion, "zh-CN",
            "ratio", "contextual_scalar", approve_contextual_scalar_ratio,
        },
        {
            kContextualSettingValueRule, kContextualSettingValueRuleVersion, "zh-CN",
            "number", "contextual_setting_value", approve_contextual_setting_value,
        },
        {
            kContextualScoreRule, kContextualScoreRuleVersion, "zh-CN",
            "number", "contextual_score", approve_contextual_score,
        },
        {
            kContextualRankRule, kContextualRankRuleVersion, "zh-CN",
            "ordinal", "contextual_rank", approve_contextual_rank,
        },
        {
            kContextualDocumentPageRule, kContextualDocumentPageRuleVersion, "zh-CN",
            "ordinal", "contextual_document_page", approve_contextual_document_page,
        },
        {
            kContextualStructuredOrdinalRule, kContextualStructuredOrdinalRuleVersion, "zh-CN",
            "ordinal", "contextual_structured_reference",
            approve_contextual_structured_ordinal,
        },
        {
            kContextualAnchoredCardinalRule, kContextualAnchoredCardinalRuleVersion, "zh-CN",
            "number", "contextual_anchored_cardinal",
            approve_contextual_anchored_cardinal,
        },
        {
            kTechnicalIdentifierRule, kTechnicalIdentifierRuleVersion, "zh-CN",
            "identifier", "contextual_technical_notation",
            approve_technical_identifier,
        },
        {
            kStructuredSpokenIdentifierRule, kStructuredSpokenIdentifierRuleVersion, "zh-CN",
            "identifier", "structured_spoken_locator",
            approve_structured_spoken_identifier,
        },
        {
            kExplicitGeographicCoordinateRule, kExplicitGeographicCoordinateRuleVersion, "zh-CN",
            "measure", "explicit_geographic_coordinate",
            approve_explicit_geographic_coordinate,
        },
        {
            kSoftwareVersionRule, kSoftwareVersionRuleVersion, "zh-CN",
            "identifier", "software_version", approve_software_version,
        },
        {
            kTelephoneRule, kTelephoneRuleVersion, "zh-CN",
            "identifier", "explicit_telephone_number", approve_telephone_number,
        },
        {
            kTelephoneListRule, kTelephoneListRuleVersion, "zh-CN",
            "identifier", "explicit_telephone_number_list",
            approve_telephone_number_list,
        },
        {
            kPublicServiceNumberRule, kPublicServiceNumberRuleVersion, "zh-CN",
            "identifier", "contextual_public_service_number",
            approve_contextual_public_service_number,
        },
        {
            kExplicitIdentifierRule, kExplicitIdentifierRuleVersion, "zh-CN",
            "identifier", "explicit_digit_sequence", approve_explicit_identifier,
        },
        {
            kTrainNumberRule, kTrainNumberRuleVersion, "zh-CN",
            "identifier", "train_number", approve_train_number,
        },
        {
            kFlightNumberRule, kFlightNumberRuleVersion, "zh-CN",
            "identifier", "flight_number", approve_flight_number,
        },
        {
            kIpv4AddressRule, kIpv4AddressRuleVersion, "zh-CN",
            "identifier", "ipv4_address", approve_ipv4_address,
        },
        {
            kNetworkPortRule, kNetworkPortRuleVersion, "zh-CN",
            "identifier", "network_port", approve_network_port,
        },
        {
            kCenturyRule, kCenturyRuleVersion, "zh-CN",
            "date_time", "explicit_century", approve_explicit_century,
        },
        {
            kClockTimeRule, kClockTimeRuleVersion, "zh-CN",
            "date_time", "explicit_clock_time", approve_explicit_clock_time,
        },
        {
            kClockTimeRangeRule, kClockTimeRangeRuleVersion, "zh-CN",
            "date_time", "explicit_clock_time_range", approve_explicit_clock_time_range,
        },
        {
            kContextualClockTimeRule, kContextualClockTimeRuleVersion, "zh-CN",
            "date_time", "contextual_unperioded_clock_time",
            approve_contextual_clock_time,
        },
        {
            kContextualApproximateClockTimeRule,
            kContextualApproximateClockTimeRuleVersion,
            "zh-CN", "date_time", "contextual_approximate_unperioded_clock_time",
            approve_contextual_approximate_clock_time,
        },
        {
            kExplicitTimecodeRule, kExplicitTimecodeRuleVersion, "zh-CN",
            "date_time", "explicit_video_timecode", approve_explicit_video_timecode,
        },
        {
            kClinicalThresholdRule, kClinicalThresholdRuleVersion, "zh-CN",
            "number", "explicit_clinical_threshold", approve_clinical_threshold,
        },
        {
            kExplicitSignedNumberRule, kExplicitSignedNumberRuleVersion, "zh-CN",
            "number", "explicit_signed_number", approve_explicit_signed_number,
        },
        {
            kCalendarTimestampRule, kCalendarTimestampRuleVersion, "zh-CN",
            "date_time", "explicit_calendar_timestamp",
            approve_explicit_calendar_timestamp,
        },
        {
            kCalendarTimestampRangeRule, kCalendarTimestampRangeRuleVersion, "zh-CN",
            "date_time", "explicit_calendar_timestamp_range",
            approve_explicit_calendar_timestamp_range,
        },
        {
            kApproximateCalendarTimestampRule,
            kApproximateCalendarTimestampRuleVersion,
            "zh-CN", "date_time", "explicit_approximate_calendar_timestamp",
            approve_explicit_approximate_calendar_timestamp,
        },
        {
            kContextualMonthDayTimestampRule,
            kContextualMonthDayTimestampRuleVersion,
            "zh-CN", "date_time", "explicit_contextual_month_day_timestamp",
            approve_contextual_month_day_timestamp,
        },
        {
            kContextualFullDateTimestampRule,
            kContextualFullDateTimestampRuleVersion,
            "zh-CN", "date_time", "explicit_contextual_full_date_timestamp",
            approve_contextual_full_date_timestamp,
        },
        {
            kContextualApproximateMonthDayTimestampRule,
            kContextualApproximateMonthDayTimestampRuleVersion,
            "zh-CN", "date_time",
            "explicit_contextual_approximate_month_day_timestamp",
            approve_contextual_approximate_month_day_timestamp,
        },
        {
            kDecimalDurationRule, kDecimalDurationRuleVersion, "zh-CN",
            "date_time", "explicit_decimal_duration",
            approve_explicit_decimal_duration,
        },
        {
            kDecimalDurationApproximateRule, kDecimalDurationApproximateRuleVersion, "zh-CN",
            "date_time", "explicit_decimal_duration_approximation",
            approve_explicit_decimal_duration,
        },
        {
            kDecimalDurationThresholdRule, kDecimalDurationThresholdRuleVersion, "zh-CN",
            "date_time", "explicit_decimal_duration_threshold",
            approve_explicit_decimal_duration,
        },
        {
            kDecimalMeasureRule, kDecimalMeasureRuleVersion, "zh-CN",
            "measure", "explicit_decimal_measure",
            approve_explicit_decimal_measure,
        },
        {
            kDecimalMeasureApproximateRule, kDecimalMeasureApproximateRuleVersion, "zh-CN",
            "measure", "explicit_decimal_measure_approximation",
            approve_explicit_decimal_measure,
        },
        {
            kDecimalMeasureThresholdRule, kDecimalMeasureThresholdRuleVersion, "zh-CN",
            "measure", "explicit_decimal_measure_threshold",
            approve_explicit_decimal_measure,
        },
        {
            kDecimalMoneyRule, kDecimalMoneyRuleVersion, "zh-CN",
            "money", "explicit_decimal_money", approve_explicit_decimal_money,
        },
        {
            kDecimalMoneyApproximateRule, kDecimalMoneyApproximateRuleVersion, "zh-CN",
            "money", "explicit_decimal_money_approximation", approve_explicit_decimal_money,
        },
        {
            kDecimalMoneyThresholdRule, kDecimalMoneyThresholdRuleVersion, "zh-CN",
            "money", "explicit_decimal_money_threshold", approve_explicit_decimal_money,
        },
        {
            kContextualStockPriceRule, kContextualStockPriceRuleVersion, "zh-CN",
            "money", "contextual_stock_price", approve_contextual_stock_price,
        },
        {
            kExplicitTemperatureRule, kExplicitTemperatureRuleVersion, "zh-CN",
            "measure", "explicit_temperature", approve_explicit_temperature,
        },
        {
            kExplicitTemperatureApproximateRule,
            kExplicitTemperatureApproximateRuleVersion,
            "zh-CN", "measure", "explicit_temperature_approximation",
            approve_explicit_temperature,
        },
        {
            kExplicitTemperatureThresholdRule,
            kExplicitTemperatureThresholdRuleVersion,
            "zh-CN", "measure", "explicit_temperature_threshold",
            approve_explicit_temperature,
        },
        {
            kExplicitTemperatureRangeRule,
            kExplicitTemperatureRangeRuleVersion,
            "zh-CN", "measure", "explicit_temperature_range",
            approve_explicit_temperature_range,
        },
        {
            kExplicitIntegerMoneyRule, kExplicitIntegerMoneyRuleVersion, "zh-CN",
            "money", "explicit_integer_money", approve_explicit_integer_money,
        },
        {
            kExplicitIntegerMoneyThresholdRule,
            kExplicitIntegerMoneyThresholdRuleVersion,
            "zh-CN", "money", "explicit_integer_money_threshold",
            approve_explicit_integer_money,
        },
        {
            kExplicitIntegerMoneyApproximateRule,
            kExplicitIntegerMoneyApproximateRuleVersion,
            "zh-CN", "money", "explicit_integer_money_approximation",
            approve_explicit_integer_money,
        },
        {
            kExplicitIntegerMeasureRule, kExplicitIntegerMeasureRuleVersion, "zh-CN",
            "measure", "explicit_integer_measure", approve_explicit_integer_measure,
        },
        {
            kExplicitIntegerMeasureThresholdRule,
            kExplicitIntegerMeasureThresholdRuleVersion,
            "zh-CN", "measure", "explicit_integer_measure_threshold",
            approve_explicit_integer_measure,
        },
        {
            kExplicitIntegerMeasureApproximateRule,
            kExplicitIntegerMeasureApproximateRuleVersion,
            "zh-CN", "measure", "explicit_integer_measure_approximation",
            approve_explicit_integer_measure,
        },
        {
            kContextualApproximateRateDistanceRule,
            kContextualApproximateRateDistanceRuleVersion,
            "zh-CN", "measure", "contextual_approximate_rate_distance",
            approve_explicit_integer_measure,
        },
        {
            kExplicitIntegerMeasurePeriodicRule,
            kExplicitIntegerMeasurePeriodicRuleVersion,
            "zh-CN", "measure", "explicit_integer_measure_periodic",
            approve_explicit_integer_measure,
        },
        {
            kExplicitIntegerDurationRule, kExplicitIntegerDurationRuleVersion, "zh-CN",
            "date_time", "explicit_integer_duration", approve_explicit_integer_duration,
        },
        {
            kExplicitIntegerDurationThresholdRule,
            kExplicitIntegerDurationThresholdRuleVersion,
            "zh-CN", "date_time", "explicit_integer_duration_threshold",
            approve_explicit_integer_duration,
        },
        {
            kExplicitIntegerDurationApproximateRule,
            kExplicitIntegerDurationApproximateRuleVersion,
            "zh-CN", "date_time", "explicit_integer_duration_approximation",
            approve_explicit_integer_duration,
        },
        {
            kExplicitIntegerDurationPeriodicRule,
            kExplicitIntegerDurationPeriodicRuleVersion,
            "zh-CN", "date_time", "explicit_integer_duration_periodic",
            approve_explicit_integer_duration,
        },
        {
            kExplicitLockPeriodMonthsRule,
            kExplicitLockPeriodMonthsRuleVersion,
            "zh-CN", "date_time", "explicit_lock_period_months",
            approve_explicit_lock_period_months,
        },
        {
            kLargeApproximateYearSpanRule,
            kLargeApproximateYearSpanRuleVersion,
            "zh-CN", "date_time", "explicit_large_approximate_year_span",
            approve_large_approximate_year_span,
        },
        {
            kExplicitValidityYearsRule,
            kExplicitValidityYearsRuleVersion,
            "zh-CN", "date_time", "explicit_validity_years",
            approve_explicit_anchored_year_or_day_duration,
        },
        {
            kExplicitRenewalApproximateDaysRule,
            kExplicitRenewalApproximateDaysRuleVersion,
            "zh-CN", "date_time", "explicit_renewal_approximate_days",
            approve_explicit_anchored_year_or_day_duration,
        },
        {
            kExplicitCompoundDurationRule, kExplicitCompoundDurationRuleVersion, "zh-CN",
            "date_time", "explicit_compound_duration", approve_explicit_compound_duration,
        },
        {
            kExplicitMinuteSecondDurationRule,
            kExplicitMinuteSecondDurationRuleVersion,
            "zh-CN", "date_time", "explicit_compound_duration",
            approve_explicit_compound_duration,
        },
        {
            kExplicitMoneyRangeRule, kExplicitMoneyRangeRuleVersion, "zh-CN",
            "money", "explicit_money_range", approve_explicit_money_range,
        },
        {
            kExplicitMeasureRangeRule, kExplicitMeasureRangeRuleVersion, "zh-CN",
            "measure", "explicit_measure_range", approve_explicit_measure_range,
        },
        {
            kExplicitDurationRangeRule, kExplicitDurationRangeRuleVersion, "zh-CN",
            "date_time", "explicit_duration_range", approve_explicit_duration_range,
        },
    };
    return rules;
}

int semantic_candidate_priority(const TransformationCandidate& candidate) {
    if (candidate.rule_id == kExplicitSignedNumberRule) return 108;
    if (candidate.rule_id == kApproximateCalendarTimestampRule) return 125;
    if (candidate.rule_id == kContextualApproximateClockTimeRule) return 122;
    if (candidate.rule_id == kCalendarTimestampRangeRule) return 135;
    if (candidate.rule_id == kCrossSegmentMonthRule) return 108;
    if (candidate.rule_id == kContextualFullDateTimestampRule) return 125;
    if (candidate.rule_id == kContextualApproximateMonthDayTimestampRule) return 120;
    if (candidate.rule_id == kContextualMonthDayTimestampRule) return 115;
    if (candidate.rule_id == kCalendarTimestampRule) return 110;
    if (candidate.rule_id == kClockTimeRule
        || candidate.rule_id == kContextualClockTimeRule) return 110;
    if (candidate.rule_id == kTelephoneListRule) return 105;
    if (candidate.rule_id == kRecurringCrossYearMonthRangeRule) return 105;
    if (candidate.rule_id == kFullDateRangeRule
        || candidate.rule_id == kSharedYearDateRangeRule
        || candidate.rule_id == kSharedMonthDateRangeRule
        || candidate.rule_id == kFullDateSharedMonthRangeRule
        || candidate.rule_id == kYearMonthRangeRule
        || candidate.rule_id == kYearRangeRule
        || candidate.rule_id == kMonthRangeRule
        || candidate.rule_id == kPercentageRangeRule
        || candidate.rule_id == kSharedPrefixPercentageRangeRule
        || candidate.rule_id == kExplicitMoneyRangeRule
        || candidate.rule_id == kExplicitMeasureRangeRule
         || candidate.rule_id == kExplicitDurationRangeRule
         || candidate.rule_id == kExplicitTemperatureRangeRule
         || candidate.rule_id == kClockTimeRangeRule
         || candidate.rule_id == kExplicitCompoundDurationRule
         || candidate.rule_id == kExplicitMinuteSecondDurationRule) {
        return 100;
    }
    if (candidate.rule_id == kFullDateRule || candidate.rule_id == kYearMonthRule
        || candidate.rule_id == kMonthDayRule || candidate.rule_id == kMonthPeriodRule
        || candidate.rule_id == kRelativeYearMonthRule) {
        return 90;
    }
    if (candidate.rule_id == kDecimalMoneyRule
        || candidate.rule_id == kDecimalMoneyApproximateRule
        || candidate.rule_id == kDecimalMoneyThresholdRule
        || candidate.rule_id == kDecimalMeasureRule
        || candidate.rule_id == kDecimalMeasureApproximateRule
        || candidate.rule_id == kDecimalMeasureThresholdRule
        || candidate.rule_id == kDecimalDurationRule
        || candidate.rule_id == kDecimalDurationApproximateRule
        || candidate.rule_id == kDecimalDurationThresholdRule) {
        return 80;
    }
    return 70;
}

bool contains_candidate(
    const TransformationCandidate& outer,
    const TransformationCandidate& inner
) {
    return outer.semantic_source.start <= inner.semantic_source.start
        && outer.semantic_source.end >= inner.semantic_source.end;
}

void add_unreleased_composite_blockers(
    const std::string& input,
    std::vector<TransformationCandidate>* candidates
) {
    const auto boundaries = unicode_scalar_boundaries(input);
    if (!boundaries.has_value()) return;
    std::vector<TransformationCandidate> ordered = *candidates;
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return left.semantic_source.start < right.semantic_source.start
            || (left.semantic_source.start == right.semantic_source.start
                && left.semantic_source.end < right.semantic_source.end);
    });
    std::vector<TransformationCandidate> blockers;
    for (size_t first_index = 0; first_index < ordered.size(); ++first_index) {
        const auto& first = ordered[first_index];
        for (size_t second_index = first_index + 1; second_index < ordered.size(); ++second_index) {
            const auto& second = ordered[second_index];
            if (second.semantic_source.start < first.semantic_source.end) continue;
            const size_t gap_length = second.semantic_source.start - first.semantic_source.end;
            if (gap_length == 0 || gap_length > 12) break;
            const std::string gap = unicode_scalar_substring(
                input,
                *boundaries,
                first.semantic_source.end,
                second.semantic_source.start
            );
            if (contains_any(gap, {"，", "。", "；", "：", "！", "？", ",", ";", ":", "!", "?"})) {
                continue;
            }
            const bool lexical_dao = starts_with_any(gap, {
                "到站", "到期", "到达", "到场", "到账", "到手", "到底",
            });
            const bool direct_range = !lexical_dao
                && starts_with_any(gap, {"至", "到", "-", "－", "—", "~", "～"});
            const bool comparison_range = ends_with_any(gap, {"至", "到"})
                && contains_any(gap, {
                    "增长", "增加", "升至", "上升", "下降", "降至", "减少", "缩短",
                    "延长", "提高", "降低", "扩大", "收窄", "调整", "变为", "改为",
                });
            if (!direct_range && !comparison_range) continue;
            const size_t start = first.semantic_source.start;
            const size_t end = second.semantic_source.end;
            if (std::any_of(candidates->begin(), candidates->end(), [&](const auto& item) {
                    return (item.rule_id == kCalendarTimestampRangeRule
                            || item.rule_id == kClockTimeRangeRule)
                        && item.semantic_source.start <= start
                        && item.semantic_source.end >= end;
                })) {
                continue;
            }
            blockers.push_back({
                "zh-unreleased-composite-" + std::to_string(start) + "-" + std::to_string(end),
                {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
                "zh-CN",
                "guard",
                "unreleased_composite",
                "zh.preserve.unreleased_composite",
                "1",
            });
            break;
        }
    }
    const size_t length = boundaries->size() - 1;
    for (size_t connector = 0; connector < length; ++connector) {
        const std::string connector_text = unicode_scalar_substring(
            input, *boundaries, connector, connector + 1
        );
        const std::string connector_prefix = unicode_scalar_substring(
            input, *boundaries, connector, std::min(length, connector + 2)
        );
        if (!range_connector(connector_text)
            || starts_with_any(connector_prefix, {
                "到站", "到期", "到达", "到场", "到账", "到手", "到底",
            })
            || std::any_of(candidates->begin(), candidates->end(), [&](const auto& candidate) {
                return candidate.semantic_source.start <= connector
                    && candidate.semantic_source.end > connector
                    && (candidate.rule_id == kYearRangeRule
                        || candidate.rule_id == kYearMonthRangeRule
                        || candidate.rule_id == kFullDateRangeRule
                        || candidate.rule_id == kSharedYearDateRangeRule
                        || candidate.rule_id == kSharedMonthDateRangeRule
                        || candidate.rule_id == kFullDateSharedMonthRangeRule
                         || candidate.rule_id == kCalendarTimestampRangeRule
                         || candidate.rule_id == kClockTimeRangeRule
                        || candidate.rule_id == kPercentageRangeRule
                        || candidate.rule_id == kSharedPrefixPercentageRangeRule
                        || candidate.rule_id == kExplicitMoneyRangeRule
                        || candidate.rule_id == kExplicitMeasureRangeRule
                        || candidate.rule_id == kExplicitDurationRangeRule);
            })) {
            continue;
        }
        const TransformationCandidate* left_candidate = nullptr;
        const TransformationCandidate* right_candidate = nullptr;
        for (const auto& candidate : *candidates) {
            if (candidate.semantic_source.end <= connector
                && connector - candidate.semantic_source.end <= 12
                && (left_candidate == nullptr
                    || candidate.semantic_source.end > left_candidate->semantic_source.end)) {
                left_candidate = &candidate;
            }
            if (candidate.semantic_source.start > connector
                && candidate.semantic_source.start - connector <= 12
                && (right_candidate == nullptr
                    || candidate.semantic_source.start < right_candidate->semantic_source.start)) {
                right_candidate = &candidate;
            }
        }
        if (left_candidate == nullptr || right_candidate == nullptr) continue;
        const std::string left_bridge = unicode_scalar_substring(
            input,
            *boundaries,
            left_candidate->semantic_source.end,
            connector
        );
        const bool temporal_endpoint_bridge = contains_any(left_bridge, {
            "点", "时", "分", "秒",
        });
        if (!left_bridge.empty() && !temporal_endpoint_bridge
            && !ends_with_any(left_bridge, {
                "增长", "增加", "上涨", "上升", "下降", "降至", "减少", "缩短",
                "延长", "提高", "降低", "扩大", "收窄", "调整", "变为", "改为",
            })) {
            continue;
        }
        size_t start = left_candidate->semantic_source.start;
        size_t end = right_candidate->semantic_source.end;
        bool expanded = true;
        while (expanded) {
            expanded = false;
            for (const auto& candidate : *candidates) {
                if (candidate.semantic_source.end < start
                    || candidate.semantic_source.start > end) {
                    continue;
                }
                const size_t expanded_start = std::min(start, candidate.semantic_source.start);
                const size_t expanded_end = std::max(end, candidate.semantic_source.end);
                if (expanded_start != start || expanded_end != end) {
                    start = expanded_start;
                    end = expanded_end;
                    expanded = true;
                }
            }
        }
        blockers.push_back({
            "zh-unreleased-range-clause-" + std::to_string(start) + "-" + std::to_string(end),
            {start, end, unicode_scalar_substring(input, *boundaries, start, end)},
            "zh-CN",
            "guard",
            "unreleased_composite",
            "zh.preserve.unreleased_composite",
            "1",
        });
    }
    candidates->insert(candidates->end(), blockers.begin(), blockers.end());
}

std::vector<TransformationCandidate> resolve_complete_semantic_candidates(
    std::vector<TransformationCandidate> candidates
) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.semantic_source.start < right.semantic_source.start
            || (left.semantic_source.start == right.semantic_source.start
                && left.semantic_source.end > right.semantic_source.end);
    });
    std::vector<TransformationCandidate> resolved;
    for (size_t begin = 0; begin < candidates.size();) {
        size_t end = begin + 1;
        size_t component_end = candidates[begin].semantic_source.end;
        while (end < candidates.size()
            && candidates[end].semantic_source.start < component_end) {
            component_end = std::max(component_end, candidates[end].semantic_source.end);
            ++end;
        }
        if (end == begin + 1) {
            resolved.push_back(std::move(candidates[begin]));
            begin = end;
            continue;
        }

        size_t selected = end;
        int selected_priority = -1;
        bool conflicting_selection = false;
        for (size_t index = begin; index < end; ++index) {
            const bool contains_component = std::all_of(
                candidates.begin() + static_cast<std::ptrdiff_t>(begin),
                candidates.begin() + static_cast<std::ptrdiff_t>(end),
                [&](const auto& other) {
                    return contains_candidate(candidates[index], other);
                }
            );
            if (!contains_component) continue;
            const int priority = semantic_candidate_priority(candidates[index]);
            if (selected == end || priority > selected_priority) {
                selected = index;
                selected_priority = priority;
                conflicting_selection = false;
            } else if (priority == selected_priority
                && (candidates[index].semantic_source.start
                        != candidates[selected].semantic_source.start
                    || candidates[index].semantic_source.end
                        != candidates[selected].semantic_source.end
                    || candidates[index].rule_id != candidates[selected].rule_id)) {
                conflicting_selection = true;
            }
        }
        if (selected != end && !conflicting_selection) {
            resolved.push_back(std::move(candidates[selected]));
        }
        begin = end;
    }
    return resolved;
}

}  // namespace

static SafeNormalizationResult normalize_zh_conservative_v2_impl(
    const V2SegmentRequest& request,
    bool shadow_atomic
) {
    if (request.locale != "zh-CN") {
        auto result = normalize_fail_closed(request.text, {}, {});
        result.valid = false;
        result.failure_reason = "unsupported_locale";
        result.decision_reason.clear();
        return result;
    }
    const auto full_dates = detect_full_dates(request.text);
    const auto full_date_ranges = detect_full_date_ranges(request.text, full_dates);
    const auto month_days = detect_month_days(request.text, full_dates);
    const auto month_periods = detect_month_periods(request.text);
    const auto month_ranges = detect_month_ranges(request.text);
    const auto recurring_cross_year_month_ranges =
        detect_recurring_cross_year_month_ranges(request.text);
    const auto relative_year_months = detect_relative_year_months(
        request.text, recurring_cross_year_month_ranges
    );
    const auto contextual_followup_months = detect_contextual_followup_months(request.text);
    const auto shared_year_date_ranges = detect_shared_year_date_ranges(
        request.text, full_dates, month_days
    );
    const auto shared_month_date_ranges = detect_shared_month_date_ranges(
        request.text, month_days
    );
    const auto full_date_shared_month_ranges = detect_full_date_shared_month_ranges(
        request.text, full_dates
    );
    std::vector<TransformationCandidate> standalone_full_dates;
    std::copy_if(
        full_dates.begin(),
        full_dates.end(),
        std::back_inserter(standalone_full_dates),
        [&](const auto& full_date) {
            return std::none_of(
                full_date_ranges.begin(),
                full_date_ranges.end(),
                [&](const auto& date_range) {
                    return overlaps(
                        full_date.semantic_source.start,
                        full_date.semantic_source.end,
                        date_range.semantic_source
                    );
                }
            ) && std::none_of(
                shared_year_date_ranges.begin(),
                shared_year_date_ranges.end(),
                [&](const auto& date_range) {
                    return overlaps(
                        full_date.semantic_source.start,
                        full_date.semantic_source.end,
                        date_range.semantic_source
                    );
                }
            ) && std::none_of(
                full_date_shared_month_ranges.begin(),
                full_date_shared_month_ranges.end(),
                [&](const auto& date_range) {
                    return overlaps(
                        full_date.semantic_source.start,
                        full_date.semantic_source.end,
                        date_range.semantic_source
                    );
                }
            );
        }
    );
    std::vector<TransformationCandidate> standalone_month_days;
    std::copy_if(
        month_days.begin(),
        month_days.end(),
        std::back_inserter(standalone_month_days),
        [&](const auto& month_day) {
            return std::none_of(
                shared_year_date_ranges.begin(),
                shared_year_date_ranges.end(),
                [&](const auto& date_range) {
                    return overlaps(
                        month_day.semantic_source.start,
                        month_day.semantic_source.end,
                        date_range.semantic_source
                    );
                }
            ) && std::none_of(
                shared_month_date_ranges.begin(),
                shared_month_date_ranges.end(),
                [&](const auto& date_range) {
                    return overlaps(
                        month_day.semantic_source.start,
                        month_day.semantic_source.end,
                        date_range.semantic_source
                    );
                }
            );
        }
    );
    const auto year_ranges = detect_year_ranges(request.text);
    const auto year_month_ranges = detect_year_month_ranges(request.text);
    auto year_month_exclusions = full_dates;
    year_month_exclusions.insert(
        year_month_exclusions.end(), year_month_ranges.begin(), year_month_ranges.end()
    );
    const auto year_months = detect_year_months(request.text, year_month_exclusions);
    const auto cross_segment_month = detect_cross_segment_month(
        request.text, request.left_context
    );
    auto date_candidates = standalone_full_dates;
    date_candidates.insert(
        date_candidates.end(), standalone_month_days.begin(), standalone_month_days.end()
    );
    const auto clock_times = detect_explicit_clock_times(request.text);
    const auto contextual_clock_times = detect_contextual_clock_times(request.text);
    const auto video_timecodes = detect_explicit_video_timecodes(request.text);
    const auto clinical_thresholds = detect_clinical_thresholds(request.text);
    const auto clock_time_ranges = detect_explicit_clock_time_ranges(request.text);
    auto timestamp_clock_times = clock_times;
    std::copy_if(
        contextual_clock_times.begin(),
        contextual_clock_times.end(),
        std::back_inserter(timestamp_clock_times),
        [](const auto& clock) {
            return clock.rule_id == kContextualClockTimeRule;
        }
    );
    auto timestamps = detect_explicit_calendar_timestamps(
        request.text, date_candidates, timestamp_clock_times
    );
    const auto timestamp_ranges = detect_explicit_calendar_timestamp_ranges(
        request.text, timestamps
    );
    const auto approximate_timestamps = detect_explicit_approximate_calendar_timestamps(
        request.text, timestamps
    );
    timestamps.insert(
        timestamps.end(), approximate_timestamps.begin(), approximate_timestamps.end()
    );
    const auto contextual_timestamps = detect_contextual_month_day_timestamps(
        request.text, date_candidates
    );
    timestamps.insert(
        timestamps.end(), contextual_timestamps.begin(), contextual_timestamps.end()
    );
    auto candidates = date_candidates;
    candidates.insert(candidates.end(), month_periods.begin(), month_periods.end());
    candidates.insert(candidates.end(), month_ranges.begin(), month_ranges.end());
    candidates.insert(
        candidates.end(), relative_year_months.begin(), relative_year_months.end()
    );
    candidates.insert(
        candidates.end(), contextual_followup_months.begin(), contextual_followup_months.end()
    );
    candidates.insert(
        candidates.end(),
        recurring_cross_year_month_ranges.begin(),
        recurring_cross_year_month_ranges.end()
    );
    candidates.insert(candidates.end(), timestamps.begin(), timestamps.end());
    candidates.insert(candidates.end(), timestamp_ranges.begin(), timestamp_ranges.end());
    candidates.insert(candidates.end(), full_date_ranges.begin(), full_date_ranges.end());
    candidates.insert(
        candidates.end(), shared_year_date_ranges.begin(), shared_year_date_ranges.end()
    );
    candidates.insert(
        candidates.end(), shared_month_date_ranges.begin(), shared_month_date_ranges.end()
    );
    candidates.insert(
        candidates.end(),
        full_date_shared_month_ranges.begin(),
        full_date_shared_month_ranges.end()
    );
    candidates.insert(candidates.end(), year_ranges.begin(), year_ranges.end());
    candidates.insert(candidates.end(), year_month_ranges.begin(), year_month_ranges.end());
    candidates.insert(candidates.end(), year_months.begin(), year_months.end());
    candidates.insert(
        candidates.end(), cross_segment_month.begin(), cross_segment_month.end()
    );
    const auto years = detect_explicit_years(request.text, candidates);
    candidates.insert(candidates.end(), years.begin(), years.end());
    const auto percentages = detect_percentages(request.text);
    candidates.insert(candidates.end(), percentages.begin(), percentages.end());
    const auto percentage_ranges = detect_percentage_ranges(request.text, percentages);
    candidates.insert(candidates.end(), percentage_ranges.begin(), percentage_ranges.end());
    const auto shared_prefix_percentage_ranges = detect_shared_prefix_percentage_ranges(
        request.text, percentages
    );
    candidates.insert(
        candidates.end(),
        shared_prefix_percentage_ranges.begin(),
        shared_prefix_percentage_ranges.end()
    );
    const auto decimal_durations = detect_explicit_decimal_durations(request.text);
    candidates.insert(candidates.end(), decimal_durations.begin(), decimal_durations.end());
    const auto decimal_measures = detect_explicit_decimal_measures(request.text);
    candidates.insert(candidates.end(), decimal_measures.begin(), decimal_measures.end());
    const auto signed_numbers = detect_explicit_signed_numbers(request.text);
    candidates.insert(candidates.end(), signed_numbers.begin(), signed_numbers.end());
    const auto isolated_digit_g = detect_isolated_digit_g_identifiers(request.text);
    candidates.insert(candidates.end(), isolated_digit_g.begin(), isolated_digit_g.end());
    const auto respirators = detect_respirator_standards(request.text);
    candidates.insert(candidates.end(), respirators.begin(), respirators.end());
    const auto automotive_stores = detect_automotive_stores(request.text);
    candidates.insert(candidates.end(), automotive_stores.begin(), automotive_stores.end());
    const auto playstation_models = detect_playstation_models(request.text);
    candidates.insert(candidates.end(), playstation_models.begin(), playstation_models.end());
    const auto closed_product_models = detect_closed_product_models(request.text);
    candidates.insert(candidates.end(), closed_product_models.begin(), closed_product_models.end());
    const auto contextual_product_models = detect_contextual_product_models(request.text);
    candidates.insert(
        candidates.end(), contextual_product_models.begin(), contextual_product_models.end()
    );
    const auto land_parcel_identifiers = detect_land_parcel_identifiers(request.text);
    candidates.insert(
        candidates.end(), land_parcel_identifiers.begin(), land_parcel_identifiers.end()
    );
    const auto display_resolutions = detect_display_resolutions(request.text);
    candidates.insert(candidates.end(), display_resolutions.begin(), display_resolutions.end());
    const auto contextual_ratios = detect_explicit_contextual_ratios(request.text);
    candidates.insert(candidates.end(), contextual_ratios.begin(), contextual_ratios.end());
    const auto contextual_scalar_ratios = detect_contextual_scalar_ratios(request.text);
    candidates.insert(
        candidates.end(), contextual_scalar_ratios.begin(), contextual_scalar_ratios.end()
    );
    const auto contextual_setting_values = detect_contextual_setting_values(request.text);
    candidates.insert(
        candidates.end(), contextual_setting_values.begin(), contextual_setting_values.end()
    );
    const auto contextual_scores = detect_contextual_scores(request.text);
    candidates.insert(candidates.end(), contextual_scores.begin(), contextual_scores.end());
    const auto contextual_ranks = detect_contextual_ranks(request.text);
    candidates.insert(candidates.end(), contextual_ranks.begin(), contextual_ranks.end());
    const auto contextual_document_pages = detect_contextual_document_pages(request.text);
    candidates.insert(
        candidates.end(), contextual_document_pages.begin(), contextual_document_pages.end()
    );
    const auto contextual_structured_ordinals =
        detect_contextual_structured_ordinals(request.text);
    candidates.insert(
        candidates.end(), contextual_structured_ordinals.begin(),
        contextual_structured_ordinals.end()
    );
    const auto contextual_anchored_cardinals =
        detect_contextual_anchored_cardinals(request.text);
    candidates.insert(
        candidates.end(), contextual_anchored_cardinals.begin(),
        contextual_anchored_cardinals.end()
    );
    const auto technical_identifiers = detect_technical_identifiers(request.text);
    candidates.insert(
        candidates.end(), technical_identifiers.begin(), technical_identifiers.end()
    );
    const auto structured_spoken_identifiers =
        detect_structured_spoken_identifiers(request.text);
    candidates.insert(
        candidates.end(), structured_spoken_identifiers.begin(),
        structured_spoken_identifiers.end()
    );
    const auto geographic_coordinates = detect_explicit_geographic_coordinates(request.text);
    candidates.insert(
        candidates.end(), geographic_coordinates.begin(), geographic_coordinates.end()
    );
    const auto software_versions = detect_software_versions(request.text);
    candidates.insert(candidates.end(), software_versions.begin(), software_versions.end());
    const auto telephone_numbers = detect_telephone_numbers(request.text);
    candidates.insert(candidates.end(), telephone_numbers.begin(), telephone_numbers.end());
    const auto telephone_lists = detect_telephone_number_lists(request.text);
    candidates.insert(candidates.end(), telephone_lists.begin(), telephone_lists.end());
    const auto public_service_numbers = detect_contextual_public_service_numbers(request.text);
    candidates.insert(
        candidates.end(), public_service_numbers.begin(), public_service_numbers.end()
    );
    const auto explicit_identifiers = detect_explicit_identifiers(request.text);
    candidates.insert(candidates.end(), explicit_identifiers.begin(), explicit_identifiers.end());
    const auto train_numbers = detect_train_numbers(request.text);
    candidates.insert(candidates.end(), train_numbers.begin(), train_numbers.end());
    const auto flight_numbers = detect_flight_numbers(request.text);
    candidates.insert(candidates.end(), flight_numbers.begin(), flight_numbers.end());
    const auto ipv4_addresses = detect_ipv4_addresses(request.text);
    candidates.insert(candidates.end(), ipv4_addresses.begin(), ipv4_addresses.end());
    const auto network_ports = detect_network_ports(request.text);
    candidates.insert(candidates.end(), network_ports.begin(), network_ports.end());
    const auto centuries = detect_explicit_centuries(request.text);
    candidates.insert(candidates.end(), centuries.begin(), centuries.end());
    candidates.insert(candidates.end(), clock_time_ranges.begin(), clock_time_ranges.end());
    candidates.insert(candidates.end(), clock_times.begin(), clock_times.end());
    candidates.insert(
        candidates.end(), contextual_clock_times.begin(), contextual_clock_times.end()
    );
    candidates.insert(candidates.end(), video_timecodes.begin(), video_timecodes.end());
    candidates.insert(
        candidates.end(), clinical_thresholds.begin(), clinical_thresholds.end()
    );
    const auto decimal_money = detect_explicit_decimal_money(request.text);
    candidates.insert(candidates.end(), decimal_money.begin(), decimal_money.end());
    const auto stock_prices = detect_contextual_stock_prices(request.text);
    candidates.insert(candidates.end(), stock_prices.begin(), stock_prices.end());
    const auto temperatures = detect_explicit_temperatures(request.text);
    candidates.insert(candidates.end(), temperatures.begin(), temperatures.end());
    const auto temperature_ranges = detect_explicit_temperature_ranges(
        request.text, temperatures
    );
    candidates.insert(
        candidates.end(), temperature_ranges.begin(), temperature_ranges.end()
    );
    const auto integer_money = detect_explicit_integer_money(request.text);
    candidates.insert(candidates.end(), integer_money.begin(), integer_money.end());
    const auto integer_measures = detect_explicit_integer_measures(request.text);
    candidates.insert(candidates.end(), integer_measures.begin(), integer_measures.end());
    const auto compound_durations = detect_explicit_compound_durations(request.text);
    candidates.insert(
        candidates.end(), compound_durations.begin(), compound_durations.end()
    );
    const auto integer_durations = detect_explicit_integer_durations(request.text);
    candidates.insert(candidates.end(), integer_durations.begin(), integer_durations.end());
    const auto lock_period_months = detect_explicit_lock_period_months(request.text);
    candidates.insert(candidates.end(), lock_period_months.begin(), lock_period_months.end());
    const auto large_approximate_year_spans =
        detect_large_approximate_year_spans(request.text);
    candidates.insert(
        candidates.end(),
        large_approximate_year_spans.begin(),
        large_approximate_year_spans.end()
    );
    const auto anchored_year_or_day_durations =
        detect_explicit_anchored_year_or_day_durations(request.text);
    candidates.insert(
        candidates.end(),
        anchored_year_or_day_durations.begin(),
        anchored_year_or_day_durations.end()
    );
    const auto money_ranges = detect_explicit_unit_ranges(
        ExplicitUnitRangeKind::money,
        request.text,
        "money",
        "explicit_money_range",
        kExplicitMoneyRangeRule,
        kExplicitMoneyRangeRuleVersion
    );
    candidates.insert(candidates.end(), money_ranges.begin(), money_ranges.end());
    const auto measure_ranges = detect_explicit_unit_ranges(
        ExplicitUnitRangeKind::measure,
        request.text,
        "measure",
        "explicit_measure_range",
        kExplicitMeasureRangeRule,
        kExplicitMeasureRangeRuleVersion
    );
    candidates.insert(candidates.end(), measure_ranges.begin(), measure_ranges.end());
    const auto duration_ranges = detect_explicit_unit_ranges(
        ExplicitUnitRangeKind::duration,
        request.text,
        "date_time",
        "explicit_duration_range",
        kExplicitDurationRangeRule,
        kExplicitDurationRangeRuleVersion
    );
    candidates.insert(candidates.end(), duration_ranges.begin(), duration_ranges.end());
    add_unreleased_composite_blockers(request.text, &candidates);
    candidates = resolve_complete_semantic_candidates(std::move(candidates));
    const size_t total_candidate_count = candidates.size();
    std::vector<std::string> duplicate_rule_ids;
    std::vector<std::pair<std::string, size_t>> duplicate_rule_candidate_counts;
    for (const auto& candidate : candidates) {
        const size_t count = std::count_if(
            candidates.begin(), candidates.end(), [&](const auto& other) {
                return other.rule_id == candidate.rule_id;
            }
        );
        if (count > 1 && std::find(
                duplicate_rule_ids.begin(), duplicate_rule_ids.end(), candidate.rule_id
            ) == duplicate_rule_ids.end()) {
            duplicate_rule_ids.push_back(candidate.rule_id);
            duplicate_rule_candidate_counts.emplace_back(candidate.rule_id, count);
        }
    }
    auto result = normalize_fail_closed(
        request.text,
        {
            request.left_context,
            request.right_context,
            request.has_left_neighbor,
            request.has_right_neighbor,
        },
        candidates,
        released_rules()
    );
    result.audit_duplicate_rule_candidate_counts = duplicate_rule_candidate_counts;
    result.audit_total_candidate_count = total_candidate_count;
    if (result.valid && result.normalized.applied.size() > 1) {
        result.decision_reason = shadow_atomic
            ? "shadow_atomic_applied_multiple_transformations"
            : "applied_multiple_semantic_objects";
    }
    return result;
}

SafeNormalizationResult normalize_zh_conservative_v2(const V2SegmentRequest& request) {
    return normalize_zh_conservative_v2_impl(request, false);
}

SafeNormalizationResult normalize_zh_conservative_v2_shadow_atomic(
    const V2SegmentRequest& request
) {
    auto result = normalize_zh_conservative_v2_impl(request, true);
    if (result.valid && result.normalized.applied.size() > 1) {
        result.decision_reason = "shadow_atomic_applied_multiple_transformations";
    }
    return result;
}

}  // namespace zh_itn::itn
