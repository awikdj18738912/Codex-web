/*
 * itn_safe_test.cc - Fail-closed ITN composition checks.
 */

#include "itn_safe.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace zh_itn::itn;

RuleApproval approve_test_year(
    const std::string&,
    const SafePolicyContext&,
    const TransformationCandidate& candidate
) {
    if (candidate.candidate_id.rfind("preserve-", 0) == 0) {
        return {RuleDecision::preserve, "", {}, {}, "ambiguous_test_candidate"};
    }
    if (candidate.semantic_source.text != "二零二六年") {
        return {RuleDecision::error, "", {}, {}, "unexpected_test_year"};
    }
    return {
        RuleDecision::approve,
        "2026",
        {{"suffix", "年", "zh-CN"}},
        {{
            {
            candidate.semantic_source.start,
            candidate.semantic_source.end - 1,
            "二零二六",
            },
            "2026",
        }},
        "explicit_calendar_year",
    };
}

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

TransformationCandidate year_candidate() {
    return {
        "candidate-2026",
        {1, 6, "二零二六年"},
        "zh-CN",
        "date_time",
        "calendar_year",
        "zh.date.calendar_year",
        "1",
    };
}

std::vector<ReleasedRule> released_year_rule() {
    return {{
        "zh.date.calendar_year", "1", "zh-CN",
        "date_time", "calendar_year", approve_test_year,
    }};
}

}  // namespace

int main() {
    bool passed = true;

    const auto identity = normalize_fail_closed("一些问题", {}, {});
    passed &= expect(identity.valid, "identity normalization failed");
    passed &= expect(identity.normalized.output == "一些问题", "identity changed input");
    passed &= expect(identity.normalized.applied.empty(), "identity applied a transformation");
    passed &= expect(
        identity.decision_reason == "no_candidates_detected" && identity.failure_reason.empty(),
        "identity diagnostic reason is missing"
    );

    auto unreleased = year_candidate();
    const auto unreleased_result = normalize_fail_closed(
        "在二零二六年发布",
        {unreleased},
        {}
    );
    passed &= expect(unreleased_result.valid, "unreleased rule caused structural failure");
    passed &= expect(
        unreleased_result.normalized.output == "在二零二六年发布",
        "unreleased rule changed output"
    );
    passed &= expect(
        unreleased_result.decision_reason == "no_released_rule_match"
            && unreleased_result.failure_reason.empty(),
        "unreleased-rule diagnostic reason is missing"
    );

    auto ambiguous = year_candidate();
    ambiguous.candidate_id = "preserve-ambiguous-year";
    const auto ambiguous_result = normalize_fail_closed(
        "在二零二六年发布",
        {ambiguous},
        released_year_rule()
    );
    passed &= expect(ambiguous_result.valid, "ambiguous candidate caused structural failure");
    passed &= expect(
        ambiguous_result.normalized.output == "在二零二六年发布",
        "ambiguous candidate changed output"
    );
    passed &= expect(
        ambiguous_result.decision_reason.rfind("all_candidates_preserved", 0) == 0
            && ambiguous_result.failure_reason.empty(),
        "preserved-candidate diagnostic reason is missing"
    );

    const auto applied = normalize_fail_closed(
        "在二零二六年发布",
        {year_candidate()},
        released_year_rule()
    );
    passed &= expect(applied.valid, "released transformation failed");
    passed &= expect(applied.normalized.output == "在2026年发布", "released output mismatch");
    passed &= expect(applied.normalized.applied.size() == 1, "released trace missing");
    passed &= expect(
        applied.decision_reason == "applied" && applied.failure_reason.empty(),
        "applied result has invalid diagnostic state"
    );

    auto invalid = year_candidate();
    invalid.semantic_source.text = "二零二五年";
    const auto invalid_result = normalize_fail_closed(
        "在二零二六年发布",
        {invalid},
        released_year_rule()
    );
    passed &= expect(!invalid_result.valid, "invalid source was accepted");
    passed &= expect(
        invalid_result.normalized.output == "在二零二六年发布",
        "invalid source did not fail closed"
    );

    auto overlap_left = year_candidate();
    auto overlap_right = year_candidate();
    overlap_right.candidate_id = "candidate-overlap";
    const auto overlap_result = normalize_fail_closed(
        "在二零二六年发布",
        {overlap_left, overlap_right},
        released_year_rule()
    );
    passed &= expect(!overlap_result.valid, "overlapping candidates were accepted");
    passed &= expect(
        overlap_result.normalized.output == "在二零二六年发布",
        "overlap did not fail closed"
    );

    auto ambiguous_overlap = year_candidate();
    ambiguous_overlap.candidate_id = "preserve-ambiguous-overlap";
    const auto ambiguous_overlap_result = normalize_fail_closed(
        "在二零二六年发布",
        {year_candidate(), ambiguous_overlap},
        released_year_rule()
    );
    passed &= expect(!ambiguous_overlap_result.valid, "ambiguous overlap was ignored");
    passed &= expect(
        ambiguous_overlap_result.normalized.output == "在二零二六年发布",
        "ambiguous overlap did not fail closed"
    );

    auto corrupted = applied.normalized;
    corrupted.output = "在2025年发布";
    std::string validation_failure;
    passed &= expect(
        !validate_safe_normalization(corrupted, released_year_rule(), &validation_failure),
        "corrupted output passed provenance validation"
    );

    auto fully_corrupted = applied.normalized;
    fully_corrupted.output = "在2025年发布";
    fully_corrupted.applied[0].approval.parsed_value = "2025";
    fully_corrupted.applied[0].approval.edits[0].replacement = "2025";
    validation_failure.clear();
    passed &= expect(
        !validate_safe_normalization(
            fully_corrupted,
            released_year_rule(),
            &validation_failure
        ),
        "self-consistent forged approval passed trusted approver replay"
    );

    auto duplicate_rules = released_year_rule();
    duplicate_rules.push_back(duplicate_rules.front());
    const auto invalid_registry = normalize_fail_closed(
        "在二零二六年发布",
        {year_candidate()},
        duplicate_rules
    );
    passed &= expect(!invalid_registry.valid, "duplicate released rule was accepted");
    passed &= expect(
        invalid_registry.normalized.output == "在二零二六年发布",
        "invalid registry did not fail closed"
    );

    return passed ? 0 : 1;
}
