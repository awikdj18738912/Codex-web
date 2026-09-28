/*
 * itn_safe.h - Fail-closed ITN transformation and composition contracts.
 */

#ifndef ZH_ITN_SAFE_H
#define ZH_ITN_SAFE_H

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace zh_itn::itn {

struct V2SourceRange {
    size_t start;
    size_t end;
    std::string text;
};

struct V2RuleEvidence {
    std::string kind;
    std::string value;
    std::string source;
};

std::optional<std::vector<size_t>> unicode_scalar_boundaries(const std::string& text);

std::string unicode_scalar_substring(
    const std::string& text,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
);

struct AtomicEdit {
    V2SourceRange source;
    std::string replacement;
};

struct SafePolicyContext {
    std::string left_context;
    std::string right_context;
    bool has_left_neighbor;
    bool has_right_neighbor;
};

struct TransformationCandidate {
    std::string candidate_id;
    V2SourceRange semantic_source;
    std::string locale;
    std::string semantic_type;
    std::string subtype;
    std::string rule_id;
    std::string rule_version;
};

enum class RuleDecision {
    preserve,
    approve,
    error,
};

struct RuleApproval {
    RuleDecision decision;
    std::string parsed_value;
    std::vector<V2RuleEvidence> evidence;
    std::vector<AtomicEdit> edits;
    std::string reason;
};

using RuleApprover = RuleApproval (*)(
    const std::string& input,
    const SafePolicyContext& context,
    const TransformationCandidate& candidate
);

struct ReleasedRule {
    std::string rule_id;
    std::string rule_version;
    std::string locale;
    std::string semantic_type;
    std::string subtype;
    RuleApprover approver;
};

struct SafeMapping {
    std::string kind;
    size_t input_start;
    size_t input_end;
    size_t output_start;
    size_t output_end;
    std::string candidate_id;
    std::string rule_id;
    std::string rule_version;
};

struct AppliedTransformation {
    TransformationCandidate candidate;
    RuleApproval approval;
};

struct SafeNormalizedSegment {
    std::string input;
    SafePolicyContext context;
    std::string output;
    std::vector<SafeMapping> mappings;
    std::vector<AppliedTransformation> applied;
};

struct SafeNormalizationResult {
    SafeNormalizedSegment normalized;
    bool valid;
    std::string failure_reason;
    std::string decision_reason;
    std::vector<std::pair<std::string, size_t>> audit_duplicate_rule_candidate_counts;
    size_t audit_total_candidate_count;
};

SafeNormalizationResult normalize_fail_closed(
    const std::string& input,
    const std::vector<TransformationCandidate>& candidates,
    const std::vector<ReleasedRule>& released_rules
);

SafeNormalizationResult normalize_fail_closed(
    const std::string& input,
    const SafePolicyContext& context,
    const std::vector<TransformationCandidate>& candidates,
    const std::vector<ReleasedRule>& released_rules
);

bool validate_safe_normalization(
    const SafeNormalizedSegment& normalized,
    const std::vector<ReleasedRule>& released_rules,
    std::string* failure_reason = nullptr
);

}  // namespace zh_itn::itn

#endif  // ZH_ITN_SAFE_H
