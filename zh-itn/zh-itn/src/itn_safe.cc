/*
 * itn_safe.cc - Fail-closed ITN transformation and composition.
 */

#include "itn_safe.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace zh_itn::itn {
namespace {

struct ScalarText {
    std::vector<size_t> boundaries;
};

std::optional<ScalarText> scalar_text(const std::string& text) {
    ScalarText decoded = {{0}};
    for (size_t offset = 0; offset < text.size();) {
        const auto lead = static_cast<unsigned char>(text[offset]);
        size_t length = 0;
        uint32_t codepoint = 0;
        if (lead < 0x80) {
            length = 1;
            codepoint = lead;
        } else if ((lead & 0xe0) == 0xc0) {
            length = 2;
            codepoint = lead & 0x1f;
        } else if ((lead & 0xf0) == 0xe0) {
            length = 3;
            codepoint = lead & 0x0f;
        } else if ((lead & 0xf8) == 0xf0) {
            length = 4;
            codepoint = lead & 0x07;
        } else {
            return std::nullopt;
        }
        if (offset + length > text.size()) return std::nullopt;
        for (size_t index = 1; index < length; ++index) {
            const auto value = static_cast<unsigned char>(text[offset + index]);
            if ((value & 0xc0) != 0x80) return std::nullopt;
            codepoint = (codepoint << 6) | (value & 0x3f);
        }
        if ((length == 2 && codepoint < 0x80)
            || (length == 3 && codepoint < 0x800)
            || (length == 4 && codepoint < 0x10000)
            || (codepoint >= 0xd800 && codepoint <= 0xdfff)
            || codepoint > 0x10ffff) {
            return std::nullopt;
        }
        offset += length;
        decoded.boundaries.push_back(offset);
    }
    return decoded;
}

std::string scalar_substring(
    const std::string& text,
    const ScalarText& scalars,
    size_t start,
    size_t end
) {
    if (start > end || end >= scalars.boundaries.size()) return {};
    return text.substr(
        scalars.boundaries[start],
        scalars.boundaries[end] - scalars.boundaries[start]
    );
}

void set_failure(std::string* failure_reason, const std::string& value) {
    if (failure_reason != nullptr) *failure_reason = value;
}

const ReleasedRule* matching_rule(
    const TransformationCandidate& candidate,
    const std::vector<ReleasedRule>& released_rules
) {
    const auto found = std::find_if(
        released_rules.begin(),
        released_rules.end(),
        [&](const ReleasedRule& rule) {
            return rule.rule_id == candidate.rule_id
                && rule.rule_version == candidate.rule_version
                && rule.locale == candidate.locale
                && rule.semantic_type == candidate.semantic_type
                && rule.subtype == candidate.subtype;
        }
    );
    return found == released_rules.end() ? nullptr : &*found;
}

bool valid_rule_registry(const std::vector<ReleasedRule>& released_rules) {
    for (size_t index = 0; index < released_rules.size(); ++index) {
        const auto& rule = released_rules[index];
        if (rule.rule_id.empty() || rule.rule_version.empty() || rule.locale.empty()
            || rule.semantic_type.empty() || rule.subtype.empty() || rule.approver == nullptr) {
            return false;
        }
        for (size_t other = index + 1; other < released_rules.size(); ++other) {
            if (rule.rule_id == released_rules[other].rule_id
                && rule.rule_version == released_rules[other].rule_version) {
                return false;
            }
        }
    }
    return true;
}

SafeNormalizationResult identity_result(
    const std::string& input,
    const SafePolicyContext& context,
    bool valid,
    const std::string& failure_reason
) {
    SafeNormalizedSegment normalized = {input, context, input, {}, {}};
    const auto scalars = scalar_text(input);
    if (scalars.has_value() && scalars->boundaries.size() > 1) {
        const size_t length = scalars->boundaries.size() - 1;
        normalized.mappings.push_back({"equal", 0, length, 0, length, "", "", ""});
    }
    return {
        std::move(normalized),
        valid,
        valid ? "" : failure_reason,
        valid ? failure_reason : "",
        {},
        0,
    };
}

bool valid_source_range(
    const std::string& input,
    const ScalarText& scalars,
    const V2SourceRange& range
) {
    const size_t length = scalars.boundaries.size() - 1;
    return range.start < range.end
        && range.end <= length
        && scalar_substring(input, scalars, range.start, range.end) == range.text;
}

bool valid_proposed_candidate(
    const std::string& input,
    const ScalarText& scalars,
    const TransformationCandidate& candidate
) {
    return !candidate.candidate_id.empty()
        && !candidate.locale.empty()
        && !candidate.semantic_type.empty()
        && !candidate.subtype.empty()
        && valid_source_range(input, scalars, candidate.semantic_source);
}

bool valid_approval(
    const std::string& input,
    const ScalarText& scalars,
    const TransformationCandidate& candidate,
    const RuleApproval& approval,
    std::string* failure_reason
) {
    if (approval.decision != RuleDecision::approve || approval.parsed_value.empty()
        || approval.evidence.empty() || approval.edits.empty()) {
        set_failure(failure_reason, "released_approval_incomplete");
        return false;
    }
    if (std::any_of(approval.evidence.begin(), approval.evidence.end(), [](const auto& item) {
            return item.kind.empty() || item.value.empty() || item.source.empty();
        })) {
        set_failure(failure_reason, "invalid_rule_evidence");
        return false;
    }
    if (approval.edits.empty()) {
        set_failure(failure_reason, "released_candidate_has_no_edits");
        return false;
    }
    size_t edit_cursor = candidate.semantic_source.start;
    for (const auto& edit : approval.edits) {
        if (!valid_source_range(input, scalars, edit.source)
            || edit.source.start < candidate.semantic_source.start
            || edit.source.end > candidate.semantic_source.end
            || edit.source.start < edit_cursor
            || edit.replacement.empty()
            || edit.replacement == edit.source.text
            || !scalar_text(edit.replacement).has_value()) {
            set_failure(failure_reason, "invalid_atomic_edit");
            return false;
        }
        edit_cursor = edit.source.end;
    }
    return true;
}

struct SelectedEdit {
    const TransformationCandidate* candidate;
    const AtomicEdit* edit;
};

bool same_atomic_edit(const AtomicEdit& left, const AtomicEdit& right) {
    return left.source.start == right.source.start
        && left.source.end == right.source.end
        && left.source.text == right.source.text
        && left.replacement == right.replacement;
}

bool same_evidence(const V2RuleEvidence& left, const V2RuleEvidence& right) {
    return left.kind == right.kind
        && left.value == right.value
        && left.source == right.source;
}

bool same_approval(const RuleApproval& left, const RuleApproval& right) {
    return left.decision == right.decision
        && left.parsed_value == right.parsed_value
        && left.reason == right.reason
        && left.evidence.size() == right.evidence.size()
        && left.edits.size() == right.edits.size()
        && std::equal(
            left.evidence.begin(),
            left.evidence.end(),
            right.evidence.begin(),
            same_evidence
        )
        && std::equal(
            left.edits.begin(),
            left.edits.end(),
            right.edits.begin(),
            same_atomic_edit
        );
}

bool same_edit(const SafeMapping& mapping, const SelectedEdit& selected) {
    return mapping.input_start == selected.edit->source.start
        && mapping.input_end == selected.edit->source.end
        && mapping.candidate_id == selected.candidate->candidate_id
        && mapping.rule_id == selected.candidate->rule_id
        && mapping.rule_version == selected.candidate->rule_version;
}

}  // namespace

std::optional<std::vector<size_t>> unicode_scalar_boundaries(const std::string& text) {
    const auto decoded = scalar_text(text);
    return decoded.has_value()
        ? std::optional<std::vector<size_t>>(decoded->boundaries)
        : std::nullopt;
}

std::string unicode_scalar_substring(
    const std::string& text,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    return scalar_substring(text, {boundaries}, start, end);
}

SafeNormalizationResult normalize_fail_closed(
    const std::string& input,
    const std::vector<TransformationCandidate>& candidates,
    const std::vector<ReleasedRule>& released_rules
) {
    return normalize_fail_closed(input, {"", "", false, false}, candidates, released_rules);
}

SafeNormalizationResult normalize_fail_closed(
    const std::string& input,
    const SafePolicyContext& context,
    const std::vector<TransformationCandidate>& candidates,
    const std::vector<ReleasedRule>& released_rules
) {
    const auto scalars = scalar_text(input);
    if (!scalars.has_value()) return identity_result(input, context, false, "invalid_utf8");
    if (!valid_rule_registry(released_rules)) {
        return identity_result(input, context, false, "invalid_rule_registry");
    }

    std::unordered_map<std::string, size_t> candidate_ids;
    std::vector<const TransformationCandidate*> ordered_candidates;
    for (const auto& candidate : candidates) {
        if (!valid_proposed_candidate(input, *scalars, candidate)) {
            return identity_result(input, context, false, "invalid_proposed_candidate");
        }
        if (!candidate_ids.emplace(candidate.candidate_id, 1).second) {
            return identity_result(input, context, false, "duplicate_candidate_id");
        }
        ordered_candidates.push_back(&candidate);
    }
    std::sort(
        ordered_candidates.begin(),
        ordered_candidates.end(),
        [](const auto* left, const auto* right) {
            return left->semantic_source.start < right->semantic_source.start
                || (left->semantic_source.start == right->semantic_source.start
                    && left->semantic_source.end < right->semantic_source.end);
        }
    );
    for (size_t index = 1; index < ordered_candidates.size(); ++index) {
        if (ordered_candidates[index]->semantic_source.start
            < ordered_candidates[index - 1]->semantic_source.end) {
            return identity_result(input, context, false, "overlapping_candidate_ranges");
        }
    }

    std::vector<AppliedTransformation> approved;
    std::string failure_reason;
    std::string first_preserve_reason;
    size_t matching_rule_count = 0;
    size_t preserved_candidate_count = 0;
    for (const auto& candidate : candidates) {
        const auto* rule = matching_rule(candidate, released_rules);
        if (rule == nullptr) continue;
        ++matching_rule_count;
        const auto approval = rule->approver(input, context, candidate);
        if (approval.decision == RuleDecision::preserve) {
            ++preserved_candidate_count;
            if (first_preserve_reason.empty()) first_preserve_reason = approval.reason;
            continue;
        }
        if (approval.decision == RuleDecision::error) {
            return identity_result(input, context, false, approval.reason.empty()
                ? "rule_approver_error" : approval.reason);
        }
        if (!valid_approval(input, *scalars, candidate, approval, &failure_reason)) {
            return identity_result(input, context, false, failure_reason);
        }
        approved.push_back({candidate, approval});
    }

    std::sort(approved.begin(), approved.end(), [](const auto& left, const auto& right) {
        return left.candidate.semantic_source.start < right.candidate.semantic_source.start
            || (left.candidate.semantic_source.start == right.candidate.semantic_source.start
                && left.candidate.semantic_source.end < right.candidate.semantic_source.end);
    });
    for (size_t index = 1; index < approved.size(); ++index) {
        if (approved[index].candidate.semantic_source.start
            < approved[index - 1].candidate.semantic_source.end) {
            return identity_result(input, context, false, "overlapping_semantic_ranges");
        }
    }

    std::vector<SelectedEdit> edits;
    for (const auto& applied : approved) {
        for (const auto& edit : applied.approval.edits) {
            edits.push_back({&applied.candidate, &edit});
        }
    }
    std::sort(edits.begin(), edits.end(), [](const auto& left, const auto& right) {
        return left.edit->source.start < right.edit->source.start;
    });
    for (size_t index = 1; index < edits.size(); ++index) {
        if (edits[index].edit->source.start < edits[index - 1].edit->source.end) {
            return identity_result(input, context, false, "overlapping_atomic_edits");
        }
    }
    if (edits.empty()) {
        if (candidates.empty()) {
            return identity_result(input, context, true, "no_candidates_detected");
        }
        if (matching_rule_count == 0) {
            return identity_result(input, context, true, "no_released_rule_match");
        }
        if (preserved_candidate_count == matching_rule_count) {
            return identity_result(
                input,
                context,
                true,
                first_preserve_reason.empty()
                    ? "all_candidates_preserved"
                    : "all_candidates_preserved:" + first_preserve_reason
            );
        }
        return identity_result(input, context, true, "no_approved_atomic_edits");
    }

    SafeNormalizedSegment normalized = {input, context, "", {}, {}};
    normalized.applied = approved;
    size_t input_cursor = 0;
    size_t output_cursor = 0;
    for (const auto& selected : edits) {
        if (input_cursor < selected.edit->source.start) {
            const std::string unchanged = scalar_substring(
                input,
                *scalars,
                input_cursor,
                selected.edit->source.start
            );
            normalized.output += unchanged;
            const size_t length = selected.edit->source.start - input_cursor;
            normalized.mappings.push_back({
                "equal", input_cursor, selected.edit->source.start,
                output_cursor, output_cursor + length, "", "", "",
            });
            output_cursor += length;
        }
        normalized.output += selected.edit->replacement;
        const auto replacement_scalars = scalar_text(selected.edit->replacement);
        const size_t output_end = output_cursor + replacement_scalars->boundaries.size() - 1;
        normalized.mappings.push_back({
            "replace", selected.edit->source.start, selected.edit->source.end,
            output_cursor, output_end, selected.candidate->candidate_id,
            selected.candidate->rule_id, selected.candidate->rule_version,
        });
        input_cursor = selected.edit->source.end;
        output_cursor = output_end;
    }
    const size_t input_length = scalars->boundaries.size() - 1;
    if (input_cursor < input_length) {
        const std::string unchanged = scalar_substring(input, *scalars, input_cursor, input_length);
        normalized.output += unchanged;
        const size_t length = input_length - input_cursor;
        normalized.mappings.push_back({
            "equal", input_cursor, input_length, output_cursor, output_cursor + length,
            "", "", "",
        });
    }
    if (!validate_safe_normalization(normalized, released_rules, &failure_reason)) {
        return identity_result(input, context, false, failure_reason);
    }
    const std::string decision_reason = preserved_candidate_count == 0
        ? "applied"
        : "applied_with_preserved:"
            + (first_preserve_reason.empty() ? "unspecified" : first_preserve_reason);
    return {std::move(normalized), true, "", decision_reason, {}, 0};
}

bool validate_safe_normalization(
    const SafeNormalizedSegment& normalized,
    const std::vector<ReleasedRule>& released_rules,
    std::string* failure_reason
) {
    const auto input_scalars = scalar_text(normalized.input);
    const auto output_scalars = scalar_text(normalized.output);
    if (!input_scalars.has_value() || !output_scalars.has_value()) {
        set_failure(failure_reason, "invalid_utf8");
        return false;
    }
    if (!valid_rule_registry(released_rules)) {
        set_failure(failure_reason, "invalid_rule_registry");
        return false;
    }

    std::vector<SelectedEdit> approved_edits;
    std::unordered_map<std::string, size_t> candidate_ids;
    std::vector<const TransformationCandidate*> applied_candidates;
    for (const auto& applied : normalized.applied) {
        const auto& candidate = applied.candidate;
        std::string candidate_failure;
        if (!valid_proposed_candidate(normalized.input, *input_scalars, candidate)) {
            set_failure(failure_reason, "invalid_applied_candidate");
            return false;
        }
        const auto* rule = matching_rule(candidate, released_rules);
        if (rule == nullptr) {
            set_failure(failure_reason, "unreleased_transformation");
            return false;
        }
        const auto expected_approval = rule->approver(
            normalized.input,
            normalized.context,
            candidate
        );
        if (!same_approval(expected_approval, applied.approval)
            || !valid_approval(
                normalized.input,
                *input_scalars,
                candidate,
                applied.approval,
                &candidate_failure
            )) {
            set_failure(failure_reason, candidate_failure.empty()
                ? "approver_replay_mismatch" : candidate_failure);
            return false;
        }
        if (!candidate_ids.emplace(candidate.candidate_id, 1).second) {
            set_failure(failure_reason, "duplicate_candidate_id");
            return false;
        }
        applied_candidates.push_back(&candidate);
        for (const auto& edit : applied.approval.edits) {
            approved_edits.push_back({&candidate, &edit});
        }
    }
    std::sort(
        applied_candidates.begin(),
        applied_candidates.end(),
        [](const auto* left, const auto* right) {
            return left->semantic_source.start < right->semantic_source.start;
        }
    );
    for (size_t index = 1; index < applied_candidates.size(); ++index) {
        if (applied_candidates[index]->semantic_source.start
            < applied_candidates[index - 1]->semantic_source.end) {
            set_failure(failure_reason, "overlapping_semantic_ranges");
            return false;
        }
    }
    std::sort(approved_edits.begin(), approved_edits.end(), [](const auto& left, const auto& right) {
        return left.edit->source.start < right.edit->source.start;
    });
    for (size_t index = 1; index < approved_edits.size(); ++index) {
        if (approved_edits[index].edit->source.start < approved_edits[index - 1].edit->source.end) {
            set_failure(failure_reason, "overlapping_atomic_edits");
            return false;
        }
    }

    size_t input_cursor = 0;
    size_t output_cursor = 0;
    size_t replace_count = 0;
    std::string replayed;
    for (const auto& mapping : normalized.mappings) {
        if (mapping.input_start != input_cursor || mapping.output_start != output_cursor
            || mapping.input_end <= mapping.input_start
            || mapping.output_end <= mapping.output_start
            || mapping.input_end >= input_scalars->boundaries.size()
            || mapping.output_end >= output_scalars->boundaries.size()) {
            set_failure(failure_reason, "mapping_coverage_invalid");
            return false;
        }
        const std::string input_piece = scalar_substring(
            normalized.input,
            *input_scalars,
            mapping.input_start,
            mapping.input_end
        );
        const std::string output_piece = scalar_substring(
            normalized.output,
            *output_scalars,
            mapping.output_start,
            mapping.output_end
        );
        if (mapping.kind == "equal") {
            if (input_piece != output_piece || !mapping.candidate_id.empty()
                || !mapping.rule_id.empty() || !mapping.rule_version.empty()) {
                set_failure(failure_reason, "invalid_equal_mapping");
                return false;
            }
            replayed += input_piece;
        } else if (mapping.kind == "replace") {
            const auto found = std::find_if(
                approved_edits.begin(),
                approved_edits.end(),
                [&](const SelectedEdit& selected) { return same_edit(mapping, selected); }
            );
            if (found == approved_edits.end() || output_piece != found->edit->replacement
                || input_piece != found->edit->source.text) {
                set_failure(failure_reason, "untraceable_replacement");
                return false;
            }
            replayed += found->edit->replacement;
            ++replace_count;
        } else {
            set_failure(failure_reason, "unsupported_mapping_kind");
            return false;
        }
        input_cursor = mapping.input_end;
        output_cursor = mapping.output_end;
    }
    const size_t input_length = input_scalars->boundaries.size() - 1;
    const size_t output_length = output_scalars->boundaries.size() - 1;
    if (input_cursor != input_length || output_cursor != output_length
        || replayed != normalized.output || replace_count != approved_edits.size()) {
        set_failure(failure_reason, "normalization_replay_mismatch");
        return false;
    }
    return true;
}

}  // namespace zh_itn::itn
