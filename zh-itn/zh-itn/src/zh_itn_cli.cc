/*
 * zh_itn_cli.cc - Strict fail-closed ITN sidecar protocol.
 */

#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "itn_zh_conservative_v2.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

using json = nlohmann::json;
using namespace zh_itn::itn;

constexpr size_t kSchemaVersion = 3;
constexpr size_t kMaxRequestBytes = 64 * 1024 * 1024;
constexpr size_t kMaxSegments = 10000;
constexpr size_t kMaxSegmentBytes = 1024 * 1024;
constexpr size_t kMaxContextCharacters = 32;

json span_json(
    const std::string& text,
    const std::vector<size_t>& boundaries,
    size_t start,
    size_t end
) {
    return {
        {"start", start},
        {"end", end},
        {"text", unicode_scalar_substring(text, boundaries, start, end)},
    };
}

json mapping_json(
    const SafeMapping& mapping,
    const SafeNormalizedSegment& normalized,
    const std::vector<size_t>& input_boundaries,
    const std::vector<size_t>& output_boundaries
) {
    const std::string output_text = unicode_scalar_substring(
        normalized.output,
        output_boundaries,
        mapping.output_start,
        mapping.output_end
    );
    json value = {
        {"kind", mapping.kind},
        {"input", span_json(
            normalized.input,
            input_boundaries,
            mapping.input_start,
            mapping.input_end
        )},
        {"output", span_json(
            normalized.output,
            output_boundaries,
            mapping.output_start,
            mapping.output_end
        )},
    };
    if (mapping.kind == "replace") {
        const AppliedTransformation* approved = nullptr;
        size_t matching_edits = 0;
        for (const auto& applied : normalized.applied) {
            if (applied.candidate.candidate_id != mapping.candidate_id
                || applied.candidate.rule_id != mapping.rule_id
                || applied.candidate.rule_version != mapping.rule_version) {
                continue;
            }
            for (const auto& edit : applied.approval.edits) {
                if (edit.source.start == mapping.input_start
                    && edit.source.end == mapping.input_end
                    && edit.replacement == output_text) {
                    approved = &applied;
                    ++matching_edits;
                }
            }
        }
        if (approved == nullptr || matching_edits != 1) {
            throw std::runtime_error("replace mapping is not backed by one approved atomic edit");
        }
        json evidence = json::array();
        for (const auto& item : approved->approval.evidence) {
            evidence.push_back({
                {"kind", item.kind},
                {"value", item.value},
                {"source", item.source},
            });
        }
        value["provenance"] = {
            {"candidate_id", mapping.candidate_id},
            {"rule_id", mapping.rule_id},
            {"rule_version", mapping.rule_version},
            {"locale", approved->candidate.locale},
            {"semantic_type", approved->candidate.semantic_type},
            {"subtype", approved->candidate.subtype},
            {"semantic_source", {
                {"start", approved->candidate.semantic_source.start},
                {"end", approved->candidate.semantic_source.end},
                {"text", approved->candidate.semantic_source.text},
            }},
            {"parsed_value", approved->approval.parsed_value},
            {"reason", approved->approval.reason},
            {"evidence", std::move(evidence)},
        };
    }
    return value;
}

bool valid_context(const std::string& value) {
    if (value.size() > kMaxSegmentBytes) return false;
    const auto boundaries = unicode_scalar_boundaries(value);
    return boundaries.has_value() && boundaries->size() - 1 <= kMaxContextCharacters;
}

json shadow_atomic_json(const SafeNormalizationResult& result) {
    size_t replacement_count = 0;
    for (const auto& applied : result.normalized.applied) {
        replacement_count += applied.approval.edits.size();
    }
    json value = {
        {"valid", result.valid},
        {"status", result.valid
            ? (result.normalized.applied.empty() ? "unchanged" : "applied")
            : "error"},
        {"output", result.normalized.output},
        {"decision_reason", result.decision_reason.empty()
            ? (result.normalized.applied.empty() ? "unchanged" : "applied")
            : result.decision_reason},
        {"replacement_count", replacement_count},
        {"mappings", json::array()},
    };
    if (!result.valid) {
        value["error"] = result.failure_reason;
        return value;
    }
    const auto input_boundaries = unicode_scalar_boundaries(result.normalized.input);
    const auto output_boundaries = unicode_scalar_boundaries(result.normalized.output);
    if (!input_boundaries.has_value() || !output_boundaries.has_value()) {
        throw std::runtime_error("shadow atomic result contains invalid UTF-8");
    }
    for (const auto& mapping : result.normalized.mappings) {
        value["mappings"].push_back(mapping_json(
            mapping,
            result.normalized,
            *input_boundaries,
            *output_boundaries
        ));
    }
    return value;
}

json normalize_request(const json& request) {
    if (!request.is_object() || request.value("schema_version", 0) != kSchemaVersion) {
        throw std::runtime_error("unsupported request schema version");
    }
    if (request.value("locale", "") != "zh-CN") {
        throw std::runtime_error("unsupported ITN locale");
    }
    if (request.value("profile", "") != "conservative") {
        throw std::runtime_error("unsupported ITN output profile");
    }
    if (!request.contains("segments") || !request["segments"].is_array()) {
        throw std::runtime_error("request must contain a segments array");
    }
    if (request.contains("audit_diagnostics")
        && !request["audit_diagnostics"].is_boolean()) {
        throw std::runtime_error("audit_diagnostics must be a boolean");
    }
    const bool audit_diagnostics = request.value("audit_diagnostics", false);
    if (request["segments"].size() > kMaxSegments) {
        throw std::runtime_error("request contains too many segments");
    }

    json response = {
        {"schema_version", kSchemaVersion},
        {"locale", "zh-CN"},
        {"profile", "conservative"},
        {"offset_encoding", "unicode_scalar_v1"},
        {"engine", "ZhITN"},
        {"engine_version", "2.0.0"},
        {"rule_pack", kZhConservativeV2RulePack},
        {"fail_closed", true},
        {"segments", json::array()},
    };

    for (size_t index = 0; index < request["segments"].size(); ++index) {
        const auto& source = request["segments"][index];
        if (!source.is_object() || !source.contains("id")
            || !source["id"].is_number_unsigned()
            || source["id"].get<size_t>() != index
            || !source.contains("text") || !source["text"].is_string()) {
            throw std::runtime_error("invalid ITN segment identity");
        }
        const std::string text = source["text"].get<std::string>();
        const std::string left_context = source.value("left_context", "");
        const std::string right_context = source.value("right_context", "");
        if (text.size() > kMaxSegmentBytes || !valid_context(left_context)
            || !valid_context(right_context)) {
            throw std::runtime_error("ITN segment or context is too large");
        }
        const V2SegmentRequest segment_request = {
            text,
            left_context,
            right_context,
            "zh-CN",
            source.value("has_left_neighbor", false),
            source.value("has_right_neighbor", false),
        };
        const auto normalized = normalize_zh_conservative_v2(segment_request);
        json segment = {
            {"id", index},
            {"input", text},
            {"output", normalized.normalized.output},
            {"mappings", json::array()},
        };
        if (audit_diagnostics) {
            const auto shadow = normalize_zh_conservative_v2_shadow_atomic(segment_request);
            json duplicate_rule_candidate_counts = json::array();
            for (const auto& [rule_id, count]
                : normalized.audit_duplicate_rule_candidate_counts) {
                duplicate_rule_candidate_counts.push_back({
                    {"rule_id", rule_id},
                    {"candidate_count", count},
                });
            }
            segment["diagnostics"] = {
                {"decision_reason", normalized.decision_reason.empty()
                    ? (normalized.normalized.applied.empty() ? "unchanged" : "applied")
                    : normalized.decision_reason},
                {"duplicate_rule_candidate_counts", std::move(
                    duplicate_rule_candidate_counts
                )},
                {"detected_candidate_count", normalized.audit_total_candidate_count},
                {"shadow_atomic", shadow_atomic_json(shadow)},
            };
        }
        if (!normalized.valid) {
            segment["status"] = "error";
            segment["error_code"] = "fail_closed";
            segment["error"] = normalized.failure_reason;
            response["segments"].push_back(std::move(segment));
            continue;
        }
        const auto input_boundaries = unicode_scalar_boundaries(normalized.normalized.input);
        const auto output_boundaries = unicode_scalar_boundaries(normalized.normalized.output);
        if (!input_boundaries.has_value() || !output_boundaries.has_value()) {
            throw std::runtime_error("validated ITN result contains invalid UTF-8");
        }
        for (const auto& mapping : normalized.normalized.mappings) {
            segment["mappings"].push_back(mapping_json(
                mapping,
                normalized.normalized,
                *input_boundaries,
                *output_boundaries
            ));
        }
        segment["status"] = normalized.normalized.applied.empty()
            ? "unchanged"
            : "applied";
        response["segments"].push_back(std::move(segment));
    }
    return response;
}

}  // namespace

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        std::ostringstream buffer;
        buffer << std::cin.rdbuf();
        const std::string payload = buffer.str();
        if (payload.size() > kMaxRequestBytes) {
            throw std::runtime_error("ITN request is too large");
        }
        std::cout << normalize_request(json::parse(payload)).dump();
        return 0;
    } catch (const std::exception& error) {
        std::cout << json({
            {"schema_version", kSchemaVersion},
            {"error", error.what()},
        }).dump();
        return 1;
    }
}
