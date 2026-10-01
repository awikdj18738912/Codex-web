#include "zh_itn_bridge.h"

#include <string>
#include <utility>

#include "itn_zh_conservative_v2.h"

namespace {

struct ResultHandle {
    zh_itn::itn::SafeNormalizationResult value;
};

const zh_itn::itn::SafeMapping* mapping_at(const void* handle, size_t index) {
    if (handle == nullptr) return nullptr;
    const auto& mappings = static_cast<const ResultHandle*>(handle)->value.normalized.mappings;
    return index < mappings.size() ? &mappings[index] : nullptr;
}

}  // namespace

extern "C" {

void* zh_itn_normalize_text(const char* text) {
    if (text == nullptr) return nullptr;
    try {
        zh_itn::itn::V2SegmentRequest request{
            text, "", "", "zh-CN", false, false
        };
        return new ResultHandle{
            zh_itn::itn::normalize_zh_conservative_v2(request)
        };
    } catch (...) {
        return nullptr;
    }
}

int zh_itn_result_valid(const void* handle) {
    return handle != nullptr && static_cast<const ResultHandle*>(handle)->value.valid;
}

const char* zh_itn_result_output(const void* handle) {
    return handle == nullptr ? nullptr
        : static_cast<const ResultHandle*>(handle)->value.normalized.output.c_str();
}

const char* zh_itn_result_error(const void* handle) {
    return handle == nullptr ? "zh-itn failed"
        : static_cast<const ResultHandle*>(handle)->value.failure_reason.c_str();
}

size_t zh_itn_result_mapping_count(const void* handle) {
    return handle == nullptr ? 0
        : static_cast<const ResultHandle*>(handle)->value.normalized.mappings.size();
}

const char* zh_itn_mapping_kind(const void* handle, size_t index) {
    const auto* mapping = mapping_at(handle, index);
    return mapping == nullptr ? nullptr : mapping->kind.c_str();
}

const char* zh_itn_mapping_rule_id(const void* handle, size_t index) {
    const auto* mapping = mapping_at(handle, index);
    return mapping == nullptr ? nullptr : mapping->rule_id.c_str();
}

size_t zh_itn_mapping_input_start(const void* handle, size_t index) {
    const auto* mapping = mapping_at(handle, index);
    return mapping == nullptr ? 0 : mapping->input_start;
}

size_t zh_itn_mapping_input_end(const void* handle, size_t index) {
    const auto* mapping = mapping_at(handle, index);
    return mapping == nullptr ? 0 : mapping->input_end;
}

size_t zh_itn_mapping_output_start(const void* handle, size_t index) {
    const auto* mapping = mapping_at(handle, index);
    return mapping == nullptr ? 0 : mapping->output_start;
}

size_t zh_itn_mapping_output_end(const void* handle, size_t index) {
    const auto* mapping = mapping_at(handle, index);
    return mapping == nullptr ? 0 : mapping->output_end;
}

const char* zh_itn_rule_pack(void) {
    return zh_itn::itn::kZhConservativeV2RulePack;
}

void zh_itn_result_free(void* handle) {
    delete static_cast<ResultHandle*>(handle);
}

}  // extern "C"
