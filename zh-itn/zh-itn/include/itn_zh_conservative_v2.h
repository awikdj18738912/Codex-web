/*
 * itn_zh_conservative_v2.h - Strict conservative Chinese ITN pipeline.
 */

#ifndef ZH_ITN_ZH_CONSERVATIVE_V2_H
#define ZH_ITN_ZH_CONSERVATIVE_V2_H

#include "itn_safe.h"

namespace zh_itn::itn {

inline constexpr const char* kZhConservativeV2RulePack =
    "zh-itn-strict-conservative-v85";

struct V2SegmentRequest {
    std::string text;
    std::string left_context;
    std::string right_context;
    std::string locale;
    bool has_left_neighbor;
    bool has_right_neighbor;
};

SafeNormalizationResult normalize_zh_conservative_v2(const V2SegmentRequest& request);
SafeNormalizationResult normalize_zh_conservative_v2_shadow_atomic(
    const V2SegmentRequest& request
);

}  // namespace zh_itn::itn

#endif  // ZH_ITN_ZH_CONSERVATIVE_V2_H
