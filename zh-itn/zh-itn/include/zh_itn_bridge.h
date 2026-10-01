#ifndef ZH_ITN_BRIDGE_H
#define ZH_ITN_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The returned handle owns all strings until zh_itn_result_free is called. */
void* zh_itn_normalize_text(const char* text);
int zh_itn_result_valid(const void* handle);
const char* zh_itn_result_output(const void* handle);
const char* zh_itn_result_error(const void* handle);
size_t zh_itn_result_mapping_count(const void* handle);
const char* zh_itn_mapping_kind(const void* handle, size_t index);
const char* zh_itn_mapping_rule_id(const void* handle, size_t index);
size_t zh_itn_mapping_input_start(const void* handle, size_t index);
size_t zh_itn_mapping_input_end(const void* handle, size_t index);
size_t zh_itn_mapping_output_start(const void* handle, size_t index);
size_t zh_itn_mapping_output_end(const void* handle, size_t index);
const char* zh_itn_rule_pack(void);
void zh_itn_result_free(void* handle);

#ifdef __cplusplus
}
#endif

#endif
