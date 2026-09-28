# 中文逆文本规范化（ZH ITN）

这是一个独立的、保守型中文 ITN 工具：将有明确语义依据的中文口语数字等文本改写为书面形式。它只处理文本，不包含语音识别模型、音频解码或分词服务。规则判定不确定时保留原文；不适合当作“把所有中文数字转成阿拉伯数字”的通用转换器。

本目录是单独的源码副本，不依赖仓库内其他源码，也不会改变原应用的 ITN 实现或构建。公开的 C++ 代码位于 `include/` 与 `src/`；命令行程序名为 `zh_itn`。

## 构建

需要 CMake 3.20+、支持 C++17 的编译器。首次配置会下载固定版本的 `nlohmann/json` 3.11.3，并校验 SHA-256；构建后运行程序不需要联网。

以下命令在本目录 `standalone/zh-itn` 中执行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

在 Windows 上可使用 Visual Studio 的 CMake 生成器；程序通常位于 `build/Release/zh_itn.exe`，其他生成器一般位于 `build/zh_itn`。

如果已离线准备好 `nlohmann/json` 3.11.3 的源码，可在配置时加入 `-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/absolute/path/to/json-source`。此路径必须包含该依赖的 `CMakeLists.txt`。

## 命令行用法

程序从标准输入读取**一个完整 JSON 请求**，向标准输出写入**一个 JSON 响应**后退出。不要把它作为逐行、常驻服务使用；多次请求应多次启动，或在自己的程序里直接链接 `zh_itn_core`。标准输入和输出均使用 UTF-8。

```sh
printf '%s' '{"schema_version":3,"locale":"zh-CN","profile":"conservative","segments":[{"id":0,"text":"在二零二六年发布","left_context":"","right_context":"","has_left_neighbor":false,"has_right_neighbor":false}]}' | ./build/zh_itn
```

典型成功响应包含 `schema_version`、`locale`、`profile`、`offset_encoding`、`engine`、`engine_version`、`rule_pack`、`fail_closed` 和 `segments`。每个段返回 `input`、`output`、`status` 与 `mappings`；`mappings` 的位置是 Unicode 标量值下标（`unicode_scalar_v1`），**不是 UTF-8 字节偏移或 UTF-16 下标**。如果调用方使用 JavaScript 字符串索引，必须先做位置转换。`replace` 映射另含规则来源 `provenance`。

请求约束：`schema_version` 必须为 `3`，`locale` 必须为 `zh-CN`，`profile` 必须为 `conservative`；`segments` 的 `id` 从 0 连续递增。`left_context`、`right_context` 和邻居布尔值可省略，默认空串与 `false`。上下文用于边界判断，**不属于被改写的文本**。可选 `audit_diagnostics: true` 输出更多审计信息。单个请求上限 64 MiB、10,000 段；单段及其上下文的字节数上限为 1 MiB，上下文最多 32 个 Unicode 标量值。

请求解析失败或参数不支持时，程序以非零状态退出，并在 stdout 返回 `{"schema_version":3,"error":"..."}`。接入时应检查进程退出状态和响应结构；不要把错误响应当作成功规范化结果。

## C++ 库用法

链接 `zh_itn_core`，包含 `itn_zh_conservative_v2.h`，调用 `zh_itn::itn::normalize_zh_conservative_v2`。例如：

```cpp
#include "itn_zh_conservative_v2.h"

zh_itn::itn::V2SegmentRequest request{
    "在二零二六年发布", "", "", "zh-CN", false, false
};
auto result = zh_itn::itn::normalize_zh_conservative_v2(request);
if (result.valid) {
    // result.normalized.output 是规范化结果。
}
```

其他项目可使用 `add_subdirectory`：

```cmake
add_subdirectory(path/to/zh-itn)
target_link_libraries(your_target PRIVATE zh_itn_core)
```

若只需要库而不需要运行副本的测试，可在配置时设置 `-DBUILD_TESTING=OFF`。当前 CMake 仍会构建命令行程序，供集成测试和人工验证使用。
