cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(zh_itn VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options(/utf-8 /EHsc)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(zh_itn_core STATIC
  src/itn_safe.cc
  src/itn_zh_conservative_v2.cc
)
target_include_directories(zh_itn_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(zh_itn src/zh_itn_cli.cc)
target_link_libraries(zh_itn PRIVATE zh_itn_core nlohmann_json::nlohmann_json)

include(CTest)
if(BUILD_TESTING)
  add_executable(zh_itn_safe_test tests/itn_safe_test.cc)
  target_link_libraries(zh_itn_safe_test PRIVATE zh_itn_core)
  add_test(NAME zh_itn_safe COMMAND zh_itn_safe_test)

  add_executable(zh_itn_rules_test tests/itn_zh_conservative_v2_test.cc)
  target_link_libraries(zh_itn_rules_test PRIVATE zh_itn_core)
  add_test(NAME zh_itn_rules COMMAND zh_itn_rules_test)

  add_test(NAME zh_itn_cli COMMAND "${CMAKE_COMMAND}"
    "-DZH_ITN_EXE=$<TARGET_FILE:zh_itn>"
    "-DZH_ITN_REQUEST=${CMAKE_CURRENT_SOURCE_DIR}/tests/request.json"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_smoke.cmake")
endif()
