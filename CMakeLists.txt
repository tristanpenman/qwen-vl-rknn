cmake_minimum_required(VERSION 3.20)
project(vlm-rknn VERSION 0.1.0 LANGUAGES C CXX)

set(COPYRIGHT "Copyright (c) 2026 Tristan Penman")
set(IDENTIFIER "com.tristanpenman.vlm-rknn")

# Force out-of-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid language extensions to ensure consistent behavior across compilers
set(CMAKE_CXX_EXTENSIONS OFF)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Default to Debug for single-config generators if user didn't specify anything
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# build config
# -----------------------------------------------------------------------------

option(VLM_RKNN_ENABLE_OPENCV "Fetch and link OpenCV as a third-party dependency" ON)
option(VLM_RKNN_ENABLE_SERVER "Build the HTTP server executable" ON)

set(VLM_RKNN_TARGET "rk3588" CACHE STRING "Rockchip target device family")

if(ANDROID)
    set(VLM_RKNN_ARCH ${CMAKE_ANDROID_ARCH_ABI})
    set(DEFAULT_RKNN_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/lib-android/${VLM_RKNN_ARCH}/librknnrt.so)
    set(DEFAULT_RKLLM_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rkllm/lib-android/${VLM_RKNN_ARCH}/librkllmrt.so)
else()
    set(VLM_RKNN_ARCH aarch64)
    set(DEFAULT_RKNN_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/lib/${VLM_RKNN_ARCH}/librknnrt.so)
    set(DEFAULT_RKLLM_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rkllm/lib/${VLM_RKNN_ARCH}/librkllmrt.so)
endif()

# -----------------------------------------------------------------------------
# rknn runtime
# -----------------------------------------------------------------------------

set(RKNN_RUNTIME_LIB ${DEFAULT_RKNN_RUNTIME_LIB} CACHE FILEPATH
    "Path to librknnrt.so for the selected target")
set(RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/include CACHE PATH
    "Path containing rknn_api.h")

if(NOT EXISTS ${RKNN_RUNTIME_LIB})
    message(FATAL_ERROR
        "RKNN runtime library not found: ${RKNN_RUNTIME_LIB}. "
        "Set -DRKNN_RUNTIME_LIB=<path-to-librknnrt.so> (for Android use arm64-v8a runtime).")
endif()

if(NOT EXISTS ${RKNN_INCLUDE_DIR}/rknn_api.h)
    message(FATAL_ERROR
        "RKNN headers not found in ${RKNN_INCLUDE_DIR}. "
        "Set -DRKNN_INCLUDE_DIR=<path-containing-rknn_api.h>.")
endif()

# -----------------------------------------------------------------------------
# rkllm runtime
# -----------------------------------------------------------------------------

set(RKLLM_RUNTIME_LIB ${DEFAULT_RKLLM_RUNTIME_LIB} CACHE FILEPATH
    "Path to librkllmrt.so for the selected target")
set(RKLLM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rkllm/include CACHE PATH
    "Path containing rkllm.h")

if(NOT EXISTS ${RKLLM_RUNTIME_LIB})
    message(FATAL_ERROR
        "RKLLM runtime library not found: ${RKLLM_RUNTIME_LIB}. "
        "Set -DRKLLM_RUNTIME_LIB=<path-to-librkllmrt.so> (for Android use arm64-v8a runtime).")
endif()

if(NOT EXISTS ${RKLLM_INCLUDE_DIR}/rkllm.h)
    message(FATAL_ERROR
        "RKLLM headers not found in ${RKLLM_INCLUDE_DIR}. "
        "Set -DRKLLM_INCLUDE_DIR=<path-containing-rkllm.h>.")
endif()

# -----------------------------------------------------------------------------
# OpenCV
# -----------------------------------------------------------------------------

if(VLM_RKNN_ENABLE_OPENCV)
    include(cmake/OpenCV.cmake)
endif()

# -----------------------------------------------------------------------------
# JSON for Modern C++
# -----------------------------------------------------------------------------

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# -----------------------------------------------------------------------------
# SentencePiece
# -----------------------------------------------------------------------------

set(SPM_ENABLE_SHARED OFF CACHE BOOL INTERNAL)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL INTERNAL)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG 11051e3b73b3a6222a52acd720e39805dc7545ab
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(sentencepiece)
set(LIBSENTENCEPIECE_INCLUDES ${sentencepiece_SOURCE_DIR}/src)

# Suppress warning in SentencePiece builds using GCC.
target_compile_options(sentencepiece-static PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>
)

# -----------------------------------------------------------------------------
# HTTP server dependencies
# -----------------------------------------------------------------------------

if(VLM_RKNN_ENABLE_SERVER)
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "Disable cpp-httplib OpenSSL support" FORCE)
    set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "Disable cpp-httplib zlib support" FORCE)
    set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "Disable cpp-httplib Brotli support" FORCE)
    set(HTTPLIB_INSTALL OFF CACHE BOOL "Disable cpp-httplib install rules" FORCE)

    FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.18.6
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )

    FetchContent_MakeAvailable(httplib)
endif()

# -----------------------------------------------------------------------------
# vlm-rknn-core
# -----------------------------------------------------------------------------

add_library(vlm-rknn-core STATIC
    cpp/src/ini.cc
    cpp/src/logger.cc
    cpp/src/rknn_utils.cc
    cpp/src/vlm_rknn.cc
)

add_library(vlm-rknn::core ALIAS vlm-rknn-core)

target_compile_definitions(vlm-rknn-core PUBLIC
    VLM_RKNN_TARGET="${VLM_RKNN_TARGET}"
)

target_include_directories(vlm-rknn-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
    ${RKNN_INCLUDE_DIR}
    ${RKLLM_INCLUDE_DIR}
    ${LIBSENTENCEPIECE_INCLUDES}
    ${VLM_RKNN_OPENCV_INCLUDE_DIRS}
)

target_link_libraries(vlm-rknn-core PUBLIC
    ${RKNN_RUNTIME_LIB}
    ${RKLLM_RUNTIME_LIB}
    nlohmann_json::nlohmann_json
    sentencepiece
    ${VLM_RKNN_OPENCV_TARGETS}
)

# -----------------------------------------------------------------------------
# vlm-rknn
# -----------------------------------------------------------------------------

add_executable(vlm-rknn
    cpp/src/main.cc
)

# OpenCV's CMake sets CMAKE_RUNTIME_OUTPUT_DIRECTORY to <build>/bin
set_target_properties(vlm-rknn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(vlm-rknn PRIVATE vlm-rknn::core)

# Strip debug symbols from release builds
target_link_options(vlm-rknn PRIVATE
    $<$<CONFIG:Release>:-Wl,--strip-all>
)

# -----------------------------------------------------------------------------
# vlm-rknn-cache
# -----------------------------------------------------------------------------

add_executable(vlm-rknn-cache
    cpp/src/cache.cc
)

set_target_properties(vlm-rknn-cache PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(vlm-rknn-cache PRIVATE vlm-rknn::core)

target_link_options(vlm-rknn-cache PRIVATE
    $<$<CONFIG:Release>:-Wl,--strip-all>
)

# -----------------------------------------------------------------------------
# vlm-rknn-server
# -----------------------------------------------------------------------------

if(VLM_RKNN_ENABLE_SERVER)
    add_executable(vlm-rknn-server
        cpp/src/registry.cc
        cpp/src/server.cc
    )

    set_target_properties(vlm-rknn-server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    target_link_libraries(vlm-rknn-server PRIVATE
        vlm-rknn::core
        httplib::httplib
        nlohmann_json::nlohmann_json
    )

    target_link_options(vlm-rknn-server PRIVATE
        $<$<CONFIG:Release>:-Wl,--strip-all>
    )
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

include(CTest)
if(BUILD_TESTING AND NOT ANDROID)
    add_executable(vlm-rknn-tests
        cpp/tests/vlm_rknn_test.cc
    )
    target_link_libraries(vlm-rknn-tests PRIVATE vlm-rknn::core)
    add_test(NAME vlm-rknn COMMAND vlm-rknn-tests)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator: ${CMAKE_GENERATOR} (multi-config: ${is_multi_config})")
message(STATUS "Rockchip target: ${VLM_RKNN_TARGET}")
message(STATUS "OpenCV enabled: ${VLM_RKNN_ENABLE_OPENCV}")
message(STATUS "RKNN runtime library: ${RKNN_RUNTIME_LIB}")
