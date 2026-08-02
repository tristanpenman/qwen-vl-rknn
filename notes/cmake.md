# CMake

The current `CMakeLists.txt` is generally solid.

It address a number of CMake best practices:

* target-oriented
* pinning dependency releases
* use of namespaced aliases
* support for configurable runtime paths
* handles single-configuration builds correctly
* scopes server dependencies

The following improvements are recommended, roughly in priority order.

## Fix or remove the OpenCV option

`VLM_RKNN_ENABLE_OPENCV=OFF` cannot currently work. `vlm_rknn.cc`, `main.cc`, the tests, and the public `vlm_rknn.h` header all include OpenCV unconditionally.

Either treat OpenCV as mandatory and remove the option, or conditionally compile all OpenCV-dependent APIs and sources. As written, the option advertises a configuration that will fail to compile.

## Model RKNN and RKLLM as imported targets

Instead of passing raw paths and include directories separately, represent each prebuilt runtime as an imported target:

```cmake
add_library(RKNN::Runtime SHARED IMPORTED)
set_target_properties(RKNN::Runtime PROPERTIES
    IMPORTED_LOCATION "${RKNN_RUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${RKNN_INCLUDE_DIR}"
)
```

Do likewise for RKLLM, then link the targets:

```cmake
target_link_libraries(vlm_rknn_core PUBLIC
    RKNN::Runtime
    RKLLM::Runtime
)
```

This encapsulates each dependency's location and usage requirements, makes missing or incorrect targets easier to diagnose, and avoids parallel variables drifting apart. Imported targets are CMake's intended representation for external prebuilt libraries. See the [CMake imported-library documentation](https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries).

## Pin fetched dependencies to commit hashes

The tags are versioned, which is already better than tracking branches, but Git tags can be moved. Use immutable commit hashes for OpenCV, cpp-httplib, and nlohmann/json, ideally with comments recording their corresponding release tags.

CMake explicitly recommends hashes for remotely fetched repositories for security and reproducibility. See the [FetchContent documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html).

## Express C++17 as a target usage requirement

The global settings apply to every subsequently created target, including fetched third-party targets. Prefer:

```cmake
target_compile_features(vlm_rknn_core PUBLIC cxx_std_17)
set_target_properties(vlm_rknn_core PROPERTIES CXX_EXTENSIONS OFF)
```

`PUBLIC` is appropriate because the public header itself contains C++17 types such as `std::optional` and `std::string_view`. Consumers then automatically inherit the minimum standard. See the [CMake compile-features documentation](https://cmake.org/cmake/help/latest/command/target_compile_features.html).

## Make `VLM_RKNN_TARGET` private

The macro appears to be used only inside `vlm_rknn.cc`, not in public headers:

```cmake
target_compile_definitions(vlm_rknn_core PRIVATE
    VLM_RKNN_TARGET="${VLM_RKNN_TARGET}"
)
```

Leaving it `PUBLIC` unnecessarily injects the macro into every executable that links the core library.

## Quote filesystem paths consistently

For example:

```cmake
if(NOT EXISTS "${RKNN_RUNTIME_LIB}")
if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
```

Also quote defaults passed to cache variables. This makes paths containing spaces or list separators safer and communicates that each value is one path.

## Decide whether `vlm_rknn_core` is internal or reusable

Its current public interface exposes the entire `cpp/src` directory as well as OpenCV, RKNN, and RKLLM. That is technically consistent with `vlm_rknn.h`, which includes all three dependencies, but it also exposes private headers such as `logger.h` and `ini.h`.

If reuse is intended, introduce something like `cpp/include/vlm-rknn/vlm_rknn.h` and use `BUILD_INTERFACE` and `INSTALL_INTERFACE` include directories. CMake supports these specifically for clean build-tree and install-tree interfaces. See the [CMake include-directory documentation](https://cmake.org/cmake/help/latest/command/target_include_directories.html).

If the library is purely internal, the current layout is acceptable, although `PUBLIC` should then be understood as build-internal propagation rather than a stable API promise.

## Avoid duplicating the core target in the Android app

`android/app/src/main/cpp/CMakeLists.txt` repeats the core sources, dependency setup, target definition, and C++ settings. That is likely to drift from the root configuration.

A useful longer-term refactor would extract core target creation into a shared CMake module or make the Android project consume the root target via `add_subdirectory()`. The OpenCV option and runtime-target work would then only need to be maintained once.

## Smaller cleanups

- `COPYRIGHT` and `IDENTIFIER` are currently unused and can be removed until packaging needs them.
- `LANGUAGES C CXX` can probably become `LANGUAGES CXX`; there are no project C sources.
- The final build-type status message is blank for multi-config generators; report `CMAKE_CONFIGURATION_TYPES` in that case.
- Consider `CMakePresets.json` to centralize native and Android configurations, although the existing build scripts already provide much of that value.
- The explicit `-Wl,--strip-all` works for the current GNU/LLVM-style toolchains but is toolchain-specific. Guard it by platform or compiler, or perform stripping as a packaging step.
