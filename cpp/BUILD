# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "rtx_sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "rtx_win_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "rtx_win_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

cc_library(
    name = "torch_tensorrt",
    srcs = [
        "src/compile_spec.cpp",
        "src/logging.cpp",
        "src/torch_tensorrt.cpp",
        "src/types.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/logging.h",
        "include/torch_tensorrt/macros.h",
        "include/torch_tensorrt/torch_tensorrt.h",
    ],
    linkstatic = True,
    strip_include_prefix = "include/",
    deps = [
        "//core",
        "//core/ir",
        "//core/util:prelude",
    ],
    alwayslink = True,
)

cc_library(
    name = "tensorrt_executorch_blob_header",
    srcs = [
        "src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBlobHeader.h",
    ],
    strip_include_prefix = "include",
)

cc_library(
    name = "tensorrt_executorch_weight_streaming_budget",
    srcs = [
        "src/torch_tensorrt/executorch/WeightStreamingBudget.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/WeightStreamingBudget.h",
    ],
    strip_include_prefix = "include",
)

cc_library(
    name = "tensorrt_executorch_binding_names",
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBindingNames.h",
    ],
    strip_include_prefix = "include",
    deps = [
        "//core/runtime:tensorrt_binding_names",
    ],
)

# Registration only, kept out of :tensorrt_executorch_backend so that linking
# the backend into both a shared library and an executable cannot register the
# allocator twice. Executables that run a TensorRT delegated program depend on
# this directly.
#
# Not part of :executorch_backend_source_files. The CMake build of the reference
# runner enables the CUDA/AOTI delegate, which already registers this allocator.
cc_library(
    name = "tensorrt_executorch_cuda_device_allocator",
    srcs = [
        "src/torch_tensorrt/executorch/RegisterCudaDeviceAllocator.cpp",
    ],
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": [
            "@executorch//:executorch_headers",
            "@executorch//:extension_cuda",
        ],
        ":sbsa": [
            "@executorch//:executorch_headers",
            "@executorch//:extension_cuda",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

# Registration only, for the same reason as the allocator above: the kernel
# registry aborts when the same kernel is registered twice, so this cannot ride
# along with the backend. Executables that run a TensorRT delegated program
# depend on this directly.
#
# Not part of :executorch_backend_source_files. The CMake build of the reference
# runner links executorch::kernels, which already registers these.
cc_library(
    name = "tensorrt_executorch_device_copy_kernels",
    srcs = [
        "src/torch_tensorrt/executorch/RegisterDeviceCopyKernels.cpp",
    ],
    deps = [
        "@executorch//:executorch_device_copy_kernels",
        "@executorch//:executorch_headers",
    ],
    alwayslink = True,
)

# Implementation detail of :tensorrt_executorch_backend, so the header sits under
# src/ and ships with the sources rather than in the installed API set. A caller
# turning the pool on needs only the option key, which is in TensorRTBackend.h.
cc_library(
    name = "tensorrt_executorch_shared_scratch_pool",
    hdrs = [
        "src/torch_tensorrt/executorch/SharedScratchPool.h",
    ],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": [
            "@cuda//:cuda_headers",
        ],
        ":sbsa": [
            "@cuda//:cuda_headers",
        ],
        "//conditions:default": [],
    }),
)

# Implementation detail of :tensorrt_executorch_backend, under src/ like the pool
# header above. Its own target so the sharing key can be tested without a GPU.
cc_library(
    name = "tensorrt_executorch_shared_engine_key",
    hdrs = [
        "src/torch_tensorrt/executorch/SharedEngineKey.h",
    ],
    strip_include_prefix = "src",
)

# The one internal entry point of :tensorrt_executorch_backend that its test calls
# directly. Under src/ for the same reason as the pool header above: it is an
# implementation detail of execute(), and an installed header declaring it would
# make it API from the next release.
#
# It carries the definition and not only the declaration, so the target stands on
# its own: a header target whose symbol another target defines links only for a
# consumer that happens to depend on that one too.
cc_library(
    name = "tensorrt_executorch_pooled_scratch_install",
    srcs = [
        "src/torch_tensorrt/executorch/PooledScratchInstall.cpp",
    ],
    hdrs = [
        "src/torch_tensorrt/executorch/PooledScratchInstall.h",
    ],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": [
            "@executorch//:executorch_headers",
            "@tensorrt//:nvinfer",
        ],
        ":sbsa": [
            "@executorch//:executorch_headers",
            "@tensorrt_sbsa//:nvinfer",
        ],
        "//conditions:default": [],
    }),
)

# The body of the pool's test-only reset. testonly and under src/, so it is in
# neither the packaged include tree nor executorch_backend_source_files below: a
# release build has no definition of it anywhere. SharedScratchPool.h keeps only
# the friend declaration that lets this reach the registry's internals.
cc_library(
    name = "tensorrt_executorch_shared_scratch_pool_reset",
    testonly = True,
    hdrs = [
        "src/torch_tensorrt/executorch/SharedScratchPoolReset.h",
    ],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":tensorrt_executorch_shared_scratch_pool",
    ],
)

# The pool's test-only entry points. A separate target, and testonly, so the
# released archive -- Bazel's :tensorrt_executorch_backend and CMake's
# executorch_trt_backend, neither of which compiles this source -- carries no
# definition of them and exports no symbol for them. One of them frees the live
# pool with no wait for work in flight, which is not something a release build
# should offer a caller.
cc_library(
    name = "tensorrt_executorch_shared_scratch_pool_test_hooks",
    testonly = True,
    srcs = [
        "src/torch_tensorrt/executorch/SharedScratchPoolTestHooks.cpp",
    ],
    hdrs = [
        "src/torch_tensorrt/executorch/SharedScratchPoolTestHooks.h",
    ],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":tensorrt_executorch_shared_scratch_pool",
        ":tensorrt_executorch_shared_scratch_pool_reset",
    ] + select({
        ":linux_x86_64": [
            "@cuda//:cudart",
        ],
        ":sbsa": [
            "@cuda//:cudart",
        ],
        "//conditions:default": [],
    }),
)

# Kept out of release archives and source packages, like the scratch pool hooks.
cc_library(
    name = "tensorrt_executorch_shared_engine_test_hooks",
    testonly = True,
    srcs = [
        "src/torch_tensorrt/executorch/SharedEngineTestHooks.cpp",
    ],
    hdrs = [
        "src/torch_tensorrt/executorch/SharedEngineTestHooks.h",
    ],
    strip_include_prefix = "src",
    deps = [
        ":tensorrt_executorch_backend",
    ],
)

cc_library(
    name = "tensorrt_executorch_execution_graph_headers",
    hdrs = ["src/torch_tensorrt/executorch/ExecutionGraph.h"],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:private"],
    deps = ["@cuda//:cuda_headers"] + select({
        ":linux_x86_64": [
            "@executorch//:executorch_headers",
            "@tensorrt//:nvinfer",
        ],
        ":sbsa": [
            "@executorch//:executorch_headers",
            "@tensorrt_sbsa//:nvinfer",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensorrt_executorch_optimization_profile_selection",
    hdrs = [
        "include/torch_tensorrt/executorch/OptimizationProfileSelection.h",
    ],
    strip_include_prefix = "include",
    # The header includes <NvInfer.h>, so it cannot build where the deps below
    # resolve to an empty list.
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": ["@tensorrt//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensorrt_executorch_backend",
    srcs = [
        # Private, deliberately not in hdrs: EngineHandle grows fields as the
        # backend gains features and is never installed, so nothing outside this
        # library may depend on its layout.
        "src/torch_tensorrt/executorch/EngineHandle.h",
        "src/torch_tensorrt/executorch/ExecutionGraph.cpp",
        "src/torch_tensorrt/executorch/TensorRTBackend.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBackend.h",
    ],
    strip_include_prefix = "include",
    # Build the TensorRT backend as a static library. The final application
    # links this target together with the ExecuTorch runtime it was compiled
    # against, avoiding any runtime plugin/dlopen dependency.
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":tensorrt_executorch_binding_names",
        ":tensorrt_executorch_blob_header",
        ":tensorrt_executorch_execution_graph_headers",
        ":tensorrt_executorch_optimization_profile_selection",
        ":tensorrt_executorch_pooled_scratch_install",
        ":tensorrt_executorch_shared_engine_key",
        ":tensorrt_executorch_shared_scratch_pool",
        ":tensorrt_executorch_weight_streaming_budget",
    ] + select({
        ":linux_x86_64": [
            "@cuda//:cudart",
            "@executorch//:executorch_headers",
            "@executorch//:extension_cuda",
            "@tensorrt//:nvinfer",
        ],
        ":sbsa": [
            "@cuda//:cudart",
            "@executorch//:executorch_headers",
            "@executorch//:extension_cuda",
            "@tensorrt_sbsa//:nvinfer",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

# For tests that inspect a live handle's state. testonly keeps the private
# EngineHandle layout from becoming something a shipped target can depend on.
cc_library(
    name = "tensorrt_executorch_engine_handle_test_access",
    testonly = True,
    hdrs = [
        "src/torch_tensorrt/executorch/EngineHandle.h",
    ],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [":tensorrt_executorch_backend"],
)

filegroup(
    name = "executorch_backend_source_files",
    srcs = [
        "src/torch_tensorrt/executorch/CMakeLists.txt",
        "src/torch_tensorrt/executorch/EngineHandle.h",
        "src/torch_tensorrt/executorch/ExecutionGraph.cpp",
        "src/torch_tensorrt/executorch/ExecutionGraph.h",
        "src/torch_tensorrt/executorch/PooledScratchInstall.cpp",
        "src/torch_tensorrt/executorch/PooledScratchInstall.h",
        "src/torch_tensorrt/executorch/README.md",
        "src/torch_tensorrt/executorch/SharedEngineKey.h",
        "src/torch_tensorrt/executorch/SharedScratchPool.h",
        "src/torch_tensorrt/executorch/TensorRTBackend.cpp",
        "src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp",
        "src/torch_tensorrt/executorch/WeightStreamingBudget.cpp",
    ],
)

filegroup(
    name = "api_headers",
    srcs = glob(
        ["include/**/*.h"],
        exclude = [
            "include/torch_tensorrt/executorch/**/*.h",
        ],
    ),
)

filegroup(
    name = "executorch_api_headers",
    srcs = [
        "include/torch_tensorrt/executorch/OptimizationProfileSelection.h",
        "include/torch_tensorrt/executorch/TensorRTBackend.h",
        "include/torch_tensorrt/executorch/TensorRTBindingNames.h",
        "include/torch_tensorrt/executorch/TensorRTBlobHeader.h",
        "include/torch_tensorrt/executorch/WeightStreamingBudget.h",
    ],
)
