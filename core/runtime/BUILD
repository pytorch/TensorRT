load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")
load("//toolchains/torch_nccl:defs.bzl", "if_torch_nccl")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl",
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
    name = "rtx_win",
    constraint_values = [
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
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

cc_library(
    name = "runtime",
    srcs = [
        "DeviceList.cpp",
        "Platform.cpp",
        "RTDevice.cpp",
        "TRTEngine.cpp",
        "TRTEngineProfiler.cpp",
        "TRTRuntimeConfig.cpp",
        "execute_engine.cpp",
        "register_jit_hooks.cpp",
        "runtime.cpp",
    ],
    hdrs = [
        "Platform.h",
        "RTDevice.h",
        "TRTEngine.h",
        "TRTEngineProfiler.h",
        "TRTRuntimeConfig.h",
        "runtime.h",
    ],
    copts = if_torch_nccl(["-DUSE_C10D_NCCL"]),
    defines = select({
        # nvinfer1::IRuntimeConfig (and the matching ICudaEngine::createRuntimeConfig
        # / createExecutionContext(IRuntimeConfig*) overloads) was introduced in
        # TensorRT 10.11. The TensorRT shipped with the Jetpack l4t-r36.4 toolchain
        # (@tensorrt_l4t) predates 10.11 and does not export this type. Every other
        # configuration here (RTX, SBSA, Windows, default x86_64 Linux) is on a
        # TensorRT >= 10.11 bundle, so it gets the macro.
        #
        # Gate every IRuntimeConfig-using site in core/runtime with
        # `#ifdef TRT_HAS_IRUNTIME_CONFIG`; the Jetpack path falls back to the
        # legacy createExecutionContext() no-arg overload.
        ":jetpack": [],
        "//conditions:default": ["TRT_HAS_IRUNTIME_CONFIG"],
    }),
    linkopts = [
        "-lstdc++fs",
    ],
    local_defines = select({
        # TensorRT-RTX builds: opt into feature-gated APIs that the runtime layer
        # depends on (e.g. IExecutionContext::isStreamCapturable).
        ":rtx_win": ["ENABLE_FEATURE_DISABLE_RUNTIME_ALLOCATION"],
        ":rtx_x86_64": ["ENABLE_FEATURE_DISABLE_RUNTIME_ALLOCATION"],
        "//conditions:default": [],
    }),
    deps = [
        "//core/plugins:torch_tensorrt_plugins",
        "//core/util:prelude",
    ] + if_torch_nccl(["@torch_nccl//:nccl_headers"]) + select({
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        ":rtx_win": ["@tensorrt_rtx_win//:nvinfer"],
        ":rtx_x86_64": ["@tensorrt_rtx//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":windows": ["@tensorrt_win//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":jetpack": ["@torch_l4t//:libtorch"],
        ":rtx_win": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":windows": ["@libtorch_win//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

filegroup(
    name = "include_files",
    srcs = [
        "Platform.h",
        "RTDevice.h",
        "TRTEngine.h",
        "TRTEngineProfiler.h",
        "TRTRuntimeConfig.h",
        "runtime.h",
    ],
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "include",
    srcs = [":include_files"],
    package_dir = "core/runtime/",
)

pkg_files(
    name = "include_pkg_files",
    srcs = [":include_files"],
    prefix = "include/torch_tensorrt/core/runtime/",
    visibility = ["//visibility:public"],
)
