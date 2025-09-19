load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")

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
    name = "prelude",
    hdrs = [
        "prelude.h",
    ],
    deps = [
        ":build_info",
        ":exception",
        ":jit_util",
        ":macros",
        ":trt_util",
        "//core/util/logging",
    ],
)

cc_library(
    name = "jit_util",
    hdrs = [
        "jit_util.h",
    ],
    deps = [
        ":macros",
    ] + select({
        ":jetpack": ["@torch_l4t//:libtorch"],
        ":rtx_win": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":windows": ["@libtorch_win//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_library(
    name = "macros",
    hdrs = [
        "macros.h",
    ],
    deps = [
        ":exception",
        "//core/util/logging",
    ],
)

cc_library(
    name = "exception",
    srcs = [
        "Exception.cpp",
    ],
    hdrs = [
        "Exception.h",
    ],
    alwayslink = True,
)

cc_library(
    name = "build_info",
    hdrs = [
        "build_info.h",
    ],
    deps = select({
        ":jetpack": ["@torch_l4t//:libtorch"],
        ":rtx_win": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":windows": ["@libtorch_win//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_library(
    name = "trt_util",
    srcs = [
        "trt_util.cpp",
    ],
    hdrs = [
        "trt_util.h",
    ],
    deps = [
        ":macros",
        "//core/util/logging",
    ] + select({
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
        "Exception.h",
        "build_info.h",
        "jit_util.h",
        "macros.h",
        "prelude.h",
        "trt_util.h",
    ],
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "include",
    srcs = ["//core/util:include_files"],
    package_dir = "core/util/",
)

pkg_zip(
    name = "include_zip",
    srcs = ["//core/util:include_files"],
    out = "include.zip",
    package_dir = "core/util/",
)
