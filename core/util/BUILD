load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
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
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
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
)

cc_library(
    name = "build_info",
    hdrs = [
        "build_info.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
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
        "@tensorrt//:nvinfer",
        "//core/util/logging",
        ":macros",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)

pkg_tar(
    name = "include",
    srcs = [
        "//core/util:Exception.h",
        "//core/util:build_info.h",
        "//core/util:jit_util.h",
        "//core/util:macros.h",
        "//core/util:prelude.h",
        "//core/util:trt_util.h",
    ],
    package_dir = "core/util/",
)
