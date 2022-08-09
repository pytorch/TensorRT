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
    name = "logging",
    srcs = [
        "TorchTRTLogger.cpp",
    ],
    hdrs = [
        "TorchTRTLogger.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)

pkg_tar(
    name = "include",
    srcs = ["TorchTRTLogger.h"],
    package_dir = "core/util/logging",
)
