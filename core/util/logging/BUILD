package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dont_use_cxx11_abi",
    values = {
        "define": "api=python",
    }
)

cc_library(
    name = "logging",
    hdrs = [
        "TRTorchLogger.h",
    ],
    srcs = [
        "TRTorchLogger.cpp",
    ],
    deps = [
        "@tensorrt//:nvinfer"
    ] + select({
        ":dont_use_cxx11_abi":  ["@libtorch_non_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    })
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/util/logging",
    srcs = ["TRTorchLogger.h"],
)
