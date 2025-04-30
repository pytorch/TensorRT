load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "logging",
    srcs = [
        "TorchTRTLogger.cpp",
    ],
    hdrs = [
        "TorchTRTLogger.h",
    ],
    deps = select({
        ":windows": [
            "@libtorch_win//:libtorch",
            "@tensorrt_win//:nvinfer",
        ],
        "//conditions:default": [
            "@libtorch",
            "@tensorrt//:nvinfer",
        ],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["TorchTRTLogger.h"],
    package_dir = "core/util/logging",
)
