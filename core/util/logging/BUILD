package(default_visibility = ["//visibility:public"])

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
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/util/logging",
    srcs = ["TRTorchLogger.h"],
)
