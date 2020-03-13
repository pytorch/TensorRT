package(default_visibility = ["//visibility:public"])

cc_library(
    name = "conversionctx",
    hdrs = [
        "ConversionCtx.h",
    ],
    srcs = [
        "ConversionCtx.cpp",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "@libtorch//:libtorch",
        "//core/util:prelude",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/conversionctx/",
    srcs = ["ConversionCtx.h"],
)
