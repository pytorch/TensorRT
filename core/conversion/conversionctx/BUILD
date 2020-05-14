package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dont_use_cxx11_abi",
    values = {
        "define": "api=python",
    }
)

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
        "//core/util:prelude",
    ] + select({
        ":dont_use_cxx11_abi":  ["@libtorch_non_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    })
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/conversionctx/",
    srcs = ["ConversionCtx.h"],
)
