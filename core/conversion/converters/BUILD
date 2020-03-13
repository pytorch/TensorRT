package(default_visibility = ["//visibility:public"])

cc_library(
    name = "converters",
    hdrs = [
        "converters.h",
    ] + glob(["impl/**/*.cpp"]),
    srcs = [
        "NodeConverterRegistry.cpp",
        "Weights.cpp",
        "Arg.cpp",
    ],
    deps = [
        "@libtorch//:libtorch",
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx"
    ],
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/converters/",
    srcs = ["converters.h"],
)
