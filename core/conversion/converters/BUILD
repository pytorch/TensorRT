package(default_visibility = ["//visibility:public"])

cc_library(
    name = "converters",
    hdrs = [
        "converters.h",
    ],
    srcs = [
        "NodeConverterRegistry.cpp",
        "Weights.cpp",
        "Arg.cpp",
        "impl/activation.cpp",
        "impl/batch_norm.cpp",
        "impl/constant.cpp",
        "impl/conv_deconv.cpp",
        "impl/element_wise.cpp",
        "impl/linear.cpp",
        "impl/pooling.cpp",
        "impl/reduce.cpp",
        "impl/softmax.cpp",
        "impl/unary.cpp",
    ],
    deps = [
        "@libtorch//:libtorch",
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx"
    ],
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/converters/",
    srcs = ["converters.h"],
)
