package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "converters",
    hdrs = [
        "converters.h"
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
        "impl/matrix_multiply.cpp",
        "impl/pooling.cpp",
        "impl/reduce.cpp",
        "impl/shape.cpp",
        "impl/shuffle.cpp",
        "impl/softmax.cpp",
        "impl/unary.cpp",
        "impl/interpolate.cpp"
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx",
        "//core/conversion/converters/impl/plugins"
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/converters/",
    srcs = ["converters.h"],
)
