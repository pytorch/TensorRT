package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "converters",
    hdrs = [
        "converters.h"
    ],
    srcs = [
        "NodeConverterRegistry.cpp",
        "Weights.cpp",
        "impl/activation.cpp",
        "impl/batch_norm.cpp",
        "impl/concat.cpp",
        "impl/constant.cpp",
        "impl/conv_deconv.cpp",
        "impl/element_wise.cpp",
        "impl/interpolate.cpp",
        "impl/linear.cpp",
        "impl/matrix_multiply.cpp",
        "impl/pooling.cpp",
        "impl/reduce.cpp",
        "impl/select.cpp",
        "impl/shuffle.cpp",
        "impl/softmax.cpp",
        "impl/stack.cpp",
        "impl/unary.cpp",
    ],
    deps = [
        "//core/util:prelude",
        "//core/conversion/var",
        "//core/conversion/tensorcontainer",
        "//core/conversion/conversionctx",
        "//core/conversion/converters/impl/plugins",
        "@tensorrt//:nvinfer",

    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    srcs = ["converters.h"],
    package_dir = "core/conversion/converters/",
)
