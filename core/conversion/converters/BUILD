package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "weights",
    hdrs = [
        "Weights.h"
    ],
    srcs = [
        "Weights.cpp"
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx"
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

cc_library(
    name = "converters",
    hdrs = [
        "converters.h"
    ],
    srcs = [
        "NodeConverterRegistry.cpp",
        "impl/activation.cpp",
        "impl/batch_norm.cpp",
        "impl/concat.cpp",
        "impl/constant.cpp",
        "impl/conv_deconv.cpp",
        "impl/element_wise.cpp",
        "impl/expand.cpp",
        "impl/linear.cpp",
        "impl/matrix_multiply.cpp",
        "impl/normalize.cpp",
        "impl/pooling.cpp",
        "impl/reduce.cpp",
        "impl/replication_pad.cpp",
        "impl/shuffle.cpp",
        "impl/softmax.cpp",
        "impl/unary.cpp",
        "impl/interpolate.cpp",
        "impl/select.cpp",
        "impl/squeeze.cpp",
        "impl/stack.cpp",
        "impl/lstm_cell.cpp",
        "impl/unsqueeze.cpp",
        "impl/topk.cpp",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/var",
        "//core/conversion/tensorcontainer",
        "//core/conversion/conversionctx",
        "//core/plugins:trtorch_plugins",
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
    srcs = [
        "converters.h",
        "Weights.h"
    ],
)
