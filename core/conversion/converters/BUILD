load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "weights",
    srcs = [
        "Weights.cpp",
    ],
    hdrs = [
        "Weights.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

cc_library(
    name = "converter_util",
    srcs = [
        "converter_util.cpp",
    ],
    hdrs = [
        "converter_util.h",
    ],
    deps = [
        ":weights",
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

cc_library(
    name = "converters",
    srcs = [
        "NodeConverterRegistry.cpp",
        "impl/activation.cpp",
        "impl/batch_norm.cpp",
        "impl/bitwise.cpp",
        "impl/cast.cpp",
        "impl/concat.cpp",
        "impl/constant.cpp",
        "impl/constant_pad.cpp",
        "impl/conv_deconv.cpp",
        "impl/cumsum.cpp",
        "impl/element_wise.cpp",
        "impl/expand.cpp",
        "impl/interpolate.cpp",
        "impl/layer_norm.cpp",
        "impl/linear.cpp",
        "impl/lstm_cell.cpp",
        "impl/matrix_multiply.cpp",
        "impl/max.cpp",
        "impl/normalize.cpp",
        "impl/pooling.cpp",
        "impl/quantization.cpp",
        "impl/reduce.cpp",
        "impl/reflection_pad.cpp",
        "impl/replication_pad.cpp",
        "impl/select.cpp",
        "impl/shuffle.cpp",
        "impl/softmax.cpp",
        "impl/squeeze.cpp",
        "impl/stack.cpp",
        "impl/topk.cpp",
        "impl/unary.cpp",
        "impl/unsqueeze.cpp",
    ],
    hdrs = [
        "converters.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/var",
        "//core/conversion/tensorcontainer",
        "//core/conversion/conversionctx",
        "//core/plugins:torch_tensorrt_plugins",
        ":converter_util",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "Weights.h",
        "converter_util.h",
        "converters.h",
    ],
    package_dir = "core/conversion/converters/",
)
