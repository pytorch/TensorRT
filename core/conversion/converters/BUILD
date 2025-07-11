load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "rtx_win",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default"
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
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
        "//core/conversion/conversionctx",
        "//core/util:prelude",
    ] + select({
        ":rtx_x86_64": ["@tensorrt_rtx//:nvinfer"],
        ":rtx_win": ["@tensorrt_rtx_win//:nvinfer"],
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
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
        "//core/conversion/conversionctx",
        "//core/util:prelude",
    ] + select({
        ":rtx_x86_64": ["@tensorrt_rtx//:nvinfer"],
        ":rtx_win": ["@tensorrt_rtx_win//:nvinfer"],
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)


cc_library(
    name = "converters",
    srcs = [
        "NodeConverterRegistry.cpp",
        "impl/activation.cpp",
        # "impl/batch_norm.cpp",
        "impl/bitwise.cpp",
        "impl/cast.cpp",
        "impl/chunk.cpp",
        "impl/concat.cpp",
        "impl/constant.cpp",
        "impl/constant_pad.cpp",
        "impl/conv_deconv.cpp",
        "impl/cumsum.cpp",
        "impl/einsum.cpp",
        "impl/element_wise.cpp",
        "impl/expand.cpp",
        "impl/internal_ops.cpp",
        # "impl/interpolate.cpp",
        "impl/layer_norm.cpp",
        "impl/linear.cpp",
        "impl/lstm_cell.cpp",
        "impl/matrix_multiply.cpp",
        "impl/max.cpp",
        # "impl/normalize.cpp",
        # "impl/pooling.cpp",
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
    ] + select({
        # exclude plugins from rtx build
        ":rtx_x86_64": [],
        ":rtx_win": [],
        "//conditions:default": [ 
            "impl/interpolate.cpp",
            "impl/normalize.cpp",
            "impl/pooling.cpp",
            "impl/batch_norm.cpp",
        ],
    }),
    hdrs = [
        "converters.h",
    ],
    deps = [
        ":converter_util",
        "//core/conversion/conversionctx",
        "//core/conversion/tensorcontainer",
        "//core/conversion/var",
        "//core/plugins:torch_tensorrt_plugins",
        "//core/util:prelude",
    ] + select({
        ":rtx_x86_64": ["@tensorrt_rtx//:nvinfer"],
        ":rtx_win": ["@tensorrt_rtx_win//:nvinfer"],
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
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
