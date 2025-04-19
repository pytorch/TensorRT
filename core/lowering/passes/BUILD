load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "convNd_to_convolution.cpp",
        "device_casting.cpp",
        "exception_elimination.cpp",
        "fuse_addmm_branches.cpp",
        "linear_to_addmm.cpp",
        "module_fallback.cpp",
        "op_aliasing.cpp",
        "reduce_gelu.cpp",
        "reduce_remainder.cpp",
        "reduce_to.cpp",
        "remove_bn_dim_check.cpp",
        "remove_contiguous.cpp",
        "remove_dropout.cpp",
        "remove_nops.cpp",
        "remove_unnecessary_casts.cpp",
        "replace_aten_pad.cpp",
        "rewrite_inputs_with_params.cpp",
        "silu_to_sigmoid_multiplication.cpp",
        "tile_to_repeat.cpp",
        "unpack_addmm.cpp",
        "unpack_batch_norm.cpp",
        "unpack_hardsigmoid.cpp",
        "unpack_hardswish.cpp",
        "unpack_log_softmax.cpp",
        "unpack_rsqrt.cpp",
        "unpack_scaled_dot_product_attention.cpp",
        "unpack_std.cpp",
        "unpack_var.cpp",
        "view_to_reshape.cpp",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        "//core/util:prelude",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["passes.h"],
    package_dir = "core/lowering/passes/",
)
