package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "passes",
    srcs = [
        "conv2d_to_convolution.cpp",
        "conv3d_to_convolution.cpp",
        "exception_elimination.cpp",
        "fuse_addmm_branches.cpp",
        "linear_to_addmm.cpp",
        "op_aliasing.cpp",
        "reduce_to.cpp",
        "remove_bn_dim_check.cpp",
        "remove_contiguous.cpp",
        "remove_dropout.cpp",
        "remove_nops.cpp",
        "silu_to_sigmoid_multiplication.cpp",
        "unpack_addmm.cpp",
        "unpack_batch_norm.cpp",
        "unpack_log_softmax.cpp",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        "//core/util:prelude",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    srcs = ["passes.h"],
    package_dir = "core/lowering/passes/",
)
