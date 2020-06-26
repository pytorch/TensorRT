package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

config_setting(
    name = "use_pre_cxx11_abi_aarch64",
    values = {
        "define": "abi=pre_cxx11_abi",
        "cpu" : "aarch64",
    }
)

config_setting(
    name="aarch64",
    values={
        "cpu":"aarch64"
    },
)

cc_library(
    name = "passes",
    hdrs = [
        "passes.h",
    ],
    srcs = [
        "conv2d_to_convolution.cpp",
        "exception_elimination.cpp",
        "fuse_addmm_branches.cpp",
        "fuse_flatten_linear.cpp",
        "remove_contiguous.cpp",
        "remove_dropout.cpp",
        "remove_to.cpp",
        "unpack_addmm.cpp",
        "unpack_batch_norm.cpp",
        "unpack_log_softmax.cpp",
    ],
    deps = [
        "//core/util:prelude",
    ] + select({
	    ":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        ":use_pre_cxx11_abi_aarch64":  ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/lowering/passes/",
    srcs = ["passes.h"],
)

