package(default_visibility = ["//visibility:public"])

cc_library(
    name = "passes",
    hdrs = [
        "passes.h",
    ],
    srcs = [
        "conv2d_to_convolution.cpp",
        "exception_elimination.cpp",
        "fuse_flatten_linear.cpp",
        "remove_dropout.cpp",
        "unpack_addmm.cpp",
        "unpack_batch_norm.cpp",
        "unpack_log_softmax.cpp",
    ],
    deps = [
        "//core/util:prelude",
        "@libtorch//:libtorch",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/lowering/passes/",
    srcs = ["passes.h"],
)

