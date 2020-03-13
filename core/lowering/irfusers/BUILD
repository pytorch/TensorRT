package(default_visibility = ["//visibility:public"])

cc_library(
    name = "irfusers",
    hdrs = [
        "irfusers.h",
    ],
    srcs = [
        "fuse_flatten_linear.cpp",
        "expand_log_softmax.cpp",
        "remove_dropout.cpp",
        "unpack_batch_norm.cpp"
    ],
    deps = [
        "@libtorch//:libtorch",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/lowering/irfusers/",
    srcs = ["irfusers.h"],
)

