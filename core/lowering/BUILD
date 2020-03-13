package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lowering",
    hdrs = [
        "lowering.h",
    ],
    srcs = [
        "lowering.cpp",
        "drop_unused_nodes.cpp",
    ],
    deps = [
        "@libtorch//:libtorch",
        "//core/lowering/irfusers"
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/lowering/",
    srcs = ["lowering.h"],
)

