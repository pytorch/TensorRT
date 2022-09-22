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
    name = "partitioningctx",
    srcs = [
        "PartitioningCtx.cpp",
    ],
    hdrs = [
        "PartitioningCtx.h",
    ],
    deps = [
        "//core/util:prelude",
        "//core/ir",
        "//core/conversion",
        "//core/partitioning/segmentedblock",
        "//core/partitioning/partitioninginfo",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "PartitioningCtx.h",
    ],
    package_dir = "core/partitioning/partitioningctx",
)
