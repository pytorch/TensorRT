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
    name = "partitioninginfo",
    srcs = [
        "PartitioningInfo.cpp",
    ],
    hdrs = [
        "PartitioningInfo.h",
    ],
    deps = [
        "//core/conversion",
        "//core/ir",
        "//core/lowering",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "PartitioningInfo.h",
    ],
    package_dir = "core/partitioning/partitioninginfo",
)
