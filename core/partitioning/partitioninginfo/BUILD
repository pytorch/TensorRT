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
    name = "partitioninginfo",
    srcs = [
        "PartitioningInfo.cpp",
    ],
    hdrs = [
        "PartitioningInfo.h",
    ],
    deps = [
        "//core/util:prelude",
        "//core/ir",
        "//core/conversion",
        "//core/lowering",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
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
