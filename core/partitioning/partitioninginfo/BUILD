load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/build_srcs:torch": "whl"
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_lib_collection": "datacenter"
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_lib_collection": "jetpack"
    },
)

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
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_jetpack//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
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
