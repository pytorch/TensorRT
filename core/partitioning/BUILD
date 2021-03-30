package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "partitioning",
    hdrs = [
        "SegmentedBlock.h",
        "shape_analysis.h",
        "partitioning.h",
    ],
    srcs = [
        "SegmentedBlock.cpp",
        "shape_analysis.cpp",
        "partitioning.cpp",
    ],
    deps = [
        "//core/conversion",
        "//core/util:prelude",
        "//core/lowering"
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/partitioning/",
    srcs = ["partitioning.h"],
)

