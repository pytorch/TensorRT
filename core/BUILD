load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "python_core",
    values = {
        "define": "target_lang=python",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "core",
    srcs = [
        "compiler.cpp",
    ],
    hdrs = [
        "compiler.h",
    ],
    deps = [
        "//core/conversion",
        "//core/lowering",
        "//core/partitioning",
        "//core/runtime",
        "//core/util/logging",
    ] + select({
        ":windows": [
            "@libtorch_win//:libtorch",
            "@tensorrt_win//:nvinfer",
        ],
        "//conditions:default": [
            "@libtorch",
            "@tensorrt//:nvinfer",
        ],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "compiler.h",
    ],
    package_dir = "core/",
)
