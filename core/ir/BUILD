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
    name = "ir",
    srcs = [
        "GraphInputs.cpp",
        "Input.cpp",
        "StaticParams.cpp",
        "ir.cpp",
    ],
    hdrs = [
        "ir.h",
    ],
    deps = [
        "//core/util:prelude",
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
        "ir.h",
    ],
    package_dir = "core/ir/",
)
