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
    name = "var",
    srcs = [
        "Var.cpp",
    ],
    hdrs = [
        "Var.h",
        "Var_inl.h",
    ],
    deps = [
        "//core/conversion/converters:converter_util",
        "//core/conversion/tensorcontainer",
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
        "Var.h",
        "Var_inl.h",
    ],
    package_dir = "core/conversion/var/",
)
