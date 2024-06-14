load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "conversion",
    srcs = [
        "conversion.cpp",
        "conversion_ignorelist.cpp",
    ],
    hdrs = [
        "conversion.h",
    ],
    deps = [
        "//core/conversion/conversionctx",
        "//core/conversion/converters",
        "//core/conversion/evaluators",
        "//core/conversion/var",
        "//core/ir",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer", "@libtorch_win//:libtorch"],
        ":use_pre_cxx11_abi": ["@tensorrt//:nvinfer", "@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@tensorrt//:nvinfer", "@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["conversion.h"],
    package_dir = "core/conversion/",
)
