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
    name = "conversion",
    srcs = [
        "conversion.cpp",
        "conversion_ignorelist.cpp",
    ],
    hdrs = [
        "conversion.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/conversion/var",
        "//core/conversion/conversionctx",
        "//core/conversion/converters",
        "//core/conversion/evaluators",
        "//core/ir",
        "//core/util:prelude",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)

pkg_tar(
    name = "include",
    srcs = ["conversion.h"],
    package_dir = "core/conversion/",
)
