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
    name = "python_core",
    values = {
        "define": "target_lang=python",
    },
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
        "//core/runtime",
        "//core/lowering",
        "//core/partitioning",
        "//core/util/logging",
        "@tensorrt//:nvinfer",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
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
