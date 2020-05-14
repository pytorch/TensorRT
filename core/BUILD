package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dont_use_cxx11_abi",
    values = {
        "define": "api=python",
    }
)

cc_library(
    name = "core",
    hdrs = [
        "compiler.h",
    ],
    srcs = [
        "compiler.cpp",
    ],
    deps = [
        "//core/conversion",
        "//core/execution",
        "//core/lowering",
        "//core/util/logging",
        "@tensorrt//:nvinfer"
    ] + select({
        ":dont_use_cxx11_abi":  ["@libtorch_non_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink=True,
)


load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/",
    srcs = ["compiler.h"],
)
