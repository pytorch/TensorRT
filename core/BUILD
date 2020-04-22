package(default_visibility = ["//visibility:public"])

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
        "@libtorch//:libtorch",
        "@tensorrt//:nvinfer"
    ],
    alwayslink=True,
)


load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/",
    srcs = ["compiler.h"],
)
