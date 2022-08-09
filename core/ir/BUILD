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
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)

pkg_tar(
    name = "include",
    srcs = [
        "ir.h",
    ],
    package_dir = "core/ir/",
)
