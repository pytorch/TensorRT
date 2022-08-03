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
    name = "lowering",
    srcs = [
        "LowerInfo.cpp",
        "drop_unused_nodes.cpp",
        "lowering.cpp",
        "register_trt_placeholder_ops.cpp",
    ],
    hdrs = [
        "lowering.h",
    ],
    deps = [
        "//core/lowering/passes",
        "//core/util:prelude",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["lowering.h"],
    package_dir = "core/lowering/",
)
