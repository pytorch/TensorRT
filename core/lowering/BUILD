package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dont_use_cxx11_abi",
    values = {
        "define": "api=python",
    }
)

cc_library(
    name = "lowering",
    hdrs = [
        "lowering.h",
    ],
    srcs = [
        "lowering.cpp",
        "drop_unused_nodes.cpp",
        "register_const_op.cpp"
    ],
    deps = [
        "//core/lowering/passes",
        "//core/util:prelude"
    ] + select({
        ":dont_use_cxx11_abi":  ["@libtorch_non_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/lowering/",
    srcs = ["lowering.h"],
)

