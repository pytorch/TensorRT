package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

config_setting(
    name = "use_pre_cxx11_abi_aarch64",
    values = {
        "define": "abi=pre_cxx11_abi",
        "cpu" : "aarch64",
    }
)

config_setting(
    name="aarch64",
    values={
        "cpu":"aarch64"
    },
)

cc_library(
    name = "lowering",
    hdrs = [
        "lowering.h",
    ],
    srcs = [
        "lowering.cpp",
        "drop_unused_nodes.cpp",
        "register_trt_placeholder_ops.cpp"
    ],
    deps = [
        "//core/lowering/passes",
        "//core/util:prelude"
    ] + select({
	    ":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        ":use_pre_cxx11_abi_aarch64":  ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
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

