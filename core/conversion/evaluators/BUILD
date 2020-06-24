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
config_setting(
    name="x86_64",
    values={
        "cpu":"x86_64"
    },
)

cc_library(
    name = "evaluators",
    hdrs = [
        "evaluators.h",
    ],
    srcs = [
        "NodeEvaluatorRegistry.cpp",
        "prim.cpp",
        "aten.cpp",
        "eval_macros.h"
    ],
    deps = [
        "//core/util:prelude",
        "//core/conversion/var",
        "//core/conversion/tensorcontainer",
    ] + select({
        ":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi_aarch64" : ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/evaluators/",
    srcs = ["evaluators.h"],
)
