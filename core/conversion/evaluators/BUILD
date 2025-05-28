load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default"
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "evaluators",
    srcs = [
        "NodeEvaluatorRegistry.cpp",
        "aten.cpp",
        "eval_macros.h",
        "eval_util.cpp",
        "eval_util.h",
        "prim.cpp",
    ],
    hdrs = [
        "evaluators.h",
    ],
    deps = [
        "//core/conversion/tensorcontainer",
        "//core/conversion/var",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["evaluators.h"],
    package_dir = "core/conversion/evaluators/",
)
