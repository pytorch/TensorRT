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
    name = "torch_tensorrt_plugins",
    srcs = [
        "impl/interpolate_plugin.cpp",
        "impl/normalize_plugin.cpp",
        "register_plugins.cpp",
    ],
    hdrs = [
        "impl/interpolate_plugin.h",
        "impl/normalize_plugin.h",
        "plugins.h",
    ],
    copts = [
        "-pthread",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        "//core/util:prelude",
    ] + select({
        ":windows": [
            "@tensorrt_win//:nvinfer",
            "@tensorrt_win//:nvinferplugin",
        ],
        ":sbsa": [
            "@tensorrt_sbsa//:nvinfer",
            "@tensorrt_sbsa//:nvinferplugin",
        ],
        ":jetpack": [
            "@tensorrt_l4t//:nvinfer",
            "@tensorrt_l4t//:nvinferplugin",
        ],
        "//conditions:default": [
            "@tensorrt//:nvinfer",
            "@tensorrt//:nvinferplugin",
        ],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["plugins.h"],
    package_dir = "core/plugins/",
)

pkg_tar(
    name = "impl_include",
    srcs = [
        "impl/interpolate_plugin.h",
        "impl/normalize_plugin.h",
    ],
    package_dir = "core/plugins/impl",
)
