load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl",
    },
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "rtx_win",
    constraint_values = [
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack",
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
    srcs = select({
        ":rtx_win": [],
        ":rtx_x86_64": [],
        "//conditions:default": [
            "impl/interpolate_plugin.cpp",
            "impl/normalize_plugin.cpp",
            "register_plugins.cpp",
        ],
    }),
    hdrs = select({
        ":rtx_win": [],
        ":rtx_x86_64": [],
        "//conditions:default": [
            "impl/interpolate_plugin.h",
            "impl/normalize_plugin.h",
            "plugins.h",
        ],
    }),
    copts = [
        "-pthread",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        "//core/util:prelude",
    ] + select({
        ":jetpack": [
            "@tensorrt_l4t//:nvinfer",
            "@tensorrt_l4t//:nvinferplugin",
        ],
        ":rtx_win": [],
        ":rtx_x86_64": [],
        ":sbsa": [
            "@tensorrt_sbsa//:nvinfer",
            "@tensorrt_sbsa//:nvinferplugin",
        ],
        ":windows": [
            "@tensorrt_win//:nvinfer",
            "@tensorrt_win//:nvinferplugin",
        ],
        "//conditions:default": [
            "@tensorrt//:nvinfer",
            "@tensorrt//:nvinferplugin",
        ],
    }) + select({
        ":jetpack": ["@torch_l4t//:libtorch"],
        ":rtx_win": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":windows": ["@libtorch_win//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

filegroup(
    name = "include_files",
    srcs = ["plugins.h"],
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "include",
    srcs = [":include_files"],
    package_dir = "core/plugins/",
)

filegroup(
    name = "impl_include_files",
    srcs = [
        "impl/interpolate_plugin.h",
        "impl/normalize_plugin.h",
    ],
    visibility = ["//visibility:public"],
)

pkg_files(
    name = "impl_include_pkg_files",
    srcs = [":impl_include_files"],
    visibility = ["//visibility:public"],
    prefix = "include/torch_tensorrt/core/plugins/impl",
)

pkg_tar(
    name = "impl_include",
    srcs = [":impl_include_files"],
    package_dir = "core/plugins/impl",
)
