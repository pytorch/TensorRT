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
        "@tensorrt//:nvinfer",
        "@tensorrt//:nvinferplugin",
        "//core/util:prelude",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
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
