package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "trtorch_plugins",
    hdrs = [
        "impl/interpolate_plugin.h",
        "impl/normalize_plugin.h",
        "plugins.h",

    ],
    srcs = [
        "impl/interpolate_plugin.cpp",
        "impl/normalize_plugin.cpp",
        "register_plugins.cpp",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "@tensorrt//:nvinferplugin",
        "//core/util:prelude",
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
    copts = [
        "-pthread"
    ],
    linkopts = [
        "-lpthread",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/plugins/",
    srcs = ["plugins.h",
            "impl/interpolate_plugin.h",
            "impl/normalize_plugin.h"],
)
