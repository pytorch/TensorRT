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
    name = "conversion",
    hdrs = [
        "conversion.h",
    ],
    srcs = [
        "conversion.cpp",
        "conversion_blacklist.cpp",
        "InterfaceTypes.cpp"
    ],
    deps = [
        "//core/conversion/var",
        "//core/conversion/conversionctx",
        "//core/conversion/converters",
        "//core/conversion/evaluators",
        "//core/util:prelude"
    ] + select({
        ":use_pre_cxx11_abi_aarch64": ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
        ":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }) + select({
        ":aarch64": ["@tensorrt_aarch64//:nvinfer"],
        "//conditions:default":  ["@tensorrt//:nvinfer"],
    }),
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/",
    srcs = ["conversion.h"],
)
