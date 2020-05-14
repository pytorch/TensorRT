package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dont_use_cxx11_abi",
    values = {
        "define": "api=python",
    }
)

cc_library(
    name = "conversion",
    hdrs = [
        "conversion.h",
    ],
    srcs = [
        "conversion.cpp",
        "conversion_blacklist.cpp",
        "string_to_type_lut.cpp",
        "InterfaceTypes.cpp"
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/conversion/conversionctx",
        "//core/conversion/converters",
        "//core/conversion/evaluators",
        "//core/util:prelude"
    ] + select({
        ":dont_use_cxx11_abi":  ["@libtorch_non_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    })
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/",
    srcs = ["conversion.h"],
)
