package(default_visibility = ["//visibility:public"])

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
        "@libtorch//:libtorch",
        "//core/conversion/conversionctx",
        "//core/conversion/converters",
        "//core/conversion/evaluators",
        "//core/util:prelude"
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/",
    srcs = ["conversion.h"],
)
