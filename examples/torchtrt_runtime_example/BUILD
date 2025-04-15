load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_lib_collection": "default"
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_lib_collection": "jetpack"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)


cc_binary(
    name = "torchtrt_runtime_example",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//core/runtime",
        "@libtorch",
        "@libtorch//:caffe2",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    })
)
