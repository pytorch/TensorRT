load("@rules_cc//cc:defs.bzl", "cc_binary")

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

cc_binary(
    name = "ptq",
    srcs = [
        "main.cpp",
    ],
    copts = [
        "-pthread",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        "//cpp:torch_tensorrt",
        "//examples/int8/benchmark",
        "//examples/int8/datasets:cifar10",
        "@libtorch",
        "@libtorch//:caffe2",
        ] + select({
            ":windows": [
                "@libtorch_win//:libtorch",
                "@libtorch_win//:caffe2",
            ],
            ":use_torch_whl": [
                "@torch_whl//:libtorch",
                "@torch_whl//:caffe2",
            ],
            ":jetpack": [
                "@torch_l4t//:libtorch",
                "@torch_l4t//:caffe2",
            ],
            "//conditions:default": [
                "@libtorch//:libtorch",
                "@libtorch//:caffe2",
            ],
        }) + select({
            ":windows": ["@tensorrt_win//:nvinfer"],
            ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
            ":jetpack": ["@tensorrt_l4t//:nvinfer"],
            "//conditions:default": ["@tensorrt//:nvinfer"],
        })
)
