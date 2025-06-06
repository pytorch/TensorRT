load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
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
    name = "ci_build_testing",
    values = {
        "define": "torchtrt_src=prebuilt",
    },
)

cc_library(
    name = "util",
    srcs = [
        "evaluate_graph.cpp",
        "run_forward.cpp",
        "run_graph.cpp",
        "run_graph_engine.cpp",
        "util.cpp",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": [
            "@libtorch_win//:caffe2",
            "@libtorch_win//:libtorch",
        ],
        ":use_torch_whl": [
            "@torch_whl//:caffe2",
            "@torch_whl//:libtorch",
        ],
        ":jetpack": [
            "@torch_l4t//:libtorch",
            "@torch_l4t//:caffe2",
        ],
        "//conditions:default": [
            "@libtorch",
            "@libtorch//:caffe2",
        ],
    }) + select({
        ":ci_build_testing": [
            "@torch_tensorrt",
            "@torch_tensorrt//:torch_tensorrt_core_hdrs",
        ],
        "//conditions:default": [
            "//core/conversion",
            "//core/util:prelude",
            "//cpp:torch_tensorrt",
        ],
    }),
)
