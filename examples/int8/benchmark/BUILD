load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl",
    },
)

cc_library(
    name = "benchmark",
    srcs = [
        "benchmark.cpp",
        "timer.h",
    ],
    hdrs = [
        "benchmark.h",
    ],
    deps = [
        "//cpp:torch_tensorrt",
    ] + select({
        ":use_torch_whl": [
            "@torch_whl//:caffe2",
            "@torch_whl//:libtorch",
        ],
        ":windows": [
            "@libtorch_win//:caffe2",
            "@libtorch_win//:libtorch",
        ],
        "//conditions:default": [
            "@libtorch",
            "@libtorch//:caffe2",
        ],
    }),
)
