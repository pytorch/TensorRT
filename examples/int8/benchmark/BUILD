load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

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
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
