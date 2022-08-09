load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "cpp_benchmark",
    srcs = [
        "main.cpp",
        "timer.h",
    ],
    deps = [
        "//cpp:torch_tensorrt",
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
