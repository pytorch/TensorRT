load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "qat",
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
        "@tensorrt//:nvinfer",
    ],
)
