load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "torchtrt_runtime_example",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//core/runtime",
        "@libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
