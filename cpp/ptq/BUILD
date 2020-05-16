package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ptq",
    srcs = [
        "main.cpp",
        "timer.h"
    ],
    deps = [
        "//cpp/ptq/datasets:cifar10",
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "//cpp/api:trtorch",
        "@tensorrt//:nvinfer"
    ],
    copts = [
        "-pthread"
    ],
    linkopts = [
        "-lpthread",
    ]
)
