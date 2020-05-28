package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ptq",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//cpp/ptq/benchmark",
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
