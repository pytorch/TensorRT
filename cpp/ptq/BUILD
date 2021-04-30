package(default_visibility = ["//visibility:public"])

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
        "//cpp/api:trtorch",
        "//cpp/ptq/benchmark",
        "//cpp/ptq/datasets:cifar10",
        "@libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
