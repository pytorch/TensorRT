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
        "//cpp/int8/benchmark",
        "//cpp/int8/datasets:cifar10",
        "@libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
