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
        "//cpp/api:trtorch",
        "//cpp/qat/benchmark",
        "//cpp/qat/datasets:cifar10",
        "@libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
