package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "trtorchrt_example",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "//core/runtime:runtime",
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
