package(default_visibility = ["//visibility:public"])

# trtorch is the directory extracted from dowloaded release tar file
# It has include, lib, bin directory and LICENSE file
cc_library(
    name = "trtorch_runtime",
    srcs = ["trtorch/lib/libtrtorchrt.so", "trtorch/lib/libtrtorch_plugins.so"],
)

cc_binary(
    name = "samplertapp",
    srcs = [
        "main.cpp"
    ],
    deps = [
        ":trtorch_runtime",
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
