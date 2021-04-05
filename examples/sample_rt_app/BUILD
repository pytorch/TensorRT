package(default_visibility = ["//visibility:public"])

# trtorch is the dowloaded tar file
# It has include, lib, bin directory and LICENSE file
cc_library(
    name = "trtorch_runtime",
    srcs = ["trtorch/lib/libtrtorchrt.so", "trtorch/lib/libtrtorch_plugins.so"],
    hdrs = ["trtorch/include/trtorch/core/runtime/runtime.h"], # "trtorch/include/trtorch/trtorch.h", "trtorch/include/trtorch/macros.h"],
    includes = ["trtorch/include/trtorch"],
    #include_prefix="trtorch/include",
)

cc_binary(
    name = "samplertapp",
    srcs = [
        "main.cpp"
    ],
    includes = ["trtorch/include", "trtorch/include/trtorch"],
    deps = [
        ":trtorch_runtime",
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "@tensorrt//:nvinfer",
    ],
)
