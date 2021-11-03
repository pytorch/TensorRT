package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "benchmark",
    srcs = [
        "main.cpp",
        "timer.h",
    ],
    deps = [
        "//cpp:torch_tensorrt",
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
