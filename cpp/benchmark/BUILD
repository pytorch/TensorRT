package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "benchmark",
    srcs = [
        "main.cpp",
        "timer.h"
    ],
    deps = [
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "//cpp/api:trtorch"
    ],
)
