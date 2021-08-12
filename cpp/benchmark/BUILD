package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "benchmark",
    srcs = [
        "main.cpp",
        "timer.h",
    ],
    deps = [
        "//cpp/api:trtorch",
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
