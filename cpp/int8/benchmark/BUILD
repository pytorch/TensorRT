package(default_visibility = ["//visibility:public"])

cc_library(
    name = "benchmark",
    srcs = [
        "benchmark.cpp",
        "timer.h",
    ],
    hdrs = [
        "benchmark.h",
    ],
    deps = [
        "//cpp/api:trtorch",
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
