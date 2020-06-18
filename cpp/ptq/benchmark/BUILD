package(default_visibility = ["//visibility:public"])

cc_library(
    name = "benchmark",
    hdrs = [
        "benchmark.h"
    ],
    srcs = [
        "benchmark.cpp",
        "timer.h"
    ],
    deps = [
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "//cpp/api:trtorch"
    ],
)
