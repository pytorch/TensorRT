package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cifar10",
    hdrs = [
        "cifar10.h"
    ],
    srcs = [
        "cifar10.cpp"
    ],
    deps = [
        "@libtorch//:libtorch"
    ],
    data = [
        ":cifar10_data"
    ]

)

filegroup(
    name = "cifar10_data",
    srcs = glob(["data/cifar-10-batches-bin/**/*.bin"])
)