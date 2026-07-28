load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cifar10",
    srcs = [
        "cifar10.cpp",
    ],
    hdrs = [
        "cifar10.h",
    ],
    data = [
        ":cifar10_data",
    ],
    deps = [
        "@libtorch",
    ],
)

filegroup(
    name = "cifar10_data",
    srcs = glob(["data/cifar-10-batches-bin/**/*.bin"]),
)
