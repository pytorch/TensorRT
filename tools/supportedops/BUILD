load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "supportedops",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//core/conversion/converters",
        "//cpp:torch_tensorrt",
    ],
)
