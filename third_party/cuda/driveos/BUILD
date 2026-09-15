load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart",
    srcs = ["lib/libcudart.so"],
    hdrs = glob([
        "cuda/targets/aarch64-linux/include/**/*",
        "cuda/thor/targets/aarch64-linux/include/**/*",
    ]),
    includes = [
        "cuda/targets/aarch64-linux/include",
        "cuda/thor/targets/aarch64-linux/include",
    ],
)
