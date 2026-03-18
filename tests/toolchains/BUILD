load("@rules_cc//cc:defs.bzl", "cc_test")
load("//toolchains/torch_nccl:defs.bzl", "if_torch_nccl")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "test_torch_nccl_detect",
    srcs = ["test_torch_nccl_detect.cpp"],
    # Mirrors exactly how downstream targets use torch_nccl_detect.
    # A false positive (USE_C10D_NCCL defined but NCCL not in the library)
    # will cause a link failure here, catching the misdetection at test time.
    copts = if_torch_nccl(["-DUSE_C10D_NCCL"]),
    deps = [
        "@cuda//:cuda",
        "@googletest//:gtest_main",
        "@libtorch",
    ],
)
