load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

cc_library(
    name = "tensorrt_executorch_backend",
    srcs = ["TensorRTBackend.cpp"],
    hdrs = [
        "TensorRTBackend.h",
        "TensorRTBlobHeader.h",
    ],
    # Build the TensorRT backend as a static library. The final application
    # links this target together with the ExecuTorch runtime it was compiled
    # against, avoiding any runtime plugin/dlopen dependency.
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": [
            "@executorch//:executorch_headers",
            "@cuda//:cudart",
            "@tensorrt//:nvinfer",
        ],
        ":sbsa": [
            "@executorch//:executorch_headers",
            "@cuda//:cudart",
            "@tensorrt_sbsa//:nvinfer",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)
