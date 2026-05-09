load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl",
    },
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "rtx_win",
    constraint_values = [
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
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
    deps = [
        "@executorch//:executorch_headers",
    ] + select({
        ":jetpack": ["@cuda_l4t//:cudart"],
        ":windows": ["@cuda_win//:cudart"],
        "//conditions:default": ["@cuda//:cudart"],
    }) + select({
        ":jetpack": ["@tensorrt_l4t//:nvinfer"],
        ":rtx_win": ["@tensorrt_rtx_win//:nvinfer"],
        ":rtx_x86_64": ["@tensorrt_rtx//:nvinfer"],
        ":sbsa": ["@tensorrt_sbsa//:nvinfer"],
        ":windows": ["@tensorrt_win//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }),
    alwayslink = True,
)
