load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "ci_rhel_x86_64_linux",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
        "@//toolchains/distro:ci_rhel",
    ],
)

cc_library(
    name = "nvinfer_headers",
    hdrs = select({
        ":ci_rhel_x86_64_linux": glob(
            [
                "include/NvInfer*.h",
            ],
            allow_empty = True,
            exclude = [
                "include/NvInferPlugin.h",
                "include/NvInferPluginUtils.h",
            ],
        ),
        ":windows": glob(
            [
                "include/NvInfer*.h",
            ],
            allow_empty = True,
            exclude = [
                "include/NvInferPlugin.h",
                "include/NvInferPluginUtils.h",
            ],
        ),
        "//conditions:default": glob(
            [
                "include/x86_64-linux-gnu/NvInfer*.h",
            ],
            exclude = [
                "include/x86_64-linux-gnu/NvInferPlugin.h",
                "include/x86_64-linux-gnu/NvInferPluginUtils.h",
            ],
        ),
    }),
    includes = select({
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_static_lib",
    static_library = select({
        ":windows": "lib/tensorrt_rtx_1_0.lib",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":ci_rhel_x86_64_linux": "lib64/libtensorrt_rtx.so",
        ":windows": "lib/tensorrt_rtx_1_0.dll",
        "//conditions:default": "lib/x86_64-linux-gnu/libtensorrt_rtx.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinfer",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer_headers",
        "nvinfer_lib",
    ] + select({
        ":windows": [
            "nvinfer_static_lib",
            "@cuda_win//:cudart",
        ],
        "//conditions:default": ["@cuda//:cudart"],
    }),
)
