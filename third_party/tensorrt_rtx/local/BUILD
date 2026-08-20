load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "rtx_win_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    flag_values = {"@//toolchains/dep_collection:compute_libs": "rtx"},
)

config_setting(
    name = "rtx_win_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:windows",
    ],
    flag_values = {"@//toolchains/dep_collection:compute_libs": "rtx"},
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {"@//toolchains/dep_collection:compute_libs": "rtx"},
)

config_setting(
    name = "rtx_sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    flag_values = {"@//toolchains/dep_collection:compute_libs": "rtx"},
)

cc_library(
    name = "nvinfer_headers",
    hdrs = select({
        ":rtx_sbsa": glob(
            [
                "include/NvInfer*.h",
            ],
            allow_empty = True,
            # exclude = [
            #     "include/NvInferPlugin.h",
            #     "include/NvInferPluginUtils.h",
            # ],
        ),
        ":rtx_win_arm64": glob(
            ["include/NvInfer*.h"],
            allow_empty = True,
        ),
        ":rtx_win_x86_64": glob(
            [
                "include/NvInfer*.h",
            ],
            allow_empty = True,
            # exclude = [
            #     "include/NvInferPlugin.h",
            #     "include/NvInferPluginUtils.h",
            # ],
        ),
        ":rtx_x86_64": glob(
            [
                "include/NvInfer*.h",
            ],
            allow_empty = True,
            # exclude = [
            #     "include/NvInferPlugin.h",
            #     "include/NvInferPluginUtils.h",
            # ],
        ),
    }),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_static_lib",
    static_library = select({
        ":rtx_win_arm64": "lib/tensorrt_rtx_1_6.lib",
        ":rtx_win_x86_64": "lib/tensorrt_rtx_1_6.lib",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":rtx_sbsa": "lib/libtensorrt_rtx.so",
        ":rtx_win_arm64": "bin/tensorrt_rtx_1_6.dll",
        ":rtx_win_x86_64": "bin/tensorrt_rtx_1_6.dll",
        ":rtx_x86_64": "lib/libtensorrt_rtx.so",
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
        ":rtx_win_arm64": [
            "nvinfer_static_lib",
            "@cuda_win_arm64//:cudart",
        ],
        ":rtx_win_x86_64": [
            "nvinfer_static_lib",
            "@cuda_win//:cudart",
        ],
        "//conditions:default": ["@cuda//:cudart"],
    }),
)
