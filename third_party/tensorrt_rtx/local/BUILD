load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "rtx_win",
    constraint_values = [
        "@platforms//os:windows",
    ],
    flag_values = {"@//toolchains/dep_collection:compute_libs": "rtx"},
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
        "@//toolchains/distro:ci_rhel",
    ],
    flag_values = {"@//toolchains/dep_collection:compute_libs": "rtx"},
)

cc_library(
    name = "nvinfer_headers",
    hdrs = select({
        ":rtx_win": glob(
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
        ":rtx_win": "lib/tensorrt_rtx_1_0.lib",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":rtx_win": "lib/tensorrt_rtx_1_0.dll",
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
        ":rtx_win": [
            "nvinfer_static_lib",
            "@cuda_win//:cudart",
        ],
        "//conditions:default": ["@cuda//:cudart"],
    }),
)
