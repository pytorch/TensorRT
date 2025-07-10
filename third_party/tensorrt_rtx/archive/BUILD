load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "nvinfer_headers",
    hdrs = glob(
        [
            "include/NvInfer*.h",
        ],
        allow_empty = True,
        exclude = [
            "include/NvInferPlugin.h",
            "include/NvInferPluginUtils.h",
        ],
    ),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":windows": "lib/tensorrt_rtx_1_0.dll",
        "//conditions:default": "lib/libtensorrt_rtx.so",
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
