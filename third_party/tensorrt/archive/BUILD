load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "@//toolchains/dep_collection:compute_libs": "jetpack",
    },
)

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
        ":windows": "lib/nvinfer_10.dll",
        "//conditions:default": "lib/libnvinfer.so",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_static_lib",
    static_library = select({
        ":windows": "lib/nvinfer_10.lib",
        "//conditions:default": "lib/libnvinfer_static.a",
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
        ":jetpack": [
            "nvinfer_static_lib",
            "@cuda_l4t//:cudart",
        ],
        ":windows": [
            "nvinfer_static_lib",
            "@cuda_win//:cudart",
        ],
        "//conditions:default": ["@cuda//:cudart"],
    }),
)

####################################################################################

cc_import(
    name = "nvparsers_lib",
    shared_library = select({
        ":windows": "lib/nvparsers.dll",
        "//conditions:default": "lib/libnvparsers.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvparsers_headers",
    hdrs = [
        "include/NvCaffeParser.h",
        "include/NvOnnxConfig.h",
        "include/NvOnnxParser.h",
        "include/NvOnnxParserRuntime.h",
        "include/NvUffParser.h",
    ],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvparsers",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvparsers_headers",
        "nvparsers_lib",
    ],
)

####################################################################################

cc_import(
    name = "nvonnxparser_lib",
    shared_library = select({
        ":windows": "lib/nvonnxparser*.dll",
        "//conditions:default": "lib/libnvonnxparser*.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_headers",
    hdrs = [
        "include/NvOnnxConfig.h",
        "include/NvOnnxParser.h",
        "include/NvOnnxParserRuntime.h",
    ],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvonnxparser_headers",
        "nvonnxparser_lib",
    ],
)

####################################################################################

cc_import(
    name = "nvonnxparser_runtime_lib",
    shared_library = select({
        ":windows": "lib/nvonnxparser_runtime.dll",
        "//conditions:default": "lib/libnvonnxparser_runtime.so",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvonnxparser_runtime_header",
    hdrs = ["include/NvOnnxParserRuntime.h"],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_runtime",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvparsers_headers",
        "nvparsers_lib",
    ],
)

####################################################################################

cc_import(
    name = "nvcaffeparser_lib",
    shared_library = select({
        ":windows": "lib/nvcaffe_parsers.dll",
        "//conditions:default": "lib/libnvcaffe_parsers.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser_headers",
    hdrs = ["include/NvCaffeParser.h"],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser",
    visibility = ["//visibility:public"],
    deps = [
        "nvcaffeparser_headers",
        "nvcaffeparser_lib",
        "nvinfer",
    ],
)

####################################################################################

cc_library(
    name = "nvinferplugin",
    srcs = select({
        ":windows": ["lib/nvinfer_plugin_10.lib"],
        "//conditions:default": ["lib/libnvinfer_plugin.so"],
    }),
    hdrs = glob(
        ["include/NvInferPlugin*.h"],
        allow_empty = True,
    ),
    copts = ["-pthread"],
    includes = ["include/"],
    linkopts = ["-lpthread"],
    deps = [
        "nvinfer",
    ] + select({
        ":jetpack": ["@cuda_l4t//:cudart"],
        ":windows": ["@cuda_win//:cudart"],
        "//conditions:default": ["@cuda//:cudart"],
    }),
    alwayslink = True,
)
