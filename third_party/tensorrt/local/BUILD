package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

cc_library(
    name = "nvinfer_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvUtils.h",
        ] + glob(
            [
                "include/aarch64-linux-gnu/NvInfer*.h",
            ],
            exclude = [
                "include/aarch64-linux-gnu/NvInferPlugin.h",
                "include/aarch64-linux-gnu/NvInferPluginUtils.h",
            ],
        ),
        "//conditions:default": [
            "include/NvUtils.h",
        ] + glob(
            [
                "include/NvInfer*.h",
            ],
            exclude = [
                "include/NvInferPlugin.h",
                "include/NvInferPluginUtils.h",
            ],
        ),
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/"],
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvinfer.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libnvinfer.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinfer",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer_headers",
        "nvinfer_lib",
        "@cublas",
        "@cuda//:cudart",
        "@cudnn",
    ],
)

####################################################################################

cc_import(
    name = "nvparsers_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvparsers.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libnvparsers.so",
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
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvonnxparser.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libnvonnxparser.so",
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
        ":aarch64_linux": "lib/x86_64-linux-gnu/libnvonnxparser_runtime.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libnvonnxparser_runtime.so",
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
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvcaffe_parsers.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libnvcaffe_parsers.so",
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

cc_import(
    name = "nvinferplugin_lib",
    shared_library = select({
        ":aarch64_linux": "lib/x86_64-linux-gnu/libnvinfer_plugin.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libnvinfer_plugin.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinferplugin_headers",
    hdrs = glob(["include/NvInferPlugin*.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinferplugin",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer",
        "nvinferplugin_headers",
        "nvinferplugin_lib",
    ],
)
