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
            "include/x86_64-linux-gnu/NvUtils.h",
        ] + glob(
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
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
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
    deps = [
        "nvinfer_headers",
        "nvinfer_lib",
        "@cuda//:cudart",
        "@cuda//:cublas",
        "@cudnn//:cudnn"
    ],
    visibility = ["//visibility:public"],
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
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvCaffeParser.h",
            "include/aarch64-linux-gnu/NvOnnxParser.h",
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
            "include/aarch64-linux-gnu/NvOnnxConfig.h",
            "include/aarch64-linux-gnu/NvUffParser.h"
        ],
        "//conditions:default": [
            "include/x86_64-linux-gnu/NvCaffeParser.h",
            "include/x86_64-linux-gnu/NvOnnxParser.h",
            "include/x86_64-linux-gnu/NvOnnxParserRuntime.h",
            "include/x86_64-linux-gnu/NvOnnxConfig.h",
            "include/x86_64-linux-gnu/NvUffParser.h"
        ]
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvparsers",
    deps = [
        "nvparsers_headers",
        "nvparsers_lib",
        "nvinfer"
    ],
    visibility = ["//visibility:public"],
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
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvOnnxParser.h",
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
            "include/aarch64-linux-gnu/NvOnnxConfig.h"
        ],
        "//conditions:default": [
            "include/x86_64-linux-gnu/NvOnnxParser.h",
            "include/x86_64-linux-gnu/NvOnnxParserRuntime.h",
            "include/x86_64-linux-gnu/NvOnnxConfig.h"
        ]
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser",
    deps = [
        "nvonnxparser_headers",
        "nvonnxparser_lib",
        "nvinfer"
    ],
    visibility = ["//visibility:public"],
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
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
        ],
        "//conditions:default": [
            "include/x86_64-linux-gnu/NvOnnxParserRuntime.h",
        ]
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_runtime",
    deps = [
        "nvparsers_headers",
        "nvparsers_lib",
        "nvinfer"
    ],
    visibility = ["//visibility:public"],
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
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvCaffeParser.h",
        ],
        "//conditions:default": [
            "include/x86_64-linux-gnu/NvCaffeParser.h",
        ]
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser",
    deps = [
        "nvcaffeparser_headers",
        "nvcaffeparser_lib",
        "nvinfer"
    ],
    visibility = ["//visibility:public"],
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
    hdrs = select({
        ":aarch64_linux": glob(["include/aarch64-linux-gnu/NvInferPlugin*.h"]),
        "//conditions:default": glob(["include/x86_64-linux-gnu/NvInferPlugin*.h"])
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        "//conditions:default": ["include/x86_64-linux-gnu/"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinferplugin",
    deps = [
        "nvinferplugin_headers",
        "nvinferplugin_lib",
        "nvinfer"
    ],
    visibility = ["//visibility:public"],
)
