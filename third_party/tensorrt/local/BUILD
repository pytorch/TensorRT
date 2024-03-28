load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
package(default_visibility = ["//visibility:public"])
config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)
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
        ":aarch64_linux": [
        ] + glob(
            [
                "include/aarch64-linux-gnu/NvInfer*.h",
            ],
            exclude = [
                "include/aarch64-linux-gnu/NvInferPlugin.h",
                "include/aarch64-linux-gnu/NvInferPluginUtils.h",
            ],
        ),
        ":ci_rhel_x86_64_linux": [
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
        ":windows": [
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
        "//conditions:default": [
        ] + glob(
            [
	        "x86_64-linux-gnu/include/NvInfer.h",
                "x86_64-linux-gnu/include/NvInfer*.h",
            ],
            exclude = [
                "x86_64-linux-gnu/include/NvInferPlugin.h",
                "x86_64-linux-gnu/include/NvInferPluginUtils.h",
            ],
        ),
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["x86_64-linux-gnu/include/"],
    }),
    visibility = ["//visibility:private"],
)
cc_import(
    name = "nvinfer_static_lib",
    static_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvinfer_static.a",
        ":ci_rhel_x86_64_linux": "lib64/libnvinfer_static.a",
        ":windows": "lib/nvinfer.lib",
        "//conditions:default": "x86_64-linux-gnu/lib/libnvinfer_static.a",
    }),
    visibility = ["//visibility:private"],
)
cc_import(
    name = "nvinfer_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvinfer.so",
        ":ci_rhel_x86_64_linux": "lib64/libnvinfer.so",
        ":windows": "lib/nvinfer.dll",
        "//conditions:default": "x86_64-linux-gnu/lib/libnvinfer.so",
    }),
    visibility = ["//visibility:private"],
)
cc_library(
    name = "nvinfer",
    visibility = ["//visibility:public"],
    deps = [
        "nvinfer_headers",
        "nvinfer_lib",
        "@cuda//:cudart",
        "@cudnn",
    ],
)
####################################################################################
cc_import(
    name = "nvparsers_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libnvparsers.so",
        ":ci_rhel_x86_64_linux": "lib64/libnvparsers.so",
        ":windows": "lib/nvparsers.dll",
        "//conditions:default": "x86_64-linux-gnu/lib/libnvparsers.so",
    }),
    visibility = ["//visibility:private"],
)
cc_library(
    name = "nvparsers_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvCaffeParser.h",
            "include/aarch64-linux-gnu/NvOnnxConfig.h",
            "include/aarch64-linux-gnu/NvOnnxParser.h",
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
            "include/aarch64-linux-gnu/NvUffParser.h",
        ],
        ":ci_rhel_x86_64_linux": [
            "include/NvCaffeParser.h",
            "include/NvOnnxConfig.h",
            "include/NvOnnxParser.h",
            "include/NvOnnxParserRuntime.h",
            "include/NvUffParser.h",
        ],
        ":windows": [
            "include/NvCaffeParser.h",
            "include/NvOnnxConfig.h",
            "include/NvOnnxParser.h",
            "include/NvOnnxParserRuntime.h",
            "include/NvUffParser.h",
        ],
        "//conditions:default": [
            "x86_64-linux-gnu/include/NvCaffeParser.h",
            "x86_64-linux-gnu/include/NvOnnxConfig.h",
            "x86_64-linux-gnu/include/NvOnnxParser.h",
            "x86_64-linux-gnu/include/NvOnnxParserRuntime.h",
            "x86_64-linux-gnu/include/NvUffParser.h",
        ],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["x86_64-linux-gnu/include/"],
    }),
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
        ":ci_rhel_x86_64_linux": "lib64/libnvonnxparser.so",
        ":windows": "lib/nvonnxparser.dll",
        "//conditions:default": "x86_64-linux-gnu/lib/libnvonnxparser.so",
    }),
    visibility = ["//visibility:private"],
)
cc_library(
    name = "nvonnxparser_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvOnnxConfig.h",
            "include/aarch64-linux-gnu/NvOnnxParser.h",
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
        ],
        ":ci_rhel_x86_64_linux": [
            "include/NvOnnxConfig.h",
            "include/NvOnnxParser.h",
            "include/NvOnnxParserRuntime.h",
        ],
        ":windows": [
            "include/NvOnnxConfig.h",
            "include/NvOnnxParser.h",
            "include/NvOnnxParserRuntime.h",
        ],
        "//conditions:default": [
            "x86_64-linux-gnu/include/NvOnnxConfig.h",
            "x86_64-linux-gnu/include/NvOnnxParser.h",
            "x86_64-linux-gnu/include/NvOnnxParserRuntime.h",
        ],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["x86_64-linux-gnu/include/"],
    }),
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
        ":aarch64_linux": "x86_64-linux-gnu/lib/libnvonnxparser_runtime.so",
        ":ci_rhel_x86_64_linux": "lib64/libnvonnxparser_runtime.so",
        ":windows": "lib/nvonnxparser_runtime.dll",
        "//conditions:default": "x86_64-linux-gnu/lib/libnvonnxparser_runtime.so",
    }),
    visibility = ["//visibility:public"],
)
cc_library(
    name = "nvonnxparser_runtime_header",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
        ],
        ":ci_rhel_x86_64_linux": [
            "include/NvOnnxParserRuntime.h",
        ],
        ":windows": [
            "include/NvOnnxParserRuntime.h",
        ],
        "//conditions:default": [
            "x86_64-linux-gnu/include/NvOnnxParserRuntime.h",
        ],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["x86_64-linux-gnu/include/"],
    }),
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
        ":ci_rhel_x86_64_linux": "lib64/libnvcaffe_parsers.so",
        ":windows": "lib/nvcaffe_parsers.dll",
        "//conditions:default": "x86_64-linux-gnu/lib/libnvcaffe_parsers.so",
    }),
    visibility = ["//visibility:private"],
)
cc_library(
    name = "nvcaffeparser_headers",
    hdrs = select({
        ":aarch64_linux": [
            "include/aarch64-linux-gnu/NvCaffeParser.h",
        ],
        ":ci_rhel_x86_64_linux": [
            "include/NvOnnxParserRuntime.h",
        ],
        ":windows": [
            "include/NvCaffeParser.h",
        ],
        "//conditions:default": [
            "x86_64-linux-gnu/include/NvCaffeParser.h",
        ],
    }),
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu"],
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["x86_64-linux-gnu/include/"],
    }),
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
        ":aarch64_linux": ["lib/aarch64-linux-gnu/libnvinfer_plugin.so"],
        ":ci_rhel_x86_64_linux": ["lib64/libnvinfer_plugin.so"],
        ":windows": ["lib/nvinfer_plugin.dll"],
        "//conditions:default": ["x86_64-linux-gnu/lib/libnvinfer_plugin.so"],
    }),
    hdrs = select({
        ":aarch64_linux": glob(["include/aarch64-linux-gnu/NvInferPlugin*.h"]),
        ":ci_rhel_x86_64_linux": glob(["include/NvInferPlugin*.h"]),
        ":windows": glob(["include/NvInferPlugin*.h"]),
        "//conditions:default": glob(["x86_64-linux-gnu/include/NvInferPlugin*.h"]),
    }),
    copts = [
        "-pthread",
    ],
    includes = select({
        ":aarch64_linux": ["include/aarch64-linux-gnu/"],
        ":ci_rhel_x86_64_linux": ["include/"],
        ":windows": ["include/"],
        "//conditions:default": ["x86_64-linux-gnu/include/"],
    }),
    linkopts = [
        "-lpthread",
    ] + select({
        ":aarch64_linux": ["-Wl,--no-as-needed -ldl -lrt -Wl,--as-needed"],
        "//conditions:default": [],
    }),
    deps = [
        "nvinfer",
        "@cuda//:cudart",
        "@cudnn",
    ],
    alwayslink = True,
)