package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvinfer_headers",
    hdrs = glob([
        "include/NvInfer*.h",
    ], exclude=[
        "include/NvInferPlugin.h",
        "include/NvInferPluginUtils.h"
    ]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = "lib/x86_64-linux-gnu/libnvinfer.so",
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
    shared_library = "lib/x86_64-linux-gnu/libnvparsers.so",
    visibility = ["//visibility:private"],
)


cc_library(
    name = "nvparsers_headers",
    hdrs = [
        "include/NvCaffeParser.h",
        "include/NvOnnxParser.h",
        "include/NvOnnxParserRuntime.h",
        "include/NvOnnxConfig.h",
        "include/NvUffParser.h"
    ],
    includes = ["include/"],
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
    shared_library = "lib/x86_64-linux-gnu/libnvonnxparser.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_headers",
    hdrs = [
        "include/NvOnnxParser.h",
        "include/NvOnnxParserRuntime.h",
        "include/NvOnnxConfig.h"
    ],
    includes = ["include/"],
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
    shared_library = "lib/x86_64-linux-gnu/libnvonnxparser_runtime.so",
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
    shared_library = "lib/x86_64-linux-gnu/libnvcaffe_parsers.so",
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
    shared_library = "lib/x86_64-linux-gnu/libnvinfer_plugin.so",
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
    deps = [
        "nvinferplugin_headers",
        "nvinferplugin_lib",
        "nvinfer"
    ],
    visibility = ["//visibility:public"],
)
