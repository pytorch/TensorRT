package(default_visibility = ["//visibility:public"])

config_setting(
    name="aarch64",
    values={
        "cpu":"aarch64"
    },
)
config_setting(
    name="x86_64",
    values={
        "cpu":"x86_64"
    },
)

config_setting(
    name = "aarch64_linux",
    values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
)

cc_library(
    name = "nvinfer_headers",
    hdrs = [
        "include/NvUtils.h"
    ] + glob([
        "include/NvInfer*.h",
    ], exclude=["include/NvInferPlugin.h", "include/NvInferPluginUtils.h"] ),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = "lib/libnvinfer.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinfer_headers_aarch64",
    hdrs = [
        "include/aarch64-linux-gnu/NvUtils.h"
    ] + glob([
        "include/aarch64-linux-gnu/NvInfer*.h",
    ], exclude=["include/aarch64-linux-gnu/NvInferPlugin.h", "include/aarch64-linux-gnu/NvInferPluginUtils.h"] ),
    includes = ["include/aarch64-linux-gnu"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib_aarch64",
    shared_library = "lib/aarch64-linux-gnu/libnvinfer.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinfer",
    deps = [
    ] + select({
        ":aarch64": [
            ":nvinfer_headers_aarch64",
            ":nvinfer_lib_aarch64",
            "@cuda_aarch64//:cudart",
            "@cuda_aarch64//:cublas",
            "@cudnn_aarch64//:cudnn"
        ],
        "//conditions:default": [
            ":nvinfer_headers",
            ":nvinfer_lib",
            "@cuda//:cudart",
            "@cuda//:cublas",
            "@cudnn//:cudnn"
        ]
    }),
    visibility = ["//visibility:public"],
)
