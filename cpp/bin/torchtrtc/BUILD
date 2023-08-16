load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_binary(
    name = "torchtrtc",
    srcs = [
        "accuracy.cpp",
        "accuracy.h",
        "fileio.cpp",
        "fileio.h",
        "luts.h",
        "main.cpp",
        "parser_util.cpp",
        "parser_util.h",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        "//cpp:torch_tensorrt",
        "//third_party/args",
    ] + select({
        ":use_pre_cxx11_abi": [
            "@libtorch_pre_cxx11_abi//:caffe2",
            "@libtorch_pre_cxx11_abi//:libtorch",
        ],
        "//conditions:default": [
            "@libtorch",
            "@libtorch//:caffe2",
        ],
    }),
)
