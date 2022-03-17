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
        "accuracy.h",
        "accuracy.cpp",
        "fileio.h",
        "fileio.cpp",
        "luts.h",
        "main.cpp",
        "parser_util.h",
        "parser_util.cpp"
    ],
    deps = [
        "//third_party/args",
        "//cpp:torch_tensorrt",
    ] + select({
        ":use_pre_cxx11_abi": [
            "@libtorch_pre_cxx11_abi//:libtorch",
            "@libtorch_pre_cxx11_abi//:caffe2",
        ],
        "//conditions:default": [
            "@libtorch//:libtorch",
            "@libtorch//:caffe2",
        ],
    }),
)
