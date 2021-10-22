package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "util",
    srcs = [
        "evaluate_graph.cpp",
        "run_forward.cpp",
        "run_graph.cpp",
        "run_graph_engine.cpp",
        "util.cpp",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
    ] + select({
        ":use_pre_cxx11_abi": [
            "@libtorch_pre_cxx11_abi//:libtorch",
            "@libtorch_pre_cxx11_abi//:caffe2",
        ],
        "//conditions:default": [
            "@libtorch//:libtorch",
            "@libtorch//:caffe2",
        ],
    }) + select({
        "//tests:ci_build_testing": [
            "@torch_tensorrt//:torch_tensorrt",
            "@torch_tensorrt//:torch_tensorrt_core_hdrs"
        ],
        "//conditions:default": [
            "//cpp:torch_tensorrt",
            "//core/conversion",
            "//core/util:prelude"
        ]
    }),
)
