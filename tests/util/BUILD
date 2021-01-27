package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "util",
    hdrs = [
        "util.h",
    ],
    srcs = [
        "evaluate_graph.cpp",
        "run_graph.cpp",
        "run_graph_engine.cpp",
        "run_forward.cpp",
        "util.cpp",
    ],
    deps = [
        "//core/conversion",
        "//core/util:prelude",
        "//cpp/api:trtorch",
        "@tensorrt//:nvinfer"
    ] + select({
        ":use_pre_cxx11_abi":  [
            "@libtorch_pre_cxx11_abi//:libtorch",
            "@libtorch_pre_cxx11_abi//:caffe2",
        ],
        "//conditions:default":  [
            "@libtorch//:libtorch",
            "@libtorch//:caffe2",
        ],
    })
)
