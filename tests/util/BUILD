package(default_visibility = ["//visibility:public"])

cc_library(
    name = "util",
    hdrs = [
        "util.h",
    ],
    srcs = [
        "util.cpp",
        "run_graph.cpp",
        "run_graph_engine.cpp",
        "run_forward.cpp"
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "@libtorch//:libtorch",
        "//core/conversion",
        "//core/util:prelude",
        "//cpp/api:trtorch",
    ]
)
