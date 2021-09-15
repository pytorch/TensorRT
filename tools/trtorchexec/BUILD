package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "trtorchexec",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//cpp/api:trtorch",
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
