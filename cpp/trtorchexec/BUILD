package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "trtorchexec",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "//cpp/api:trtorch"
    ],
)
